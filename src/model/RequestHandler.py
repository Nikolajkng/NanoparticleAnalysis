from PIL import Image
import threading
#from src.model.DataTools import *
from src.model.PlottingTools import *
#from src.model.CrossValidation import *
class request_handler:
    def __init__(self, pre_loaded_model_name=None):
        self.unet = None
        self.load_model_async(pre_loaded_model_name)

    def load_model_async(self, model_name):
        def load():
            from src.model.UNet import UNet
            self.unet = UNet()  # or UNet(pre_loaded_model_path=...)
            print("Model loaded")

        threading.Thread(target=load, daemon=True).start()
    def process_request_train(self, model_config, stop_training_event = None, loss_callback = None, test_callback = None):  
        # CHANGE CROSS VALIDATION HERE (uncomment):
        from src.model.CrossValidation import cv_holdout
        from src.model.UNet import UNet
        self.unet = UNet()
        iou, dice_score = cv_holdout(self.unet, model_config, self.unet.preferred_input_size, stop_training_event, loss_callback, test_callback)
        #cv_kfold(self.unet, images_path, masks_path)
        print(f"Model IOU: {iou}\nModel Dice Score: {dice_score}")
        return iou, dice_score


    def process_request_segment(self, image, output_folder):
        from src.model.DataTools import mirror_fill, extract_slices, construct_image_from_patches, center_crop, to_2d_image_array
        import torch
        import torchvision.transforms.functional as TF
        import numpy as np

        tensor = TF.to_tensor(image.pil_image).unsqueeze(0)
        tensor = tensor.to(self.unet.device)
        stride_length = self.unet.preferred_input_size[0]*4//5
        tensor_mirror_filled = mirror_fill(tensor, self.unet.preferred_input_size, (stride_length,stride_length))
        patches = extract_slices(tensor_mirror_filled, self.unet.preferred_input_size, (stride_length,stride_length))

        segmentations = np.empty((patches.shape[0], 2, patches.shape[2], patches.shape[3]), dtype=patches.dtype)

        self.unet.eval()
        self.unet.to(self.unet.device)
        patches_tensor = torch.tensor(patches, dtype=tensor.dtype, device=tensor.device)
        with torch.no_grad():
            if self.unet.device.type == 'cuda':
                from torch import autocast
                
                with autocast("cuda"):
                    segmentations = self.unet(patches_tensor).cpu().detach().numpy()
            else:
                segmentations = self.unet(patches_tensor).cpu().detach().numpy()


        segmented_image = construct_image_from_patches(segmentations, tensor_mirror_filled.shape[2:], (stride_length,stride_length))
        segmented_image = center_crop(segmented_image, (tensor.shape[2], tensor.shape[3])).argmax(axis=1)
        segmented_image_2d = to_2d_image_array(segmented_image)

        from src.model.SegmentationAnalyzer import SegmentationAnalyzer
        analyzer = SegmentationAnalyzer()
        num_labels, _, stats, centroids = analyzer.get_connected_components(segmented_image_2d)
        particle_count = num_labels - 1
        annotated_image = analyzer.add_annotations(segmented_image_2d, centroids)
        annotated_image_pil = Image.fromarray(annotated_image)
        segmented_image_pil = Image.fromarray(segmented_image_2d)

        
        table_data = analyzer.format_table_data(stats, image.file_info, particle_count)
        analyzer.write_stats_to_txt(stats, image.file_info, particle_count, output_folder)
        histogram_fig = analyzer.create_histogram(stats, image.file_info) 
        
        return segmented_image_pil, annotated_image_pil, table_data, histogram_fig
    
    def process_request_load_model(self, model_path):
        self.unet.load_model(model_path)
        return None
    
    def process_request_test_model(self, test_data_image_dir, test_data_mask_dir, testing_callback = None):
        from src.model.SegmentationDataset import SegmentationDataset
        from torch.utils.data import DataLoader
        dataset = SegmentationDataset(test_data_image_dir, test_data_mask_dir)
        test_dataloader = DataLoader(dataset, batch_size=1)
        from src.model.ModelEvaluator import ModelEvaluator

        iou, dice_score = ModelEvaluator.evaluate_model(self.unet, test_dataloader, testing_callback)
        print(iou)
        print(dice_score)
        return iou, dice_score
        
    def process_request_segment_folder(self, input_folder, output_parent_folder):
        import os
        for filename in os.listdir(input_folder):
            file_path = os.path.join(input_folder, filename)
            image = self.process_request_load_image(file_path)
            output_folder = f"{output_parent_folder}/{image.file_info.file_name}"
            os.makedirs(output_folder, exist_ok=True)
            segmented_image_pil, annotated_image_pil, table_data, histogram_fig = self.process_request_segment(image, output_folder)
            # Save the segmented image and annotated image
            segmented_image_pil.save(os.path.join(output_folder, f"{image.file_info.file_name}_segmented.tif"))
            annotated_image_pil.save(os.path.join(output_folder, f"{image.file_info.file_name}_annotated.tif"))
            
    def process_request_load_image(self, image_path):
        from src.shared.ParticleImage import ParticleImage
        image = ParticleImage(image_path)
        image.file_info = image.get_file_info(image_path)
        if image.pil_image.width > 1024 or image.pil_image.height > 1024:
                image.resize((1024, 1024))
        return image