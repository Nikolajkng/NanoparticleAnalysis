from PIL import Image
import numpy as np

from src.model.DataTools import *
from src.model.PlottingTools import *
from src.model.CrossValidation import *
from src.model.SegmentationAnalyzer import SegmentationAnalyzer
from src.model.ModelEvaluator import ModelEvaluator
from src.shared.ModelConfig import ModelConfig
from src.shared.ParticleImage import ParticleImage
class request_handler:
    def __init__(self, unet):
        self.unet = unet

    def process_request_train(self, model_config: ModelConfig, stop_training_event = None, loss_callback = None, test_callback = None):  
        # CHANGE CROSS VALIDATION HERE (uncomment):
        self.unet = UNet()
        iou, pixel_accuracy = cv_holdout(self.unet, model_config, self.unet.preffered_input_size, stop_training_event, loss_callback, test_callback)
        #cv_kfold(self.unet, images_path, masks_path)
        print(f"Model IOU: {iou}\nModel Pixel Accuracy: {pixel_accuracy}")
        return iou, pixel_accuracy


    def process_request_segment(self, image: ParticleImage, output_folder):
        tensor = TF.to_tensor(image.pil_image).unsqueeze(0)
        stride_length = self.unet.preffered_input_size[0]*4//5
        tensor_mirror_filled = mirror_fill(tensor, self.unet.preffered_input_size, (stride_length,stride_length))
        patches = extract_slices(tensor_mirror_filled, self.unet.preffered_input_size, (stride_length,stride_length))

        segmentations = np.empty((patches.shape[0], 2, patches.shape[2], patches.shape[3]), dtype=patches.dtype)

        self.unet.eval()
        self.unet.to(tensor.device)
        patches_tensor = torch.tensor(patches, dtype=tensor.dtype, device=tensor.device)
        with torch.no_grad():
            segmentations = self.unet(patches_tensor).detach().numpy()

            
        segmented_image = construct_image_from_patches(segmentations, tensor_mirror_filled.shape[2:], (stride_length,stride_length))
        segmented_image = center_crop(segmented_image, (tensor.shape[2], tensor.shape[3])).argmax(axis=1)
        segmented_image_2d = to_2d_image_array(segmented_image)
        
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
    
    def process_request_test_model(self, test_data_image_dir, test_data_mask_dir):
        dataset = SegmentationDataset(test_data_image_dir, test_data_mask_dir)
        test_dataloader = DataLoader(dataset, batch_size=1)
        iou, pixel_accuracy = ModelEvaluator.evaluate_model(self.unet, test_dataloader)
        print(iou)
        print(pixel_accuracy)
        return iou, pixel_accuracy
        
    def process_request_segment_folder(self, input_folder, output_parent_folder):
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
        image = ParticleImage(image_path)
        image.file_info = image.get_file_info(image_path)
        return image