import torchvision.transforms.functional as TF
from model.TensorTools import *
from model.DataTools import *
from model.PlottingTools import *
from model.CrossValidation import *
from PIL import Image
import numpy as np
from model.SegmentationAnalyzer import SegmentationAnalyzer
from shared.ScaleInfo import ScaleInfo
from model.ModelEvaluator import ModelEvaluator
from shared.ModelConfig import ModelConfig
from model.dmFileReader import dmFileReader
from shared.IOFunctions import is_dm_format
class request_handler:
    def __init__(self, unet):
        self.unet = unet

    def process_request_train(self, model_config: ModelConfig, loss_callback=None):  
        # CHANGE CROSS VALIDATION HERE (uncomment):
        iou, pixel_accuracy = cv_holdout(self.unet, model_config, self.unet.preffered_input_size, loss_callback)
        #cv_kfold(self.unet, images_path, masks_path)
        print(f"Model IOU: {iou}\nModel Pixel Accuracy: {pixel_accuracy}")
        return iou, pixel_accuracy


    def process_request_segment(self, image_path, scale_info: ScaleInfo):
        analyzer = SegmentationAnalyzer()
        reader = dmFileReader()
        tensor = None
        if is_dm_format(image_path):
            tensor = reader.get_tensor_from_dm_file(image_path)
        else:
            tensor = tensor_from_image_no_resize(image_path)
        stride_length = self.unet.preffered_input_size[0]*4//5
        tensor_mirror_filled = mirror_fill(tensor, self.unet.preffered_input_size, (stride_length,stride_length))
        patches = extract_slices(tensor_mirror_filled, self.unet.preffered_input_size, (stride_length,stride_length))

        segmentations = np.empty((patches.shape[0], 2, patches.shape[2], patches.shape[3]), dtype=patches.dtype)
        patch_idx = 0
        self.unet.eval()
        for patch in patches:
            with torch.no_grad():
                segmentation = self.unet(torch.tensor(patch, dtype=tensor.dtype, device=tensor.device).unsqueeze(0))
            segmentation_numpy = segmentation.detach().numpy()
            segmentations[patch_idx] = segmentation_numpy
            patch_idx += 1
            
        segmented_image = construct_image_from_patches(segmentations, tensor_mirror_filled.shape[2:], (stride_length,stride_length))
        segmented_image = center_crop(segmented_image, (tensor.shape[2], tensor.shape[3])).argmax(axis=1)
        segmented_image_2d = to_2d_image_array(segmented_image)
        
        # Data Analysis 
        num_labels, _, stats, centroids = analyzer.get_connected_components(segmented_image_2d)
        particle_count = num_labels - 1
        annotated_image = analyzer.add_annotations(segmented_image_2d, centroids)
        annotated_image_pil = Image.fromarray(annotated_image)
        table_data = analyzer.format_table_data(stats, scale_info, particle_count)
        analyzer.write_stats_to_txt(stats, scale_info, particle_count)
        histogram = analyzer.create_histogram(stats, scale_info)
        return segmented_image_2d, annotated_image_pil, table_data
    
    def process_request_load_model(self, model_path):
        self.unet.load_model(model_path)
        return None
        
    def process_request_calculate_image_width(self, scale_info: ScaleInfo):
        scaled_length = float(np.abs(scale_info.end_x- scale_info.start_x))
        real_length = float(scale_info.real_scale_length)
        input_image_real_width = real_length / scaled_length * scale_info.image_width
        return ScaleInfo(
            0,
            scale_info.image_width,
            input_image_real_width,
            scale_info.image_width
        )
    
    def process_request_test_model(self, test_data_image_dir, test_data_mask_dir):
        dataset = SegmentationDataset(test_data_image_dir, test_data_mask_dir)
        test_dataloader = DataLoader(dataset, batch_size=1)
        iou, pixel_accuracy = ModelEvaluator.evaluate_model(self.unet, test_dataloader)
        print(iou)
        print(pixel_accuracy)
        return iou, pixel_accuracy
    
    def process_request_get_dm_image(self, file_path):
        reader = dmFileReader()
        size_info, image = reader.get_image_from_dm_file(file_path)
        return size_info, image