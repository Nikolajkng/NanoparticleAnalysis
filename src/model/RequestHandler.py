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

class request_handler:
    def __init__(self, unet):
        self.unet = unet

    def process_request_train(self, model_config: ModelConfig, loss_callback=None):  
        # CHANGE CROSS VALIDATION HERE (uncomment):
        iou, pixel_accuracy = cv_holdout(self.unet, model_config, loss_callback)
        #cv_kfold(self.unet, images_path, masks_path)
        
        return iou, pixel_accuracy


    def process_request_segment(self, image_path, scale_info: ScaleInfo):
        analyzer = SegmentationAnalyzer()
        input = tensor_from_image(image_path)
        segmentation = self.unet.segment(input)
        segmentation_numpy = segmentation_tensor_to_numpy(segmentation)
        num_labels, labels, stats, centroids = analyzer.get_connected_components(segmentation_numpy)
        table_data = analyzer.format_table_data(stats, scale_info)
        annotated_image = analyzer.add_annotations(segmentation_numpy, centroids)
        image_pil = Image.fromarray(annotated_image)
        return image_pil, table_data
    
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