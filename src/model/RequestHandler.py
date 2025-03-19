import torchvision.transforms.functional as TF
from model.TensorTools import *
from model.PlottingTools import *
from model.CrossValidation import *
from PIL import Image
import numpy as np
from model.SegmentationAnalyzer import SegmentationAnalyzer
from shared.ScaleInfo import ScaleInfo
from model.ModelEvaluator import ModelEvaluator

class request_handler:
    def __init__(self, unet):
        self.unet = unet

    def process_request_train(self, images_path, masks_path):  
        # CHANGE CROSS VALIDATION HERE (uncomment):
        #cv_holdout(self.unet, images_path, masks_path)
        cv_kfold(self.unet, images_path, masks_path)
        
        return None


    def process_request_segment(self, image_path):
        image = Image.open(image_path).convert("L")
        image = image.resize((256,256), Image.NEAREST)
        image = TF.to_tensor(image).unsqueeze(0)
        segmentation = self.unet.segment(image)
        from model.TensorTools import segmentation_to_image
        segmentation_image = segmentation_to_image(segmentation)
        
        segmentation_numpy = (segmentation.squeeze(0).numpy() * 255).astype(np.uint8)
        print(segmentation_numpy.shape)
        analyzer = SegmentationAnalyzer()
        num_labels, labels, stats, centroids = analyzer.get_connected_components(segmentation_numpy)
        annotated_image = analyzer.add_annotations(segmentation_numpy, centroids)
        image_pil = Image.fromarray(annotated_image)
        return image_pil
    
    def process_request_load_model(self, model_path):
        self.unet.load_model(model_path)
        return None
        
    def process_request_calculate_image_width(self, scale_info: ScaleInfo):
        scaled_length = float(np.abs(scale_info.end_x- scale_info.start_x))
        real_length = float(scale_info.real_scale_length)
        input_image_real_width = real_length / scaled_length * scale_info.image_width
        return input_image_real_width
    
    def process_request_test_model(self, test_data_image_dir, test_data_mask_dir):
        dataset = SegmentationDataset(test_data_image_dir, test_data_mask_dir)
        test_dataloader = DataLoader(dataset, batch_size=1)
        iou, pixel_accuracy = ModelEvaluator.evaluate_model(self.unet, test_dataloader)
        print(iou)
        print(pixel_accuracy)
        return iou, pixel_accuracy