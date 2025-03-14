import torchvision.transforms.functional as TF
from model.TensorTools import *
from model.PlottingTools import *
from model.CrossValidation import *
from PIL import Image
import numpy as np
from model.SegmentationAnalyzer import SegmentationAnalyzer
from shared.ScaleInfo import ScaleInfo

class request_handler:
    def __init__(self, unet):
        self.unet = unet

    def process_request_train(self, images_path, masks_path):
        try:
            
            # CHANGE CROSS VALIDATION HERE (uncomment):
            #cv_holdout(self.unet, images_path, masks_path)
            cv_kfold(self.unet, images_path, masks_path)
            
            return (None, 0)
        except Exception as e:
            return (e, 1)

    def process_request_segment(self, image_path):
        try:
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
            return (image_pil, 0)
        except Exception as e:
            return (e, 1)

    def process_request_load_model(self, model_path):
        try:
            self.unet.load_model(model_path)
            return (None, 0)
        except Exception as e:
            return (e, 1)
        
    def process_request_calculate_image_width(self, scale_info: ScaleInfo):
        try:
            scaled_length = float(np.abs(scale_info.end_x- scale_info.start_x))
            real_length = float(scale_info.real_scale_length)
            input_image_real_width = real_length / scaled_length * scale_info.image_width
            return (input_image_real_width, 1)
        except Exception as e:
            return (e, 1)