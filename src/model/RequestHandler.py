from PIL import Image
import torchvision.transforms.functional as TF
from model.TensorTools import *
from model.PlottingTools import *
from src.model.CrossValidation import *

class request_handler:
    def __init__(self, unet):
        self.unet = unet

    def process_request_train(self, images_path, masks_path):
        try:
            
            # CHANGE CROSS VALIDATION HERE:
            train_model_holdout(self.unet, images_path, masks_path)
            # train_model_kfold(self.unet, images_path, masks_path)
            
            return (None, 0)
        except Exception as e:
            return (e, 1)

    def process_request_segment(self, image_path):
        try:
            image = Image.open(image_path).convert("L")
            image = TF.to_tensor(image).unsqueeze(0)
            output = self.unet(image)
            from model.TensorTools import segmentation_to_image
            segmentation = segmentation_to_image(output)
            return (segmentation, 0)
        except Exception as e:
            return (e, 1)

    def process_request_load_model(self, model_path):
        try:
            self.unet.load_model(model_path)
            return (None, 0)
        except Exception as e:
            return (e, 1)