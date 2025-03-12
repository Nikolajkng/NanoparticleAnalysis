import torchvision.transforms.functional as TF
from model.TensorTools import *
from model.PlottingTools import *
from model.CrossValidation import *
from PIL import Image

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
            output = self.unet.segment(image)
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