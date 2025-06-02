from torch import Tensor
import numpy as np
from torch.utils.data import Dataset
import random

from src.model.SegmentationDataset import RepeatDataset, SegmentationDataset
class DataAugmenter():
    def __init__(self):
        return 
    
    @staticmethod
    def get_transformer(crop: bool, rotate: bool, flip: bool, deform: bool, adjust_brightness: bool, blur: bool):
        def transformer(image, mask):
            from torchvision.transforms.v2 import ElasticTransform, RandomCrop, GaussianBlur
            import torchvision.transforms.functional as TF
            from torchvision.transforms import InterpolationMode

            # Elastic deformation
            if deform:
                size = image.shape[-1]
                params = ElasticTransform.get_params(
                    size=[512,512],
                    alpha=(20.0, 60), 
                    sigma=(4.0, 6.0)
                )
                
                image = TF.elastic_transform(image, params)
                mask = TF.elastic_transform(mask, params, interpolation=InterpolationMode.NEAREST)

            # Random crop
            if crop:
                i, j, h, w = RandomCrop.get_params(
                    image, output_size=(256, 256))
                image = TF.crop(image, i, j, h, w)
                mask = TF.crop(mask, i, j, h, w)

            # Random rotation
            if rotate:
                #angle = random.randint(-30, 30)
                angle = random.choice([0, 90, 180, 270])
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

            # Random horizontal flipping
            if flip and random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            if adjust_brightness:
                brightness_factor = random.uniform(0.8, 1.2)
                image = TF.adjust_brightness(image, brightness_factor)

            if blur:
                blur_transform = GaussianBlur(kernel_size=3, sigma=(0.5, 1.5))
                if random.random() < 0.5:
                    image = blur_transform(image)

            return image, mask
        return transformer
    
    def augment_dataset(self, dataset: Dataset, input_size: tuple[int, int], augmentations=[True,True,False,False,False,False]) -> Dataset:
        new_images = []
        new_masks = []
        
        for i in range(len(dataset)):
            image, mask = dataset[i]
            new_images.extend(image.unsqueeze(0))
            new_masks.extend(mask.unsqueeze(0))
       
    
        return RepeatDataset(dataset=SegmentationDataset.from_image_set(new_images, new_masks, transforms=DataAugmenter.get_transformer(*augmentations)), repeat_factor=10 if augmentations[0] else 20)
    


if __name__ == '__main__':
    data_augmenter = DataAugmenter()
    dataset = SegmentationDataset("data/highres_images/", "data/highres_masks/")
    dataset = data_augmenter.augment_dataset(dataset, (512, 512))
    # image, mask = dataset[0]
    # rotated_images, rotated_masks = data_augmenter.create_rotated_tensors(image, mask)
    # crop_image, crop_mask = data_augmenter.random_crop(image, mask, cropped_size=(50,50))
    # pixels = normalizeTensorToPixels(crop_image)
    # img = TF.to_pil_image(pixels.byte())
    # img.show()
    # augmented_images, augmented_masks = data_augmenter.create_hflipped_tensors(rotated_images, rotated_masks)
    
    
    # for idx, (image, mask) in enumerate(dataset):
    #     pixels = normalizeTensorToPixels(image)
    #     image = TF.to_pil_image(pixels.byte())
    #     mask_np = mask.numpy()
    #     mask_np = to_2d_image_array(mask_np)
    #     segmented_image = Image.fromarray(mask_np)
        

    #     fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    #     manager = plt.get_current_fig_manager()

    #     axes[0].imshow(image, cmap='gray')
    #     axes[0].set_title("Image")

        
    #     axes[1].imshow(segmented_image, cmap='gray')
    #     axes[1].set_title("Segmentation")
    #     manager.window.showMaximized()
    #     plt.pause(0.1)
    #     plt.tight_layout()

