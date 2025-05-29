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
            from torchvision.transforms.functional import elastic_transform, crop, rotate, hflip, adjust_brightness
            from torchvision.transforms import InterpolationMode

            # Elastic deformation
            if deform:
                size = image.shape[-1]
                params = ElasticTransform.get_params(
                    size=[512,512],
                    alpha=(20.0, 60), 
                    sigma=(4.0, 6.0)
                )
                
                image = elastic_transform(image, params)
                mask = elastic_transform(mask, params, interpolation=InterpolationMode.NEAREST)

            # Random crop
            if crop:
                i, j, h, w = RandomCrop.get_params(
                    image, output_size=(256, 256))
                image = crop(image, i, j, h, w)
                mask = crop(mask, i, j, h, w)

            # Random rotation
            if rotate:
                angle = random.randint(-30, 30)
                #angle = random.choice([0, 90, 180, 270])
                image = rotate(image, angle)
                mask = rotate(mask, angle)

            # Random horizontal flipping
            if flip and random.random() > 0.5:
                image = hflip(image)
                mask = hflip(mask)
            
            if adjust_brightness:
                brightness_factor = random.uniform(0.8, 1.2)
                adjust_brightness(image, brightness_factor)

            if blur:
                blur_transform = GaussianBlur(kernel_size=3, sigma=(0.5, 1.5))
                if random.random() < 0.5:
                    image = blur_transform(image)

            return image, mask
        return transformer

        
    def create_rotated_tensors(self, images: list[Tensor], masks: list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        from torchvision.transforms.functional import rotate

        rotated_images, rotated_masks = [], []
        for image, mask in zip(images, masks):
            internal_rotated_images = [image]
            internal_rotated_masks = [mask]
            for _ in range(3):
                new_image, new_mask = rotate(internal_rotated_images[-1], 90), rotate(internal_rotated_masks[-1], 90)
                internal_rotated_images.append(new_image)
                internal_rotated_masks.append(new_mask)
            rotated_images.extend(internal_rotated_images)
            rotated_masks.extend(internal_rotated_masks)
        return rotated_images, rotated_masks
    
    def __get_random_crop(self, image: Tensor, mask: Tensor, cropped_size=(256,256)):
        image_width, image_height = image.shape[-2:]
        start_height = np.random.randint(0, image_height - cropped_size[1])
        start_width = np.random.randint(0, image_width - cropped_size[0])
        cropped_image = image[:, start_width:start_width + cropped_size[0], start_height:start_height + cropped_size[1]]
        cropped_mask = mask[:, start_width:start_width + cropped_size[0], start_height:start_height + cropped_size[1]]
        return cropped_image, cropped_mask

    def create_random_crops(self, image: Tensor, mask: Tensor, amount_to_create: int, cropped_size=(256,256)):
        images, masks = [], []
        if image.shape[-1] <= cropped_size[1] or image.shape[-2] <= cropped_size[0]:
            return [image], [mask]
        for i in range(amount_to_create):
            cropped_image, cropped_mask = self.__get_random_crop(image, mask, cropped_size)
            images.append(cropped_image)
            masks.append(cropped_mask)
        return images, masks
    
    def create_hflipped_tensors(self, images: list[Tensor], masks: list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        from torchvision.transforms.functional import hflip

        final_flipped_images = []
        final_flipped_masks = []
        for image, mask in zip(images, masks):
            final_flipped_images.extend([image, hflip(image)])
            final_flipped_masks.extend([mask, hflip(mask)])
        return (final_flipped_images, final_flipped_masks)
    
    def augment_dataset(self, dataset: Dataset, input_size: tuple[int, int], augmentations=[True,True,False,False,False,False]) -> Dataset:
        new_images = []
        new_masks = []
        
        for i in range(len(dataset)):
            image, mask = dataset[i]  # Works for both custom and standard datasets

            # cropped_images, cropped_masks = self.create_random_crops(image, mask, 10, input_size)
            # augmented_images = cropped_images
            # augmented_masks = cropped_masks
            # if len(cropped_images) != 1: # Means we could crop the images
            #     rotated_images, rotated_masks = self.create_rotated_tensors(cropped_images, cropped_masks)
            #     flipped_images, flipped_masks = self.create_hflipped_tensors(rotated_images, rotated_masks)
            #     # augmented_images = rotated_images + flipped_images + cropped_images[4:]
            #     # augmented_masks = rotated_masks + flipped_masks + cropped_masks[4:]
            #     augmented_images, augmented_masks = flipped_images, flipped_masks
            # else: # image was correct resolution already
            #     rotated_images, rotated_masks = self.create_rotated_tensors(cropped_images, cropped_masks)
            #     augmented_images, augmented_masks = self.create_hflipped_tensors(rotated_images, rotated_masks)
            new_images.extend(image.unsqueeze(0))
            new_masks.extend(mask.unsqueeze(0))
       
       
       


        # print("Original dataset size:", len(dataset))
        # print("Augmented dataset size:", len(new_images))
        # for image in new_images:
        #     if image.shape[-1] != input_size[0] or image.shape[-2] != input_size[1]:
        #         print("Error with data augment: Final size doesn't match")

        # Wrap in a new SegmentationDataset or another compatible Dataset
        return RepeatDataset(dataset=SegmentationDataset.from_image_set(new_images, new_masks, transforms=DataAugmenter.get_transformer(*augmentations)), repeat_factor=10 if augmentations[0] else 20)

    def get_crops_for_dataset(self, dataset: Dataset, amount_to_crop, crop_size):
        new_images, new_masks = [], []
        for i in range(len(dataset)):
                image, mask = dataset[i] 
                if image.shape[-1] > 256:
                    new_imagesi, new_masksi = self.create_random_crops(image, mask, amount_to_crop, cropped_size=crop_size)
                    new_images.extend(new_imagesi)
                    new_masks.extend(new_masksi)
                else:
                    new_images.append(image)
                    new_masks.append(mask)
        return SegmentationDataset.from_image_set(new_images, new_masks)
    


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

