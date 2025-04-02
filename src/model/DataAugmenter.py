from torch import Tensor
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import rotate, hflip
from model.SegmentationDataset import SegmentationDataset
from model.TensorTools import normalizeTensorToPixels
import numpy as np
class DataAugmenter():
    def __init__(self):
        return
        
    def create_rotated_tensors(self, image: Tensor, mask: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        rotated_images = [image]
        rotated_masks = [mask]
        for _ in range(3):
            new_image, new_mask = rotate(rotated_images[-1], 90), rotate(rotated_masks[-1], 90)
            rotated_images.append(new_image)
            rotated_masks.append(new_mask)
        return (rotated_images, rotated_masks)
    
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
        final_flipped_images = []
        final_flipped_masks = []
        for image, mask in zip(images, masks):
            final_flipped_images.extend([image, hflip(image)])
            final_flipped_masks.extend([mask, hflip(mask)])
        return (final_flipped_images, final_flipped_masks)
    
    def augment_dataset(self, dataset: SegmentationDataset) -> SegmentationDataset:
        new_images = []
        new_masks = []
        for image, mask in zip(dataset.images, dataset.masks):
            rotated_images, rotated_masks = self.create_rotated_tensors(image, mask)
            augmented_images, augmented_masks = self.create_hflipped_tensors(rotated_images, rotated_masks)
            new_images.extend(augmented_images)
            new_masks.extend(augmented_masks)
        dataset.images = new_images
        dataset.masks = new_masks
        return dataset



if __name__ == '__main__':
    data_augmenter = DataAugmenter()
    dataset = SegmentationDataset("data/images/", "data/masks/")
    image, mask = dataset[0]
    rotated_images, rotated_masks = data_augmenter.create_rotated_tensors(image, mask)
    crop_image, crop_mask = data_augmenter.random_crop(image, mask, cropped_size=(50,50))
    pixels = normalizeTensorToPixels(crop_image)
    img = TF.to_pil_image(pixels.byte())
    img.show()
    augmented_images, augmented_masks = data_augmenter.create_hflipped_tensors(rotated_images, rotated_masks)
    for image in augmented_images:
        pixels = normalizeTensorToPixels(image)
        img = TF.to_pil_image(pixels.byte())
        img.show()

