from torch import Tensor
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import rotate, hflip
from SegmentationDataset import SegmentationDataset
from TensorTools import normalizeTensorToPixels
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
    
    def create_hflipped_tensors(self, images: list[Tensor], masks: list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        final_flipped_images = []
        final_flipped_masks = []
        for image, mask in zip(images, masks):
            final_flipped_images.extend([image, hflip(image)])
            final_flipped_masks.extend([mask, hflip(mask)])
        return (final_flipped_images, final_flipped_masks)


if __name__ == '__main__':
    data_augmenter = DataAugmenter()
    dataset = SegmentationDataset("data/images/", "data/masks/")
    image, mask = dataset[0]
    rotated_images, rotated_masks = data_augmenter.create_rotated_tensors(image, mask)
    augmented_images, augmented_masks = data_augmenter.create_hflipped_tensors(rotated_images, rotated_masks)
    for image in augmented_images:
        pixels = normalizeTensorToPixels(image)
        img = TF.to_pil_image(pixels.byte())
        img.show()

