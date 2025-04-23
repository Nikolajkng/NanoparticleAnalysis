from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms.functional as TF


class SegmentationDataset(Dataset):
    def __init__(self, image_dir=None, mask_dir=None, transform=None):

        self.images = []
        self.masks = []
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        if not image_dir or not mask_dir:
            return
        self.image_filenames = os.listdir(image_dir)
        self.mask_filenames = os.listdir(mask_dir)
        self.transform = transform
        
        for index in range(len(self.image_filenames)):
            img_path = os.path.join(self.image_dir, self.image_filenames[index])
            mask_path = os.path.join(self.mask_dir, self.mask_filenames[index])

            image = Image.open(img_path).convert("L")
            mask = Image.open(mask_path).convert("L")

            image = TF.to_tensor(image)
            mask = TF.to_tensor(mask)

            self.images.append(image)
            self.masks.append(mask)

    @classmethod
    def from_image_set(cls, images, masks):
        res = cls()
        res.images = images
        res.masks = masks
        return res

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return self.images[index], self.masks[index]