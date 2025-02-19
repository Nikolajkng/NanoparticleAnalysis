from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms.functional as TF


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = os.listdir(image_dir)
        self.mask_filenames = os.listdir(mask_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.image_filenames[index])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[index])

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        return image, mask
