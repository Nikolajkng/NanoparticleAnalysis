import random
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as transforms
import numpy as np
import cv2

class RepeatDataset(Dataset):
    def __init__(self, dataset, repeat_factor):
        self.dataset = dataset
        self.repeat_factor = repeat_factor

    def __len__(self):
        return len(self.dataset) * self.repeat_factor

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]

class SegmentationDataset(Dataset):
    def __init__(self, image_dir=None, mask_dir=None, transform=None):

        self.images = []
        self.masks = []
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        if not image_dir or not mask_dir:
            return
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
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
    def from_image_set(cls, images, masks, transforms=None):
        res = cls()
        res.images = images
        res.masks = masks
        res.transform = transforms
        return res

    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        if self.transform:
            image, mask = self.transform(self.images[index], self.masks[index])
        else:
            image = self.images[index]
            mask = self.masks[index]
        
        return image, mask
    
    @staticmethod
    def apply_clahe(image):
        img = np.array(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(img)
        return equalized