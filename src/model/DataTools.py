from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import os
from PIL import Image
from torch import Tensor
import torchvision.transforms.functional as TF
import numpy as np

def get_dataloaders(dataset: Dataset, train_data_size: float, validation_data_size: float) -> tuple[DataLoader, DataLoader, DataLoader]:
    
    train_data, val_data, test_data = random_split(dataset, [train_data_size, validation_data_size, 1-train_data_size-validation_data_size])
    train_dataloader = DataLoader(train_data, batch_size=12, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=3, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=1)
    return (train_dataloader, val_dataloader, test_dataloader)


def get_dataloaders_without_testset(dataset: Dataset, train_data_size: float) -> tuple[DataLoader, DataLoader]:
    
    train_data, val_data = random_split(dataset, [train_data_size, 1-train_data_size])
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)

    return (train_dataloader, val_dataloader)


def resize_and_save_images(folder_path, output_size=(256, 256), is_masks=False):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.tif')):
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path)
            img = img.convert("L")  
            if img.width == 256 and img.height == 256:
                continue
            img_resized = img.resize(output_size, Image.NEAREST)
            if is_masks:
                img_binary = img_resized.point(lambda p: 255 if p > 20 else 0)
                img_binary.save(os.path.join(folder_path,"new"+filename))
            else:    
                img_resized.save(image_path)  # You can change this line to save it elsewhere
            print(image_path)

def tensor_from_image(image_path: str, tensor_size=(256,256)) -> Tensor:
    image = Image.open(image_path).convert("L")
    image = image.resize(tensor_size, Image.NEAREST)
    image = TF.to_tensor(image).unsqueeze(0)
    return image

def segmentation_tensor_to_numpy(tensor: Tensor) -> np.ndarray:
        return (tensor.squeeze(0).numpy() * 255).astype(np.uint8)


if __name__ == '__main__':
    folder_path = 'data/masks/'
    resize_and_save_images(folder_path, is_masks=True)
