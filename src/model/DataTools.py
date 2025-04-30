import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import os
from PIL import Image
from torch import Tensor
import torchvision.transforms.functional as TF
import numpy as np


from model.DataAugmenter import DataAugmenter
from model.dmFileReader import dmFileReader
from shared.IOFunctions import is_dm_format

def get_dataloaders(dataset: Dataset, train_data_size: float, validation_data_size: float, input_size: tuple[int, int]) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_augmenter = DataAugmenter()
    
    train_data, val_data, test_data = random_split(dataset, [train_data_size, validation_data_size, 1-train_data_size-validation_data_size])
    
    train_data = data_augmenter.augment_dataset(train_data)
    val_data = data_augmenter.get_crops_for_dataset(val_data, 10, input_size)
    test_data = data_augmenter.get_crops_for_dataset(test_data, 10, input_size)

    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=1)
    return (train_dataloader, val_dataloader, test_dataloader)


def get_dataloaders_without_testset(dataset: Dataset, train_data_size: float, input_size: tuple[int, int]) -> tuple[DataLoader, DataLoader]:
    
    train_data, val_data = random_split(dataset, [train_data_size, 1-train_data_size])

    data_augmenter = DataAugmenter()
    train_data = data_augmenter.augment_dataset(train_data, input_size)

    val_data = data_augmenter.get_crops_for_dataset(val_data, 10, input_size)

    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True, drop_last=True)
    return (train_dataloader, val_dataloader)

def center_crop(image, target_size: tuple[int, int]):
    _, _, h, w = image.shape
    th, tw = target_size

    start_h = (h-th) // 2
    start_w = (w-tw) // 2
    return image[:, :, start_h:start_h + th, start_w:start_w + tw]

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

def tensor_from_image_no_resize(image_path: str):
    image = Image.open(image_path).convert("L")
    image = TF.to_tensor(image).unsqueeze(0)
    return image

def tensor_from_image(image_path: str, resize=(256,256)) -> Tensor:
    image = Image.open(image_path).convert("L")
    image.thumbnail(resize)
    image = TF.to_tensor(image).unsqueeze(0)
    return image

def to_2d_image_array(array: np.ndarray) -> np.ndarray:
    return (np.squeeze(array) * 255).astype(np.uint8)

def load_image_as_tensor(image_path: str):
    reader = dmFileReader()
    tensor = None
    if is_dm_format(image_path):
        tensor = reader.get_tensor_from_dm_file(image_path)
    else:
        tensor = tensor_from_image_no_resize(image_path)
    if tensor.shape[-1] > 1024 or tensor.shape[-2] > 1024:
        tensor = TF.resize(tensor, 1024)
    return tensor

# Made with help from https://www.programmersought.com/article/15316517340/
def mirror_fill(images: Tensor, patch_size: tuple, stride_size: tuple):
    images_np = images.cpu().numpy()
    batch_size, channels, img_width, img_height = images_np.shape
    patch_height, patch_width = patch_size
    stride_height, stride_width = stride_size

    remaining_width = (img_width - patch_width) % stride_width
    remaining_height = (img_height - patch_height) % stride_height

    needed_padding_width = (stride_width - remaining_width) % stride_width
    needed_padding_height = (stride_height - remaining_height) % stride_height

    if needed_padding_width:

        padded_images = np.empty(
            (batch_size, channels, img_width + needed_padding_width, img_height), 
            dtype=images_np.dtype
        )

        start_x = needed_padding_width // 2
        end_x = start_x + img_width

        for i, img in enumerate(images_np):
            padded_images[i, :, start_x:end_x, :] = img
            padded_images[i, :, :start_x, :] = np.flip(img[:, :start_x, :], axis=1)
            padded_images[i, :, end_x:, :] = np.flip(img[:, img_width - (needed_padding_width - needed_padding_width // 2):, :], axis=1)
        
        images_np = padded_images
    
    if needed_padding_height:
        img_width = images_np.shape[2]
        padded_images = np.empty(
            (batch_size, channels, img_width, img_height + needed_padding_height), 
            dtype=images_np.dtype
        )
        start_y = needed_padding_height // 2
        end_y = start_y + img_height

        for i, img in enumerate(images_np):
            padded_images[i, :, :, start_y:end_y] = img
            padded_images[i, :, :, :start_y] = np.flip(img[:, :, :start_y], axis=2)
            padded_images[i, :, :, end_y:] = np.flip(img[:, :, img_height - (needed_padding_height - needed_padding_height // 2):], axis=2)
        
        images_np = padded_images
    
    return torch.tensor(images_np, dtype=images.dtype, device=images.device)
    


# Made with help from https://www.programmersought.com/article/15316517340/
def extract_slices(images: Tensor, patch_size: tuple, stride_size: tuple):
    images_np = images.cpu().numpy()
    batch_size, channels, img_width, img_height = images.shape
    patch_height, patch_width = patch_size
    stride_height, stride_width = stride_size


    n_patches_y = (img_height - patch_height) // stride_height + 1
    n_patches_x = (img_width - patch_width) // stride_width + 1

    n_patches_per_image = n_patches_x * n_patches_y

    n_patches_total = n_patches_per_image * batch_size

    patches = np.empty((n_patches_total, channels, patch_width, patch_height), dtype=images_np.dtype)

    patch_idx = 0

    for img in images_np:
        for i in range(n_patches_y):
            for j in range(n_patches_x):
                start_x = j * stride_width
                start_y = i * stride_height
                end_x = start_x + patch_width
                end_y = start_y + patch_height

                patches[patch_idx] = img[:, start_x:end_x, start_y:end_y]
                patch_idx += 1
    return patches

def construct_image_from_patches(patches: np.ndarray, img_size: tuple, stride_size: tuple):
    img_width, img_height = img_size
    stride_width, stride_height = stride_size
    n_patches_total, channels, patch_width, patch_height = patches.shape
    
    n_patches_y = (img_height - patch_height) // stride_height + 1
    n_patches_x = (img_width - patch_width) // stride_width + 1

    n_patches_per_image = n_patches_x * n_patches_y

    batch_size = n_patches_total // n_patches_per_image

    images = np.zeros((batch_size, channels, img_width, img_height))
    weights = np.zeros_like(images)

    for img_idx, (img, weights) in enumerate(zip(images, weights)):
        start = img_idx * n_patches_per_image

        for i in range(n_patches_y):
            for j in range(n_patches_x):
                start_x = j * stride_width
                start_y = i * stride_height
                end_x = start_x + patch_width
                end_y = start_y + patch_height
                patch_idx = start + i * n_patches_x + j
                img[:, start_x:end_x, start_y:end_y] += patches[patch_idx]
                weights[start_x:end_x, start_y:end_y] += 1
    images /= weights

    return images


import torch.nn.functional as F
def normalizeTensorToPixels(tensor: Tensor) -> Tensor:
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    tensor = tensor * 255
    return tensor
    
def showTensor(tensor: Tensor) -> None:
    #probabilities = F.softmax(tensor, dim=1)  
    if tensor.dim == 4:
        tensor = tensor.squeeze(1)
    if tensor.size(0) == 1:
        #probabilities = tensor.squeeze(0)
        pixels = normalizeTensorToPixels(tensor[0, :, :])
    
        img = TF.to_pil_image(pixels.byte())
        img.show()

if __name__ == '__main__':
    folder_path = 'data/to_resize/'
    resize_and_save_images(folder_path, is_masks=True, output_size=(4096, 4074))
    # tensor = tensor_from_image('data/W. sample_0011.tif', (256, 256))
    # tensor = mirror_fill(tensor, (100,100), (100,100))
    # patches = extract_slices(tensor, (100,100), (100,100))
    # showTensor(tensor)
    # reconstructed = construct_image_from_patches(patches, (300,300), (100,100))
    # showTensor(reconstructed)
    
