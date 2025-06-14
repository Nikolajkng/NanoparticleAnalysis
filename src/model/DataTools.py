import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import os
from PIL import Image
from torch import Tensor
import numpy as np
import sys
from src.model.DataAugmenter import DataAugmenter
from src.model.dmFileReader import dmFileReader
from src.shared.IOFunctions import is_dm_format
from src.model.SegmentationDataset import SegmentationDataset

def slice_dataset_in_four(dataset, input_size=(256, 256)):
    images = []
    masks = []
    for img, mask in dataset:
        width = img.shape[-1]
        height = img.shape[-2]
        if width <= input_size[0] or height <= input_size[1]:
            images.append(img)
            masks.append(mask)
            continue
        new_width = width // 2
        new_height = height // 2

        image_slices = [
            img[:, :new_height, :new_width],
            img[:, new_height:, :new_width],
            img[:, :new_height, new_width:],
            img[:, new_height:, new_width:]
        ]
        mask_slices = [
            mask[:, :new_height, :new_width],
            mask[:, new_height:, :new_width],
            mask[:, :new_height, new_width:],
            mask[:, new_height:, new_width:]
        ]
        images.extend(image_slices)
        masks.extend(mask_slices)
    return SegmentationDataset.from_image_set(images, masks)

# Helper to process val/test with mirror_fill and extract_slices
def process_and_slice(data_subset, input_size=(256, 256)):
    images = []
    masks = []
    for img, mask in data_subset:
        img = img.unsqueeze(0) if img.dim() == 3 else img
        mask = mask.unsqueeze(0) if mask.dim() == 3 else mask
        filled_image = mirror_fill(img, patch_size=input_size, stride_size=input_size)
        filled_mask = mirror_fill(mask, patch_size=input_size, stride_size=input_size)

        sliced_images = extract_slices(filled_image, patch_size=input_size, stride_size=input_size)
        sliced_masks = extract_slices(filled_mask, patch_size=input_size, stride_size=input_size)
        # Convert np.ndarray -> torch.Tensor if necessary
        if isinstance(sliced_masks[0], np.ndarray):
            sliced_images = [torch.from_numpy(img) for img in sliced_images]
        if isinstance(sliced_masks[0], np.ndarray):
            sliced_masks = [torch.from_numpy(mask) for mask in sliced_masks]
        images.extend(sliced_images)
        masks.extend(sliced_masks)

    # Create list of (image, mask) tensors
    return SegmentationDataset.from_image_set(images, masks)

def get_dataloaders(dataset: Dataset, train_data_size: float, validation_data_size: float, input_size: tuple[int, int], with_data_augmentation: bool) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_augmenter = DataAugmenter()
    dataset = slice_dataset_in_four(dataset, input_size)
    train_data, val_data, test_data = random_split(dataset, [train_data_size, validation_data_size, 1-train_data_size-validation_data_size])
    print(f"Train images: {train_data.indices}")
    print(f"Validation images: {val_data.indices}")
    print(f"Test images: {test_data.indices}")
    if with_data_augmentation:
        train_data = data_augmenter.augment_dataset(train_data, input_size)
    else:
        train_data = data_augmenter.augment_dataset(train_data, input_size, [False, False, False, False, False, False])
    val_data = process_and_slice(val_data, input_size)
    test_data = process_and_slice(test_data, input_size)

    if torch.cuda.is_available():
        train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True, num_workers=24)
    else:
        train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=1)
    return (train_dataloader, val_dataloader, test_dataloader)


def get_dataloaders_without_testset(dataset: Dataset, train_data_size: float, input_size: tuple[int, int], with_data_augmentation: bool) -> tuple[DataLoader, DataLoader]:
    data_augmenter = DataAugmenter()
    dataset = slice_dataset_in_four(dataset, input_size)
    train_data, val_data = random_split(dataset, [train_data_size, 1-train_data_size])
    if with_data_augmentation:
        train_data = data_augmenter.augment_dataset(train_data, input_size)
    else:
        train_data = data_augmenter.augment_dataset(train_data, input_size, [False, False, False, False, False, False])
    val_data = process_and_slice(val_data, input_size)

    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True, drop_last=True)
    return (train_dataloader, val_dataloader)


def get_dataloaders_kfold(dataset: Dataset, train_data_size: float, batch_size: int, input_size: tuple[int, int], random_cropping: bool) -> tuple[DataLoader, DataLoader]:
    data_augmenter = DataAugmenter()
    train_data, val_data = random_split(dataset, [train_data_size, 1-train_data_size])
    if random_cropping:
        train_data = data_augmenter.augment_dataset(train_data, input_size)
    else:
        train_data = data_augmenter.augment_dataset(train_data, input_size, [False, True, True, False])
    val_data = process_and_slice(val_data, input_size)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=24)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True, drop_last=True)
    return (train_dataloader, val_dataloader)

def get_dataloaders_kfold_already_split(train_data, val_data, batch_size, input_size, augmentations=[True,True,False,False,False,False]):
    data_augmenter = DataAugmenter()
    print(augmentations)
    if not augmentations[0]: # No random cropping
        train_data = process_and_slice(train_data, input_size)
    train_data = data_augmenter.augment_dataset(train_data, input_size, augmentations)
  
    val_data = process_and_slice(val_data, input_size)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=24)
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
            if img.width == output_size[0] and img.height == output_size[1]:
                continue
            img = img.resize(output_size)
            if is_masks:
                img_binary = img.point(lambda p: 255 if p > 20 else 0)
                img_binary.save(os.path.join(folder_path,"new"+filename))
            else:    
                img.save(os.path.join(folder_path,"new"+filename))  
            print(image_path)

def tensor_from_image_no_resize(image_path: str):
    import torchvision.transforms.functional as TF
    image = Image.open(image_path).convert("L")
    image = TF.to_tensor(image).unsqueeze(0)
    return image

def tensor_from_image(image_path: str, resize=(256,256)) -> Tensor:
    import torchvision.transforms.functional as TF

    image = Image.open(image_path).convert("L")
    image.thumbnail(resize)
    image = TF.to_tensor(image).unsqueeze(0)
    return image

def to_2d_image_array(array: np.ndarray) -> np.ndarray:
    return (np.squeeze(array) * 255).astype(np.uint8)

def load_image_as_tensor(image_path: str):
    import torchvision.transforms.functional as TF

    reader = dmFileReader()
    tensor = None
    if is_dm_format(image_path):
        tensor = reader.get_tensor_from_dm_file(image_path)
    else:
        tensor = tensor_from_image_no_resize(image_path)
    if tensor.shape[-1] > 1024 or tensor.shape[-2] > 1024:
        tensor = TF.resize(tensor, 1024)
    return tensor

# Made with help from https://stackoverflow.com/questions/65754703/pillow-converting-a-tiff-from-greyscale-16-bit-to-8-bit-results-in-fully-white
def tiff_force_8bit(image, **kwargs):
    if image.format == 'TIFF' and image.mode in ('I;16', 'I;16B', 'I;16L'):
        array = np.array(image)
        normalized = (array.astype(np.uint16) - array.min()) * 255.0 / (array.max() - array.min())
        range_val = array.max() - array.min()
        if range_val == 0:
            normalized = np.zeros_like(array, dtype=np.uint8)
        else:
            normalized = (array.astype(np.uint16) - array.min()) * 255.0 / range_val
        image = Image.fromarray(normalized.astype(np.uint8))

    return image

# Made with help from https://www.programmersought.com/article/15316517340/
def mirror_fill(images: Tensor, patch_size: tuple, stride_size: tuple) -> Tensor:
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
def extract_slices(images: Tensor, patch_size: tuple, stride_size: tuple) -> np.ndarray:
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
                weights[:, start_x:end_x, start_y:end_y] += 1
    images /= weights

    return images

def get_normalizer(dataset: SegmentationDataset):
    X = torch.stack(dataset.images)
    mu = X.mean(axis=(0, 2, 3)).tolist()
    std = X.std(axis=(0, 2, 3)).tolist()
    from torchvision.transforms.v2 import Normalize
    return Normalize(mean=mu, std=std)

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS  
    else:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def normalizeTensorToPixels(tensor: Tensor) -> Tensor:
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    tensor = tensor * 255
    return tensor
    
def showTensor(tensor: Tensor) -> None:
    import torchvision.transforms.functional as TF
    if tensor.dim == 4:
        tensor = tensor.squeeze(1)
    if tensor.size(0) == 1:
        pixels = normalizeTensorToPixels(tensor[0, :, :])
    
        img = TF.to_pil_image(pixels.byte())
        img.show()
        
def calculate_class_imbalance(masks_dir: str) -> dict:
    class_counts = {}
    for filename in os.listdir(masks_dir):
        if filename.endswith(('.tif')):
            mask_path = os.path.join(masks_dir, filename)
            mask = Image.open(mask_path).convert("L")
            mask_np = np.array(mask)
            unique, counts = np.unique(mask_np, return_counts=True)
            for u, c in zip(unique, counts):
                if int(u) not in class_counts:
                    class_counts[int(u)] = 0
                class_counts[int(u)] += int(c)
    for class_id, count in class_counts.items():
        print(f"Class {class_id}: {count} pixels")
    print("Total pixels:", sum(class_counts.values()))
    print("Class imbalance ratio:", {k: v / sum(class_counts.values()) for k, v in class_counts.items()})
    
    return class_counts

def bootstrap_compare(model_A: list, model_B: list, n_bootstraps=100_000, ci=(2.5, 97.5)):
    A = np.array(model_A)
    B = np.array(model_B)
    obs_diff = B.mean() - A.mean()

    # Center both samples under H0
    pooled_mean = np.concatenate((A, B)).mean()
    A0 = A - A.mean() + pooled_mean
    B0 = B - B.mean() + pooled_mean

    boot_diffs = np.empty(n_bootstraps)
    null_diffs = np.empty(n_bootstraps)
    for i in range(n_bootstraps):
        idx = np.random.choice(len(A), size=len(A), replace=True)
        boot_diffs[i] = B[idx].mean() - A[idx].mean()
        null_diffs[i] = B0[idx].mean() - A0[idx].mean()

    p_value = np.mean(np.abs(null_diffs) >= np.abs(obs_diff))
    ci_lower, ci_upper = np.percentile(boot_diffs, ci)
    
    print(f"Mean performance of model A:        {A.mean():.4f}")
    print(f"Mean performance of model B:        {B.mean():.4f}")
    print(f"Mean difference:                    {obs_diff:.4f}")
    print(f"p_value:                            {p_value:.4f}")
    print(f"{ci[1]-ci[0]}% Confidence interval: [{ci_lower:.4f}; {ci_upper:.4f}]")
