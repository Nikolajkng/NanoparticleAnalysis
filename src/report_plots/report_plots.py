import matplotlib.pyplot as plt
import numpy as np
from src.model.DataTools import tensor_from_image_no_resize, to_2d_image_array

def show_annotation(image, mask):
    """
    Show the image alongside the mask.
    """
    # Convert tensors to numpy arrays
    image_np = to_2d_image_array(image.numpy())
    mask_np = to_2d_image_array(mask.numpy())
    # Normalize the image for display
    image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np))
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    # Display the original image
    axes[0].imshow(image_np, cmap='gray')
    axes[0].set_title("Original Image")
    
    # Display the ground truth mask
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title("Ground Truth Mask")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image = tensor_from_image_no_resize("data/highres_images/E. sample_0008_4k.tif")
    mask = tensor_from_image_no_resize("data/highres_masks/E. sample_0008_mask_4k.tif")
    show_annotation(image, mask)