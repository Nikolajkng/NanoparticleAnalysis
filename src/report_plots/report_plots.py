import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#from src.model.DataTools import tensor_from_image_no_resize, to_2d_image_array

def plot_paired_bar(iou_A, iou_B, dice_A, dice_B):
    A_label = "Horizontal flipping"
    B_label = "Flipping + rotation"
    diff_iou = iou_B - iou_A
    diff_dice = dice_B - dice_A

    indices = np.arange(1, len(diff_iou)+1)

    plt.figure(figsize=(10,4))

    plt.subplot(1, 2, 1)
    plt.bar(indices, diff_iou, color='skyblue', edgecolor='black')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title(f'Paired Differences: IOU ({B_label} - {A_label})')
    plt.xlabel('Fold')
    plt.ylabel('Difference')
    plt.xticks(indices)
    plt.ylim(-max(np.abs(diff_iou))*1.2, max(np.abs(diff_iou))*1.2)

    plt.subplot(1, 2, 2)
    plt.bar(indices, diff_dice, color='salmon', edgecolor='black')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title(f'Paired Differences: Dice ({B_label} - {A_label})')
    plt.xlabel('Fold')
    plt.ylabel('Difference')
    plt.xticks(indices)
    plt.ylim(-max(np.abs(diff_dice))*1.2, max(np.abs(diff_dice))*1.2)

    plt.tight_layout()
    plt.show()

def plot_paired_scatter(iou_A, iou_B, dice_A, dice_B):
    
    diff_iou = iou_B - iou_A
    diff_dice = dice_B - dice_A

    plt.figure(figsize=(10,4))

    plt.subplot(1, 2, 1)
    plt.scatter(range(1, len(diff_iou)+1), diff_iou, color='skyblue')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title('Paired Differences: IOU (B - A)')
    plt.xlabel('Sample')
    plt.ylabel('Difference')
    plt.ylim(min(diff_iou)*1.2, max(diff_iou)*1.2)

    plt.subplot(1, 2, 2)
    plt.scatter(range(1, len(diff_dice)+1), diff_dice, color='salmon')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title('Paired Differences: Dice (B - A)')
    plt.xlabel('Sample')
    plt.ylabel('Difference')
    plt.ylim(min(diff_dice)*1.2, max(diff_dice)*1.2)

    plt.tight_layout()
    plt.show()

def plot_paired_histogram(iou_A, iou_B, dice_A, dice_B):
    # Calculate paired differences
    diff_iou = iou_B - iou_A
    diff_dice = dice_B - dice_A

    # Plot histograms
    plt.figure(figsize=(12, 5))

    max_diff = max(np.max(np.abs(diff_iou)), np.max(np.abs(diff_dice)))


    plt.subplot(1, 2, 1)
    plt.hist(diff_iou, bins=20, color='skyblue', edgecolor='black')
    plt.title('Histogram of Paired Differences: IOU')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.xlim(-max_diff, max_diff)  # symmetric around zero


    plt.subplot(1, 2, 2)
    plt.hist(diff_dice, bins=20, color='salmon', edgecolor='black')
    plt.title('Histogram of Paired Differences: Dice')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.xlim(-max_diff, max_diff)  # symmetric around zero


    plt.tight_layout()
    plt.show()

def plot_iou_dice_boxplot(iou_A, iou_B, dice_A, dice_B):
    import pandas as pd
    A_label = "No augmentation"
    B_label = "Horizontal flip"
    # Prepare DataFrame
    data = {
        'Score': np.concatenate([iou_A, iou_B, dice_A, dice_B]),
        'Metric': ['IoU'] * 10 + ['Dice'] * 10,
        'Model': [A_label] * 5 + [B_label] * 5 + [A_label] * 5 + [B_label] * 5
    }
    df = pd.DataFrame(data)

    # Style
    sns.set_theme(style='whitegrid')
    palette = {A_label: '#1f77b4', B_label: '#ff7f0e'}

    # Create plot
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='Metric', y='Score', hue='Model', data=df, palette=palette, width=0.6, linewidth=1.5, whis=[0,100], showfliers=False)
    
    # Calculate and annotate means
    means = {
        ('IoU', A_label): np.mean(iou_A),
        ('IoU', B_label): np.mean(iou_B),
        ('Dice', A_label): np.mean(dice_A),
        ('Dice', B_label): np.mean(dice_B),
    }
    x_offsets = {('IoU', A_label): -0.15, ('IoU', B_label): 0.15,
                 ('Dice', A_label): 0.85, ('Dice', B_label): 1.15}

    for (metric, model), mean in means.items():
        x = x_offsets[(metric, model)]
        
        # Find maximum y for this group to place the mean above it
        scores = df[(df['Metric'] == metric) & (df['Model'] == model)]['Score']
        y_max = scores.max()
        
        ax.text(
            x, y_max + 0.01,  # place above the max value
            f"Mean = {mean:.3f}",
            ha='center',
            fontsize=10,
            color='dimgray',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey', boxstyle='round,pad=0.3')
        )

    # Polish
    ax.set_title(f'Comparison of IoU and Dice Scores\n {A_label} vs. {B_label}', fontsize=14, weight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Metric', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(title='Model', title_fontsize=12, fontsize=11, loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_iou_dice_boxplot3(iou_A, iou_B, iou_C, dice_A, dice_B, dice_C):
    import pandas as pd
    A_label = "No augmentation"
    B_label = "Horizontal flip"
    C_label = "Horizontal flip + rotation"
    # Prepare DataFrame
    data = {
        'Score': np.concatenate([iou_A, iou_B, iou_C, dice_A, dice_B, dice_C]),
        'Metric': ['IoU'] * 15 + ['Dice'] * 15,
        'Model': [A_label] * 5 + [B_label] * 5 + [C_label] * 5 + [A_label] * 5 + [B_label] * 5 + [C_label] * 5
    }
    df = pd.DataFrame(data)
    # Style
    sns.set_theme(style='whitegrid')
    palette = {A_label: '#1f77b4', B_label: '#ff7f0e', C_label: '#2ca02c'}
    # Create plot
    plt.figure(figsize=(10, 7))
    ax = sns.boxplot(x='Metric', y='Score', hue='Model', data=df, palette=palette, width=0.6, linewidth=1.5, whis=[0,100], showfliers=False)
    # Calculate and annotate means
    means = {
        ('IoU', A_label): np.mean(iou_A),
        ('IoU', B_label): np.mean(iou_B),
        ('IoU', C_label): np.mean(iou_C),
        ('Dice', A_label): np.mean(dice_A),
        ('Dice', B_label): np.mean(dice_B),
        ('Dice', C_label): np.mean(dice_C),
    }
    x_offsets = {('IoU', A_label): -0.2, ('IoU', B_label): 0, ('IoU', C_label): 0.2,
                    ('Dice', A_label): 0.8, ('Dice', B_label): 1, ('Dice', C_label): 1.2}
    for (metric, model), mean in means.items():
        x = x_offsets[(metric, model)]
        
        # Find maximum y for this group to place the mean above it
        scores = df[(df['Metric'] == metric) & (df['Model'] == model)]['Score']
        y_max = scores.max()
        
        ax.text(
            x, y_max + 0.005,  # place above the max value
            f"Mean = {mean:.3f}",
            ha='center',
            fontsize=10,
            color='dimgray',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey', boxstyle='round,pad=0.3')
        )
    # Polish
    ax.set_title(f'Comparison of IoU and Dice Scores\n {A_label} vs. {B_label} vs. {C_label}', fontsize=14, weight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Metric', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(title='Model', title_fontsize=12, fontsize=11, loc='upper left')
    plt.tight_layout()
    plt.show()
    

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

def bootstrap_compare(model_A: list, model_B: list, n_bootstraps=100_00, ci=(2.5, 97.5)):
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
def to_2d_image_array(array: np.ndarray) -> np.ndarray:
    return (np.squeeze(array) * 255).astype(np.uint8)

def show_example_images(images):
    """
    Show a few example images from the dataset.
    """
    fig, axes = plt.subplots(2, len(images)//2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, image in zip(axes, images):
        # Convert tensor to numpy array
        image_np = to_2d_image_array(image)
        # Normalize the image for display
        image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np))
        ax.imshow(image_np, cmap='gray')
        ax.axis('on')
    fig.suptitle("Example TEM Images", fontsize=16)
    plt.tight_layout()  # Leave space for the title
    plt.show()

def remove_border_artifacts(image, mask, crop_size=448):
    # Center crop after affine to remove black borders
    i, j, h, w = TF.center_crop.get_params(image, output_size=(crop_size, crop_size))
    image = TF.crop(image, i, j, h, w)
    mask = TF.crop(mask, i, j, h, w)

    # Resize back to original size if needed
    image = TF.resize(image, [512, 512], interpolation=TF.InterpolationMode.BILINEAR)
    mask = TF.resize(mask, [512, 512], interpolation=TF.InterpolationMode.NEAREST)

    return image, mask

def random_affine(image, mask):
    import random
    # Parameters for affine transformation
    angle = random.uniform(-10, 10)            # Small rotation
    translate = (random.uniform(-0.05, 0.05),  # Max 5% shift
                 random.uniform(-0.05, 0.05))
    scale = random.uniform(0.95, 1.05)         # Mild scaling
    shear = random.uniform(-5, 5)              # Slight shearing
    import torchvision.transforms.functional as TF

    # Apply the same transform to both image and mask
    image = TF.affine(image, angle, translate, scale, shear, interpolation=TF.InterpolationMode.BILINEAR)
    mask = TF.affine(mask, angle, translate, scale, shear, interpolation=TF.InterpolationMode.NEAREST)

    return image, mask
def tensor_from_image_no_resize(image_path: str):
    from PIL import Image
    import torchvision.transforms.functional as TF

    image = Image.open(image_path).convert("L")
    image = TF.to_tensor(image).unsqueeze(0)
    return image

def compute_collected_histogram(image_paths):
    total_histogram = np.zeros(256, dtype=np.uint64)
    from PIL import Image
    for path in image_paths:
        img = Image.open(path).convert("L")  # Ensure grayscale
        hist = np.array(img.histogram(), dtype=np.uint64)
        total_histogram += hist
    
    return total_histogram

def get_image_paths(directory, extensions={'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                image_paths.append(os.path.join(root, file))
    return image_paths

def get_sorted_image_mask_paths(image_dir, mask_dir):
    import os
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])

    assert len(image_files) == len(mask_files), "Number of images and masks must match."

    image_paths = [os.path.join(image_dir, f) for f in image_files]
    mask_paths = [os.path.join(mask_dir, f) for f in mask_files]

    return image_paths, mask_paths

def apply_clahe(pil_image):
    import cv2
    import numpy as np
    from PIL import Image
    img = np.array(pil_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(img)
    return Image.fromarray(equalized)

def contrast_stretch(image, low=0.01, high=0.99):
    """
    Apply contrast stretching to the image.
    """
    import numpy as np
    from PIL import Image

    # Convert to numpy array
    img_array = np.array(image)

    # Compute low and high percentiles
    p_low = np.percentile(img_array, low * 100)
    p_high = np.percentile(img_array, high * 100)

    # Stretch the contrast
    stretched = (img_array - p_low) / (p_high - p_low) * 255.0
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)

    return Image.fromarray(stretched)

def compute_foreground_background_histograms(image_paths, mask_paths):
    import numpy as np
    from PIL import Image
    fg_hist = np.zeros(256, dtype=np.uint64)
    bg_hist = np.zeros(256, dtype=np.uint64)

    for img_path, mask_path in zip(image_paths, mask_paths):
        img = np.array(Image.open(img_path).convert("L"))
        img = np.array(apply_clahe(img))
        mask = np.array(Image.open(mask_path).convert("1"))  # Binary mask (0 or 255)

        fg_pixels = img[mask == 1]
        bg_pixels = img[mask == 0]

        fg_hist += np.bincount(fg_pixels.astype(np.uint8), minlength=256).astype(np.uint64)
        bg_hist += np.bincount(bg_pixels.astype(np.uint8), minlength=256).astype(np.uint64)

    return fg_hist, bg_hist

def compute_foreground_background_histograms2(images, masks):
    import numpy as np
    from PIL import Image
    fg_hist = np.zeros(256, dtype=np.uint64)
    bg_hist = np.zeros(256, dtype=np.uint64)

    for img_path, mask_path in zip(image_paths, mask_paths):
        img = np.array(Image.open(img_path).convert("L"))
        img = np.array(apply_clahe(img))
        mask = np.array(Image.open(mask_path).convert("1"))  # Binary mask (0 or 255)

        fg_pixels = img[mask == 1]
        bg_pixels = img[mask == 0]

        fg_hist += np.bincount(fg_pixels.astype(np.uint8), minlength=256).astype(np.uint64)
        bg_hist += np.bincount(bg_pixels.astype(np.uint8), minlength=256).astype(np.uint64)

    return fg_hist, bg_hist

def array_to_tensor(array: np.ndarray):
    """
    Convert a 2D numpy array to a PyTorch tensor.
    """
    import torch
    return torch.tensor(array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

if __name__ == "__main__":
    #show_example_images(images[0:6])

    image_paths = get_image_paths("data/highres_images/")
    histogram = compute_collected_histogram(image_paths)
    # Print or plot the histogram
    import matplotlib.pyplot as plt
    plt.plot(histogram)
    plt.title("Collected Pixel Value Histogram")
    plt.xlabel("Pixel Value (0-255)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


    image_paths, mask_paths = get_sorted_image_mask_paths("data/highres_images", "data/highres_masks")
    
    fg_hist, bg_hist = compute_foreground_background_histograms(image_paths, mask_paths)
    # Plotting the histograms
    import matplotlib.pyplot as plt
    plt.plot(fg_hist, label='Foreground', color='red')
    plt.plot(bg_hist, label='Background', color='blue')
    plt.title("Foreground vs Background Pixel Histograms")
    plt.xlabel("Pixel Value (0-255)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()



    # image = tensor_from_image_no_resize("data/highres_images/E. sample_0008_4k.tif")
    # mask = tensor_from_image_no_resize("data/highres_masks/E. sample_0008_mask_4k.tif")
    # image2, mask2 = random_affine(image, mask)
    # show_example_images([image, mask, image2, mask2])
    # show_annotation(image, mask)
    # import pandas as pd
    
    # iou_A = np.array([0.7106041312217712, 0.7450249195098877, 0.7225610613822937, 0.7385781407356262, 0.7855268716812134])
    # iou_B = np.array([0.7667941451072693, 0.6404618620872498, 0.7248525619506836, 0.7824148535728455, 0.8072212338447571] )
    # iou_C = np.array([0.7660102844238281, 0.786454975605011, 0.749537467956543, 0.759172797203064, 0.7851026654243469])
    # dice_A = np.array([0.7884710431098938, 0.8103492856025696, 0.8125852346420288, 0.8322637677192688, 0.8616784811019897])
    # dice_B = np.array([0.8451998233795166, 0.7093654274940491, 0.8108140826225281, 0.8679991960525513, 0.8851615786552429])
    # dice_C = np.array([0.8413566946983337, 0.8447999954223633, 0.8332952260971069, 0.8431856036186218, 0.8538570404052734])
    # #plot_iou_dice_boxplot(iou_A, iou_B, dice_A, dice_B) 
    # plot_iou_dice_boxplot(iou_A, iou_B, dice_A, dice_B)
    # plot_paired_bar(iou_A, iou_B, dice_A, dice_B)
    # plot_paired_bar(iou_B, iou_C, dice_B, dice_C)
    # diff = iou_B - iou_A
    # bootstrap_compare(iou_A, iou_B)
    images = []
    masks = []
    from PIL import Image
    image_dir = "data/highres_images/"
    masks_dir = "data/highres_masks/"
    image_filenames = sorted(os.listdir(image_dir))
    mask_filenames = sorted(os.listdir(masks_dir))
    indices = [4, 0, 11]#, 9, 2, 10]#[0, 2, 4, 9, 10, 11]
    selected_filenames = selected_items = [image_filenames[i] for i in indices]
    import torchvision.transforms.functional as TF
    for image_name, mask_name in zip(image_filenames, mask_filenames):
        img_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(masks_dir, mask_name)


        image = Image.open(img_path).convert("L")
        image2 = contrast_stretch(image)
        mask = Image.open(mask_path).convert("L")
    
        #image = TF.to_tensor(image)
        #image2 = TF.to_tensor(image2)
        #images.append(image)
        images.append(image2)
        masks.append(mask)
    #show_example_images(images)
    fg_hist, bg_hist = compute_foreground_background_histograms2(images, masks)
    plt.plot(fg_hist, label='Foreground', color='red')
    plt.plot(bg_hist, label='Background', color='blue')
    plt.title("Foreground vs Background Pixel Histograms")
    plt.xlabel("Pixel Value (0-255)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

