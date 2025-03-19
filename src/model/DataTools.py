from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import os
from PIL import Image

def get_dataloaders(dataset: Dataset, train_data_size: float) -> tuple[DataLoader, DataLoader]:
    
    train_data, val_data = random_split(dataset, [train_data_size, 1-train_data_size])
    train_dataloader = DataLoader(train_data, batch_size=12, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=3, shuffle=True, drop_last=True)
    
    return (train_dataloader, val_dataloader)

def resize_and_save_images(folder_path, output_size=(256, 256), is_masks=False):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.tif')):
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path)
            img = img.convert("L")  
            img_resized = img.resize(output_size, Image.NEAREST)
            if is_masks:
                img_binary = img_resized.point(lambda p: 255 if p > 20 else 0)
                img_binary.save(os.path.join(folder_path,"new"+filename))
            else:    
                img_resized.save(image_path)  # You can change this line to save it elsewhere
            print(image_path)

if __name__ == '__main__':
    folder_path = 'data/images/'
    resize_and_save_images(folder_path, is_masks=False)
