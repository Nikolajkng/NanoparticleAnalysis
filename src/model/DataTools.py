from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split

def get_dataloaders(dataset: Dataset, train_data_size: float) -> tuple[DataLoader, DataLoader]:
    
    train_data, val_data = random_split(dataset, [train_data_size, 1-train_data_size])
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True)

    return (train_dataloader, val_dataloader)
