import torch, os
import pandas as pd
from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader, Subset
from torchvision import transforms
from PIL import Image


def create_class_names_idx(data_dir: str):
    # Scan directory
    obj = os.scandir(data_dir)
    
    # Create class name: index dictionary
    class_names_idx = {class_name.name: i for i, class_name in enumerate(obj)}
    return class_names_idx


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, targ_dir, transform=None):
        self.image_paths = dataframe.image.values
        self.labels = dataframe.label.values
        self.transform = transform
        self.class_names_idx = create_class_names_idx(targ_dir)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        img = self.image_paths[index]
        
        # Retrieve corresponding label to image
        label_name = self.labels[index]
        
        # Instead label name (str) we need label index (int)
        label_index = self.class_names_idx[label_name]
        
        # Open img as PIL object
        img = Image.open(img).convert("RGB")
        
        # Apply transform if present
        if self.transform:
            return self.transform(img), label_index
        return img, label_index


def create_dataloaders(targ_dir: str, test_size: float, batch_size: int, transform: transforms.Compose):
    # Set seed
    torch.manual_seed(42)
    
    # Initialize dataset before split
    if transform:
        dataset_folder = ImageFolder(root=targ_dir, transform=transform)
    else:
        dataset_folder = ImageFolder(root=targ_dir)
    class_names = dataset_folder.classes
    
    # Setup train and test size
    train_size = 1.0 - test_size
    
    # Random split data to train and test
    train_dataset, test_dataset = random_split(dataset_folder, [train_size, test_size])
    
    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader, class_names


def create_dataframe(targ_dir: str,
                     samples_per_label=None):
    # Setup list variables
    imgs, labels = [], []

    # Iterate through directories and count number of images and corresponding labels
    for p in Path(targ_dir).glob("*/*[.jpg.jpeg.png]"):
        imgs.append(p)
        labels.append(p.parent.name)

    # Create dataframe
    df = pd.DataFrame({"image": imgs, "label": labels})
    minimal_label_count = df.label.value_counts().min()
    
    # Create subset of data
    if samples_per_label:
        assert samples_per_label < minimal_label_count, "Samples per label higher than minimal label count."
        class_counts = df.label.value_counts()
        df = pd.concat([df[df['label'] == label].sample(samples_per_label) for label in class_counts.index])
    
    return df