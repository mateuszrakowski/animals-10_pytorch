import torch, os
import pandas as pd
from pathlib import Path


def create_class_names_idx(data_dir: str):
    # Scan directory
    obj = os.scandir(data_dir)
    
    # Create class name: index dictionary
    class_names_idx = {class_name.name: i for i, class_name in enumerate(obj)}
    return class_names_idx


def save_model(model: torch.nn.Module,
               dir_path: str,
               model_name: str):
    # Initialize Path object and create directory
    dir_path = Path(dir_path)
    dir_path.mkdir(exist_ok=True, parents=True)
    
    # Model path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_path = dir_path / model_name
    
    # Save model
    print(f"[INFO] Saving model to: {model_path}")
    torch.save(model.state_dict(), model_path)

    
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