import torch, random
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from mlxtend.plotting import plot_confusion_matrix
from torchmetrics import ConfusionMatrix
from pathlib import Path


def plot_labels(dataframe: pd.DataFrame):
    # Retrieve columns count as dictionary
    labels_dict = dict(dataframe["label"].value_counts())

    # Create bar plot
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.bar(labels_dict.keys(), labels_dict.values(), edgecolor="white");
    ax.set_xlabel("Class names")
    ax.set_ylabel("Number of images")
    ax.set_title("Number of images in each class")
    return fig

    
def plot_random_images(data_dir: str):
    # Iterate through directories and get img paths
    img_paths = list(Path(data_dir).glob("*/*[.jpg.jpeg.png]"))
    
    # Pick random images sample 
    random_imgs = random.sample(img_paths, 9)

    # Plot images in 3x3 matrix
    fig = plt.figure(figsize=(9, 9))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        img = Image.open(random_imgs[i])
        plt.imshow(img)
        plt.axis("off")
    return fig

        
def confusion_matrix(predictions: list, targets: list, num_classes: int, class_names: list):
    # Initialize Confusion Matrix object
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)
    
    # Move to tensor
    preds_tensor = torch.Tensor(predictions)
    targets_tensor = torch.Tensor(targets)
    
    # Pytorch Lightning Confmat
    confusion_matrix = confmat(preds_tensor, targets_tensor).numpy()
    
    # Plot confusion matrix
    fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix,
                                    class_names=class_names)
    return fig