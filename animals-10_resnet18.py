import torch, requests
import numpy as np
import matplotlib.pyplot as plt
import src.engine as engine
import src.prepare_data as prep_data
import src.visualize as visualize
from PIL import Image
from pathlib import Path
from src.utils import save_model
from src.prediction import make_prediction
from src.create_model import resnet18_model
from torchvision import transforms
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split 


# Dataset used for the training is available on Kaggle (28k images)
dataset_link =  "https://www.kaggle.com/datasets/alessiocorrado99/animals10"

# Set path to data (if available locally)
data_path = ""

# Setup hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
SEED = 42

# Setup transform 
transform = transforms.Compose([
    transforms.Resize(size=(384, 384)),
    transforms.CenterCrop(size=(384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Take only 250 images from every class to provide faster & balanced training
df = prep_data.create_dataframe(targ_dir=data_path, 
                                samples_per_label=250)

# Visualize the data distribution after creating dataframe subset
# plot_labels(dataframe=df)

train_df, test_df = train_test_split(df, test_size=0.2, 
                                     random_state=SEED)

# Create Pytorch Dataset from Pandas Dataframe using custom class
train_custom_dataset = prep_data.CustomDataset(dataframe=train_df, 
                                               transform=transform)

test_custom_dataset = prep_data.CustomDataset(dataframe=test_df, 
                                              transform=transform)

# Create Dataloaders
train_custom_dataloader = DataLoader(dataset=train_custom_dataset, 
                                     batch_size=32, 
                                     shuffle=True)

test_custom_dataloader = DataLoader(dataset=test_custom_dataset, 
                                    batch_size=32, 
                                    shuffle=False)

# Create dictionary with class names and indexes
class_names_idx = prep_data.create_class_names_idx(data_path)
class_names = list(class_names_idx.keys())

# Device agnostic code 
device = "cuda" if torch.cuda.is_available() else "cpu"

# Download and initialize model with pretrained weights
model = resnet18_model(output_features=len(class_names), 
                       device=device)

# Setup metrics, loss function and optimizer
accuracy_fn = Accuracy(task="multiclass", 
                       num_classes=len(class_names)).to(device)
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop on dataloaders
resnet18_model, metrics = engine.train(model=model,
                train_dataloader=train_custom_dataloader,
                test_dataloader=test_custom_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                num_epochs=NUM_EPOCHS,
                accuracy_fn=accuracy_fn,
                device=device)

# Set model path
model_dir_path = Path("models")

# Save model to file
save_model(model=resnet18_model,
           dir_path=model_dir_path,
           model_name="resnet18_v0.pth")

# Evaluation retrieves preds/targets for further analysis
results = engine.evaluation_model(model=resnet18_model,
                                  test_dataloader=test_custom_dataloader,
                                  device=device)

# Plot confusion matrix with corresponding classes
visualize.confusion_matrix(predictions=results["predictions"], 
                           targets=results["targets"], 
                           num_classes=len(class_names),
                           class_names=class_names_idx.keys())

# Download custom image for prediction
request = requests.get("https://images.unsplash.com/photo-1561948955-570b270e7c36?q=80&w=2101&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")
with open("unsplash_cat.jpg", "wb") as f:
    f.write(request.content)

# Make a prediction and plot image with predicted label & probability
make_prediction(model=resnet18_model,
                img_path="unsplash_cat.jpg",
                class_names=class_names,
                device=device)


# Create list containing corresponding indexes to classes in Dataframe
class_to_idx = [class_names_idx[idx] for idx in test_df.label.values]

# Copy previously created Test Dataframe
predictions_df = test_df.copy()

# Change list type elements from tensor -> numpy
predictions_numpy = np.stack(results["predictions"]) 
probabilities_numpy = np.stack(results["probability"])

# Create new columns containing indexes and results from evaluation on test data
predictions_df["labels_idx"] = class_to_idx
predictions_df["predictions"] = predictions_numpy
predictions_df["probability"] = probabilities_numpy


### Visualize predictions
# Sort probability based to visualize wrong and worst predictions
worst_predictions = predictions_df.sort_values(by=["probability"], ascending=True)[:30]

# Reverse dictionary to contain index: class_name
idx_to_class_name = dict((v,k) for k,v in class_names_idx.items())

# Get sample of 9 images
random_img_sample = worst_predictions.sample(n=9)

plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i+1)
    
    img_data = random_img_sample.iloc[i]
    img = img_data["image"]
    img = Image.open(img)
    
    plt.imshow(img)
    plt.title(f"Predicted: {idx_to_class_name[img_data['predictions']]}\nActual: {img_data['label']}\nProbability: {img_data['probability']:.2f}")
    plt.axis("off")