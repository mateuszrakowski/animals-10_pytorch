import torch
from torchvision import models


def efficientnetv2_model(output_features: int, device: torch.cuda.device):
    # Initialize model object
    model = models.efficientnet_v2_s(weights="DEFAULT").to(device)
    
    # Turn off gradient computation
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Modify last layer to fit number of classes
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280, out_features=output_features, bias=True)).to(device)
    
    return model


def resnet18_model(output_features: int, device: torch.cuda.device):
    # Initialize model object
    model = models.resnet18(weights="DEFAULT").to(device)
    
    # Turn off gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    # Modify last layer to fit number of classes
    model.fc = torch.nn.Linear(in_features=512, out_features=output_features, bias=True).to(device)
    
    return model


def resnet50_model(output_features: int, device: torch.cuda.device):
    # Initialize model object
    model = models.resnet50(weights="DEFAULT").to(device)
    
    # Turn off gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    # Modify last layer to fit number of classes
    model.fc = torch.nn.Linear(in_features=2048, out_features=output_features, bias=True).to(device)
    
    return model