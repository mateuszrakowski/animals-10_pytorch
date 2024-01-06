import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


def make_prediction(model: torch.nn.Module,
                    img_path: str,
                    class_names: list,
                    device=torch.cuda.device,
                    transform=None):
    # Set PIL Image object
    img = Image.open(img_path)
    
    # Setup transform
    if transform:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
                transforms.Resize(size=(384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Perform transform
    img_tensor = image_transform(img)
        
    # Set evaluation mode
    model.eval()
    
    # Forward pass
    with torch.inference_mode():
        # Add an additional dimension (batch dimension)
        y_pred = model(img_tensor.unsqueeze(dim=0).to(device))
        
    y_prob = torch.softmax(y_pred, dim=1)
    y_label = torch.argmax(y_prob, dim=1)
    
    # Plot an image with predicted label 
    plt.figure(figsize=(3, 3))
    plt.title(f"Pred: {class_names[y_label]} | Prob: {y_prob.max():.2f}")
    plt.imshow(img)
    plt.axis("off")