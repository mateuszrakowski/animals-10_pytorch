import torch
import torchmetrics
import numpy as np
from tqdm.auto import tqdm

def train_step(model: torch.nn.Module, 
               train_dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn,
               optimizer: torch.optim,
               accuracy_fn: torchmetrics.Accuracy,
               device: torch.cuda.device):
    
    torch.manual_seed(42)

    # Set training mode
    model.train()
    
    # Setup loss & accuracy variables
    model_loss = 0
    model_accuracy = 0
    
    # Training batch loop
    for batch_idx, (X, y) in enumerate(train_dataloader):
        # Push data to device
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        y_pred = model(X)
        
        # Calculate loss & accuracy
        loss = loss_fn(y_pred, y)
        model_loss += loss.item()
        
        accuracy = accuracy_fn(y_pred, y)
        model_accuracy += accuracy
        
        # Optimizer zero-grad
        optimizer.zero_grad()
        
        # Backpropagation
        loss.backward()
        
        # Optimizer step
        optimizer.step()
    
    # Adjust the metrics
    model_loss = model_loss / len(train_dataloader)
    model_accuracy = model_accuracy / len(train_dataloader)
    
    return model_loss, model_accuracy


def test_step(model: torch.nn.Module,
             test_dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn,
             accuracy_fn: torchmetrics.Accuracy,
             device: torch.cuda.device):
    
    torch.manual_seed(42)
    
    # Setup eval mode
    model.eval()
    
    # Setup loss & accuracy variables
    model_loss = 0
    model_accuracy = 0
    
    with torch.inference_mode():
        # Test batch loop
        for batch_idx, (X, y) in enumerate(test_dataloader):
            # Send data to device
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_pred = model(X)

            # Calculate loss & accuracy
            loss = loss_fn(y_pred, y)
            model_loss += loss.item()

            accuracy = accuracy_fn(y_pred, y).to(device)
            model_accuracy += accuracy
    
        # Adjust the metrics
        model_loss = model_loss / len(test_dataloader)
        model_accuracy = model_accuracy / len(test_dataloader)
    
    return model_loss, model_accuracy


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn,
          optimizer: torch.optim,
          num_epochs: int,
          accuracy_fn: torchmetrics.Accuracy,
          device: torch.cuda.device):
    
    torch.manual_seed(42)
    
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
        
    for epoch in tqdm(range(num_epochs)):
        train_loss, train_acc = train_step(model=model,
                                           train_dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           accuracy_fn=accuracy_fn,
                                           device=device)
        
        test_loss, test_acc = test_step(model=model,
                                        test_dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        accuracy_fn=accuracy_fn,
                                        device=device)
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
        tqdm.write(f'Epoch {epoch+1}/{num_epochs} - '
                   f'Train Loss: {train_loss:.4f} - '
                   f'Train Accuracy: {train_acc:.2%} - '
                   f'Validation Loss: {test_loss:.4f} - '
                   f'Validation Accuracy: {test_acc:.2%}')

    return model, results


def evaluation_model(model: torch.nn.Module,
                     test_dataloader: torch.utils.data.DataLoader,
                     device: torch.cuda.device):
    
    torch.manual_seed(42)
    
    # Setup eval mode
    model.eval()
    
    # Initialize results dictionary
    results = {"predictions": [],
               "targets": [],
               "probability": []}
    
    with torch.inference_mode():
        # Test batch loop
        for batch_idx, (X, y) in enumerate(test_dataloader):
            # Send data to device
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_pred = model(X)
            y_prob = torch.softmax(y_pred, dim=1)
            y_label = torch.argmax(y_prob, dim=1)

            # Append targets and predictions
            results["predictions"] += y_label.cpu()
            results["probability"] += y_prob.cpu()
            results["targets"] += y.cpu()
    results["probability"] = [prob.max() for prob in results["probability"]]
    
    return results
