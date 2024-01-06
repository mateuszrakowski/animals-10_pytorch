import torch, os
import pandas as pd
from pathlib import Path


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