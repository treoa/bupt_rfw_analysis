"""
Module for managing face analysis models.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pathlib import Path
from rich.console import Console

console = Console()

class AgeGenderModel(nn.Module):
    """
    A model for joint age and gender prediction.
    """
    def __init__(self, backbone='resnet50'):
        """
        Initialize the model with a specified backbone.
        
        Args:
            backbone (str): Backbone architecture ('resnet18', 'resnet34', 'resnet50', etc.)
        """
        super(AgeGenderModel, self).__init__()
        
        # Load backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            num_features = self.backbone.fc.in_features
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            num_features = self.backbone.fc.in_features
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            num_features = self.backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Replace the final layer
        self.backbone.fc = nn.Identity()
        
        # Create separate heads for age and gender
        self.shared = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.age_head = nn.Linear(256, 1)
        self.gender_head = nn.Linear(256, 2)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            tuple: (age_prediction, gender_prediction)
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Pass through shared layers
        shared_features = self.shared(features)
        
        # Age prediction (regression)
        age = self.age_head(shared_features)
        
        # Gender prediction (binary classification)
        gender = self.gender_head(shared_features)
        
        return age, gender

def download_pretrained_model(model_dir='models', model_name='age_gender_model.pth'):
    """
    Download pretrained model or create a new one if not available.
    
    Args:
        model_dir (str): Directory to save/load model
        model_name (str): Name of the model file
        
    Returns:
        str: Path to the model file
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True, parents=True)
    
    model_path = model_dir / model_name
    
    if model_path.exists():
        console.print(f"[green]Found existing model at {model_path}[/green]")
        return str(model_path)
    
    console.print("[yellow]No pretrained model found. Creating a new model...[/yellow]")
    
    # Create a new model
    model = AgeGenderModel(backbone='resnet50')
    
    # Save the model
    torch.save(model.state_dict(), model_path)
    
    console.print(f"[green]Created new model at {model_path}[/green]")
    
    return str(model_path)

def load_model(model_path, device=None):
    """
    Load model from path.
    
    Args:
        model_path (str): Path to model file
        device (torch.device, optional): Device to load model to
        
    Returns:
        AgeGenderModel: Loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    console.print(f"[blue]Loading model from {model_path} to {device}[/blue]")
    
    # Create model
    model = AgeGenderModel(backbone='resnet50')
    
    # Load weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        console.print("[green]Model loaded successfully[/green]")
    except Exception as e:
        console.print(f"[yellow]Error loading model weights: {e}. Using default initialization.[/yellow]")
    
    # Set model to evaluation mode
    model.eval()
    
    return model.to(device)
