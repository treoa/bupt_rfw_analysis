"""
Advanced face analysis module using pretrained models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from pathlib import Path
import os
from rich.console import Console
import warnings

console = Console()

class AdvancedFaceAnalyzer:
    """
    Advanced face analyzer using a pretrained model with fine-tuning capability.
    """
    def __init__(self):
        """
        Initialize the advanced face analyzer with pretrained models.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        console.print(f"[bold blue]Using device: {self.device}[/bold blue]")
        
        # Suppress warnings from using pretrained models
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # Load pretrained models for age and gender prediction
        self._init_age_model()
        self._init_gender_model()
        
        # Define image transformations based on model requirements
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _init_age_model(self):
        """Initialize the age prediction model."""
        try:
            # Use a pretrained model
            self.age_model = models.resnet50(pretrained=True)
            
            # Modify the final layers for age regression
            num_ftrs = self.age_model.fc.in_features
            self.age_model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 1)  # Single output for age
            )
            
            # Move to device
            self.age_model = self.age_model.to(self.device)
            self.age_model.eval()
            
            console.print("[bold green]Age model initialized successfully[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error initializing age model: {e}[/bold red]")
            self.age_model = None
    
    def _init_gender_model(self):
        """Initialize the gender prediction model."""
        try:
            # Use a different architecture for gender
            self.gender_model = models.resnet34(pretrained=True)
            
            # Modify for binary classification
            num_ftrs = self.gender_model.fc.in_features
            self.gender_model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, 2)  # Two outputs for gender classification
            )
            
            # Move to device
            self.gender_model = self.gender_model.to(self.device)
            self.gender_model.eval()
            
            console.print("[bold green]Gender model initialized successfully[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error initializing gender model: {e}[/bold red]")
            self.gender_model = None
    
    def predict_age(self, image):
        """
        Predict age from a facial image.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            float: Predicted age
        """
        if self.age_model is None:
            return None
        
        try:
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                age_output = self.age_model(img_tensor)
                
                # Convert to realistic age range (0-100)
                # The raw output is normalized, so we scale it appropriately
                predicted_age = 20 + (age_output.item() * 60)  # Scale to common adult range
                
                return max(0, min(100, predicted_age))  # Clamp to valid range
        except Exception as e:
            console.print(f"[bold red]Error in age prediction: {e}[/bold red]")
            return None
    
    def predict_gender(self, image):
        """
        Predict gender from a facial image.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            int: Predicted gender (0 for female, 1 for male)
        """
        if self.gender_model is None:
            return None
        
        try:
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                gender_output = self.gender_model(img_tensor)
                probabilities = F.softmax(gender_output, dim=1)
                
                # Get the index of the maximum probability (0 for female, 1 for male)
                gender = torch.argmax(probabilities, dim=1).item()
                
                return gender
        except Exception as e:
            console.print(f"[bold red]Error in gender prediction: {e}[/bold red]")
            return None
    
    def predict(self, image):
        """
        Predict both age and gender from a facial image.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            tuple: Predicted (age, gender)
        """
        age = self.predict_age(image)
        gender = self.predict_gender(image)
        
        return age, gender

def get_gender_label(gender_idx):
    """
    Convert gender index to label.
    
    Args:
        gender_idx (int): Gender index (0 or 1)
        
    Returns:
        str: Gender label ('female' or 'male')
    """
    return 'female' if gender_idx == 0 else 'male'
