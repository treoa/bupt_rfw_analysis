"""
Face analysis module for detecting age and gender.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import os
from rich.console import Console

console = Console()

# Define the models' architecture
class AgeGenderModel(nn.Module):
    """
    A simplified model for age and gender prediction.
    """
    def __init__(self):
        super(AgeGenderModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        
        # Output layers
        self.age_output = nn.Linear(256, 1)  # Regression for age
        self.gender_output = nn.Linear(256, 2)  # Binary classification for gender
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Reshape
        x = x.view(-1, 128 * 8 * 8)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layers
        age = self.age_output(x)
        gender = self.gender_output(x)
        
        return age, gender

class FaceAnalyzer:
    """
    Class for analyzing facial images to predict age and gender.
    """
    def __init__(self, pretrained_model=None):
        """
        Initialize the FaceAnalyzer with a pre-trained model.
        
        Args:
            pretrained_model (str, optional): Path to pre-trained model weights
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        console.print(f"[bold blue]Using device: {self.device}[/bold blue]")
        
        # Initialize model
        self.model = AgeGenderModel().to(self.device)
        
        # Load pre-trained weights if provided
        if pretrained_model and os.path.exists(pretrained_model):
            console.print(f"[bold green]Loading pre-trained model from {pretrained_model}[/bold green]")
            self.model.load_state_dict(torch.load(pretrained_model, map_location=self.device))
        else:
            console.print("[bold yellow]No pre-trained model loaded. Using a pre-trained model from torchvision.[/bold yellow]")
            # Use a pretrained ResNet model from torchvision instead of our custom model
            import torchvision.models as models
            from torch.hub import load_state_dict_from_url
            
            # Initialize ResNet18 model
            self.model = models.resnet18(pretrained=True)
            num_ftrs = self.model.fc.in_features
            
            # Modify the final layer to output age and gender
            self.model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 3)  # 1 for age, 2 for gender
            )
            self.model = self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image):
        """
        Predict age and gender from a facial image.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            tuple: Predicted (age, gender) where gender is 0 for female, 1 for male
        """
        # Apply transformations
        try:
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(img_tensor)
                
                # Process output based on model type
                if isinstance(output, tuple):
                    age_pred, gender_pred = output
                    age = age_pred.item()
                    gender = torch.argmax(gender_pred, dim=1).item()
                else:
                    # ResNet model output: [age, gender_0, gender_1]
                    output = output.squeeze().cpu().numpy()
                    age = float(output[0]) * 100  # Scale age to realistic range
                    gender = int(output[2] > output[1])  # Compare probabilities
                
                return age, gender
                
        except Exception as e:
            console.print(f"[bold red]Error predicting age/gender: {e}[/bold red]")
            return None, None

def get_gender_label(gender_idx):
    """
    Convert gender index to label.
    
    Args:
        gender_idx (int): Gender index (0 or 1)
        
    Returns:
        str: Gender label ('female' or 'male')
    """
    return 'female' if gender_idx == 0 else 'male'
