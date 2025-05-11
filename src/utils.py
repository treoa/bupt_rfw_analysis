"""
Utility functions for the RFW analysis project.
"""

import os
import glob
import json
from pathlib import Path
from PIL import Image
import numpy as np
from rich.console import Console
from rich.progress import track

console = Console()

def get_race_paths():
    """
    Get paths to race directories.
    
    Returns:
        dict: Dictionary with race names as keys and paths as values
    """
    # Get current full directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Check if in the current dir, that is in lower case, substring "BUPT"
    if "bupt" in str(current_dir).lower():
        base_path = Path("images")
    elif "rfw" in str(current_dir).lower():
        base_path = Path("images/test/data")
    races = ["African", "Asian", "Caucasian", "Indian"]
    race_paths = {}
    
    for race in races:
        race_path = base_path / race
        race_paths[race] = race_path
    
    return race_paths

def get_identities(race_path):
    """
    Get all identity folders for a given race.
    
    Args:
        race_path (Path): Path to race directory
        
    Returns:
        list: List of paths to identity directories
    """
    return [p for p in race_path.iterdir() if p.is_dir()]

def get_images(identity_path):
    """
    Get all image paths for a given identity.
    
    Args:
        identity_path (Path): Path to identity directory
        
    Returns:
        list: List of paths to image files
    """
    return list(identity_path.glob("*.jpg"))

def load_image(image_path):
    """
    Load image from file.
    
    Args:
        image_path (Path): Path to image file
        
    Returns:
        PIL.Image: Loaded image
    """
    try:
        return Image.open(image_path)
    except Exception as e:
        console.print(f"[bold red]Error loading image {image_path}: {e}[/bold red]")
        return None

def calculate_statistics(data):
    """
    Calculate basic statistics for a list of values.
    
    Args:
        data (list): List of numerical values
        
    Returns:
        dict: Dictionary with statistics
    """
    if not data:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None
        }
    
    return {
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data))
    }

def save_results(results, output_path="results.json"):
    """
    Save analysis results to a JSON file.
    
    Args:
        results (dict): Analysis results
        output_path (str): Path to output file
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    console.print(f"[bold green]Results saved to {output_path}[/bold green]")
