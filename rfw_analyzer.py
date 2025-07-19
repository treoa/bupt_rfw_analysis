import os
import gc
import csv
import sys
import json
import time
import types
import torch
import shutil
import random
import string
import asyncio
import logging
import requests
import warnings
import argparse
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from glob import glob
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from datetime import datetime
from torchvision import transforms
from collections import defaultdict
from matplotlib import pyplot as plt
from torch.utils.data import random_split
from torch.optim.lr_scheduler import StepLR
from typing import Dict, Tuple, List, Optional, Union
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import vit_l_32, ViT_L_32_Weights, list_models
from transformers import AutoImageProcessor, SiglipForImageClassification
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from rich import print
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn, TaskID


console = Console()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("rich")
logger.setLevel(logging.INFO)

CONFIG = {
    # "images_dir": "/workspace/BalancedFace/images",
    "images_dir": "/workspace/BalancedFace/images",
    "model_name": "prithivMLmods/Gender-Classifier-Mini",
    "max_workers": 16,
    "race_dirs": ["Asian", "Indian", "African", "Caucasian"],
    "batch_size": 400,
    "supported_extensions": ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'],
    "skip_existing_csv": True,
    "output_csv": "balancedface_results.csv",
    "plots_path": "plots",
    "use_mps": False,
}

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    logger.info("🚀 NVIDIA GPU detected! Using CUDA for acceleration")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    CONFIG["use_mps"] = True
    CONFIG["max_workers"] = 8 # Reduce workers on MPS to avoid memory issues
    logger.info("🚀 Apple Silicon GPU detected! Using Metal Performance Shaders for acceleration")
else:
    DEVICE = torch.device("cpu")
    logger.info("💻 Using CPU for processing")

def setup_model():
    """
    Sets up the model, processor, and device.
    Returns:
        tuple: (model, processor, device)
    """
    console.log("[bold cyan]Setting up the model...[/bold cyan]")
    # Determine the device to run the model on (GPU if available, otherwise CPU)
    console.log(f"Using device: [bold yellow]{DEVICE}[/bold yellow]")

    try:
        # Load the processor and model from Hugging Face Hub
        processor = AutoImageProcessor.from_pretrained(CONFIG["model_name"])
        model = SiglipForImageClassification.from_pretrained(CONFIG["model_name"])
        # Move the model to the selected device
        model.to(DEVICE)
        # Set the model to evaluation mode
        model.eval()
        console.log("[bold green]Model setup complete![/bold green] :rocket:")
        return model, processor, DEVICE
    except Exception as e:
        console.log(f"[bold red]Error setting up the model: {e}[/bold red]")
        exit()

def discover_images(data_root):
    """
    Discovers all image files in the specified directory structure.
    Args:
        data_root (str): The path to the data directory.
    Returns:
        list: A list of dictionaries, each containing image_path, race, and identity.
    """
    console.log(f"Discovering images in [bold magenta]{data_root}[/bold magenta]...")
    image_list = []
    # Use Pathlib for robust path manipulation
    root_path = Path(data_root)
    if not root_path.is_dir():
        console.log(f"[bold red]Error: The directory {data_root} does not exist.[/bold red]")
        return []

    # Iterate through race folders
    for race_path in root_path.iterdir():
        if race_path.is_dir():
            race = race_path.name
            # Iterate through identity folders
            for identity_path in race_path.iterdir():
                if identity_path.is_dir():
                    identity = identity_path.name
                    # Find all image files in the identity folder
                    for image_file in identity_path.glob('*.*'):
                        if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                            image_list.append({
                                "image_path": str(image_file),
                                "race": race,
                                "identity": identity
                            })
    
    console.log(f"Found [bold blue]{len(image_list)}[/bold blue] total images.")
    return image_list

def process_batch(batch_data, model, processor, device):
    """
    Processes a batch of images and returns the classification results.
    Args:
        batch_data (list): A list of dictionaries with image data.
        model: The loaded classification model.
        processor: The model's image processor.
        device: The device to run inference on.
    Returns:
        list: A list of dictionaries with detailed results for each image.
    """
    batch_results = []
    images = []
    valid_paths = []

    # Open images and collect valid ones
    for item in batch_data:
        try:
            img = Image.open(item["image_path"]).convert("RGB")
            images.append(img)
            valid_paths.append(item)
        except Exception as e:
            # Handle cases where an image file is corrupt or unreadable
            console.log(f"[yellow]Warning: Could not open {item['image_path']}. Skipping. Error: {e}[/yellow]")
            batch_results.append({
                "image_path": item["image_path"], "identity": item["identity"],
                "race": item["race"], "conf_score": None, "gender": "Error",
                "img_width": None, "img_height": None
            })

    if not images:
        return batch_results

    # Prepare inputs for the model
    inputs = processor(images=images, return_tensors="pt").to(device)

    # Perform inference without calculating gradients
    with torch.no_grad():
        outputs = model(**inputs)

    # Get probabilities and predicted labels
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    conf_scores, predicted_indices = torch.max(probs, dim=1)
    
    # Get labels from model config
    id2label = model.config.id2label

    # Process results for each image in the batch
    for i in range(len(images)):
        item = valid_paths[i]
        img = images[i]
        
        # Get the full label (e.g., "Male ♂") and clean it
        full_gender_label = id2label[predicted_indices[i].item()]
        cleaned_gender_label = full_gender_label.split(' ')[0]

        batch_results.append({
            "image_path": item["image_path"],
            "identity": item["identity"],
            "race": item["race"],
            "conf_score": round(conf_scores[i].item(), 4),
            "gender": cleaned_gender_label, # <-- MODIFIED VALUE USED HERE
            "img_width": img.width,
            "img_height": img.height,
        })
        
    return batch_results


def main():
    """
    Main function to run the entire classification pipeline.
    """
    model, processor, device = setup_model()
    
    image_files = discover_images(CONFIG["images_dir"])
    if not image_files:
        console.log("[bold red]No images found. Exiting.[/bold red]")
        return
        
    all_results = []
    
    # Process images in batches with a progress bar
    console.log(f"Starting batch processing with a batch size of {CONFIG["batch_size"]}...")
    for i in tqdm(range(0, len(image_files), CONFIG["batch_size"]), desc="Processing Batches"):
        batch = image_files[i:i + CONFIG["batch_size"]]
        batch_results = process_batch(batch, model, processor, device)
        all_results.extend(batch_results)

    # Convert results to a pandas DataFrame and save to CSV
    console.log(f"\n[bold cyan]Processing complete. Saving results to {CONFIG['output_csv']}...[/bold cyan]")
    df = pd.DataFrame(all_results)
    df.to_csv(CONFIG["output_csv"], index=False)
    
    console.log(f"[bold green]Successfully saved {len(df)} results to {CONFIG["output_csv"]}[/bold green] :heavy_check_mark:")
    console.print("\n[bold]Preview of the first 5 results:[/bold]")
    console.print(df.head())


if __name__ == "__main__":
    main()

