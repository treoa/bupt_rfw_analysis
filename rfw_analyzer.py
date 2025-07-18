import os
import gc
import csv
import sys
import cv2
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
import omegaconf
import ultralytics
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
import onnxruntime as ort  # Added for YOLOv11 ONNX model

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
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List, Optional, Union
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.models import vit_l_32, ViT_L_32_Weights, list_models
from transformers import AutoImageProcessor, SiglipForImageClassification
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

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
    "images_dir": "/workspace/BalancedFace/images",
    # "images_dir": "/workspace/images/test/data",
    "detection_model_path": "weights/yolov8x_person_face.pt",
    "attribute_model_path": "weights/mivolo_imdb.pth.tar",
    "max_workers": 16,
    "race_dirs": ["Asian", "Indian", "African", "Caucasian"],
    "batch_size": 100,
    "supported_extensions": ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'],
    "skip_existing_csv": True,
    "image_csv_path": "images.csv",
    "identity_csv_path": "identities.csv",
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

def setup_model(device):
    """
    Loads the Gender-Classifier-Mini model and processor from Hugging Face.

    Args:
        device (str): The device to load the model onto ('cuda' or 'cpu').

    Returns:
        tuple: A tuple containing the loaded model and processor.
    """
    print("Loading the Gender-Classifier-Mini model...")
    model_name = "prithivMLmods/Gender-Classifier-Mini"
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = SiglipForImageClassification.from_pretrained(model_name)
        model.to(device)
        print("Model loaded successfully.")
        return model, processor
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have an active internet connection and the 'transformers' library is installed.")
        return None, None





def classify_gender_batch(image_paths, model, processor, device):
    """
    Classifies the gender for a batch of images. [1]

    Args:
        image_paths (list): A list of paths to the input images.
        model: The pre-trained classification model.
        processor: The image processor for the model.
        device (str): The device the model is on ('cuda' or 'cpu').

    Returns:
        list: A list of dictionaries, each containing the image path and its prediction.
    """
    batch_results = list()
    images_to_process = list()
    valid_paths = list()

    # Open images and filter out any that can't be opened
    for image_path in image_paths:
        try:
            img = Image.open(image_path).convert("RGB")
            images_to_process.append(img)
            valid_paths.append(image_path)
        except Exception as e:
            print(f"Warning: Could not open or process image {image_path}. Error: {e}")
            batch_results.append({
                "image_path": image_path,
                "dominant_gender": "Error",
                "confidence": 0.0,
                "female_confidence": 0.0,
                "male_confidence": 0.0
            })

    if not images_to_process:
        return batch_results

    # Process the batch of images
    inputs = processor(images=images_to_process, return_tensors="pt").to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    
    labels = model.config.id2label
    
    for i, image_path in enumerate(valid_paths):
        # Get probabilities for the current image in the batch
        image_probs = probs[i].cpu().tolist()
        
        # Determine dominant gender and confidence
        dominant_index = torch.argmax(probs[i]).item()
        dominant_gender = labels[dominant_index]
        confidence = image_probs[dominant_index]

        # Create a result dictionary
        result = {
            "image_path": image_path,
            "dominant_gender": dominant_gender,
            "confidence": round(confidence, 4),
        }
        # Add individual class confidences
        for j, label in labels.items():
            clean_label = label.lower().split(" ") + "_confidence"
            result[clean_label] = round(image_probs[j], 4)
            
        batch_results.append(result)

    return batch_results

def main(input_dir, output_csv, batch_size=32):
    """
    Main function to orchestrate the batch classification process.

    Args:
        input_dir (str): The directory containing images to classify.
        output_csv (str): The path to save the output CSV file.
        batch_size (int): The number of images to process in each batch.
    """
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the model
    model, processor = setup_model(device)
    if model is None:
        return

    # Find all image files in the input directory
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_paths = list()
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    if not image_paths:
        print(f"No images found in directory: {input_dir}")
        return

    print(f"Found {len(image_paths)} images to process.")

    all_results = list()
    # Process images in batches with a progress bar
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing Batches"):
        batch_paths = image_paths[i:i + batch_size]
        batch_results = classify_gender_batch(batch_paths, model, processor, device)
        all_results.extend(batch_results)

    # Convert results to a pandas DataFrame and save to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"\nProcessing complete. Results saved to {output_csv}")


def main_cli():
    
    input_dir = "./"
    output_csv = "./output.csv"
    
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Batch Gender Classification using Gender-Classifier-Mini.")
    parser.add_argument("--input_dir", type=str, help="Directory containing the images to be classified.")
    parser.add_argument("--output_csv", type=str, help="Path to the output CSV file to save results.")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of images to process in a single batch. Adjust based on your GPU memory.")
    
    args = parser.parse_args()

    main(args.input_dir or input_dir, args.output_csv or output_csv, args.batch_size)