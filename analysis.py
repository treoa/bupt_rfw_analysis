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

from PIL import Image
from glob import glob
from tqdm import tqdm
from pathlib import Path
from pprint import pprint
from datetime import datetime
from torchvision import transforms
from collections import defaultdict
from matplotlib import pyplot as plt
from torch.utils.data import random_split
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Set, Dict, Tuple, List, Any, Optional
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torchvision.models import vit_l_32, ViT_L_32_Weights, list_models, resnext101_64x4d, ResNeXt101_64X4D_Weights

from rich import print
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn

try:
    from facial_analysis.utils.helpers import Face, draw_face_info
    from facial_analysis.models import SCRFD, Attribute, ArcFace, YOLO
except ImportError:
    print("[yellow]Warning: facial_analysis package not found.[/yellow]")
    raise ImportError

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("rich")
logger.setLevel(logging.INFO)

# Set matplotlib style for modern looking plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Check for Apple Silicon GPU availability
DEVICE = None
USE_MPS = False

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    USE_MPS = True
    logger.info("🚀 Apple Silicon GPU detected! Using Metal Performance Shaders for acceleration")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    logger.info("🚀 NVIDIA GPU detected! Using CUDA for acceleration")
else:
    DEVICE = torch.device("cpu")
    logger.info("💻 Using CPU for processing")

CONFIG = {
    # "images_dir": "BalancedFace/images",
    "images_dir": "images/test/data",
    "detection_model_path": "facial_analysis/weights/yolov8l_100e.pt",
    "attribute_model_path": "facial_analysis/weights/genderage.onnx",
    "max_workers": 8 if USE_MPS else 16,  # Reduce workers on MPS to avoid memory issues
    "race_dirs": ["Asian", "Indian", "African", "Caucasian"],
    "batch_size": 100,
    "gender_batch_size": 10,
    "gpu_batch_size": 32,
    "supported_extensions": ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'],
    "skip_existing_csv": True,
    "use_gpu_acceleration": True,
    "image_size": (224, 224),
}

# Define consistent color palette
RACE_COLORS = {
    'African': '#FF6B6B',    # Red
    'Asian': '#4ECDC4',      # Teal
    'Caucasian': '#45B7D1',  # Blue
    'Indian': '#96CEB4'      # Green
}

GENDER_COLORS = {
    'Male': '#3498db',       # Blue
    'Female': '#e74c3c',     # Red
    'Unknown': '#95a5a6'     # Gray
}


class GenderDetectorGPU:
    """GPU-accelerated gender detection for facial analysis."""
    
    def __init__(self, detection_model_path: str, attribute_model_path: str):
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'

        try:
            # self.detection_model = SCRFD(model_path=detection_model_path).to(device)
            self.detection_model = YOLO(model=detection_model_path).to(device=device)
            self.attribute_model = Attribute(model_path=attribute_model_path).to(device)
            logger.info("Gender detection models loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load models: {e}. Using simulation mode.")
            self.use_simulation = True
        
    
    def detect_gender(self, image_path: Path) -> Tuple[str, float]:
        """
        Detect gender from image.
        Returns: (gender, confidence) where gender is 'Male', 'Female', or 'Unknown'
        """
        try:
            # Read image
            frame = cv2.imread(str(image_path))
            if frame is None:
                logger.warning(f"Could not read image: {image_path}")
                return "Unknown", 0.0
            
            # Detect faces using YOLOv8
            crops, boxes, scores, cls_ids = self.detection_model.detect_faces([frame])
            
            # Handle case where no faces are detected
            if len(boxes[0]) == 0:
                logger.debug(f"No faces detected in: {image_path}")
                return "Unknown", 0.0
            
            # Get the highest confidence detection
            first_image_boxes = boxes[0]
            first_image_scores = scores[0]
            
            if len(first_image_scores) == 0:
                return "Unknown", 0.0
                
            # Find index of highest confidence detection
            best_idx = np.argmax(first_image_scores)
            bbox = first_image_boxes[best_idx]
            conf_score = first_image_scores[best_idx]
            
            # Convert bbox to list if it's numpy array
            bbox = bbox.tolist() if hasattr(bbox, 'tolist') else bbox
            
            # Get gender and age
            gender, age = self.attribute_model.get(frame, bbox)
            
            gender = "Male" if gender == 1 else "Female"
            
            return str(gender), float(conf_score)
            
        except Exception as e:
            logger.error(f"Error detecting gender for {image_path}: {e}")
            return "Unknown", 0.0


class GPUOptimizedDimensionExtractor:
    """GPU-optimized dimension extraction using batch processing."""
    
    def __init__(self):
        self.device = DEVICE
        self.use_gpu = CONFIG["use_gpu_acceleration"] and DEVICE.type != "cpu"
    
    def extract_dimensions_batch(self, image_paths: List[Path]) -> List[Tuple[int, int]]:
        """Extract dimensions for multiple images efficiently."""
        dimensions = []
        
        if self.use_gpu and USE_MPS:
            # Process in batches on GPU
            batch_size = CONFIG["gpu_batch_size"]
            
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                batch_dims = []
                
                # Load images and extract dimensions
                for path in batch_paths:
                    try:
                        with Image.open(path) as img:
                            batch_dims.append((img.width, img.height))
                    except Exception as e:
                        logger.error(f"Error processing {path}: {e}")
                        batch_dims.append((0, 0))
                
                dimensions.extend(batch_dims)
                
                # Force memory cleanup on MPS
                if USE_MPS and i % (batch_size * 5) == 0:
                    torch.mps.empty_cache()
        else:
            # CPU fallback
            for path in image_paths:
                try:
                    with Image.open(path) as img:
                        dimensions.append((img.width, img.height))
                except Exception as e:
                    logger.error(f"Error processing {path}: {e}")
                    dimensions.append((0, 0))
        
        return dimensions


class StatisticalAnalyzer:
    """Handles statistical analysis and plotting of the dataset."""
    
    @staticmethod
    def calculate_stats(data: List[float]) -> Dict[str, float]:
        """Calculate statistical metrics for a list of values."""
        if not data:
            return {
                'min': 0, 'max': 0, 'mean': 0, 
                'median': 0, 'std': 0, 'count': 0
            }
        
        data_array = np.array(data)
        return {
            'min': float(np.min(data_array)),
            'max': float(np.max(data_array)),
            'mean': float(np.mean(data_array)),
            'median': float(np.median(data_array)),
            'std': float(np.std(data_array)),
            'count': len(data)
        }
    
    @staticmethod
    def plot_dimension_distribution_per_race(images_df: pd.DataFrame, output_dir: Path):
        """Plot the distribution of image dimensions separately for each race."""
        output_dir.mkdir(exist_ok=True)
        
        races = sorted(images_df['race'].unique())
        
        # Create a figure for each race
        for race in races:
            race_df = images_df[images_df['race'] == race]
            
            if len(race_df) == 0:
                continue
            
            # Calculate resolution
            race_df['resolution'] = race_df['width'] * race_df['height']
            
            # Create figure with subplots
            fig = plt.figure(figsize=(16, 10))
            fig.suptitle(f'Image Dimensions Distribution - {race}', fontsize=16, fontweight='bold')
            
            # Subplot 1: Width histogram
            ax1 = plt.subplot(2, 3, 1)
            width_data = race_df['width'].values
            ax1.hist(width_data, bins=50, alpha=0.7, color=RACE_COLORS.get(race, '#333'), edgecolor='black')
            ax1.set_xlabel('Width (pixels)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Width Distribution')
            ax1.grid(True, alpha=0.3)
            
            # Add statistics text
            width_stats = StatisticalAnalyzer.calculate_stats(width_data.tolist())
            stats_text = f"μ={width_stats['mean']:.0f}, σ={width_stats['std']:.0f}\nmed={width_stats['median']:.0f}\nrange=[{width_stats['min']:.0f}, {width_stats['max']:.0f}]"
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Subplot 2: Height histogram
            ax2 = plt.subplot(2, 3, 2)
            height_data = race_df['height'].values
            ax2.hist(height_data, bins=50, alpha=0.7, color=RACE_COLORS.get(race, '#333'), edgecolor='black')
            ax2.set_xlabel('Height (pixels)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Height Distribution')
            ax2.grid(True, alpha=0.3)
            
            # Add statistics text
            height_stats = StatisticalAnalyzer.calculate_stats(height_data.tolist())
            stats_text = f"μ={height_stats['mean']:.0f}, σ={height_stats['std']:.0f}\nmed={height_stats['median']:.0f}\nrange=[{height_stats['min']:.0f}, {height_stats['max']:.0f}]"
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Subplot 3: Resolution histogram
            ax3 = plt.subplot(2, 3, 3)
            resolution_data = race_df['resolution'].values
            ax3.hist(resolution_data, bins=50, alpha=0.7, color=RACE_COLORS.get(race, '#333'), edgecolor='black')
            ax3.set_xlabel('Resolution (pixels²)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Resolution Distribution')
            ax3.grid(True, alpha=0.3)
            
            # Subplot 4: Width vs Height scatter
            ax4 = plt.subplot(2, 3, 4)
            ax4.scatter(race_df['width'], race_df['height'], alpha=0.5, 
                       color=RACE_COLORS.get(race, '#333'), s=10)
            ax4.set_xlabel('Width (pixels)')
            ax4.set_ylabel('Height (pixels)')
            ax4.set_title('Width vs Height Scatter Plot')
            ax4.grid(True, alpha=0.3)
            
            # Add diagonal line for square images
            min_dim = min(race_df['width'].min(), race_df['height'].min())
            max_dim = max(race_df['width'].max(), race_df['height'].max())
            ax4.plot([min_dim, max_dim], [min_dim, max_dim], 'k--', alpha=0.5, label='Square images')
            ax4.legend()
            
            # Subplot 5: Aspect ratio distribution
            ax5 = plt.subplot(2, 3, 5)
            aspect_ratios = race_df['width'] / race_df['height']
            ax5.hist(aspect_ratios, bins=50, alpha=0.7, color=RACE_COLORS.get(race, '#333'), edgecolor='black')
            ax5.set_xlabel('Aspect Ratio (Width/Height)')
            ax5.set_ylabel('Frequency')
            ax5.set_title('Aspect Ratio Distribution')
            ax5.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Square (1:1)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # Subplot 6: Statistics summary table
            ax6 = plt.subplot(2, 3, 6)
            ax6.axis('tight')
            ax6.axis('off')
            
            # Create comprehensive statistics table
            stats_data = [
                ['Metric', 'Width', 'Height', 'Resolution'],
                ['Count', f"{width_stats['count']}", f"{height_stats['count']}", f"{len(resolution_data)}"],
                ['Mean', f"{width_stats['mean']:.0f}", f"{height_stats['mean']:.0f}", f"{np.mean(resolution_data):.0f}"],
                ['Std Dev', f"{width_stats['std']:.0f}", f"{height_stats['std']:.0f}", f"{np.std(resolution_data):.0f}"],
                ['Median', f"{width_stats['median']:.0f}", f"{height_stats['median']:.0f}", f"{np.median(resolution_data):.0f}"],
                ['Min', f"{width_stats['min']:.0f}", f"{height_stats['min']:.0f}", f"{np.min(resolution_data):.0f}"],
                ['Max', f"{width_stats['max']:.0f}", f"{height_stats['max']:.0f}", f"{np.max(resolution_data):.0f}"],
            ]
            
            table = ax6.table(cellText=stats_data, cellLoc='center', loc='center',
                            colColours=['lightblue']*4)
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'dimension_distribution_{race.lower()}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Saved dimension distribution plots for {len(races)} races")
    
    @staticmethod
    def plot_identity_distribution_per_race(images_df: pd.DataFrame, output_dir: Path):
        """Plot the distribution of images per identity separately for each race."""
        output_dir.mkdir(exist_ok=True)
        
        # Calculate images per identity
        identity_counts = images_df.groupby(['race', 'identity']).size().reset_index(name='image_count')
        races = sorted(identity_counts['race'].unique())
        
        for race in races:
            race_data = identity_counts[identity_counts['race'] == race]
            
            if len(race_data) == 0:
                continue
            
            fig = plt.figure(figsize=(14, 8))
            fig.suptitle(f'Images per Identity Distribution - {race}', fontsize=16, fontweight='bold')
            
            # Subplot 1: Histogram
            ax1 = plt.subplot(2, 2, 1)
            counts = race_data['image_count'].values
            ax1.hist(counts, bins=min(30, len(np.unique(counts))), alpha=0.7, 
                    color=RACE_COLORS.get(race, '#333'), edgecolor='black')
            ax1.set_xlabel('Images per Identity')
            ax1.set_ylabel('Number of Identities')
            ax1.set_title('Distribution Histogram')
            ax1.grid(True, alpha=0.3)
            
            # Add statistics
            stats = StatisticalAnalyzer.calculate_stats(counts.tolist())
            stats_text = f"μ={stats['mean']:.1f}, σ={stats['std']:.1f}\nmed={stats['median']:.0f}\nrange=[{stats['min']:.0f}, {stats['max']:.0f}]"
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Subplot 2: Box plot
            ax2 = plt.subplot(2, 2, 2)
            bp = ax2.boxplot(counts, vert=True, patch_artist=True, showmeans=True,
                           meanprops=dict(marker='o', markerfacecolor='red', markersize=8))
            bp['boxes'][0].set_facecolor(RACE_COLORS.get(race, '#333'))
            bp['boxes'][0].set_alpha(0.7)
            ax2.set_ylabel('Images per Identity')
            ax2.set_xticklabels([race])
            ax2.set_title('Box Plot with Outliers')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Subplot 3: Cumulative distribution
            ax3 = plt.subplot(2, 2, 3)
            sorted_counts = np.sort(counts)
            cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
            ax3.plot(sorted_counts, cumulative, color=RACE_COLORS.get(race, '#333'), linewidth=2)
            ax3.set_xlabel('Images per Identity')
            ax3.set_ylabel('Cumulative Probability')
            ax3.set_title('Cumulative Distribution Function')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1)
            
            # Add percentile markers
            percentiles = [25, 50, 75]
            for p in percentiles:
                val = np.percentile(counts, p)
                ax3.axvline(x=val, color='red', linestyle='--', alpha=0.5)
                ax3.text(val, 0.05, f'{p}%', rotation=90, verticalalignment='bottom')
            
            # Subplot 4: Statistics table
            ax4 = plt.subplot(2, 2, 4)
            ax4.axis('tight')
            ax4.axis('off')
            
            # Create detailed statistics
            unique_counts, freq = np.unique(counts, return_counts=True)
            mode_val = unique_counts[np.argmax(freq)]
            
            stats_table = [
                ['Statistic', 'Value'],
                ['Total Identities', f"{stats['count']}"],
                ['Total Images', f"{np.sum(counts)}"],
                ['Mean Images/Identity', f"{stats['mean']:.2f}"],
                ['Std Deviation', f"{stats['std']:.2f}"],
                ['Median', f"{stats['median']:.0f}"],
                ['Mode', f"{mode_val}"],
                ['Min', f"{stats['min']:.0f}"],
                ['Max', f"{stats['max']:.0f}"],
                ['25th Percentile', f"{np.percentile(counts, 25):.0f}"],
                ['75th Percentile', f"{np.percentile(counts, 75):.0f}"],
            ]
            
            table = ax4.table(cellText=stats_table, cellLoc='center', loc='center',
                            colColours=['lightblue', 'lightgreen'])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.2)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'identity_distribution_{race.lower()}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Saved identity distribution plots for {len(races)} races")
    
    @staticmethod
    def plot_gender_distribution_comparison(identities_df: pd.DataFrame, output_dir: Path):
        """Plot gender distribution comparison across all races."""
        output_dir.mkdir(exist_ok=True)
        
        # Calculate gender counts per race
        gender_counts = identities_df.groupby(['race', 'gender']).size().unstack(fill_value=0)
        
        # Create comprehensive comparison figure
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('Gender Distribution Comparison Across Races', fontsize=16, fontweight='bold')
        
        # Subplot 1: Stacked bar chart
        ax1 = plt.subplot(2, 3, 1)
        gender_counts.plot(kind='bar', stacked=True, ax=ax1,
                          color=[GENDER_COLORS.get(col, '#333') for col in gender_counts.columns])
        ax1.set_xlabel('Race')
        ax1.set_ylabel('Number of Identities')
        ax1.set_title('Stacked Bar Chart')
        ax1.legend(title='Gender')
        ax1.grid(True, alpha=0.3, axis='y')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Subplot 2: Grouped bar chart
        ax2 = plt.subplot(2, 3, 2)
        gender_counts.plot(kind='bar', ax=ax2,
                          color=[GENDER_COLORS.get(col, '#333') for col in gender_counts.columns])
        ax2.set_xlabel('Race')
        ax2.set_ylabel('Number of Identities')
        ax2.set_title('Grouped Bar Chart')
        ax2.legend(title='Gender')
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Subplot 3: Percentage stacked bar
        ax3 = plt.subplot(2, 3, 3)
        gender_pct = gender_counts.div(gender_counts.sum(axis=1), axis=0) * 100
        gender_pct.plot(kind='bar', stacked=True, ax=ax3,
                       color=[GENDER_COLORS.get(col, '#333') for col in gender_pct.columns])
        ax3.set_xlabel('Race')
        ax3.set_ylabel('Percentage (%)')
        ax3.set_title('Percentage Distribution')
        ax3.legend(title='Gender')
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3, axis='y')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Subplot 4-7: Individual pie charts for each race
        races = sorted(gender_counts.index)
        for i, race in enumerate(races):
            ax = plt.subplot(2, 4, 5 + i)
            race_data = gender_counts.loc[race]
            race_data = race_data[race_data > 0]
            
            wedges, texts, autotexts = ax.pie(
                race_data.values,
                labels=race_data.index,
                colors=[GENDER_COLORS.get(g, '#333') for g in race_data.index],
                autopct='%1.1f%%',
                startangle=90
            )
            ax.set_title(f'{race}', fontsize=12, fontweight='bold')
            
            # Make percentage text readable
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'gender_distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create individual race gender plots
        for race in races:
            race_df = identities_df[identities_df['race'] == race]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f'Gender Distribution - {race}', fontsize=14, fontweight='bold')
            
            # Gender counts
            gender_dist = race_df['gender'].value_counts()
            
            # Pie chart
            colors = [GENDER_COLORS.get(g, '#333') for g in gender_dist.index]
            wedges, texts, autotexts = ax1.pie(
                gender_dist.values,
                labels=gender_dist.index,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90
            )
            ax1.set_title('Gender Proportion')
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            # Bar chart with counts
            ax2.bar(gender_dist.index, gender_dist.values, color=colors, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Gender')
            ax2.set_ylabel('Count')
            ax2.set_title('Gender Counts')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (gender, count) in enumerate(gender_dist.items()):
                ax2.text(i, count + 10, str(count), ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_dir / f'gender_distribution_{race.lower()}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("Saved gender distribution plots")


class ImageProcessor:
    """Handles asynchronous image processing with GPU acceleration."""
    
    def __init__(self, data_dir: str = CONFIG["images_dir"], max_workers: int = CONFIG["max_workers"]):
        self.data_dir = Path(os.getcwd(), data_dir)
        self.max_workers = max_workers
        self.processed_count = 0
        self.error_count = 0
        self.start_time = None
        
        # Initialize components
        self.gender_detector = GenderDetectorGPU(
            CONFIG["detection_model_path"],
            CONFIG["attribute_model_path"]
        )
        
        self.dimension_extractor = GPUOptimizedDimensionExtractor()
        
        # Validate directory structure
        self._validate_structure()
    
    def _validate_structure(self):
        """Validate the expected directory structure."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        found_races = {d.name for d in self.data_dir.iterdir() if d.is_dir()}
        missing_races = set(CONFIG["race_dirs"]) - found_races
        
        if missing_races:
            logger.warning(f"Missing race directories: {missing_races}")
        
        # Create rich table for directory summary
        table = Table(title="Dataset Directory Structure")
        table.add_column("Race", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        
        for race in CONFIG["race_dirs"]:
            status = "✓ Found" if race in found_races else "✗ Missing"
            color = "green" if race in found_races else "red"
            table.add_row(race, f"[{color}]{status}[/{color}]")
        
        console.print(table)
        
        # Display GPU status
        gpu_table = Table(title="Hardware Acceleration Status")
        gpu_table.add_column("Component", style="cyan")
        gpu_table.add_column("Status", style="green")
        gpu_table.add_column("Details", style="yellow")
        
        if USE_MPS:
            gpu_table.add_row("GPU", "✓ Apple Silicon", "Metal Performance Shaders")
        elif DEVICE.type == "cuda":
            gpu_table.add_row("GPU", "✓ NVIDIA CUDA", torch.cuda.get_device_name())
        else:
            gpu_table.add_row("GPU", "✗ Not Available", "CPU Processing")
        
        gpu_table.add_row("Batch Size", str(CONFIG["gpu_batch_size"]), "Images per GPU batch")
        gpu_table.add_row("Max Workers", str(self.max_workers), "Concurrent threads")
        
        console.print(gpu_table)
        logger.info(f"Found race directories: {found_races}")
    
    def check_existing_csv_files(self) -> Tuple[bool, bool]:
        """Check if CSV files already exist."""
        images_csv = Path(os.getcwd(), 'images.csv')
        identities_csv = Path(os.getcwd(), 'identities.csv')
        
        return images_csv.exists(), identities_csv.exists()
    
    def discover_images(self) -> List[Tuple[str, str, Path]]:
        """Discover all images in the dataset structure with rich progress."""
        console.print(Panel.fit("🔍 Discovering Images in Dataset", style="bold blue"))
        
        image_paths = []
        race_summary = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
            transient=True
        ) as progress:
            
            race_task = progress.add_task("Scanning races...", total=len(CONFIG["race_dirs"]))
            
            for race_dir in self.data_dir.iterdir():
                if not race_dir.is_dir() or race_dir.name not in CONFIG["race_dirs"]:
                    continue
                
                race = race_dir.name
                race_identities = 0
                race_images = 0
                
                progress.update(race_task, description=f"Processing {race}...")
                
                identity_dirs = [d for d in race_dir.iterdir() if d.is_dir()]
                identity_task = progress.add_task(f"Scanning {race} identities...", total=len(identity_dirs))
                
                for identity_dir in identity_dirs:
                    identity = identity_dir.name
                    identity_images = 0
                    
                    for image_file in identity_dir.iterdir():
                        if (image_file.is_file() and 
                            image_file.suffix.lower() in CONFIG["supported_extensions"]):
                            image_paths.append((race, identity, image_file))
                            identity_images += 1
                            race_images += 1
                    
                    if identity_images > 0:
                        race_identities += 1
                    
                    progress.advance(identity_task)
                
                progress.remove_task(identity_task)
                progress.advance(race_task)
                
                race_summary.append({
                    'race': race,
                    'identities': race_identities,
                    'images': race_images
                })
        
        # Display summary table
        summary_table = Table(title="Dataset Discovery Summary")
        summary_table.add_column("Race", style="cyan", no_wrap=True)
        summary_table.add_column("Identities", style="magenta", justify="right")
        summary_table.add_column("Images", style="green", justify="right")
        
        total_identities = 0
        total_images = 0
        
        for race_data in sorted(race_summary, key=lambda x: x['race']):
            summary_table.add_row(
                race_data['race'],
                str(race_data['identities']),
                str(race_data['images'])
            )
            total_identities += race_data['identities']
            total_images += race_data['images']
        
        summary_table.add_row("", "", "", style="dim")
        summary_table.add_row("TOTAL", str(total_identities), str(total_images), style="bold")
        
        console.print(summary_table)
        logger.info(f"Discovery complete: {total_images} images from {total_identities} identities")
        
        return image_paths
    
    async def process_image_batch(self, batch: List[Tuple[str, str, Path]], progress, task_id) -> List[Dict]:
        """Process a batch of images asynchronously with GPU optimization."""
        loop = asyncio.get_event_loop()
        
        # Extract paths for batch processing
        paths = [item[2] for item in batch]
        
        # Get dimensions using GPU-optimized extractor
        dimensions = await loop.run_in_executor(
            None, self.dimension_extractor.extract_dimensions_batch, paths
        )
        
        # Combine results
        batch_results = []
        for (race, identity, image_path), (width, height) in zip(batch, dimensions):
            if width > 0 and height > 0:
                batch_results.append({
                    'race': race,
                    'identity': identity,
                    'full_image_path': str(image_path),
                    'width': width,
                    'height': height
                })
                self.processed_count += 1
            else:
                self.error_count += 1
            
            progress.advance(task_id)
        
        # Memory cleanup
        if USE_MPS and self.processed_count % 500 == 0:
            torch.mps.empty_cache()
        gc.collect()
        
        return batch_results
    
    def find_highest_resolution_images(self, image_data: List[Dict]) -> Dict[Tuple[str, str], Dict]:
        """Find the highest resolution image for each identity."""
        console.print(Panel.fit("🔎 Finding Highest Resolution Images per Identity", style="bold yellow"))
        
        identity_images = defaultdict(list)
        
        # Group images by identity
        for record in image_data:
            if record['width'] > 0 and record['height'] > 0:
                key = (record['race'], record['identity'])
                identity_images[key].append(record)
        
        highest_res_images = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Processing identities...", total=len(identity_images))
            
            for (race, identity), images in identity_images.items():
                # Find image with highest resolution
                highest_res_image = max(images, key=lambda x: x['width'] * x['height'])
                highest_res_images[(race, identity)] = highest_res_image
                progress.advance(task)
        
        logger.info(f"Found highest resolution images for {len(highest_res_images)} identities")
        return highest_res_images
    
    async def detect_genders_for_identities(self, highest_res_images: Dict[Tuple[str, str], Dict]) -> Dict[Tuple[str, str], str]:
        """Detect gender for each identity using their highest resolution image."""
        console.print(Panel.fit("👤 Detecting Gender for Each Identity", style="bold magenta"))
        
        gender_results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Detecting genders...", total=len(highest_res_images))
            
            # Process in batches
            batch_size = CONFIG["gender_batch_size"]
            items = list(highest_res_images.items())
            
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                
                # Process batch
                loop = asyncio.get_event_loop()
                
                # Use ProcessPoolExecutor for CPU-bound gender detection
                with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batch))) as executor:
                    futures = []
                    
                    for (race, identity), image_data in batch:
                        image_path = Path(image_data['full_image_path'])
                        future = loop.run_in_executor(
                            executor, self.gender_detector.detect_gender, image_path
                        )
                        futures.append(((race, identity), future))
                    
                    # Collect results
                    for (race, identity), future in futures:
                        try:
                            gender, confidence = await future
                            gender_results[(race, identity)] = gender
                            
                            # Update progress with color coding
                            if gender == "Male":
                                progress.update(task, description=f"[blue]{race} - {identity}: {gender}[/blue]")
                            elif gender == "Female":
                                progress.update(task, description=f"[color(206)]{race} - {identity}: {gender}[/color(206)]")
                            else:
                                progress.update(task, description=f"[dim]{race} - {identity}: {gender}[/dim]")
                        except Exception as e:
                            logger.error(f"Error detecting gender for {race}/{identity}: {e}")
                            gender_results[(race, identity)] = "Unknown"
                        
                        progress.advance(task)
                
                # Memory cleanup
                if USE_MPS and i % (batch_size * 10) == 0:
                    torch.mps.empty_cache()
                gc.collect()
        
        # Log statistics
        gender_counts = defaultdict(int)
        for gender in gender_results.values():
            gender_counts[gender] += 1
        
        stats_table = Table(title="Gender Detection Results")
        stats_table.add_column("Gender", style="cyan")
        stats_table.add_column("Count", style="green", justify="right")
        stats_table.add_column("Percentage", style="yellow", justify="right")
        
        total = len(gender_results)
        for gender, count in sorted(gender_counts.items()):
            percentage = (count / total * 100) if total > 0 else 0
            stats_table.add_row(gender, str(count), f"{percentage:.1f}%")
        
        console.print(stats_table)
        
        return gender_results
    
    def extract_identities_with_gender(self, image_data: List[Dict], gender_results: Dict[Tuple[str, str], str]) -> List[Dict]:
        """Extract unique identities with gender information."""
        identities_set = set()
        identities_list = []
        
        for record in image_data:
            identity_key = (record['race'], record['identity'])
            if identity_key not in identities_set:
                identities_set.add(identity_key)
                gender = gender_results.get(identity_key, "Unknown")
                identities_list.append({
                    'race': record['race'],
                    'identity': record['identity'],
                    'gender': gender
                })
        
        return identities_list
    
    def save_to_csv(self, data: List[Dict], filename: str):
        """Save data to CSV file with rich output."""
        if not data:
            logger.warning(f"No data to save to {filename}")
            return
        
        output_path = Path(os.getcwd(), filename)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        
        console.print(f"💾 Saved {len(data)} records to [bold green]{output_path}[/bold green]")
    
    async def process_all_images(self) -> Tuple[List[Dict], List[Dict]]:
        """Process all images in the dataset asynchronously."""
        console.print(Panel.fit("🚀 Starting Image Processing Pipeline", style="bold green"))
        self.start_time = time.time()
        
        # Discover all images
        image_paths = self.discover_images()
        total_images = len(image_paths)
        
        if total_images == 0:
            logger.error("No images found in the dataset!")
            return [], []
        
        # Process images in batches
        all_image_data = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn("Rate: {task.fields[rate]:.1f} img/s"),
            console=console
        ) as progress:
            
            main_task = progress.add_task(
                "Processing images...", 
                total=total_images,
                rate=0.0
            )
            
            processed = 0
            batch_size = CONFIG["batch_size"] if not USE_MPS else min(CONFIG["batch_size"], 64)
            
            for i in range(0, total_images, batch_size):
                batch = image_paths[i:i + batch_size]
                batch_data = await self.process_image_batch(batch, progress, main_task)
                all_image_data.extend(batch_data)
                
                processed += len(batch)
                elapsed = time.time() - self.start_time
                rate = processed / elapsed if elapsed > 0 else 0
                progress.update(main_task, rate=rate)
        
        # Find highest resolution images
        highest_res_images = self.find_highest_resolution_images(all_image_data)
        
        # Detect genders
        gender_results = await self.detect_genders_for_identities(highest_res_images)
        
        # Extract unique identities
        identities_data = self.extract_identities_with_gender(all_image_data, gender_results)
        
        # Final statistics
        elapsed = time.time() - self.start_time
        
        final_table = Table(title="Processing Complete - Final Statistics")
        final_table.add_column("Metric", style="cyan")
        final_table.add_column("Value", style="green", justify="right")
        
        final_table.add_row("Processing Time", f"{elapsed:.2f} seconds")
        final_table.add_row("Successfully Processed", f"{self.processed_count:,} images")
        final_table.add_row("Errors Encountered", f"{self.error_count:,}")
        final_table.add_row("Average Rate", f"{self.processed_count/elapsed:.2f} images/second")
        final_table.add_row("Total Identities", f"{len(identities_data):,}")
        
        # Gender detection rate
        known_genders = len([i for i in identities_data if i['gender'] != 'Unknown'])
        detection_rate = (known_genders/len(identities_data)*100) if len(identities_data) > 0 else 0
        final_table.add_row("Gender Detection Rate", f"{known_genders}/{len(identities_data)} ({detection_rate:.1f}%)")
        
        console.print(final_table)
        
        return all_image_data, identities_data


async def main():
    """Main function to run the image processing pipeline."""
    console.print(Panel.fit("🎯 RFW Dataset Analysis Pipeline", style="bold blue"))
    
    working_directory = CONFIG['images_dir']
    
    try:
        # Initialize processor
        processor = ImageProcessor(working_directory, max_workers=CONFIG["max_workers"])
        
        # Check for existing CSV files
        images_exists, identities_exists = processor.check_existing_csv_files()
        
        if CONFIG["skip_existing_csv"] and images_exists and identities_exists:
            console.print(Panel.fit("📄 CSV files already exist. Loading from disk...", style="bold yellow"))
            
            # Load existing data
            images_df = pd.read_csv('images.csv')
            identities_df = pd.read_csv('identities.csv')
            
            console.print(f"✅ Loaded {len(images_df)} image records and {len(identities_df)} identity records")
            
        else:
            # Process all images
            images_data, identities_data = await processor.process_all_images()
            
            if not images_data:
                console.print("[bold red]❌ No image data to save. Exiting.[/bold red]")
                return
            
            # Save results
            console.print(Panel.fit("💾 Saving Results to CSV Files", style="bold yellow"))
            processor.save_to_csv(images_data, 'images.csv')
            processor.save_to_csv(identities_data, 'identities.csv')
            
            # Convert to DataFrames
            images_df = pd.DataFrame(images_data)
            identities_df = pd.DataFrame(identities_data)
        
        # Generate analysis plots
        console.print(Panel.fit("📊 Generating Statistical Analysis and Plots", style="bold cyan"))
        
        analyzer = StatisticalAnalyzer()
        output_dir = Path(os.getcwd(), 'analysis_plots')
        output_dir.mkdir(exist_ok=True)
        
        # Generate plots with progress indicators
        with console.status("[bold cyan]Creating dimension distribution plots..."):
            analyzer.plot_dimension_distribution_per_race(images_df, output_dir)
        
        with console.status("[bold cyan]Creating identity distribution plots..."):
            analyzer.plot_identity_distribution_per_race(images_df, output_dir)
        
        with console.status("[bold cyan]Creating gender distribution plots..."):
            analyzer.plot_gender_distribution_comparison(identities_df, output_dir)
        
        # Final summary by race
        race_stats = {}
        for _, row in images_df.iterrows():
            race = row['race']
            if race not in race_stats:
                race_stats[race] = {'images': 0, 'identities': set(), 'total_resolution': 0}
            race_stats[race]['images'] += 1
            race_stats[race]['identities'].add(row['identity'])
            race_stats[race]['total_resolution'] += row['width'] * row['height']
        
        # Gender distribution
        gender_stats = defaultdict(lambda: defaultdict(int))
        for _, identity in identities_df.iterrows():
            gender_stats[identity['race']][identity['gender']] += 1
        
        # Create final summary table
        race_table = Table(title="Final Dataset Summary by Race")
        race_table.add_column("Race", style="cyan")
        race_table.add_column("Images", style="green", justify="right")
        race_table.add_column("Identities", style="magenta", justify="right")
        race_table.add_column("Avg Resolution", style="yellow", justify="right")
        race_table.add_column("Male", style="blue", justify="right")
        race_table.add_column("Female", style="color(206)", justify="right")
        race_table.add_column("Unknown", style="dim", justify="right")
        
        for race in sorted(race_stats.keys()):
            stats = race_stats[race]
            genders = gender_stats[race]
            avg_resolution = stats['total_resolution'] / stats['images'] if stats['images'] > 0 else 0
            
            race_table.add_row(
                race,
                f"{stats['images']:,}",
                f"{len(stats['identities']):,}",
                f"{int(avg_resolution):,}",
                f"{genders.get('Male', 0):,}",
                f"{genders.get('Female', 0):,}",
                f"{genders.get('Unknown', 0):,}"
            )
        
        console.print(race_table)
        
        # Print plot locations
        plot_files = list(output_dir.glob('*.png'))
        if plot_files:
            plot_table = Table(title="Generated Analysis Plots")
            plot_table.add_column("Plot Type", style="cyan")
            plot_table.add_column("Files", style="green")
            
            # Group plots by type
            dimension_plots = [f for f in plot_files if 'dimension' in f.name]
            identity_plots = [f for f in plot_files if 'identity' in f.name]
            gender_plots = [f for f in plot_files if 'gender' in f.name]
            
            if dimension_plots:
                plot_table.add_row("Dimension Distribution", f"{len(dimension_plots)} files")
            if identity_plots:
                plot_table.add_row("Identity Distribution", f"{len(identity_plots)} files")
            if gender_plots:
                plot_table.add_row("Gender Distribution", f"{len(gender_plots)} files")
            
            console.print(plot_table)
            console.print(f"\n📁 All plots saved in: [bold green]{output_dir}[/bold green]")
        
        console.print(Panel.fit("✅ Analysis completed successfully!", style="bold green"))
        
    except Exception as e:
        console.print(f"[bold red]❌ Fatal error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Run the main pipeline
    asyncio.run(main())