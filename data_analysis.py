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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from rich import print
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn, TaskID

try:
    from gender_predict import GenderPredictor
    from mivolo.predictor import Predictor
    from visualizer import AdvancedDatasetAnalyzer as DatasetAnalyzer
except ImportError:
    print("[yellow]Warning: mivolo package not found.[/yellow]")
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

# Check for Apple Silicon GPU availability
DEVICE = None

CONFIG = {
    # "images_dir": "BalancedFace/images",
    "images_dir": "/workspace/RFW/images/test/data",
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
# elif torch.backends.mps.is_available():
#     DEVICE = torch.device("mps")
#     CONFIG["use_mps"] = True
#     CONFIG["max_workers"] = 8 # Reduce workers on MPS to avoid memory issues
#     logger.info("🚀 Apple Silicon GPU detected! Using Metal Performance Shaders for acceleration")
else:
    DEVICE = torch.device("cpu")
    logger.info("💻 Using CPU for processing")

RACE_COLORS = {
    'African': '#FF6B6B',    # Coral Red
    'Asian': '#4ECDC4',      # Teal
    'Caucasian': '#45B7D1',  # Sky Blue  
    'Indian': '#FFA07A'      # Light Salmon
}


class ImageDatasetProcessor:
    """
    A comprehensive processor for image datasets organized by race and identity.
    Generates CSV files with image and identity metadata, includes gender prediction,
    and creates detailed statistical plots.
    """
    
    def __init__(self, batch_size: int = 100, max_workers: int = 4):
        """
        Initialize the processor with configuration parameters.
        
        Args:
            batch_size: Number of images to process in each batch
            max_workers: Maximum number of threads for parallel processing
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.console = Console()
        self.setup_logging()
        
        # Initialize components
        self.gender_predictor = GenderPredictor(config=CONFIG)
        self.analyzer = DatasetAnalyzer(self.console, RACE_COLORS)
        
        # Data storage
        self.images_data = []
        self.identities_data = []
        
        # Supported image extensions
        self.image_extensions = CONFIG["supported_extensions"]
        self.data_dir = Path(CONFIG["images_dir"])
        
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
        
        if CONFIG["use_mps"]:
            gpu_table.add_row("GPU", "✓ Apple Silicon", "Metal Performance Shaders")
        elif DEVICE.type == "cuda":
            gpu_table.add_row("GPU", "✓ NVIDIA CUDA", torch.cuda.get_device_name())
        else:
            gpu_table.add_row("GPU", "✗ Not Available", "CPU Processing")
        
        gpu_table.add_row("Max Workers", str(self.max_workers), "Concurrent threads")
        
        console.print(gpu_table)
        logger.info(f"Found race directories: {found_races}")
    
    def check_existing_csv_files(self) -> Tuple[bool, bool]:
        """Check if CSV files already exist."""
        images_csv = Path(os.getcwd(), CONFIG["image_csv_path"])
        identities_csv = Path(os.getcwd(), CONFIG["identity_csv_path"])
        
        return images_csv.exists(), identities_csv.exists()
    
    def setup_logging(self):
        """Configure rich logging with enhanced formatting."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=self.console, rich_tracebacks=True, markup=True)]
        )
        self.logger = logging.getLogger("ImageProcessor")
    
    def get_image_dimensions(self, image_path: Path) -> Tuple[int, int, int]:
        """Extract image dimensions with enhanced error handling."""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                total_pixels = width * height
                return width, height, total_pixels
        except Exception as e:
            self.logger.warning(f"Could not process image {image_path}: {e}")
            return 0, 0, 0
    
    def process_image_batch(self, image_paths: List[Path], race: str) -> List[Dict]:
        """Process image batch with optimized parallel execution."""
        batch_results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(self.get_image_dimensions, img_path): img_path 
                for img_path in image_paths
            }
            
            for future in as_completed(future_to_path):
                img_path = future_to_path[future]
                try:
                    width, height, total_pixels = future.result()
                    identity_id = img_path.parent.name
                    
                    batch_results.append({
                        'full_image_path': str(img_path),
                        'race': race,
                        'identity_id': identity_id,
                        'image_width': width,
                        'image_height': height,
                        'image_total_pixels': total_pixels
                    })
                except Exception as e:
                    self.logger.error(f"Error processing {img_path}: {e}")
        
        return batch_results
    
    def process_identity_gender_batch(self, identity_paths: List[Path], race: str) -> List[str]:
        """Process gender prediction with advanced batch handling."""
        genders = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(self.gender_predictor.predict_gender_for_identity, identity_path): identity_path
                for identity_path in identity_paths
            }
            
            for future in as_completed(future_to_path):
                try:
                    gender = future.result()
                    genders.append(gender)
                except Exception as e:
                    self.logger.warning(f"Error predicting gender: {e}")
                    genders.append("Unknown")
        
        return genders
    
    def collect_race_data(self, race_path: Path, race: str) -> Tuple[List[Dict], List[Dict]]:
        """Comprehensive data collection with optimized processing."""
        race_images = []
        identity_dirs = []
        
        # Collect identity directories
        for identity_dir in race_path.iterdir():
            if identity_dir.is_dir():
                identity_images = [
                    f for f in identity_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in self.image_extensions
                ]
                
                if identity_images:
                    identity_dirs.append(identity_dir)
                    race_images.extend(identity_images)
        
        # Process gender prediction in optimized batches
        race_identities = []
        gender_batch_size = 32
        
        for i in range(0, len(identity_dirs), gender_batch_size):
            batch_dirs = identity_dirs[i:i + gender_batch_size]
            batch_genders = self.process_identity_gender_batch(batch_dirs, race)
            
            for identity_dir, gender in zip(batch_dirs, batch_genders):
                identity_images = [
                    f for f in identity_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in self.image_extensions
                ]
                
                race_identities.append({
                    'full_identity_path': str(identity_dir),
                    'race': race,
                    'num_images': len(identity_images),
                    'gender': gender
                })
        
        return race_images, race_identities
    
    def process_race(self, race_path: Path, race: str, progress: Progress, race_task: TaskID) -> Dict:
        """Process race with comprehensive progress tracking."""
        start_time = time.time()
        
        self.console.print(f"\n[bold {RACE_COLORS.get(race, 'blue')}]🚀 Processing race: {race}[/bold {RACE_COLORS.get(race, 'blue')}]")
        
        race_images, race_identities = self.collect_race_data(race_path, race)
        
        if not race_images:
            self.logger.warning(f"No images found for race: {race}")
            progress.update(race_task, completed=1)
            return {'race': race, 'images_processed': 0, 'identities_processed': 0, 'processing_time': 0}
        
        self.identities_data.extend(race_identities)
        
        total_images = len(race_images)
        progress.update(race_task, total=total_images)
        processed_images = 0
        
        # Process images in optimized batches
        for i in range(0, total_images, self.batch_size):
            batch = race_images[i:i + self.batch_size]
            batch_results = self.process_image_batch(batch, race)
            self.images_data.extend(batch_results)
            
            processed_images += len(batch)
            progress.update(race_task, completed=processed_images)
        
        processing_time = time.time() - start_time
        
        stats = {
            'race': race,
            'images_processed': len(race_images),
            'identities_processed': len(race_identities),
            'processing_time': processing_time
        }
        
        self.console.print(f"[green]✅[/green] Completed {race}: "
                          f"{stats['images_processed']} images, "
                          f"{stats['identities_processed']} identities "
                          f"in {processing_time:.2f}s")
        
        return stats
    
    def save_to_csv(self, data: List[Dict], filename: str, fieldnames: List[str]):
        """Save data with enhanced error handling and validation."""
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            
            self.console.print(f"[green]💾[/green] Saved {len(data)} entries to [cyan]{filename}[/cyan]")
            
        except Exception as e:
            self.logger.error(f"Error saving {filename}: {e}")
            raise
    
    def display_summary_table(self, race_stats: List[Dict]):
        """Display enhanced summary with rich styling."""
        table = Table(title="🏆 Processing Summary", show_header=True, header_style="bold magenta")
        table.add_column("Race", style="cyan", width=12)
        table.add_column("Images", justify="right", style="green", width=10)
        table.add_column("Identities", justify="right", style="blue", width=12)
        table.add_column("Time (s)", justify="right", style="yellow", width=10)
        table.add_column("Img/Sec", justify="right", style="red", width=10)
        
        total_images = 0
        total_identities = 0
        total_time = 0
        
        for stats in race_stats:
            img_per_sec = stats['images_processed'] / stats['processing_time'] if stats['processing_time'] > 0 else 0
            table.add_row(
                stats['race'],
                str(stats['images_processed']),
                str(stats['identities_processed']),
                f"{stats['processing_time']:.2f}",
                f"{img_per_sec:.1f}"
            )
            total_images += stats['images_processed']
            total_identities += stats['identities_processed']
            total_time += stats['processing_time']
        
        table.add_row("", "", "", "", "", style="bold")
        avg_speed = total_images / total_time if total_time > 0 else 0
        table.add_row("TOTAL", str(total_images), str(total_identities), 
                     f"{total_time:.2f}", f"{avg_speed:.1f}", style="bold")
        
        self.console.print("\n")
        self.console.print(table)
    
    def generate_premium_visualizations(self, output_dir: str = "premium_plots"):
        """Generate stunning visualizations with premium aesthetics."""
        images_df_exist, identities_df_exist = self.check_existing_csv_files()
        
        if not (self.images_data and self.identities_data) and not (images_df_exist and identities_df_exist):
            self.logger.warning("No data available for visualization")
            return
        elif images_df_exist and identities_df_exist and not (self.images_data and self.identities_data):
            images_df = pd.read_csv(CONFIG["image_csv_path"])
            identities_df = pd.read_csv(CONFIG["identity_csv_path"])
        else:
            # Convert to DataFrames
            images_df = pd.DataFrame(self.images_data)
            identities_df = pd.DataFrame(self.identities_data)
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.console.print("\n[bold magenta]🎨 Generating visualizations...[/bold magenta]")
        
        # Generate masterpiece visualizations including gender analysis
        self.analyzer.create_images_per_identity_masterpiece(identities_df, output_dir)
        self.analyzer.create_image_dimensions_masterpiece(images_df, output_dir)
        self.analyzer.create_gender_analysis_masterpiece(identities_df, images_df, output_dir)
        self.analyzer.create_statistical_dashboard(images_df, identities_df, output_dir)
        
        self.console.print(f"[green]🎭[/green] Premium visualizations saved to [cyan]'{output_dir}'[/cyan]")
        self.console.print(f"[yellow]📊[/yellow] Interactive plots available as HTML files")
    
    def process_dataset(self, data_path: str, generate_plots: bool = True, output_dir: str = "premium_plots") -> None:
        """Main processing pipeline with premium features."""
        start_time = time.time()
        
        # Display premium welcome message
        self.console.print(Panel.fit(
            "[bold magenta]🎭 Premium Image Dataset Processor[/bold magenta]\n"
            f"[cyan]Dataset Path:[/cyan] {data_path}\n"
            f"[yellow]Batch Size:[/yellow] {self.batch_size} | [yellow]Workers:[/yellow] {self.max_workers}\n"
            f"[green]Premium Plots:[/green] {generate_plots}",
            border_style="magenta",
            title="🚀 Advanced Analytics Engine",
            title_align="center"
        ))
        
        data_dir = Path(data_path)
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")
        
        race_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        
        if not race_dirs:
            raise ValueError(f"No race directories found in: {data_path}")
        
        self.logger.info(f"Found {len(race_dirs)} race directories: {[d.name for d in race_dirs]}")
        
        # Process with premium progress tracking
        race_stats = []
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
            console=self.console
        ) as progress:
            
            # Create tasks for each race
            race_tasks = {}
            for race_dir in race_dirs:
                race_name = race_dir.name
                race_tasks[race_name] = progress.add_task(
                    f"[{RACE_COLORS.get(race_name, 'white')}]Processing {race_name}",
                    total=1
                )
            
            overall_task = progress.add_task("[bold magenta]Overall Progress", total=len(race_dirs))
            
            # Process each race
            for i, race_dir in enumerate(race_dirs):
                race_name = race_dir.name
                stats = self.process_race(race_dir, race_name, progress, race_tasks[race_name])
                race_stats.append(stats)
                progress.update(overall_task, completed=i + 1)
        
        # Save results with premium formatting
        self.console.print("\n[bold yellow]💾 Saving results to CSV files...[/bold yellow]")
        
        if self.images_data:
            self.save_to_csv(
                self.images_data,
                CONFIG["image_csv_path"],
                ['full_image_path', 'race', 'identity_id', 'image_width', 'image_height', 'image_total_pixels']
            )
        
        if self.identities_data:
            self.save_to_csv(
                self.identities_data,
                'identities.csv',
                ['full_identity_path', 'race', 'num_images', 'gender']
            )
        
        # Generate premium visualizations
        if generate_plots:
            self.generate_premium_visualizations(output_dir)
        
        # Display premium summary
        total_time = time.time() - start_time
        self.display_summary_table(race_stats)
        
        # Final success message
        success_panel = Panel.fit(
            f"[bold green]🎉 Dataset processing completed successfully![/bold green]\n"
            f"[cyan]Processing Time:[/cyan] {total_time:.2f} seconds\n"
            f"[yellow]Images Processed:[/yellow] {len(self.images_data)}\n"
            f"[blue]Identities Processed:[/blue] {len(self.identities_data)}\n"
            f"[magenta]Premium Plots:[/magenta] {generate_plots}",
            border_style="green",
            title="✨ Success",
            title_align="center"
        )
        self.console.print(f"\n{success_panel}")


async def main(data_path: str = CONFIG['images_dir'], batch_size: int = CONFIG["batch_size"], 
         max_workers: int = CONFIG["max_workers"], output_dir: str = CONFIG["plots_path"]):
    """
    Premium dataset processing with stunning visualizations.
    
    Args:
        data_path: Path to the main data directory
        batch_size: Number of images to process in each batch
        max_workers: Maximum number of threads for parallel processing
        generate_plots: Whether to generate premium visualizations
        output_dir: Directory to save premium plots
    
    Features:
        🎨 Premium seaborn/plotly visualizations
        🚀 Parallel processing with progress tracking
        🤖 Gender prediction system (PyTorch ready)
        📊 Interactive HTML plots
        📈 Comprehensive statistical analysis
        🎭 Publication-quality graphics
    
    Examples:
        # Premium processing with all features
        process_image_dataset('./test/data')
        
        # High-performance mode
        process_image_dataset('./test/data', batch_size=200, max_workers=8)
        
        # Custom visualization directory
        process_image_dataset('./test/data', output_dir='custom_analysis')
    """
    try:
        processor = ImageDatasetProcessor(batch_size=batch_size, max_workers=max_workers)
        
        images_exists, identities_exists = processor.check_existing_csv_files()
        if CONFIG["skip_existing_csv"] and images_exists and identities_exists:
            console.print(Panel.fit("📄 CSV files already exist. Loading from disk...", style="bold yellow"))
            
            # Load existing data
            images_df = pd.read_csv(CONFIG["image_csv_path"])
            identities_df = pd.read_csv(CONFIG["identity_csv_path"])
            
            console.print(f"✅ Loaded {len(images_df)} image records and {len(identities_df)} identity records")
            
            # Generate analysis plots
            console.print(Panel.fit("📊 Generating Statistical Analysis and Plots", style="bold cyan"))
            processor.generate_premium_visualizations(output_dir=CONFIG["plots_path"])
        else:
            processor.process_dataset(data_path, output_dir=CONFIG["plots_path"])
            
            # Generate analysis plots
            console.print(Panel.fit("📊 Generating Statistical Analysis and Plots", style="bold cyan"))
            
            console.print(Panel.fit("✅ Analysis completed successfully!", style="bold green"))
    except Exception as e:
        console.print(f"[bold red]❌ Fatal error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    asyncio.run(main())