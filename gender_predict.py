import os
import cv2
import sys
import torch
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from glob import glob
from PIL import Image
from pathlib import Path
from torchvision import transforms
from typing import List, Optional, Any, Dict, Union

from rich import print
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn, TaskID

console = Console()

try:
    from mivolo.predictor import Predictor
except ImportError:
    console.print(Panel.fit(f"[bold red] Could not import mivolo package",
                            border_style="magenta",
                            title="🚀 Gender Predictor",
                            title_align="center"))
    sys.exit(1)


class GenderPredictor:
    """
    Gender prediction module with placeholder implementation.
    Ready for PyTorch model integration.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize gender predictor.
        
        Args:
            batch_size: Number of images to process in each batch
            max_workers: Maximum number of threads for parallel processing
        """
        try:
            class Args:
                def __init__(self):
                    self.detector_weights = config["detection_model_path"]
                    self.checkpoint = config['attribute_model_path']
                    # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    self.device = "mps" if torch.backends.mps.is_available() and config['use_mps'] else "cpu"
                    self.with_persons = True,
                    self.disable_faces = False,
                    self.draw = False,
        except Exception as e:
            console.print(Panel.fit("[bold red] Model weights are not indicated",
                                    border_style="red",
                                    title="🚀 Gender Predictor",
                                    title_align="center"))
        
        self.config = config or {}
        self.batch_size = self.config['batch_size'] if 'batch_size' in self.config else 32
        self.max_workers = self.config['max_workers'] if 'max_workers' in self.config else 4
        # set env YOLO_VERBOSE to be false
        os.environ["YOLO_VERBOSE"] = "false"
        
        self.args = Args()
        self.predictor = Predictor(self.args)
        
    def load_model(self, model_path: Optional[str] = None):
        """
        Load gender prediction model.
        
        Args:
            model_path: Path to the trained model file
        """
        # Placeholder for model loading
        # TODO: Implement PyTorch model loading
        # self.model = torch.load(model_path)
        # self.model.eval()
        pass
    
    def preprocess_image(self, image_path: Path) -> np.ndarray:
        """
        Preprocess image for gender prediction.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Placeholder preprocessing
                # TODO: Implement actual preprocessing for the model
                # - Resize to model input size
                # - Normalize pixel values
                # - Convert to tensor
                
                return np.array(img)
        except Exception:
            return None
    
    def predict_batch(self, image_paths: List[Path]) -> List[str]:
        """
        Predict gender for a batch of images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of predicted genders
        """
        predictions = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit preprocessing tasks
            future_to_path = {
                executor.submit(self.preprocess_image, img_path): img_path 
                for img_path in image_paths
            }
            
            # Collect preprocessed images
            preprocessed_images = []
            valid_paths = []
            
            for future in as_completed(future_to_path):
                img_path = future_to_path[future]
                try:
                    processed_img = future.result()
                    if processed_img is not None:
                        preprocessed_images.append(processed_img)
                        valid_paths.append(img_path)
                except Exception:
                    pass
            
            # Placeholder for model inference
            # TODO: Implement actual model inference
            # if self.model and preprocessed_images:
            #     with torch.no_grad():
            #         batch_tensor = torch.stack(preprocessed_images)
            #         outputs = self.model(batch_tensor)
            #         predictions = torch.argmax(outputs, dim=1)
            #         return ['Male' if p == 1 else 'Female' for p in predictions]
            
            # For now, return "Unknown" for all images
            predictions = ["Unknown"] * len(image_paths)
        
        return predictions
    
    def predict_gender_for_identity(self, identity_path: Path) -> str:
        """
        Predict gender for all images in an identity directory.
        Uses majority voting if multiple images exist.
        
        Args:
            identity_path: Path to identity directory
            
        Returns:
            Predicted gender for the identity
        """
        # Get all image files in the identity directory
        try:
            image_extensions = self.config['supported_extensions']
        except KeyError:
            image_extensions = ['.jpg', '.jpeg', '.png']
        
        image_files = [
            f for f in identity_path.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            return "Unknown"
        
        # Find the image with the highest resolution via heuristic approach
        highest_res_image = max(image_files, key=lambda x: x.stat().st_size)
        
        # Predict
        image = cv2.imread(str(highest_res_image))
        
        if not image.any():
            raise ValueError("Could not load image")
    
        detected_objects, out_image = self.predictor.recognize(image)
        
        if not detected_objects:
            print("No faces detected")
            return "Unknown"
        
        for obj in detected_objects.genders:
            if obj is not None:
                return obj.capitalize()
        
        return "Unknown"
    
        
        # From the list of detected_objects.genders extract the one 