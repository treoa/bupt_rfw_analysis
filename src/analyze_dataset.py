"""
Main script for analyzing the RFW dataset.
"""

import os
import glob
import torch
import numpy as np
from PIL import Image
import json
from pathlib import Path
from rich.console import Console
from rich.progress import track
from rich.table import Table
from collections import defaultdict

from src.utils import (
    get_race_paths, get_identities, get_images, 
    load_image, calculate_statistics, save_results
)
from src.face_analyzer import FaceAnalyzer, get_gender_label

console = Console()

def analyze_dataset(pretrained_model=None):
    """
    Analyze the RFW dataset for age, gender, resolution, and identity statistics.
    
    Args:
        pretrained_model (str, optional): Path to pre-trained model for age/gender prediction
        
    Returns:
        dict: Analysis results
    """
    # Initialize face analyzer
    face_analyzer = FaceAnalyzer(pretrained_model)
    
    # Get race paths
    race_paths = get_race_paths()
    
    # Initialize results structure
    results = {
        "races": {},
        "overall": {}
    }
    
    # Track all ages and genders for overall statistics
    all_ages = []
    all_genders = {'male': 0, 'female': 0}
    total_identities = 0
    total_images = 0
    
    # Process each race
    for race, race_path in race_paths.items():
        console.print(f"[bold blue]Processing {race} race...[/bold blue]")
        
        # Get identities for this race
        identity_paths = get_identities(race_path)
        console.print(f"Found {len(identity_paths)} identities for {race}")
        total_identities += len(identity_paths)
        
        # Initialize race statistics
        race_stats = {
            "identities_count": len(identity_paths),
            "total_images": 0,
            "age": {
                "values": [],
                "statistics": {}
            },
            "gender": {
                "male": 0,
                "female": 0
            },
            "resolution": {
                "width": [],
                "height": []
            },
            "images_per_identity": []
        }
        
        # Process each identity
        for identity_path in track(identity_paths, description=f"Analyzing {race} identities"):
            # Get images for this identity
            image_paths = get_images(identity_path)
            race_stats["images_per_identity"].append(len(image_paths))
            race_stats["total_images"] += len(image_paths)
            total_images += len(image_paths)
            
            # Identity-level gender consensus (majority vote)
            identity_genders = []
            
            # Process each image
            for img_path in image_paths:
                image = load_image(img_path)
                if image is None:
                    continue
                
                # Get image resolution
                width, height = image.size
                race_stats["resolution"]["width"].append(width)
                race_stats["resolution"]["height"].append(height)
                
                # Predict age and gender
                age, gender = face_analyzer.predict(image)
                
                if age is not None:
                    race_stats["age"]["values"].append(age)
                    all_ages.append(age)
                
                if gender is not None:
                    identity_genders.append(gender)
            
            # Determine the majority gender for this identity
            if identity_genders:
                # Convert to numpy for convenient majority voting
                identity_genders = np.array(identity_genders)
                majority_gender = int(np.round(np.mean(identity_genders)))
                gender_label = get_gender_label(majority_gender)
                
                race_stats["gender"][gender_label] += 1
                all_genders[gender_label] += 1
        
        # Calculate statistics for this race
        race_stats["age"]["statistics"] = calculate_statistics(race_stats["age"]["values"])
        race_stats["resolution"]["width_statistics"] = calculate_statistics(race_stats["resolution"]["width"])
        race_stats["resolution"]["height_statistics"] = calculate_statistics(race_stats["resolution"]["height"])
        race_stats["images_per_identity_statistics"] = calculate_statistics(race_stats["images_per_identity"])
        
        # Remove raw values to keep the results file smaller
        del race_stats["age"]["values"]
        del race_stats["resolution"]["width"]
        del race_stats["resolution"]["height"]
        
        # Store race results
        results["races"][race] = race_stats
    
    # Calculate overall statistics
    results["overall"] = {
        "total_identities": total_identities,
        "total_images": total_images,
        "age_statistics": calculate_statistics(all_ages),
        "gender_distribution": all_genders
    }
    
    return results

def display_results(results):
    """
    Display analysis results in a nicely formatted table.
    
    Args:
        results (dict): Analysis results
    """
    console.print("[bold green]===== RFW Dataset Analysis Results =====[/bold green]")
    
    # Overall statistics
    console.print("\n[bold]Overall Statistics:[/bold]")
    console.print(f"Total Identities: {results['overall']['total_identities']}")
    console.print(f"Total Images: {results['overall']['total_images']}")
    
    gender_table = Table(title="Overall Gender Distribution")
    gender_table.add_column("Gender")
    gender_table.add_column("Count")
    gender_table.add_column("Percentage")
    
    total = sum(results['overall']['gender_distribution'].values())
    for gender, count in results['overall']['gender_distribution'].items():
        percentage = (count / total) * 100 if total > 0 else 0
        gender_table.add_row(gender, str(count), f"{percentage:.2f}%")
    
    console.print(gender_table)
    
    # Age statistics
    age_table = Table(title="Age Statistics by Race")
    age_table.add_column("Race")
    age_table.add_column("Min")
    age_table.add_column("Max")
    age_table.add_column("Mean")
    age_table.add_column("Median")
    age_table.add_column("Std")
    
    for race, stats in results['races'].items():
        age_stats = stats['age']['statistics']
        age_table.add_row(
            race,
            f"{age_stats['min']:.2f}" if age_stats['min'] is not None else "N/A",
            f"{age_stats['max']:.2f}" if age_stats['max'] is not None else "N/A",
            f"{age_stats['mean']:.2f}" if age_stats['mean'] is not None else "N/A",
            f"{age_stats['median']:.2f}" if age_stats['median'] is not None else "N/A",
            f"{age_stats['std']:.2f}" if age_stats['std'] is not None else "N/A"
        )
    
    # Add overall row
    overall_age = results['overall']['age_statistics']
    age_table.add_row(
        "Overall",
        f"{overall_age['min']:.2f}" if overall_age['min'] is not None else "N/A",
        f"{overall_age['max']:.2f}" if overall_age['max'] is not None else "N/A",
        f"{overall_age['mean']:.2f}" if overall_age['mean'] is not None else "N/A",
        f"{overall_age['median']:.2f}" if overall_age['median'] is not None else "N/A",
        f"{overall_age['std']:.2f}" if overall_age['std'] is not None else "N/A"
    )
    
    console.print(age_table)
    
    # Resolution statistics
    resolution_table = Table(title="Image Resolution Statistics by Race")
    resolution_table.add_column("Race")
    resolution_table.add_column("Dimension")
    resolution_table.add_column("Min")
    resolution_table.add_column("Max")
    resolution_table.add_column("Mean")
    resolution_table.add_column("Median")
    resolution_table.add_column("Std")
    
    for race, stats in results['races'].items():
        width_stats = stats['resolution']['width_statistics']
        height_stats = stats['resolution']['height_statistics']
        
        resolution_table.add_row(
            race, "Width",
            f"{width_stats['min']:.2f}" if width_stats['min'] is not None else "N/A",
            f"{width_stats['max']:.2f}" if width_stats['max'] is not None else "N/A",
            f"{width_stats['mean']:.2f}" if width_stats['mean'] is not None else "N/A",
            f"{width_stats['median']:.2f}" if width_stats['median'] is not None else "N/A",
            f"{width_stats['std']:.2f}" if width_stats['std'] is not None else "N/A"
        )
        
        resolution_table.add_row(
            "", "Height",
            f"{height_stats['min']:.2f}" if height_stats['min'] is not None else "N/A",
            f"{height_stats['max']:.2f}" if height_stats['max'] is not None else "N/A",
            f"{height_stats['mean']:.2f}" if height_stats['mean'] is not None else "N/A",
            f"{height_stats['median']:.2f}" if height_stats['median'] is not None else "N/A",
            f"{height_stats['std']:.2f}" if height_stats['std'] is not None else "N/A"
        )
    
    console.print(resolution_table)
    
    # Images per identity statistics
    images_table = Table(title="Images per Identity Statistics by Race")
    images_table.add_column("Race")
    images_table.add_column("Min")
    images_table.add_column("Max")
    images_table.add_column("Mean")
    images_table.add_column("Median")
    images_table.add_column("Std")
    
    for race, stats in results['races'].items():
        img_stats = stats['images_per_identity_statistics']
        images_table.add_row(
            race,
            f"{img_stats['min']:.2f}" if img_stats['min'] is not None else "N/A",
            f"{img_stats['max']:.2f}" if img_stats['max'] is not None else "N/A",
            f"{img_stats['mean']:.2f}" if img_stats['mean'] is not None else "N/A",
            f"{img_stats['median']:.2f}" if img_stats['median'] is not None else "N/A",
            f"{img_stats['std']:.2f}" if img_stats['std'] is not None else "N/A"
        )
    
    console.print(images_table)

def main():
    """
    Main function to run the analysis.
    """
    console.print("[bold green]Starting RFW Dataset Analysis...[/bold green]")
    
    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Check for pretrained model
    pretrained_model = None
    if Path("models/age_gender_model.pth").exists():
        pretrained_model = "models/age_gender_model.pth"
    
    # Run analysis
    results = analyze_dataset(pretrained_model)
    
    # Save results
    save_results(results, output_dir / "rfw_analysis_results.json")
    
    # Display results
    display_results(results)

if __name__ == "__main__":
    main()
