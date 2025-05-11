"""
Main script for analyzing the RFW dataset with advanced face analysis.
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import json
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, track, TaskID
from rich.table import Table
from collections import defaultdict, Counter
import time
import glob

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import (
    get_race_paths, get_identities, get_images, 
    load_image, calculate_statistics, save_results
)
from src.advanced_face_analyzer import AdvancedFaceAnalyzer, get_gender_label

console = Console()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze RFW dataset for age, gender, and other statistics.')
    parser.add_argument('--output', type=str, default='results/rfw_analysis_results.json',
                       help='Path to save the analysis results')
    parser.add_argument('--show-plots', action='store_true', default=True,
                       help='Generate and display plots of the results')
    parser.add_argument('--race', type=str, choices=['African', 'Asian', 'Caucasian', 'Indian', 'all'],
                       default='all', help='Specific race to analyze')
    parser.add_argument('--max-identities', type=int, default=None,
                       help='Maximum number of identities to analyze per race (for testing)')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Skip analysis and just display results from existing file')
    return parser.parse_args()

def analyze_dataset(args):
    """
    Analyze the RFW dataset for age, gender, resolution, and identity statistics.
    
    Args:
        args: Command line arguments
        
    Returns:
        dict: Analysis results
    """
    # Initialize face analyzer
    face_analyzer = AdvancedFaceAnalyzer()
    
    # Get race paths
    race_paths = get_race_paths()
    
    # Filter by race if specified
    if args.race != 'all':
        race_paths = {args.race: race_paths[args.race]}
    
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
    
    # Start progress tracking
    with Progress() as progress:
        race_task = progress.add_task("[bold blue]Processing races...", total=len(race_paths))
        
        # Process each race
        for race, race_path in race_paths.items():
            console.print(f"[bold blue]Processing {race} race...[/bold blue]")
            
            # Get identities for this race
            identity_paths = get_identities(race_path)
            console.print(f"Found {len(identity_paths)} identities for {race}")
            
            # Limit identities if requested (for testing)
            if args.max_identities:
                identity_paths = identity_paths[:args.max_identities]
                console.print(f"[bold yellow]Limited to {args.max_identities} identities for testing[/bold yellow]")
                
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
            
            # Add subtask for identities
            identity_task = progress.add_task(f"[green]Analyzing {race} identities...", total=len(identity_paths))
            
            # Process each identity
            for identity_path in identity_paths:
                # Get images for this identity
                image_paths = get_images(identity_path)
                race_stats["images_per_identity"].append(len(image_paths))
                race_stats["total_images"] += len(image_paths)
                total_images += len(image_paths)
                
                # Identity-level gender consensus (majority vote)
                identity_genders = []
                identity_ages = []
                
                # Add subtask for images
                image_task = progress.add_task(f"[cyan]Processing images for {identity_path.name}...", total=len(image_paths), visible=False)
                
                # Process each image
                for img_path in image_paths:
                    progress.update(image_task, advance=1)
                    
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
                        identity_ages.append(age)
                        all_ages.append(age)
                    
                    if gender is not None:
                        identity_genders.append(gender)
                
                # Complete image task
                progress.update(image_task, visible=False)
                
                # Determine the majority gender for this identity
                if identity_genders:
                    # Convert to numpy for convenient majority voting
                    identity_genders = np.array(identity_genders)
                    majority_gender = int(np.round(np.mean(identity_genders)))
                    gender_label = get_gender_label(majority_gender)
                    
                    race_stats["gender"][gender_label] += 1
                    all_genders[gender_label] += 1
                
                # Update identity progress
                progress.update(identity_task, advance=1)
            
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
            
            # Update race progress
            progress.update(race_task, advance=1)
    
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

def generate_plots(results, output_dir):
    """
    Generate plots from the analysis results.
    
    Args:
        results (dict): Analysis results
        output_dir (Path): Directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        sns.set(style="whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
        
        # Create plots directory
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Gender distribution by race
        genders_by_race = {}
        for race, stats in results['races'].items():
            genders_by_race[race] = stats['gender']
        
        # Convert to dataframe
        import pandas as pd
        gender_data = []
        for race, counts in genders_by_race.items():
            for gender, count in counts.items():
                gender_data.append({'Race': race, 'Gender': gender, 'Count': count})
        
        gender_df = pd.DataFrame(gender_data)
        
        # Plot gender distribution
        plt.figure()
        gender_plot = sns.barplot(x='Race', y='Count', hue='Gender', data=gender_df)
        plt.title('Gender Distribution by Race')
        plt.tight_layout()
        plt.savefig(plots_dir / "gender_distribution.png", dpi=300)
        
        # 2. Age distribution by race
        age_data = []
        for race, stats in results['races'].items():
            age_stats = stats['age']['statistics']
            if age_stats['mean'] is not None:
                age_data.append({
                    'Race': race,
                    'Mean Age': age_stats['mean'],
                    'Min Age': age_stats['min'],
                    'Max Age': age_stats['max']
                })
        
        age_df = pd.DataFrame(age_data)
        
        plt.figure()
        age_plot = sns.barplot(x='Race', y='Mean Age', data=age_df)
        # Add error bars from min to max
        for i, row in age_df.iterrows():
            age_plot.errorbar(i, row['Mean Age'], 
                             yerr=[[row['Mean Age'] - row['Min Age']], [row['Max Age'] - row['Mean Age']]],
                             fmt='none', c='black', capsize=5)
        
        plt.title('Mean Age by Race (with Min-Max Range)')
        plt.tight_layout()
        plt.savefig(plots_dir / "age_distribution.png", dpi=300)
        
        # 3. Resolution distribution
        res_data = []
        for race, stats in results['races'].items():
            width_stats = stats['resolution']['width_statistics']
            height_stats = stats['resolution']['height_statistics']
            if width_stats['mean'] is not None and height_stats['mean'] is not None:
                res_data.append({
                    'Race': race,
                    'Mean Width': width_stats['mean'],
                    'Mean Height': height_stats['mean'],
                    'Mean Resolution': width_stats['mean'] * height_stats['mean']
                })
        
        res_df = pd.DataFrame(res_data)
        
        plt.figure()
        plt.bar(res_df['Race'], res_df['Mean Resolution'] / 1000000)  # Convert to megapixels
        plt.title('Mean Resolution by Race (Megapixels)')
        plt.ylabel('Megapixels')
        plt.tight_layout()
        plt.savefig(plots_dir / "resolution_distribution.png", dpi=300)
        
        # 4. Images per identity
        img_data = []
        for race, stats in results['races'].items():
            img_stats = stats['images_per_identity_statistics']
            if img_stats['mean'] is not None:
                img_data.append({
                    'Race': race,
                    'Mean Images': img_stats['mean'],
                    'Min Images': img_stats['min'],
                    'Max Images': img_stats['max']
                })
        
        img_df = pd.DataFrame(img_data)
        
        plt.figure()
        img_plot = sns.barplot(x='Race', y='Mean Images', data=img_df)
        for i, row in img_df.iterrows():
            img_plot.errorbar(i, row['Mean Images'], 
                             yerr=[[row['Mean Images'] - row['Min Images']], [row['Max Images'] - row['Mean Images']]],
                             fmt='none', c='black', capsize=5)
        
        plt.title('Mean Images per Identity by Race (with Min-Max Range)')
        plt.tight_layout()
        plt.savefig(plots_dir / "images_per_identity.png", dpi=300)
        
        console.print(f"[bold green]Plots saved to {plots_dir}[/bold green]")
        
    except ImportError:
        console.print("[bold yellow]Could not generate plots: matplotlib or seaborn not installed[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]Error generating plots: {e}[/bold red]")

def main():
    """
    Main function to run the analysis.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    output_dir = Path(os.path.dirname(args.output))
    output_dir.mkdir(exist_ok=True)
    
    # Load existing results or run analysis
    if args.skip_analysis and Path(args.output).exists():
        console.print(f"[bold yellow]Loading existing results from {args.output}[/bold yellow]")
        with open(args.output, 'r') as f:
            results = json.load(f)
    else:
        console.print("[bold green]Starting RFW Dataset Analysis...[/bold green]")
        start_time = time.time()
        
        # Run analysis
        results = analyze_dataset(args)
        
        # Save results
        save_results(results, args.output)
        
        end_time = time.time()
        console.print(f"[bold green]Analysis completed in {end_time - start_time:.2f} seconds[/bold green]")
    
    # Display results
    display_results(results)
    
    # Generate plots if requested
    if args.show_plots:
        generate_plots(results, output_dir)

if __name__ == "__main__":
    main()
