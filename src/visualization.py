"""
Visualization utilities for RFW dataset analysis results.
"""

import json
import numpy as np
from pathlib import Path
from rich.console import Console

console = Console()

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    VISUALIZATION_AVAILABLE = True
except ImportError:
    console.print("[yellow]Matplotlib, Seaborn, or Pandas not available. Visualization features disabled.[/yellow]")
    VISUALIZATION_AVAILABLE = False

def plot_gender_distribution(results, output_path=None):
    """
    Plot gender distribution by race.
    
    Args:
        results (dict): Analysis results
        output_path (Path, optional): Path to save the plot
    """
    if not VISUALIZATION_AVAILABLE:
        console.print("[yellow]Visualization libraries not available. Cannot create gender distribution plot.[/yellow]")
        return
    
    # Set style
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Extract gender data
    gender_data = []
    for race, stats in results['races'].items():
        for gender, count in stats['gender'].items():
            gender_data.append({'Race': race, 'Gender': gender, 'Count': count})
    
    gender_df = pd.DataFrame(gender_data)
    
    # Create plot
    ax = sns.barplot(x='Race', y='Count', hue='Gender', data=gender_df)
    
    # Add percentage annotations
    for race in gender_df['Race'].unique():
        race_data = gender_df[gender_df['Race'] == race]
        total = race_data['Count'].sum()
        
        for i, row in race_data.iterrows():
            percentage = (row['Count'] / total) * 100 if total > 0 else 0
            ax.text(
                row.name % len(race_data['Race'].unique()), 
                row['Count'], 
                f"{percentage:.1f}%", 
                ha='center', 
                va='bottom'
            )
    
    plt.title('Gender Distribution by Race')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        console.print(f"[green]Gender distribution plot saved to {output_path}[/green]")
    
    plt.close()

def plot_age_distribution(results, output_path=None):
    """
    Plot age distribution by race.
    
    Args:
        results (dict): Analysis results
        output_path (Path, optional): Path to save the plot
    """
    if not VISUALIZATION_AVAILABLE:
        console.print("[yellow]Visualization libraries not available. Cannot create age distribution plot.[/yellow]")
        return
    
    # Set style
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Extract age data
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
    
    # Create plot
    ax = sns.barplot(x='Race', y='Mean Age', data=age_df)
    
    # Add error bars from min to max
    for i, row in age_df.iterrows():
        plt.errorbar(
            i, row['Mean Age'], 
            yerr=[[row['Mean Age'] - row['Min Age']], [row['Max Age'] - row['Mean Age']]],
            fmt='none', c='black', capsize=5
        )
        
        # Add text annotations
        ax.text(i, row['Mean Age'] + 2, f"Mean: {row['Mean Age']:.1f}", ha='center')
        ax.text(i, row['Min Age'] - 2, f"Min: {row['Min Age']:.1f}", ha='center')
        ax.text(i, row['Max Age'] + 2, f"Max: {row['Max Age']:.1f}", ha='center')
    
    plt.title('Age Distribution by Race')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        console.print(f"[green]Age distribution plot saved to {output_path}[/green]")
    
    plt.close()

def plot_resolution_distribution(results, output_path=None):
    """
    Plot resolution distribution by race.
    
    Args:
        results (dict): Analysis results
        output_path (Path, optional): Path to save the plot
    """
    if not VISUALIZATION_AVAILABLE:
        console.print("[yellow]Visualization libraries not available. Cannot create resolution distribution plot.[/yellow]")
        return
    
    # Set style
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Extract resolution data
    res_data = []
    for race, stats in results['races'].items():
        width_stats = stats['resolution']['width_statistics']
        height_stats = stats['resolution']['height_statistics']
        if width_stats['mean'] is not None and height_stats['mean'] is not None:
            res_data.append({
                'Race': race,
                'Mean Width': width_stats['mean'],
                'Mean Height': height_stats['mean'],
                'Mean Resolution': width_stats['mean'] * height_stats['mean'] / 1000000,  # Megapixels
                'Min Width': width_stats['min'],
                'Max Width': width_stats['max'],
                'Min Height': height_stats['min'],
                'Max Height': height_stats['max']
            })
    
    res_df = pd.DataFrame(res_data)
    
    # Create plot
    ax = sns.barplot(x='Race', y='Mean Resolution', data=res_df)
    
    # Add annotations
    for i, row in res_df.iterrows():
        ax.text(
            i, row['Mean Resolution'] + 0.05, 
            f"{row['Mean Width']:.0f}×{row['Mean Height']:.0f}", 
            ha='center'
        )
    
    plt.title('Mean Resolution by Race (Megapixels)')
    plt.ylabel('Megapixels')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        console.print(f"[green]Resolution distribution plot saved to {output_path}[/green]")
    
    plt.close()

def plot_images_per_identity(results, output_path=None):
    """
    Plot images per identity distribution by race.
    
    Args:
        results (dict): Analysis results
        output_path (Path, optional): Path to save the plot
    """
    if not VISUALIZATION_AVAILABLE:
        console.print("[yellow]Visualization libraries not available. Cannot create images per identity plot.[/yellow]")
        return
    
    # Set style
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Extract images per identity data
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
    
    # Create plot
    ax = sns.barplot(x='Race', y='Mean Images', data=img_df)
    
    # Add error bars and annotations
    for i, row in img_df.iterrows():
        plt.errorbar(
            i, row['Mean Images'], 
            yerr=[[row['Mean Images'] - row['Min Images']], [row['Max Images'] - row['Mean Images']]],
            fmt='none', c='black', capsize=5
        )
        
        # Add text annotations
        ax.text(i, row['Mean Images'] + 0.1, f"Mean: {row['Mean Images']:.1f}", ha='center')
        ax.text(i, row['Min Images'] - 0.2, f"Min: {row['Min Images']:.0f}", ha='center')
        ax.text(i, row['Max Images'] + 0.2, f"Max: {row['Max Images']:.0f}", ha='center')
    
    plt.title('Images per Identity by Race')
    plt.ylabel('Number of Images')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        console.print(f"[green]Images per identity plot saved to {output_path}[/green]")
    
    plt.close()

def create_all_plots(results, output_dir=None):
    """
    Create all plots from analysis results.
    
    Args:
        results (dict): Analysis results
        output_dir (Path, optional): Directory to save plots
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create plots
    plot_gender_distribution(
        results, 
        output_dir / "gender_distribution.png" if output_dir else None
    )
    
    plot_age_distribution(
        results, 
        output_dir / "age_distribution.png" if output_dir else None
    )
    
    plot_resolution_distribution(
        results, 
        output_dir / "resolution_distribution.png" if output_dir else None
    )
    
    plot_images_per_identity(
        results, 
        output_dir / "images_per_identity.png" if output_dir else None
    )
    
    if output_dir:
        console.print(f"[bold green]All plots saved to {output_dir}[/bold green]")

def load_and_visualize(results_path, output_dir=None):
    """
    Load results from file and create visualizations.
    
    Args:
        results_path (str): Path to results JSON file
        output_dir (str, optional): Directory to save plots
    """
    try:
        # Load results
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Create plots
        create_all_plots(results, output_dir)
        
        return True
    except Exception as e:
        console.print(f"[bold red]Error loading results or creating visualizations: {e}[/bold red]")
        return False
