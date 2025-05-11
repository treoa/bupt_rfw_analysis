#!/usr/bin/env python3
"""
Script to visualize RFW analysis results.
"""

import os
import sys
import argparse
from pathlib import Path
from rich.console import Console

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.visualization import load_and_visualize

console = Console()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize RFW dataset analysis results.')
    parser.add_argument('--results', type=str, default='results/rfw_analysis_results.json',
                       help='Path to the analysis results file')
    parser.add_argument('--output', type=str, default='results/plots',
                       help='Directory to save visualization plots')
    return parser.parse_args()

def main():
    """
    Main entry point for visualization.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Check if results file exists
    results_path = Path(args.results)
    if not results_path.exists():
        console.print(f"[bold red]Results file not found: {results_path}[/bold red]")
        console.print("[yellow]Please run the analysis first or specify the correct path using --results[/yellow]")
        return
    
    # Create output directory
    output_dir = Path(args.output)
    
    # Load results and create visualizations
    console.print(f"[bold blue]Loading results from {results_path}...[/bold blue]")
    success = load_and_visualize(results_path, output_dir)
    
    if success:
        console.print(f"[bold green]Visualizations created successfully in {output_dir}[/bold green]")
    else:
        console.print("[bold red]Failed to create visualizations[/bold red]")

if __name__ == "__main__":
    main()
