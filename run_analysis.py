#!/usr/bin/env python3
"""
Quick entry script for RFW Analysis to make running even easier.
"""

import os
import sys
import subprocess
from pathlib import Path
from rich.console import Console

console = Console()

def main():
    """
    Main entry point for the RFW analysis tool.
    """
    # Check if requirements are installed
    try:
        import torch
        import torchvision
        import rich
        import numpy
        from PIL import Image
    except ImportError as e:
        console.print(f"[bold red]Missing dependency: {e}[/bold red]")
        console.print("[yellow]Installing dependencies from requirements.txt...[/yellow]")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Run the main analysis script
    console.print("[bold green]Starting RFW Analysis...[/bold green]")
    
    # Get command line arguments, skipping the script name
    args = sys.argv[1:]
    
    # Run the main script with arguments
    subprocess.run([sys.executable, "main.py"] + args)

if __name__ == "__main__":
    main()
