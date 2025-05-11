#!/usr/bin/env python3
"""
Script to run all tests for the RFW project.
"""

import os
import sys
import unittest
from rich.console import Console

console = Console()

def run_tests():
    """
    Run all tests in the tests directory.
    """
    console.print("[bold blue]Running tests for RFW project...[/bold blue]")
    
    # Discover and run tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests')
    
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Display results
    if result.wasSuccessful():
        console.print("[bold green]All tests passed![/bold green]")
        return 0
    else:
        console.print("[bold red]Some tests failed![/bold red]")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())
