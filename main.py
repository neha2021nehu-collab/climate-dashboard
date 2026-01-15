#!/usr/bin/env python3
"""
Main script to generate climate dashboard
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.dashboard import main as dashboard_main

if __name__ == "__main__":
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    
    # Run dashboard generation
    dashboard_main()