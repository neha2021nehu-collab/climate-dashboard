#!/usr/bin/env python3
"""
Main script to run the climate dashboard
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dashboard import main

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('outputs/dashboards', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/animations', exist_ok=True)
    
    # Run the dashboard
    main()