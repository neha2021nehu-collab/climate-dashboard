"""
Animated scatter plot visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

class AnimatedScatter:
    """Create animated scatter plot for climate-economy relationships"""
    
    def __init__(self, data: pd.DataFrame, config: Dict[str, Any]):
        self.data = data
        self.config = config
    
    def plot(self, ax: plt.Axes):
        """Plot static version of animated scatter"""
        
        # Aggregate data by year and country
        if 'country' in self.data.columns:
            yearly_data = self.data.groupby(['year', 'country']).agg({
                'generation_TWh': 'sum',
                'carbon_intensity': 'mean',
                'share': 'mean'
            }).reset_index()
        else:
            # Use simplified data if country column not present
            yearly_data = self.data.copy()
        
        # Select a specific year for static plot
        plot_year = 2020
        year_data = yearly_data[yearly_data['year'] == plot_year]
        
        if year_data.empty:
            # Use latest available year
            plot_year = yearly_data['year'].max()
            year_data = yearly_data[yearly_data['year'] == plot_year]
        
        # Create scatter plot
        if 'generation_TWh' in year_data.columns and 'carbon_intensity' in year_data.columns:
            scatter = ax.scatter(
                year_data['generation_TWh'],
                year_data['carbon_intensity'],
                s=year_data.get('share', 100) * 10,  # Size proportional to share
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
            
            ax.set_xlabel('Total Generation (TWh)', fontweight='bold')
            ax.set_ylabel('Carbon Intensity (gCO₂/kWh)', fontweight='bold')
            
            # Add trend line
            if len(year_data) > 1:
                x = year_data['generation_TWh'].values
                y = year_data['carbon_intensity'].values
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), 'r--', alpha=0.7, linewidth=2,
                       label=f'Trend: y = {z[0]:.3f}x + {z[1]:.1f}')
            
            ax.legend(fontsize=8)
        
        else:
            # Fallback if columns don't exist
            ax.text(0.5, 0.5, 'Energy Data Visualization\n(Animated scatter placeholder)',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=11, fontweight='bold')
        
        ax.set_title(f'Energy Generation vs Carbon Intensity ({plot_year})',
                    fontsize=12, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)