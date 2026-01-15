"""
Heatmap calendar visualization for monthly anomalies
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import calendar

class HeatmapCalendar:
    """Create heatmap calendar for temperature anomalies"""
    
    def __init__(self, data: pd.DataFrame, config: Dict[str, Any]):
        self.data = data
        self.config = config
        self.setup_colormap()
    
    def setup_colormap(self):
        """Setup temperature colormap"""
        temp_config = self.config.get('colors', {}).get('temperature', {})
        diverging_cmap = temp_config.get('diverging', 'RdBu_r')
        
        if diverging_cmap.endswith('_r'):
            self.cmap = plt.cm.get_cmap(diverging_cmap[:-2]).reversed()
        else:
            self.cmap = plt.cm.get_cmap(diverging_cmap)
    
    def prepare_heatmap_data(self) -> pd.DataFrame:
        """Prepare data for heatmap visualization"""
        # Pivot to year x month format
        pivot_data = self.data.pivot_table(
            values='anomaly',
            index='year',
            columns='month',
            aggfunc='mean'
        )
        
        # Fill missing months with NaN
        pivot_data = pivot_data.reindex(columns=range(1, 13))
        
        return pivot_data
    
    def plot(self, ax: plt.Axes):
        """Plot heatmap calendar"""
        
        # Prepare data
        heatmap_data = self.prepare_heatmap_data()
        
        # Create heatmap
        im = ax.imshow(heatmap_data.T,  # Transpose for year x month
                      aspect='auto',
                      cmap=self.cmap,
                      vmin=-2, vmax=2,
                      interpolation='nearest',
                      origin='lower')
        
        # Configure axes
        ax.set_xlabel('Year', fontweight='bold', fontsize=10)
        ax.set_ylabel('Month', fontweight='bold', fontsize=10)
        
        # Set x-ticks (every 10 years)
        years = heatmap_data.index
        if len(years) > 50:
            tick_step = max(1, len(years) // 20) * 10
        else:
            tick_step = 10
        
        tick_indices = []
        tick_labels = []
        for i, year in enumerate(years):
            if year % tick_step == 0:
                tick_indices.append(i)
                tick_labels.append(str(year))
        
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)
        
        # Set y-ticks (months)
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_yticks(np.arange(12))
        ax.set_yticklabels(month_names, fontsize=9)
        
        # Add grid
        ax.set_xticks(np.arange(len(years)) - 0.5, minor=True)
        ax.set_yticks(np.arange(12) - 0.5, minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.2)
        
        # Add title
        start_year = self.data['year'].min()
        end_year = self.data['year'].max()
        ax.set_title(f'Monthly Temperature Anomalies\n{start_year}-{end_year}', 
                    fontsize=12, fontweight='bold', pad=15)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)
        cbar.set_label('Temperature Anomaly (°C)', fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        
        # Highlight extreme years
        self._highlight_extreme_years(ax, heatmap_data)
        
        return im
    
    def _highlight_extreme_years(self, ax: plt.Axes, heatmap_data: pd.DataFrame):
        """Highlight years with extreme temperatures"""
        # Calculate annual averages
        annual_avg = heatmap_data.mean(axis=1)
        
        # Find top 5 warmest and coldest years
        warmest_years = annual_avg.nlargest(5).index.tolist()
        coldest_years = annual_avg.nsmallest(5).index.tolist()
        
        # Get indices for highlighting
        years = heatmap_data.index
        for year in warmest_years:
            if year in years:
                idx = list(years).index(year)
                # Add red border for warmest years
                rect = plt.Rectangle((idx - 0.5, -0.5), 1, 12,
                                   fill=False, edgecolor='red', linewidth=2, alpha=0.7)
                ax.add_patch(rect)
                
                # Add label
                ax.text(idx, -0.7, str(year), 
                       ha='center', va='top', fontsize=7, fontweight='bold',
                       color='red', alpha=0.8)
        
        for year in coldest_years:
            if year in years and year not in warmest_years:
                idx = list(years).index(year)
                # Add blue border for coldest years
                rect = plt.Rectangle((idx - 0.5, -0.5), 1, 12,
                                   fill=False, edgecolor='blue', linewidth=2, alpha=0.7)
                ax.add_patch(rect)
                
                # Add label
                ax.text(idx, 12.5, str(year), 
                       ha='center', va='bottom', fontsize=7, fontweight='bold',
                       color='blue', alpha=0.8)