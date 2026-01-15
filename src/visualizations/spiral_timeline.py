"""
Spiral timeline visualization for temperature anomalies
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes
import pandas as pd
from typing import Dict, Any, Optional
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap

class SpiralTimeline:
    """Create spiral visualization of temperature anomalies"""
    
    def __init__(self, data: pd.DataFrame, config: Dict[str, Any]):
        self.data = data
        self.config = config
        self.setup_colormap()
    
    def setup_colormap(self):
        """Setup temperature colormap"""
        temp_config = self.config.get('colors', {}).get('temperature', {})
        diverging_cmap = temp_config.get('diverging', 'RdBu_r')
        
        # Get colormap
        if diverging_cmap.endswith('_r'):
            self.cmap = plt.cm.get_cmap(diverging_cmap[:-2]).reversed()
        else:
            self.cmap = plt.cm.get_cmap(diverging_cmap)
    
    def create_spiral_coordinates(self, start_year: int = 1850):
        """Create spiral coordinates from data"""
        # Convert dates to spiral coordinates
        years = self.data['year'] + (self.data['month'] - 1) / 12
        theta = 2 * np.pi * ((years - start_year) % 1)  # Month angle
        radius = 1 + (years - start_year) / 10  # Increasing radius
        
        return theta, radius
    
    def plot(self, ax: PolarAxes, highlight_years: Optional[list] = None):
        """Plot spiral timeline"""
        
        # Get spiral coordinates
        theta, radius = self.create_spiral_coordinates()
        
        # Get temperature anomalies
        anomalies = self.data['anomaly'].values
        
        # Determine color range
        vmin = max(-2.0, np.nanmin(anomalies))
        vmax = min(2.0, np.nanmax(anomalies))
        
        # Create scatter plot
        sc = ax.scatter(
            theta, radius,
            c=anomalies,
            cmap=self.cmap,
            s=8,
            alpha=0.7,
            edgecolors='none',
            vmin=vmin,
            vmax=vmax,
            zorder=2
        )
        
        # Configure polar plot
        ax.set_theta_offset(np.pi / 2)  # Start at top
        ax.set_theta_direction(-1)  # Clockwise
        
        # Set radial limits
        max_year = self.data['year'].max()
        max_radius = 1 + (max_year - 1850) / 10
        ax.set_rlim(1, max_radius)
        ax.set_rticks([])  # Remove radial labels
        
        # Add month labels
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticks(np.linspace(0, 2*np.pi, 12, endpoint=False))
        ax.set_xticklabels(months, fontsize=7)
        
        # Add decade rings and labels
        for decade in range(1850, 2021, 50):
            if 1850 <= decade <= max_year:
                r = 1 + (decade - 1850) / 10
                ax.plot(np.linspace(0, 2*np.pi, 100), 
                       [r] * 100, 
                       'k-', alpha=0.2, linewidth=0.5, zorder=1)
                
                # Add decade label at 0 degrees
                ax.text(0, r, f'{decade}s', 
                       fontsize=6, 
                       va='center', 
                       ha='right',
                       transform=ax.get_yaxis_transform(),
                       bbox=dict(boxstyle='round,pad=0.1', 
                                facecolor='white', 
                                edgecolor='none',
                                alpha=0.7))
        
        # Highlight specific years
        if highlight_years:
            for year in highlight_years:
                if year in self.data['year'].values:
                    idx = self.data['year'] == year
                    ax.scatter(theta[idx], radius[idx],
                             s=20, facecolors='none', 
                             edgecolors='gold', linewidths=1.5,
                             zorder=3, label=f'{year}')
        
        # Add colorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1, axes_class=plt.Axes)
        cbar = plt.colorbar(sc, cax=cax, orientation='vertical')
        cbar.set_label('Temperature Anomaly (°C)', fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        
        # Set title
        ax.set_title('Global Temperature Timeline\n1850-2023', 
                    fontsize=10, fontweight='bold', pad=20)
        
        # Add significant events annotations
        self._add_event_annotations(ax, theta, radius)
        
        return sc
    
    def _add_event_annotations(self, ax: PolarAxes, theta: np.ndarray, radius: np.ndarray):
        """Add annotations for significant climate events"""
        events = {
            1883: ('Krakatoa', -0.5),
            1912: ('Novarupta', -0.4),
            1991: ('Pinatubo', -0.5),
            1997: ('El Niño', 0.4),
            2015: ('Paris Agreement', 0.3),
            2020: ('COVID-19', -0.1),
        }
        
        for year, (event, impact) in events.items():
            if year in self.data['year'].values:
                idx = self.data['year'] == year
                if idx.any():
                    # Find approximate position
                    r_mean = radius[idx].mean()
                    t_mean = theta[idx].mean()
                    
                    # Add annotation
                    ax.annotate(event, 
                               xy=(t_mean, r_mean),
                               xytext=(10, 10),
                               textcoords='offset points',
                               fontsize=6,
                               arrowprops=dict(arrowstyle='->', 
                                             lw=0.5,
                                             connectionstyle='arc3,rad=0.1'),
                               ha='left', va='bottom',
                               bbox=dict(boxstyle='round,pad=0.2', 
                                        facecolor='white', 
                                        edgecolor='gray',
                                        alpha=0.8))