"""
Ridgeline plot visualization for climate model ensembles
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

class RidgelinePlot:
    """Create ridgeline plot for temperature distributions"""
    
    def __init__(self, data: pd.DataFrame, config: Dict[str, Any]):
        self.data = data
        self.config = config
    
    def plot(self, ax: plt.Axes):
        """Plot ridgeline plot of temperature distributions by decade"""
        
        # Create decades
        self.data['decade'] = (self.data['year'] // 10) * 10
        decades = sorted(self.data['decade'].unique())
        
        # Filter to complete decades
        plot_decades = []
        plot_data = []
        
        for decade in decades:
            if decade >= 1850:
                decade_data = self.data[self.data['decade'] == decade]['anomaly'].values
                if len(decade_data) > 50:  # Require sufficient data
                    plot_decades.append(str(decade))
                    plot_data.append(decade_data)
        
        if len(plot_data) < 3:
            # Not enough data for ridgeline plot
            ax.text(0.5, 0.5, 'Temperature Distribution by Decade\n(Ridgeline plot placeholder)',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=11, fontweight='bold')
            ax.axis('off')
            return
        
        # Create ridgeline plot
        n = len(plot_data)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, n))
        
        # Create KDE plots for each decade
        from scipy.stats import gaussian_kde
        
        x_range = np.linspace(-2, 2, 200)
        
        max_density = 0
        kde_plots = []
        
        for i, (data, color) in enumerate(zip(plot_data, colors)):
            # Calculate KDE
            try:
                kde = gaussian_kde(data)
                y = kde(x_range)
                max_density = max(max_density, y.max())
                
                # Offset for ridgeline effect
                offset = i * 0.3
                
                # Fill under curve
                ax.fill_between(x_range, offset, offset + y, 
                               color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # Add decade label
                ax.text(-2.2, offset + 0.15, plot_decades[i],
                       ha='right', va='center', fontsize=8, fontweight='bold')
                
                kde_plots.append((x_range, y, offset))
                
                # Add mean line - FIXED: y_at_mean should be scalar
                mean_val = np.mean(data)
                y_at_mean = float(kde(mean_val))  # Convert to scalar
                
                ax.plot([mean_val, mean_val], [offset, offset + y_at_mean],
                       'k-', linewidth=1, alpha=0.8)
                ax.plot(mean_val, offset + y_at_mean, 'ko', markersize=4)
                
            except Exception as e:
                print(f"Warning: Could not create KDE for decade {plot_decades[i]}: {e}")
                continue
        
        # Configure plot
        ax.set_xlabel('Temperature Anomaly (°C)', fontweight='bold')
        ax.set_ylabel('Decade', fontweight='bold')
        ax.set_yticks([])
        ax.set_title('Temperature Distribution by Decade', 
                    fontsize=12, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Set limits
        ax.set_xlim(-2.5, 2.5)
        if kde_plots:
            max_offset = max([plot[2] for plot in kde_plots])
            ax.set_ylim(-0.5, max_offset + 0.5)
        
        # Add colorbar for timeline
        if plot_decades:
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            
            sm = ScalarMappable(cmap='viridis', 
                               norm=Normalize(vmin=int(plot_decades[0]), 
                                             vmax=int(plot_decades[-1])))
            sm.set_array([])
            
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.8)
            cbar.set_label('Decade', fontsize=9)
            cbar.ax.tick_params(labelsize=8)