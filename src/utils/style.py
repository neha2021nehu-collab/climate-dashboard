"""
Plot styling utilities for consistent visualizations
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, Any

def setup_plot_style(config: Dict[str, Any]):
    """
    Setup matplotlib style based on configuration
    
    Parameters:
    -----------
    config : dict
        Style configuration dictionary
    """
    # Try to use the specified theme
    theme = config.get('theme', 'default')
    
    # List of available styles
    available_styles = plt.style.available
    
    # Try preferred styles in order
    preferred_styles = [
        'seaborn-v0_8-whitegrid',
        'seaborn-whitegrid', 
        'ggplot',
        'seaborn',
        'default'
    ]
    
    used_style = 'default'
    for style in preferred_styles:
        if style in available_styles:
            try:
                plt.style.use(style)
                used_style = style
                break
            except:
                continue
    
    # Get font configuration
    fonts = config.get('fonts', {})
    title_font = fonts.get('title', {})
    axis_font = fonts.get('axis', {})
    
    # Update rcParams
    rc_params = {
        # Font settings
        'font.size': axis_font.get('size', 9),
        'font.family': axis_font.get('family', 'sans-serif'),
        
        # Title settings
        'axes.titlesize': title_font.get('size', 12),
        'axes.titleweight': title_font.get('weight', 'bold'),
        'axes.titlepad': 10,
        
        # Axis settings
        'axes.labelsize': axis_font.get('size', 10),
        'axes.labelweight': 'bold',
        'axes.grid': True,
        'axes.grid.which': 'both',
        'axes.grid.axis': 'both',
        
        # Tick settings
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        
        # Legend settings
        'legend.fontsize': 8,
        'legend.frameon': True,
        'legend.framealpha': 0.8,
        'legend.edgecolor': 'black',
        
        # Figure settings
        'figure.titlesize': 16,
        'figure.titleweight': 'bold',
        'figure.dpi': 100,
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
        
        # Save settings
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',
        
        # Line settings
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        
        # Patch settings
        'patch.linewidth': 0.5,
        'patch.facecolor': '#348ABD',
        'patch.edgecolor': '#FFFFFF',
    }
    
    mpl.rcParams.update(rc_params)
    
    return used_style

def get_style_summary():
    """Get summary of current matplotlib style"""
    current_params = {
        'style': mpl.rcParams['axes.grid'],
        'font_family': mpl.rcParams['font.family'],
        'font_size': mpl.rcParams['font.size'],
        'figure_dpi': mpl.rcParams['figure.dpi'],
    }
    return current_params