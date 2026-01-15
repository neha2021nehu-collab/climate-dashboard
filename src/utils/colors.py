"""
Color palette utilities for consistent coloring
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Tuple
import matplotlib.colors as mcolors

def get_color_palette(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get color palette from configuration
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
        
    Returns:
    --------
    dict : Color palette dictionary
    """
    colors_config = config.get('colors', {})
    
    # Create color palettes
    palettes = {
        'temperature': _get_temperature_colors(colors_config.get('temperature', {})),
        'emissions': colors_config.get('emissions', {}),
        'continents': colors_config.get('continents', {}),
        'sources': _get_energy_source_colors(),
        'sequential': _get_sequential_palettes(),
        'diverging': _get_diverging_palettes(),
        'categorical': _get_categorical_palettes(),
    }
    
    return palettes

def _get_temperature_colors(temp_config: Dict[str, Any]) -> Dict[str, Any]:
    """Get temperature-specific colors"""
    return {
        'sequential': temp_config.get('sequential', ['#053061', '#2166ac', '#4393c3', 
                                                    '#92c5de', '#d1e5f0', '#f7f7f7', 
                                                    '#fddbc7', '#f4a582', '#d6604d', 
                                                    '#b2182b', '#67001f']),
        'diverging': temp_config.get('diverging', 'RdBu_r'),
        'cool': ['#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0'],
        'warm': ['#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f'],
        'neutral': '#f7f7f7'
    }

def _get_energy_source_colors() -> Dict[str, str]:
    """Get colors for energy sources"""
    return {
        'Coal': '#333333',
        'Oil': '#8B0000',
        'Gas': '#FF8C00',
        'Nuclear': '#FFD700',
        'Hydro': '#1E90FF',
        'Wind': '#32CD32',
        'Solar': '#FF4500',
        'Other Renewables': '#9370DB',
        'Biomass': '#8B4513',
        'Geothermal': '#DC143C'
    }

def _get_sequential_palettes() -> Dict[str, List[str]]:
    """Get sequential color palettes"""
    return {
        'viridis': plt.cm.viridis(np.linspace(0, 1, 256)),
        'plasma': plt.cm.plasma(np.linspace(0, 1, 256)),
        'inferno': plt.cm.inferno(np.linspace(0, 1, 256)),
        'magma': plt.cm.magma(np.linspace(0, 1, 256)),
        'cividis': plt.cm.cividis(np.linspace(0, 1, 256)),
        'summer': plt.cm.summer(np.linspace(0, 1, 256)),
        'wistia': plt.cm.wistia(np.linspace(0, 1, 256)),
    }

def _get_diverging_palettes() -> Dict[str, List[str]]:
    """Get diverging color palettes"""
    return {
        'RdBu_r': plt.cm.RdBu_r(np.linspace(0, 1, 256)),
        'coolwarm': plt.cm.coolwarm(np.linspace(0, 1, 256)),
        'bwr': plt.cm.bwr(np.linspace(0, 1, 256)),
        'seismic': plt.cm.seismic(np.linspace(0, 1, 256)),
        'PuOr': plt.cm.PuOr(np.linspace(0, 1, 256)),
        'RdYlBu': plt.cm.RdYlBu(np.linspace(0, 1, 256)),
    }

def _get_categorical_palettes() -> Dict[str, List[str]]:
    """Get categorical color palettes"""
    return {
        'tab10': plt.cm.tab10(np.linspace(0, 1, 10)),
        'tab20': plt.cm.tab20(np.linspace(0, 1, 20)),
        'Set3': plt.cm.Set3(np.linspace(0, 1, 12)),
        'Set2': plt.cm.Set2(np.linspace(0, 1, 8)),
        'Paired': plt.cm.Paired(np.linspace(0, 1, 12)),
        'Pastel1': plt.cm.Pastel1(np.linspace(0, 1, 9)),
        'Pastel2': plt.cm.Pastel2(np.linspace(0, 1, 8)),
    }

def get_color_for_value(value: float, vmin: float, vmax: float, 
                       cmap_name: str = 'RdBu_r') -> Tuple[float, float, float, float]:
    """
    Get color for a value within a range
    
    Parameters:
    -----------
    value : float
        Value to color
    vmin : float
        Minimum value in range
    vmax : float
        Maximum value in range
    cmap_name : str
        Colormap name
        
    Returns:
    --------
    tuple : RGBA color
    """
    # Normalize value
    norm_value = (value - vmin) / (vmax - vmin)
    norm_value = max(0, min(1, norm_value))  # Clamp to [0, 1]
    
    # Get colormap
    if cmap_name.endswith('_r'):
        cmap = plt.cm.get_cmap(cmap_name[:-2]).reversed()
    else:
        cmap = plt.cm.get_cmap(cmap_name)
    
    return cmap(norm_value)

def get_accessible_colors():
    """Get colorblind-friendly colors"""
    return {
        'blue': '#377eb8',
        'orange': '#ff7f00',
        'green': '#4daf4a',
        'pink': '#f781bf',
        'brown': '#a65628',
        'purple': '#984ea3',
        'gray': '#999999',
        'red': '#e41a1c',
        'yellow': '#dede00'
    }

def create_custom_colormap(colors: List[str], name: str = 'custom'):
    """
    Create a custom colormap from list of colors
    
    Parameters:
    -----------
    colors : list of str
        List of color hex codes
    name : str
        Name for the colormap
        
    Returns:
    --------
    LinearSegmentedColormap
    """
    return mcolors.LinearSegmentedColormap.from_list(name, colors)