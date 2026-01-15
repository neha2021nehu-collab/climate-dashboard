"""
Utility modules for climate visualization dashboard
"""

from .style import setup_plot_style, get_style_summary
from .colors import get_color_palette, get_color_for_value, create_custom_colormap
from .helpers import (normalize_data, resample_time_series, calculate_trend,
                     detect_outliers, smooth_data, format_large_number)

__all__ = [
    'setup_plot_style',
    'get_style_summary',
    'get_color_palette',
    'get_color_for_value',
    'create_custom_colormap',
    'normalize_data',
    'resample_time_series',
    'calculate_trend',
    'detect_outliers',
    'smooth_data',
    'format_large_number'
]