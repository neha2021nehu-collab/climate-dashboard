"""
Helper functions for data processing and visualization
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def normalize_data(data: np.ndarray, 
                   method: str = 'minmax',
                   axis: int = 0) -> np.ndarray:
    """
    Normalize data using specified method
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    method : str
        Normalization method: 'minmax', 'zscore', 'robust', 'log'
    axis : int
        Axis to normalize along
        
    Returns:
    --------
    np.ndarray : Normalized data
    """
    if method == 'minmax':
        min_val = np.nanmin(data, axis=axis, keepdims=True)
        max_val = np.nanmax(data, axis=axis, keepdims=True)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1  # Avoid division by zero
        return (data - min_val) / range_val
    
    elif method == 'zscore':
        mean_val = np.nanmean(data, axis=axis, keepdims=True)
        std_val = np.nanstd(data, axis=axis, keepdims=True)
        std_val[std_val == 0] = 1  # Avoid division by zero
        return (data - mean_val) / std_val
    
    elif method == 'robust':
        median_val = np.nanmedian(data, axis=axis, keepdims=True)
        iqr_val = np.nanpercentile(data, 75, axis=axis, keepdims=True) - \
                  np.nanpercentile(data, 25, axis=axis, keepdims=True)
        iqr_val[iqr_val == 0] = 1  # Avoid division by zero
        return (data - median_val) / iqr_val
    
    elif method == 'log':
        # Add small constant to avoid log(0)
        data_positive = data - np.nanmin(data) + 1e-10
        return np.log(data_positive)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def resample_time_series(data: pd.DataFrame,
                        freq: str = 'M',
                        agg_func: str = 'mean') -> pd.DataFrame:
    """
    Resample time series data
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data with datetime index
    freq : str
        Resampling frequency
    agg_func : str
        Aggregation function
        
    Returns:
    --------
    pd.DataFrame : Resampled data
    """
    if 'date' in data.columns:
        data = data.set_index('date')
    
    if agg_func == 'mean':
        resampled = data.resample(freq).mean()
    elif agg_func == 'sum':
        resampled = data.resample(freq).sum()
    elif agg_func == 'median':
        resampled = data.resample(freq).median()
    else:
        resampled = data.resample(freq).apply(agg_func)
    
    return resampled.reset_index()

def calculate_trend(data: pd.Series,
                   method: str = 'linear') -> Tuple[float, float, np.ndarray]:
    """
    Calculate trend in time series data
    
    Parameters:
    -----------
    data : pd.Series
        Time series data
    method : str
        Trend calculation method: 'linear', 'theilsen', 'mannkendall'
        
    Returns:
    --------
    tuple : (slope, intercept, trend_values)
    """
    x = np.arange(len(data))
    y = data.values
    
    # Remove NaN values
    mask = ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    if method == 'linear':
        # Linear regression
        A = np.vstack([x_clean, np.ones(len(x_clean))]).T
        slope, intercept = np.linalg.lstsq(A, y_clean, rcond=None)[0]
        trend = slope * x + intercept
    
    elif method == 'theilsen':
        # Theil-Sen estimator (robust)
        from scipy.stats import theilslopes
        slope, intercept, _, _ = theilslopes(y_clean, x_clean)
        trend = slope * x + intercept
    
    elif method == 'mannkendall':
        # Mann-Kendall trend test
        from scipy.stats import kendalltau
        tau, p_value = kendalltau(x_clean, y_clean)
        slope = tau  # Simplified slope
        intercept = np.median(y_clean) - slope * np.median(x_clean)
        trend = slope * x + intercept
    
    else:
        raise ValueError(f"Unknown trend method: {method}")
    
    return slope, intercept, trend

def detect_outliers(data: np.ndarray,
                   method: str = 'iqr',
                   threshold: float = 1.5) -> np.ndarray:
    """
    Detect outliers in data
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    method : str
        Outlier detection method: 'iqr', 'zscore', 'mad'
    threshold : float
        Detection threshold
        
    Returns:
    --------
    np.ndarray : Boolean mask of outliers
    """
    if method == 'iqr':
        # Interquartile range method
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outliers = (data < lower_bound) | (data > upper_bound)
    
    elif method == 'zscore':
        # Z-score method
        mean_val = np.mean(data)
        std_val = np.std(data)
        z_scores = np.abs((data - mean_val) / std_val)
        outliers = z_scores > threshold
    
    elif method == 'mad':
        # Median Absolute Deviation method
        median_val = np.median(data)
        mad = np.median(np.abs(data - median_val))
        modified_z_scores = 0.6745 * (data - median_val) / mad
        outliers = np.abs(modified_z_scores) > threshold
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    return outliers

def smooth_data(data: np.ndarray,
               window_size: int = 5,
               method: str = 'moving_average') -> np.ndarray:
    """
    Smooth data using specified method
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    window_size : int
        Smoothing window size
    method : str
        Smoothing method: 'moving_average', 'exponential', 'savitzky_golay'
        
    Returns:
    --------
    np.ndarray : Smoothed data
    """
    if method == 'moving_average':
        window = np.ones(window_size) / window_size
        smoothed = np.convolve(data, window, mode='same')
    
    elif method == 'exponential':
        # Exponential smoothing
        alpha = 2 / (window_size + 1)
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
    
    elif method == 'savitzky_golay':
        from scipy.signal import savgol_filter
        smoothed = savgol_filter(data, window_size, 2)  # 2nd order polynomial
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    
    return smoothed

def format_large_number(num: float) -> str:
    """
    Format large numbers with SI prefixes
    
    Parameters:
    -----------
    num : float
        Number to format
        
    Returns:
    --------
    str : Formatted number string
    """
    if num == 0:
        return "0"
    
    # Determine magnitude
    abs_num = abs(num)
    sign = "-" if num < 0 else ""
    
    if abs_num >= 1e12:
        return f"{sign}{abs_num/1e12:.2f}T"
    elif abs_num >= 1e9:
        return f"{sign}{abs_num/1e9:.2f}B"
    elif abs_num >= 1e6:
        return f"{sign}{abs_num/1e6:.2f}M"
    elif abs_num >= 1e3:
        return f"{sign}{abs_num/1e3:.2f}k"
    elif abs_num < 0.001:
        return f"{sign}{abs_num:.2e}"
    else:
        return f"{sign}{abs_num:.2f}"

def get_season_from_month(month: int) -> str:
    """Get season name from month number"""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

def create_time_bins(dates: pd.DatetimeIndex,
                    bin_size: str = 'decade') -> pd.Series:
    """
    Create time bins for grouping
    
    Parameters:
    -----------
    dates : pd.DatetimeIndex
        Dates to bin
    bin_size : str
        Bin size: 'year', 'decade', 'century', 'custom'
        
    Returns:
    --------
    pd.Series : Bin labels
    """
    if bin_size == 'year':
        return dates.year
    elif bin_size == 'decade':
        return (dates.year // 10) * 10
    elif bin_size == 'century':
        return (dates.year // 100) * 100
    elif bin_size == 'half_century':
        return (dates.year // 50) * 50
    else:
        raise ValueError(f"Unknown bin size: {bin_size}")