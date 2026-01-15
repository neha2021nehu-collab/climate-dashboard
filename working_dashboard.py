#!/usr/bin/env python3
"""
Working dashboard - fixed version
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import os
import matplotlib as mpl

class WorkingDashboard:
    """Simple working dashboard"""
    
    def __init__(self):
        self.setup_style()
        self.data = self.generate_data()
    
    def setup_style(self):
        """Setup matplotlib with available style"""
        # List available styles and use one that exists
        available = plt.style.available
        print(f"Available styles: {available[:5]}...")  # Show first 5
        
        # Try common styles
        for style in ['seaborn-v0_8-whitegrid', 'seaborn-whitegrid', 'ggplot', 'default']:
            if style in available:
                plt.style.use(style)
                print(f"Using style: {style}")
                break
        
        # Basic rcParams
        mpl.rcParams.update({
            'font.size': 9,
            'axes.titlesize': 11,
            'axes.labelsize': 10,
            'legend.fontsize': 8,
            'figure.titlesize': 14,
            'figure.dpi': 100,
        })
    
    def generate_data(self):
        """Generate synthetic climate data"""
        print("Generating data...")
        
        # Temperature data (1850-2023)
        # Use 'ME' instead of 'M' for month end frequency
        dates = pd.date_range('1850-01', '2023-12', freq='ME')
        years = (dates.year - 1850).values / 100
        
        # Create warming trend
        trend = 1.0 * years  # 1°C per century
        acceleration = np.where(dates.year >= 1970, 0.5 * (dates.year - 1970).values / 100, 0)
        seasonality = 0.3 * np.sin(2 * np.pi * dates.month / 12)
        noise = np.random.normal(0, 0.2, len(dates))
        
        anomaly = trend + acceleration + seasonality + noise
        
        return pd.DataFrame({
            'date': dates,
            'year': dates.year,
            'month': dates.month,
            'anomaly': anomaly,
            'anomaly_smooth': pd.Series(anomaly).rolling(12, center=True).mean()
        })
    
    def create_spiral_chart(self, fig, position):
        """Create a spiral temperature chart with polar projection"""
        data = self.data
        
        # Create polar subplot
        ax = fig.add_subplot(position, projection='polar')
        
        # Convert to polar coordinates for spiral
        years_normalized = (data['year'] + (data['month'] - 1) / 12 - 1850)
        theta = 2 * np.pi * (years_normalized % 1)  # Month angle
        radius = 1 + years_normalized / 50  # Spiral outwards
        
        # Create spiral plot
        sc = ax.scatter(theta, radius, 
                       c=data['anomaly'], 
                       cmap='RdBu_r',
                       s=10,
                       alpha=0.7,
                       vmin=-2, vmax=2)
        
        # Configure polar plot
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rticks([])
        
        # Add month labels
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticks(np.linspace(0, 2*np.pi, 12, endpoint=False))
        ax.set_xticklabels(months, fontsize=7)
        
        # Add decade rings
        for decade in range(1850, 2021, 50):
            if 1850 <= decade <= data['year'].max():
                r = 1 + (decade - 1850) / 50
                ax.plot(np.linspace(0, 2*np.pi, 100), 
                       [r] * 100, 'k-', alpha=0.2, linewidth=0.5)
        
        ax.set_title('Temperature Spiral (1850-2023)', fontweight='bold')
        
        return ax, sc
    
    def create_timeline_chart(self, ax):
        """Create timeline of temperature anomalies"""
        data = self.data
        
        # Group by year
        yearly = data.groupby('year')['anomaly'].mean().reset_index()
        
        # Plot
        ax.fill_between(yearly['year'], 
                       yearly['anomaly'].rolling(10, center=True).min(),
                       yearly['anomaly'].rolling(10, center=True).max(),
                       alpha=0.2, color='gray', label='10-year range')
        
        ax.plot(yearly['year'], 
               yearly['anomaly'].rolling(10, center=True).mean(),
               'r-', linewidth=2, label='10-year mean')
        
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
        
        # Highlight warming periods
        ax.axvspan(1970, 2023, alpha=0.1, color='red', label='Rapid warming')
        ax.axvspan(1850, 1900, alpha=0.1, color='blue', label='Pre-industrial')
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Temperature Anomaly (°C)')
        ax.set_title('Global Temperature Timeline', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    def create_heatmap_chart(self, ax):
        """Create monthly heatmap"""
        data = self.data
        
        # Reshape for heatmap
        pivot = data.pivot_table(values='anomaly', 
                                index='year', 
                                columns='month', 
                                aggfunc='mean')
        
        # Create heatmap
        im = ax.imshow(pivot.T, 
                      aspect='auto',
                      cmap='RdBu_r',
                      vmin=-2, vmax=2,
                      interpolation='nearest',
                      origin='lower')
        
        # Configure axes
        ax.set_xlabel('Year')
        ax.set_ylabel('Month')
        
        # Set x ticks every 20 years
        years = pivot.index
        tick_indices = np.arange(0, len(years), 20)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([str(years[i]) for i in tick_indices], rotation=45)
        
        # Set y ticks for months
        month_names = ['J', 'F', 'M', 'A', 'M', 'J', 
                      'J', 'A', 'S', 'O', 'N', 'D']
        ax.set_yticks(np.arange(12))
        ax.set_yticklabels(month_names)
        
        ax.set_title('Monthly Anomalies Heatmap', fontweight='bold')
        
        return im
    
    def create_dashboard(self):
        """Create complete dashboard"""
        print("Creating dashboard...")
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(15, 10))
        
        # Define grid for subplots
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Create visualizations
        # Spiral chart in top-left (2 rows tall)
        ax1, sc = self.create_spiral_chart(fig, gs[:, 0])
        
        # Timeline chart in top-right
        ax2 = fig.add_subplot(gs[0, 1:])
        self.create_timeline_chart(ax2)
        
        # Heatmap in bottom-right
        ax3 = fig.add_subplot(gs[1, 1:])
        im = self.create_heatmap_chart(ax3)
        
        # Add colorbars
        plt.colorbar(sc, ax=ax1, label='Anomaly (°C)', shrink=0.8)
        plt.colorbar(im, ax=ax3, label='Anomaly (°C)', shrink=0.8)
        
        # Add overall title
        fig.suptitle('Climate Change Dashboard\nGlobal Temperature Analysis (1850-2023)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Add footer
        fig.text(0.5, 0.01, 
                f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")} | Data: Synthetic',
                ha='center', fontsize=9, alpha=0.7)
        
        return fig
    
    def save_dashboard(self, filename='climate_dashboard.png'):
        """Save dashboard to file"""
        os.makedirs('outputs', exist_ok=True)
        
        fig = self.create_dashboard()
        output_path = f'outputs/{filename}'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Dashboard saved to: {output_path}")
        plt.close(fig)

def main():
    """Main function"""
    try:
        print("Starting dashboard generation...")
        
        dashboard = WorkingDashboard()
        dashboard.save_dashboard('working_dashboard.png')
        
        print("\nDashboard generated successfully!")
        print("Check the 'outputs' folder for the PNG file.")
        
        # Optional: Display the plot
        # plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()