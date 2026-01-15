#!/usr/bin/env python3
"""
Simple Climate Dashboard - No Polar Plot
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import os
import matplotlib as mpl

class SimpleClimateDashboard:
    """Simple dashboard without polar plots"""
    
    def __init__(self):
        self.setup_style()
        self.data = self.generate_data()
    
    def setup_style(self):
        """Setup matplotlib"""
        plt.style.use('default')
        
        mpl.rcParams.update({
            'font.size': 9,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'legend.fontsize': 8,
            'figure.titlesize': 14,
            'figure.dpi': 100,
        })
    
    def generate_data(self):
        """Generate synthetic climate data"""
        print("Generating climate data...")
        
        # Generate dates
        dates = pd.date_range('1850-01', '2023-12', freq='ME')
        
        # Create realistic temperature trend
        years = (dates.year - 1850).values
        
        # Base warming trend
        trend = 0.8 * (years / 100)
        
        # Accelerated warming after 1970
        acceleration = np.where(dates.year >= 1970, 
                               0.6 * ((dates.year - 1970).values / 100), 
                               0)
        
        # Seasonality
        seasonality = 0.25 * np.sin(2 * np.pi * dates.month / 12 - np.pi/6)
        
        # Natural variability
        variability = 0.15 * np.sin(2 * np.pi * years / 60)  # ~60-year cycle
        
        # Random noise
        noise = np.random.normal(0, 0.1, len(dates))
        
        # Combine all components
        anomaly = trend + acceleration + seasonality + variability + noise
        
        return pd.DataFrame({
            'date': dates,
            'year': dates.year,
            'month': dates.month,
            'anomaly': anomaly,
            'smoothed': pd.Series(anomaly).rolling(12, center=True).mean()
        })
    
    def create_time_series(self, ax):
        """Create time series plot"""
        data = self.data
        
        # Plot individual months (transparent)
        ax.plot(data['date'], data['anomaly'], 
               'b-', alpha=0.2, linewidth=0.5, label='Monthly data')
        
        # Plot smoothed trend
        ax.plot(data['date'], data['smoothed'], 
               'r-', linewidth=2.5, label='12-month average')
        
        # Add zero line and background
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
        ax.fill_between(data['date'], -0.5, 0.5, alpha=0.1, color='gray')
        
        # Highlight significant periods
        ax.axvspan(1850, 1900, alpha=0.1, color='blue', label='Pre-industrial')
        ax.axvspan(1970, 2023, alpha=0.1, color='red', label='Accelerated warming')
        
        # Labels and formatting
        ax.set_xlabel('Year', fontweight='bold')
        ax.set_ylabel('Temperature Anomaly (°C)', fontweight='bold')
        ax.set_title('Global Temperature Timeline (1850-2023)', 
                    fontsize=13, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', framealpha=0.9)
        
        # Improve x-axis labels
        ax.xaxis.set_major_locator(plt.MultipleLocator(20))
    
    def create_heatmap(self, ax):
        """Create monthly heatmap"""
        data = self.data
        
        # Pivot to year x month format
        pivot = data.pivot_table(values='anomaly', 
                                index='year', 
                                columns='month', 
                                aggfunc='mean')
        
        # Create heatmap
        im = ax.imshow(pivot.T, 
                      aspect='auto',
                      cmap='RdBu_r',
                      vmin=-1.5, vmax=1.5,
                      interpolation='nearest',
                      origin='lower')
        
        # Configure axes
        ax.set_xlabel('Year', fontweight='bold')
        ax.set_ylabel('Month', fontweight='bold')
        
        # Year labels (every 20 years)
        years = pivot.index
        tick_positions = np.arange(0, len(years), 20)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(years[i]) for i in tick_positions], 
                          rotation=45, fontsize=8)
        
        # Month labels
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_yticks(np.arange(12))
        ax.set_yticklabels(month_names, fontsize=9)
        
        ax.set_title('Monthly Temperature Anomalies Heatmap', 
                    fontsize=13, fontweight='bold', pad=15)
        
        return im
    
    def create_decadal_trends(self, ax):
        """Create decadal trends bar chart"""
        data = self.data
        
        # Calculate decadal averages
        data['decade'] = (data['year'] // 10) * 10
        decades = data.groupby('decade')['anomaly'].agg(['mean', 'std', 'count']).reset_index()
        
        # Filter to complete decades
        decades = decades[decades['count'] >= 100]  # At least 100 months
        
        # Create bar chart with error bars
        x_pos = np.arange(len(decades))
        bars = ax.bar(x_pos, decades['mean'], 
                     yerr=decades['std'],
                     capsize=5,
                     color=plt.cm.viridis(np.linspace(0, 1, len(decades))),
                     edgecolor='black',
                     linewidth=0.5,
                     alpha=0.8,
                     error_kw=dict(elinewidth=1, ecolor='black'))
        
        # Add value labels on top of bars
        for i, (_, row) in enumerate(decades.iterrows()):
            ax.text(i, row['mean'] + 0.05, 
                   f'{row["mean"]:.2f}°C', 
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Add trend line
        if len(decades) > 1:
            z = np.polyfit(x_pos, decades['mean'], 1)
            p = np.poly1d(z)
            ax.plot(x_pos, p(x_pos), 'r--', linewidth=2, alpha=0.8,
                   label=f'Trend: {z[0]:.3f}°C/decade')
        
        # Labels and formatting
        ax.set_xlabel('Decade', fontweight='bold')
        ax.set_ylabel('Average Anomaly (°C)', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(int(d)) for d in decades['decade']], rotation=45)
        ax.set_title('Decadal Temperature Averages', 
                    fontsize=13, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=1)
        ax.legend(loc='upper left', framealpha=0.9)
    
    def create_statistics_panel(self, ax):
        """Create statistics summary panel"""
        data = self.data
        
        # Calculate statistics
        stats = {
            'Mean anomaly': f"{data['anomaly'].mean():.2f}°C",
            'Warming since 1900': f"{(data[data['year'] >= 2000]['anomaly'].mean() - data[data['year'] < 1900]['anomaly'].mean()):.2f}°C",
            'Warmest year': f"{data.groupby('year')['anomaly'].mean().idxmax()}: {data.groupby('year')['anomaly'].mean().max():.2f}°C",
            'Coldest year': f"{data.groupby('year')['anomaly'].mean().idxmin()}: {data.groupby('year')['anomaly'].mean().min():.2f}°C",
            'Trend (1850-2023)': f"{np.polyfit(data['year'].unique(), data.groupby('year')['anomaly'].mean().values, 1)[0]*100:.2f}°C/century",
            'Recent trend (2000-2023)': f"{np.polyfit(data[data['year'] >= 2000]['year'].unique(), data[data['year'] >= 2000].groupby('year')['anomaly'].mean().values, 1)[0]*100:.2f}°C/century",
        }
        
        # Clear axis
        ax.clear()
        ax.axis('off')
        
        # Add title
        ax.text(0.5, 0.95, 'Climate Statistics Summary', 
               fontsize=14, fontweight='bold', 
               ha='center', va='top', transform=ax.transAxes)
        
        # Add statistics
        y_pos = 0.85
        for key, value in stats.items():
            ax.text(0.05, y_pos, f'{key}:', 
                   fontsize=10, fontweight='bold',
                   ha='left', va='top', transform=ax.transAxes)
            ax.text(0.6, y_pos, value, 
                   fontsize=10, 
                   ha='left', va='top', transform=ax.transAxes)
            y_pos -= 0.1
        
        # Add key findings
        ax.text(0.05, 0.25, 'Key Findings:', 
               fontsize=11, fontweight='bold',
               ha='left', va='top', transform=ax.transAxes)
        
        findings = [
            '• Warming accelerated post-1970',
            '• Recent decades are warmest on record',
            '• Monthly variability has increased',
            '• All decades since 1980s above average'
        ]
        
        y_pos = 0.15
        for finding in findings:
            ax.text(0.1, y_pos, finding, 
                   fontsize=9,
                   ha='left', va='top', transform=ax.transAxes)
            y_pos -= 0.07
    
    def create_dashboard(self):
        """Create complete dashboard"""
        print("Creating dashboard visualization...")
        
        # Create figure with 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        ax1, ax2, ax3, ax4 = axes.flat
        
        # Create visualizations
        self.create_time_series(ax1)
        im = self.create_heatmap(ax2)
        self.create_decadal_trends(ax3)
        self.create_statistics_panel(ax4)
        
        # Add colorbar for heatmap
        cbar = plt.colorbar(im, ax=ax2, orientation='vertical', shrink=0.8)
        cbar.set_label('Temperature Anomaly (°C)', fontweight='bold')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Add main title
        fig.suptitle('Climate Change Dashboard: Global Temperature Analysis 1850-2023', 
                    fontsize=16, fontweight='bold', y=0.99)
        
        # Add footer
        fig.text(0.5, 0.01, 
                f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")} | Data: Synthetic | Visualization: Matplotlib',
                ha='center', fontsize=9, alpha=0.7)
        
        return fig
    
    def save_dashboard(self, filename='climate_dashboard_simple.png'):
        """Save dashboard to file"""
        os.makedirs('outputs', exist_ok=True)
        
        fig = self.create_dashboard()
        output_path = f'outputs/{filename}'
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\nDashboard saved to: {output_path}")
        plt.close(fig)
        return output_path

def main():
    """Main function"""
    try:
        print("=" * 60)
        print("CLIMATE CHANGE DASHBOARD GENERATOR")
        print("=" * 60)
        
        dashboard = SimpleClimateDashboard()
        output_path = dashboard.save_dashboard('climate_dashboard.png')
        
        print("\n" + "=" * 60)
        print("GENERATION COMPLETE!")
        print("=" * 60)
        print(f"\nDashboard saved to: {output_path}")
        print("\nThe dashboard includes:")
        print("1. Temperature timeline with trends")
        print("2. Monthly anomalies heatmap")
        print("3. Decadal temperature averages")
        print("4. Statistical summary")
        
        # Optional: Ask if user wants to display
        display = input("\nDo you want to display the dashboard? (y/n): ").lower()
        if display == 'y':
            plt.show()
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()