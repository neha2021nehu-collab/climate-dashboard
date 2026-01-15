#!/usr/bin/env python3
"""
Final Working Climate Dashboard
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== DATA GENERATION ====================

class DataGenerator:
    """Generate climate data"""
    
    def __init__(self):
        pass
    
    def generate_temperature(self):
        """Generate temperature data"""
        dates = pd.date_range('1850-01', '2023-12', freq='MS')
        years = (dates.year - 1850).values / 100
        
        trend = 0.8 * years
        acceleration = np.where(dates.year >= 1970, 0.5 * ((dates.year - 1970).values / 100), 0)
        seasonality = 0.3 * np.sin(2 * np.pi * dates.month / 12)
        noise = np.random.normal(0, 0.15, len(dates))
        
        anomaly = trend + acceleration + seasonality + noise
        
        return pd.DataFrame({
            'date': dates,
            'year': dates.year,
            'month': dates.month,
            'anomaly': anomaly,
            'smoothed': pd.Series(anomaly).rolling(12, center=True).mean()
        })

# ==================== VISUALIZATIONS ====================

def create_spiral_plot(ax, data):
    """Create spiral temperature plot"""
    years = data['year'] + (data['month'] - 1) / 12
    theta = 2 * np.pi * ((years - 1850) % 1)
    radius = 1 + (years - 1850) / 10
    
    sc = ax.scatter(theta, radius,
                   c=data['anomaly'],
                   cmap='RdBu_r',
                   s=5,
                   alpha=0.7,
                   vmin=-2, vmax=2)
    
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_rticks([])
    
    # Month labels
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(np.linspace(0, 2*np.pi, 12, endpoint=False))
    ax.set_xticklabels(months, fontsize=7)
    
    ax.set_title('Temperature Spiral (1850-2023)', fontsize=10, fontweight='bold')
    
    return sc

def create_timeline_plot(ax, data):
    """Create temperature timeline"""
    ax.plot(data['date'], data['anomaly'], 'b-', alpha=0.2, linewidth=0.3)
    ax.plot(data['date'], data['smoothed'], 'r-', linewidth=2, label='12-month average')
    
    ax.axhline(y=0, color='k', alpha=0.3, linewidth=0.5)
    ax.fill_between(data['date'], -0.5, 0.5, alpha=0.1, color='gray')
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Temperature Anomaly (°C)')
    ax.set_title('Global Temperature Timeline', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

def create_heatmap_plot(ax, data):
    """Create heatmap of monthly anomalies"""
    pivot = data.pivot_table(values='anomaly', index='year', columns='month', aggfunc='mean')
    
    im = ax.imshow(pivot.T,
                  aspect='auto',
                  cmap='RdBu_r',
                  vmin=-2, vmax=2,
                  interpolation='nearest',
                  origin='lower')
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Month')
    
    # Year ticks
    years = pivot.index
    tick_indices = np.arange(0, len(years), 20)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([str(years[i]) for i in tick_indices], rotation=45, fontsize=7)
    
    # Month ticks
    month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    ax.set_yticks(np.arange(12))
    ax.set_yticklabels(month_names, fontsize=8)
    
    ax.set_title('Monthly Temperature Anomalies', fontsize=10, fontweight='bold')
    
    return im

def create_decadal_plot(ax, data):
    """Create decadal averages plot"""
    data['decade'] = (data['year'] // 10) * 10
    decades = data.groupby('decade')['anomaly'].agg(['mean', 'std', 'count']).reset_index()
    decades = decades[decades['count'] > 50]
    
    x_pos = np.arange(len(decades))
    colors = plt.cm.viridis(np.linspace(0, 1, len(decades)))
    
    ax.bar(x_pos, decades['mean'], 
           yerr=decades['std'],
           color=colors,
           edgecolor='black',
           alpha=0.8,
           capsize=5)
    
    # Add trend line
    if len(decades) > 1:
        z = np.polyfit(x_pos, decades['mean'], 1)
        p = np.poly1d(z)
        ax.plot(x_pos, p(x_pos), 'r--', linewidth=2, alpha=0.7,
               label=f'Trend: {z[0]:.3f}°C/decade')
    
    ax.set_xlabel('Decade')
    ax.set_ylabel('Average Anomaly (°C)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(int(d)) for d in decades['decade']], rotation=45)
    ax.set_title('Decadal Temperature Averages', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=8)

def create_statistics_panel(ax, data):
    """Create statistics summary"""
    ax.axis('off')
    
    # Calculate statistics
    stats_text = []
    
    # Overall statistics
    overall_mean = data['anomaly'].mean()
    overall_std = data['anomaly'].std()
    stats_text.append(f"Mean Anomaly: {overall_mean:.2f}°C")
    stats_text.append(f"Standard Deviation: {overall_std:.2f}°C")
    
    # Warming since 1900
    pre_1900 = data[data['year'] < 1900]['anomaly'].mean()
    post_2000 = data[data['year'] >= 2000]['anomaly'].mean()
    warming = post_2000 - pre_1900
    stats_text.append(f"Warming since 1900: {warming:.2f}°C")
    
    # Warmest year
    yearly_avg = data.groupby('year')['anomaly'].mean()
    warmest_year = yearly_avg.idxmax()
    warmest_value = yearly_avg.max()
    stats_text.append(f"Warmest Year: {warmest_year} ({warmest_value:.2f}°C)")
    
    # Coldest year
    coldest_year = yearly_avg.idxmin()
    coldest_value = yearly_avg.min()
    stats_text.append(f"Coldest Year: {coldest_year} ({coldest_value:.2f}°C)")
    
    # Add text to panel
    ax.text(0.05, 0.95, 'Climate Statistics', 
           fontsize=12, fontweight='bold',
           transform=ax.transAxes)
    
    y_pos = 0.85
    for stat in stats_text:
        ax.text(0.1, y_pos, stat,
               fontsize=9,
               transform=ax.transAxes)
        y_pos -= 0.08
    
    # Add key findings
    ax.text(0.05, 0.4, 'Key Findings:', 
           fontsize=11, fontweight='bold',
           transform=ax.transAxes)
    
    findings = [
        '• Clear warming trend since 1850',
        '• Acceleration post-1970',
        '• Recent decades warmest on record',
        '• Increased variability over time'
    ]
    
    y_pos = 0.32
    for finding in findings:
        ax.text(0.1, y_pos, finding,
               fontsize=8,
               transform=ax.transAxes)
        y_pos -= 0.06

# ==================== MAIN DASHBOARD ====================

def create_dashboard():
    """Create complete dashboard"""
    print("🌍 Creating Climate Change Dashboard...")
    
    # Generate data
    generator = DataGenerator()
    temp_data = generator.generate_temperature()
    
    # Create figure
    fig = plt.figure(figsize=(20, 15))
    
    # Create GridSpec layout
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel 1: Spiral plot (top-left, 2 rows)
    ax1 = fig.add_subplot(gs[0:2, 0], projection='polar')
    sc = create_spiral_plot(ax1, temp_data)
    
    # Panel 2: Timeline (top-middle)
    ax2 = fig.add_subplot(gs[0, 1])
    create_timeline_plot(ax2, temp_data)
    
    # Panel 3: Heatmap (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    im = create_heatmap_plot(ax3, temp_data)
    
    # Panel 4: Decadal plot (middle-right)
    ax4 = fig.add_subplot(gs[1, 1:])
    create_decadal_plot(ax4, temp_data)
    
    # Panel 5: Statistics (bottom-left)
    ax5 = fig.add_subplot(gs[2, 0])
    create_statistics_panel(ax5, temp_data)
    
    # Panel 6: Placeholder for emissions (bottom-middle)
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.text(0.5, 0.5, 'Emissions Data\n(Placeholder)',
            ha='center', va='center', transform=ax6.transAxes,
            fontsize=11, fontweight='bold')
    ax6.axis('off')
    
    # Panel 7: Placeholder for energy (bottom-right)
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.text(0.5, 0.5, 'Energy Mix\n(Placeholder)',
            ha='center', va='center', transform=ax7.transAxes,
            fontsize=11, fontweight='bold')
    ax7.axis('off')
    
    # Add colorbars
    plt.colorbar(sc, ax=ax1, orientation='vertical', shrink=0.8, label='Anomaly (°C)')
    plt.colorbar(im, ax=ax3, orientation='vertical', shrink=0.8, label='Anomaly (°C)')
    
    # Add panel labels
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
    
    for ax, label in zip(axes, labels):
        ax.text(-0.1, 1.05, label,
               transform=ax.transAxes,
               fontsize=14, fontweight='bold',
               va='bottom', ha='right',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black'))
    
    # Add main title
    fig.suptitle('Climate Change Dashboard\nGlobal Temperature Analysis 1850-2023',
                fontsize=18, fontweight='bold', y=0.98)
    
    # Add footer
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M')
    fig.text(0.5, 0.01, 
            f'Generated: {current_time} | Data: Synthetic | Visualization: Matplotlib',
            ha='center', fontsize=9, alpha=0.7)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    return fig

def main():
    """Main function"""
    print("=" * 60)
    print("CLIMATE CHANGE DASHBOARD GENERATOR")
    print("=" * 60)
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    try:
        # Create dashboard
        fig = create_dashboard()
        
        # Save dashboard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'outputs/climate_dashboard_{timestamp}.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        
        print(f"\n✅ Dashboard saved to: {output_path}")
        print(f"📏 Figure size: 20x15 inches")
        print(f"🎨 Visualizations: 7 panels")
        
        # Ask if user wants to display
        show = input("\nDisplay dashboard? (y/n): ").lower()
        if show == 'y':
            plt.show()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()