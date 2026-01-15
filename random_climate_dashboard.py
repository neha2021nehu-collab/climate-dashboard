#!/usr/bin/env python3
"""
TRULY RANDOM Climate Dashboard
- Different data every run
- Real stochastic patterns
- More realistic climate variability
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import matplotlib as mpl
import warnings
warnings.filterwarnings('ignore')

class RandomClimateDashboard:
    """Dashboard with truly random climate data"""
    
    def __init__(self, seed=None):
        # Use time-based seed for true randomness
        if seed is None:
            seed = int(datetime.now().timestamp() * 1000) % 2**32
        self.seed = seed
        np.random.seed(seed)
        
        print(f"Using random seed: {seed}")
        self.setup_style()
        self.data = self.generate_random_data()
    
    def setup_style(self):
        """Setup matplotlib"""
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('default')
        
        mpl.rcParams.update({
            'font.size': 9,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'legend.fontsize': 8,
            'figure.titlesize': 16,
            'figure.dpi': 100,
        })
    
    def generate_random_data(self):
        """Generate TRULY random climate data with realistic patterns"""
        print("Generating RANDOM climate data...")
        
        # Generate dates
        dates = pd.date_range('1850-01', '2023-12', freq='MS')
        
        # 1. BASE TREND - Random warming rate between 0.6 and 1.2°C per century
        warming_rate = np.random.uniform(0.6, 1.2)
        base_trend = warming_rate * ((dates.year - 1850).values / 100)
        
        # 2. ACCELERATION - Random start year and rate
        acceleration_start = np.random.randint(1970, 1990)
        acceleration_rate = np.random.uniform(0.3, 0.8)
        acceleration = np.where(
            dates.year >= acceleration_start,
            acceleration_rate * ((dates.year - acceleration_start).values / 100),
            0
        )
        
        # 3. SEASONALITY - Random amplitude and phase
        season_amplitude = np.random.uniform(0.15, 0.35)
        season_phase = np.random.uniform(0, 2*np.pi)
        seasonality = season_amplitude * np.sin(2 * np.pi * dates.month / 12 + season_phase)
        
        # 4. MULTI-DECADAL OSCILLATIONS - Random period and amplitude
        # AMO-like oscillation (~60 years)
        amo_period = np.random.uniform(50, 70)
        amo_amplitude = np.random.uniform(0.05, 0.15)
        amo_phase = np.random.uniform(0, 2*np.pi)
        amo = amo_amplitude * np.sin(2 * np.pi * (dates.year - 1850).values / amo_period + amo_phase)
        
        # PDO-like oscillation (~20 years)
        pdo_period = np.random.uniform(15, 25)
        pdo_amplitude = np.random.uniform(0.03, 0.10)
        pdo_phase = np.random.uniform(0, 2*np.pi)
        pdo = pdo_amplitude * np.sin(2 * np.pi * (dates.year - 1850).values / pdo_period + pdo_phase)
        
        # 5. VOLCANIC ERUPTIONS - Random timing and magnitude
        volcanic_cooling = np.zeros(len(dates))
        n_eruptions = np.random.randint(8, 15)
        eruption_years = np.sort(np.random.choice(range(1850, 2020), n_eruptions, replace=False))
        
        for year in eruption_years:
            if year in dates.year.values:
                # Random magnitude between -0.3 and -1.0°C
                magnitude = -np.random.uniform(0.3, 1.0)
                duration = np.random.randint(12, 36)  # 1-3 years
                
                eruption_date = pd.Timestamp(f'{year}-06-01')
                idx_start = max(0, (eruption_date - dates[0]).days // 30)
                idx_end = min(len(dates), idx_start + duration)
                
                # Exponential decay cooling
                for i in range(idx_start, idx_end):
                    months_since = (i - idx_start)
                    decay = np.exp(-months_since / 6)  # 6-month decay constant
                    volcanic_cooling[i] += magnitude * decay
        
        # 6. RANDOM NOISE - Different variance over time (more recent = more noise)
        noise_var = 0.05 + 0.1 * ((dates.year - 1850).values / (2023 - 1850))
        noise = np.random.normal(0, noise_var, len(dates))
        
        # 7. CLIMATE SHIFTS - Random regime shifts
        n_shifts = np.random.randint(2, 5)
        shift_years = np.sort(np.random.choice(range(1860, 2010), n_shifts, replace=False))
        shift_magnitudes = np.random.uniform(-0.2, 0.3, n_shifts)
        
        shifts = np.zeros(len(dates))
        current_shift = 0
        shift_idx = 0
        
        for i, date in enumerate(dates):
            if shift_idx < len(shift_years) and date.year >= shift_years[shift_idx]:
                current_shift += shift_magnitudes[shift_idx]
                shift_idx += 1
            shifts[i] = current_shift
        
        # COMBINE ALL COMPONENTS
        anomaly = (base_trend + acceleration + seasonality + 
                  amo + pdo + volcanic_cooling + shifts + noise)
        
        # Add extreme events (random heatwaves/cold spells)
        extreme_events = np.zeros(len(dates))
        n_extremes = np.random.randint(20, 40)
        
        for _ in range(n_extremes):
            idx = np.random.randint(0, len(dates))
            magnitude = np.random.uniform(-1.5, 2.0)  # Can be hot or cold
            duration = np.random.randint(1, 4)  # 1-3 months
            
            for j in range(max(0, idx), min(len(dates), idx + duration)):
                extreme_events[j] = magnitude * (1 - (j - idx) / duration)
        
        anomaly += extreme_events
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'year': dates.year,
            'month': dates.month,
            'anomaly': anomaly,
            'smoothed': pd.Series(anomaly).rolling(12, center=True).mean(),
            'base_trend': base_trend,
            'seasonality': seasonality,
            'amo_cycle': amo,
            'volcanic': volcanic_cooling
        })
        
        # Calculate some statistics for display
        self.warming_rate = warming_rate
        self.acceleration_start = acceleration_start
        self.acceleration_rate = acceleration_rate
        self.n_eruptions = n_eruptions
        
        return df
    
    def create_time_series(self, ax):
        """Create detailed time series with components"""
        data = self.data
        
        # Plot components
        ax.fill_between(data['date'], data['base_trend'], alpha=0.3, 
                       color='orange', label='Base warming trend')
        ax.plot(data['date'], data['smoothed'], 'r-', linewidth=2.5, 
               label='12-month average')
        ax.plot(data['date'], data['anomaly'], 'b-', alpha=0.15, linewidth=0.3,
               label='Monthly data')
        
        # Highlight volcanic eruptions
        volcanic_idx = np.abs(data['volcanic']) > 0.1
        if volcanic_idx.any():
            ax.scatter(data['date'][volcanic_idx], 
                      data['anomaly'][volcanic_idx],
                      c='purple', s=10, alpha=0.6, 
                      label='Volcanic cooling', zorder=5)
        
        # Add zero line
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
        
        # Labels and formatting
        ax.set_xlabel('Year', fontweight='bold')
        ax.set_ylabel('Temperature Anomaly (°C)', fontweight='bold')
        ax.set_title('Global Temperature with Random Components', 
                    fontsize=13, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=7, framealpha=0.9)
        
        # Set x-axis ticks
        ax.xaxis.set_major_locator(plt.MultipleLocator(20))
    
    def create_component_breakdown(self, ax):
        """Show breakdown of different components"""
        data = self.data.sample(n=500)  # Sample for clarity
        
        components = ['base_trend', 'seasonality', 'amo_cycle', 'volcanic']
        colors = ['orange', 'green', 'blue', 'purple']
        
        bottom = np.zeros(len(data))
        for comp, color in zip(components, colors):
            values = data[comp].values
            ax.bar(range(len(data)), values, bottom=bottom, 
                  color=color, alpha=0.7, width=1, label=comp)
            bottom += values
        
        # Add total anomaly line
        ax.plot(range(len(data)), data['anomaly'].values, 
               'r-', linewidth=2, alpha=0.8, label='Total anomaly')
        
        ax.set_xlabel('Sample Points', fontweight='bold')
        ax.set_ylabel('Contribution (°C)', fontweight='bold')
        ax.set_title('Breakdown of Temperature Components', 
                    fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')
    
    def create_distribution_plot(self, ax):
        """Show distribution of anomalies over time"""
        data = self.data
        
        # Create decades
        data['decade'] = (data['year'] // 10) * 10
        decades = sorted(data['decade'].unique())
        
        # Prepare data for violin plot
        plot_data = []
        decade_labels = []
        
        for decade in decades:
            if decade >= 1850 and data[data['decade'] == decade].shape[0] > 50:
                plot_data.append(data[data['decade'] == decade]['anomaly'].values)
                decade_labels.append(str(decade))
        
        # Create violin plot
        parts = ax.violinplot(plot_data, showmeans=True, showmedians=True)
        
        # Color violins by temperature
        for i, pc in enumerate(parts['bodies']):
            # Color based on mean temperature
            mean_temp = np.mean(plot_data[i])
            color = plt.cm.RdYlBu_r((mean_temp + 2) / 4)  # Normalize to -2 to +2°C range
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
        
        # Customize
        ax.set_xlabel('Decade', fontweight='bold')
        ax.set_ylabel('Temperature Anomaly (°C)', fontweight='bold')
        ax.set_xticks(range(1, len(decade_labels) + 1))
        ax.set_xticklabels(decade_labels, rotation=45)
        ax.set_title('Distribution of Temperatures by Decade', 
                    fontsize=13, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    def create_statistics_panel(self, ax):
        """Create statistics panel with random parameters"""
        data = self.data
        
        # Calculate statistics
        warming_1850_1900 = data[data['year'] < 1900]['anomaly'].mean()
        warming_2000_2023 = data[data['year'] >= 2000]['anomaly'].mean()
        total_warming = warming_2000_2023 - warming_1850_1900
        
        warmest_year = data.groupby('year')['anomaly'].mean().idxmax()
        warmest_value = data.groupby('year')['anomaly'].mean().max()
        
        # Random parameters from generation
        stats = {
            'Random Seed': f"{self.seed}",
            'Warming Rate': f"{self.warming_rate:.2f}°C/century",
            'Acceleration Start': f"{self.acceleration_start}",
            'Acceleration Rate': f"{self.acceleration_rate:.2f}°C/century",
            'Volcanic Eruptions': f"{self.n_eruptions} events",
            'Total Warming (since 1850-1900)': f"{total_warming:.2f}°C",
            'Warmest Year': f"{warmest_year}: {warmest_value:.2f}°C",
            'Recent Trend': f"{np.polyfit(data[data['year'] >= 2000]['year'].unique(), data[data['year'] >= 2000].groupby('year')['anomaly'].mean().values, 1)[0]*100:.2f}°C/century",
        }
        
        # Clear axis
        ax.clear()
        ax.axis('off')
        
        # Add title
        ax.text(0.5, 0.95, 'RANDOM CLIMATE SIMULATION STATISTICS', 
               fontsize=14, fontweight='bold', 
               ha='center', va='top', transform=ax.transAxes, color='darkred')
        
        # Add statistics
        y_pos = 0.85
        for key, value in stats.items():
            ax.text(0.05, y_pos, f'{key}:', 
                   fontsize=10, fontweight='bold',
                   ha='left', va='top', transform=ax.transAxes)
            ax.text(0.55, y_pos, value, 
                   fontsize=10, fontweight='bold',
                   ha='left', va='top', transform=ax.transAxes,
                   color='darkblue')
            y_pos -= 0.08
        
        # Add note about randomness
        ax.text(0.05, 0.25, 'NOTE: This is a RANDOM climate simulation.', 
               fontsize=11, fontweight='bold', style='italic',
               ha='left', va='top', transform=ax.transAxes,
               color='darkred')
        
        ax.text(0.05, 0.18, 'Each run produces different:', 
               fontsize=9, ha='left', va='top', transform=ax.transAxes)
        
        random_elements = [
            '• Warming rate and acceleration',
            '• Volcanic eruption timing/magnitude',
            '• Climate oscillation patterns',
            '• Extreme event frequency',
            '• Noise characteristics'
        ]
        
        y_pos = 0.10
        for element in random_elements:
            ax.text(0.1, y_pos, element, 
                   fontsize=8,
                   ha='left', va='top', transform=ax.transAxes)
            y_pos -= 0.05
    
    def create_dashboard(self):
        """Create complete dashboard with random data"""
        print("Creating RANDOM climate dashboard...")
        
        # Create figure with 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        ax1, ax2, ax3, ax4 = axes.flat
        
        # Create visualizations
        self.create_time_series(ax1)
        self.create_component_breakdown(ax2)
        self.create_distribution_plot(ax3)
        self.create_statistics_panel(ax4)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Add main title with seed info
        fig.suptitle(f'RANDOM CLIMATE CHANGE SIMULATION (Seed: {self.seed})\nGlobal Temperature Analysis 1850-2023', 
                    fontsize=16, fontweight='bold', y=0.99)
        
        # Add footer
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.5, 0.01, 
                f'Generated: {current_time} | Data: RANDOM SIMULATION | Each run is different!',
                ha='center', fontsize=9, alpha=0.7, style='italic')
        
        return fig
    
    def save_dashboard(self, filename=None):
        """Save dashboard to file with timestamp"""
        os.makedirs('outputs', exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'random_climate_{timestamp}.png'
        
        fig = self.create_dashboard()
        output_path = f'outputs/{filename}'
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\nRANDOM dashboard saved to: {output_path}")
        plt.close(fig)
        return output_path

def main():
    """Main function"""
    print("=" * 70)
    print("RANDOM CLIMATE CHANGE DASHBOARD GENERATOR")
    print("=" * 70)
    print("\nEach run produces a UNIQUE climate simulation with:")
    print("- Different warming rates")
    print("- Random volcanic eruptions")
    print("- Varying climate oscillations")
    print("- Unique extreme events")
    print("- Stochastic noise patterns")
    
    # Ask for seed or use random
    seed_input = input("\nEnter seed number (or press Enter for random): ").strip()
    if seed_input:
        try:
            seed = int(seed_input)
        except:
            print("Invalid seed, using random...")
            seed = None
    else:
        seed = None
    
    try:
        dashboard = RandomClimateDashboard(seed=seed)
        output_path = dashboard.save_dashboard()
        
        print("\n" + "=" * 70)
        print("RANDOM SIMULATION COMPLETE!")
        print("=" * 70)
        print(f"\nDashboard saved to: {output_path}")
        print(f"\nKey random parameters used:")
        print(f"- Warming rate: {dashboard.warming_rate:.2f}°C/century")
        print(f"- Acceleration started: {dashboard.acceleration_start}")
        print(f"- Volcanic eruptions: {dashboard.n_eruptions}")
        
        # Run again option
        rerun = input("\nGenerate another random simulation? (y/n): ").lower()
        if rerun == 'y':
            # Create new random seed
            new_seed = int(datetime.now().timestamp() * 1000) % 2**32
            print(f"\nUsing new seed: {new_seed}")
            dashboard2 = RandomClimateDashboard(seed=new_seed)
            output_path2 = dashboard2.save_dashboard(f'random_climate_{new_seed}.png')
            print(f"Second dashboard saved to: {output_path2}")
        
        # Display option
        display = input("\nDo you want to display the dashboard? (y/n): ").lower()
        if display == 'y':
            plt.show()
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()