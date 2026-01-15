"""
Sankey diagram visualization for emissions flows
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import pandas as pd
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

class SankeyDiagram:
    """Create simplified Sankey diagram for emissions"""
    
    def __init__(self, data: pd.DataFrame, config: Dict[str, Any]):
        self.data = data
        self.config = config
        self.year = config.get('visualizations', {}).get('sankey', {}).get('year', 2020)
        self.setup_colors()
    
    def setup_colors(self):
        """Setup color scheme from config"""
        colors_config = self.config.get('colors', {})
        self.colors = {
            'CO2': colors_config.get('emissions', {}).get('co2', '#FF6B35'),
            'CH4': colors_config.get('emissions', {}).get('ch4', '#C73E3A'),
            'N2O': colors_config.get('emissions', {}).get('n2o', '#6A0572'),
            'Energy': '#3A86FF',
            'Industry': '#8338EC',
            'Agriculture': '#06D6A0',
            'Waste': '#FFBE0B',
            'Land Use Change': '#118AB2'
        }
    
    def aggregate_emissions_data(self) -> Dict[str, Any]:
        """Aggregate emissions data for Sankey diagram"""
        # Filter for selected year
        yearly_data = self.data[self.data['year'] == self.year].copy()
        
        if yearly_data.empty:
            # Use latest available year
            self.year = self.data['year'].max()
            yearly_data = self.data[self.data['year'] == self.year].copy()
        
        # Aggregate by gas
        gas_totals = yearly_data.groupby('gas')['emissions_MT'].sum()
        
        # Aggregate by sector for each gas
        sector_flows = {}
        for gas in ['CO2', 'CH4', 'N2O']:
            gas_data = yearly_data[yearly_data['gas'] == gas]
            sector_totals = gas_data.groupby('sector')['emissions_MT'].sum()
            sector_flows[gas] = sector_totals.to_dict()
        
        # Calculate total emissions
        total_emissions = gas_totals.sum()
        
        # Normalize for plotting
        normalized_gas = gas_totals / total_emissions
        normalized_sectors = {}
        for gas in sector_flows:
            gas_total = sum(sector_flows[gas].values())
            normalized_sectors[gas] = {s: v/gas_total for s, v in sector_flows[gas].items()}
        
        return {
            'year': self.year,
            'gas_totals': gas_totals,
            'sector_flows': sector_flows,
            'normalized_gas': normalized_gas,
            'normalized_sectors': normalized_sectors,
            'total_emissions': total_emissions
        }
    
    def plot(self, ax: plt.Axes):
        """Plot Sankey diagram"""
        
        # Get aggregated data
        aggregated = self.aggregate_emissions_data()
        
        # Setup plot dimensions
        left_margin = 0.1
        right_margin = 0.1
        top_margin = 0.1
        bottom_margin = 0.1
        
        available_width = 0.8
        available_height = 0.8
        
        gas_width = available_width * 0.25
        sector_width = available_width * 0.25
        gap_width = available_width * 0.5  # Space for flows
        
        # Gas sources (left column)
        gas_sources = ['CO2', 'CH4', 'N2O']
        gas_heights = [aggregated['normalized_gas'].get(gas, 0) * available_height 
                      for gas in gas_sources]
        
        # Filter out zero heights
        gas_positions = []
        current_y = bottom_margin
        
        for gas, height in zip(gas_sources, gas_heights):
            if height > 0.01:  # Only plot if significant
                gas_positions.append({
                    'gas': gas,
                    'y_start': current_y,
                    'y_end': current_y + height,
                    'y_center': current_y + height / 2,
                    'height': height
                })
                current_y += height + 0.01
        
        # Sectors (right column)
        all_sectors = ['Energy', 'Industry', 'Agriculture', 'Waste', 'Land Use Change']
        sector_heights = []
        
        for sector in all_sectors:
            sector_total = 0
            for gas in gas_sources:
                sector_total += aggregated['sector_flows'][gas].get(sector, 0)
            sector_heights.append(sector_total / aggregated['total_emissions'] * available_height)
        
        sector_positions = []
        current_y = bottom_margin
        
        for sector, height in zip(all_sectors, sector_heights):
            if height > 0.01:  # Only plot if significant
                sector_positions.append({
                    'sector': sector,
                    'y_start': current_y,
                    'y_end': current_y + height,
                    'y_center': current_y + height / 2,
                    'height': height
                })
                current_y += height + 0.01
        
        # Draw gas boxes (left)
        for gas_info in gas_positions:
            rect = Rectangle((left_margin, gas_info['y_start']), 
                           gas_width, gas_info['height'],
                           facecolor=self.colors[gas_info['gas']], 
                           edgecolor='black', 
                           linewidth=0.8,
                           alpha=0.8)
            ax.add_patch(rect)
            
            # Add gas label
            ax.text(left_margin + gas_width/2, 
                   gas_info['y_center'],
                   gas_info['gas'],
                   ha='center', va='center',
                   fontsize=9, fontweight='bold',
                   color='white')
            
            # Add emissions value
            emissions = aggregated['gas_totals'].get(gas_info['gas'], 0)
            ax.text(left_margin + gas_width/2,
                   gas_info['y_center'] - 0.02,
                   f'{emissions:,.0f} MT',
                   ha='center', va='top',
                   fontsize=7, color='white')
        
        # Draw sector boxes (right)
        x_sector = 1 - right_margin - sector_width
        for sector_info in sector_positions:
            rect = Rectangle((x_sector, sector_info['y_start']), 
                           sector_width, sector_info['height'],
                           facecolor=self.colors[sector_info['sector']], 
                           edgecolor='black', 
                           linewidth=0.8,
                           alpha=0.8)
            ax.add_patch(rect)
            
            # Add sector label
            label = sector_info['sector']
            if len(label) > 8:
                # Break long labels
                words = label.split()
                if len(words) > 1:
                    label = '\n'.join(words)
            
            ax.text(x_sector + sector_width/2,
                   sector_info['y_center'],
                   label,
                   ha='center', va='center',
                   fontsize=8, fontweight='bold',
                   color='white',
                   rotation=0)
        
        # Draw SIMPLIFIED flow connections (rectangles instead of curves)
        for gas_info in gas_positions:
            for sector_info in sector_positions:
                gas = gas_info['gas']
                sector = sector_info['sector']
                
                flow_value = aggregated['sector_flows'][gas].get(sector, 0)
                if flow_value > aggregated['total_emissions'] * 0.01:  # Only significant flows
                    # Calculate flow width
                    flow_width = (flow_value / aggregated['total_emissions']) * available_height
                    
                    # Draw a simple rectangle for the flow
                    flow_x = left_margin + gas_width
                    flow_y = (gas_info['y_center'] + sector_info['y_center']) / 2 - flow_width/2
                    flow_height = flow_width
                    flow_rect_width = x_sector - flow_x
                    
                    rect = Rectangle((flow_x, flow_y),
                                   flow_rect_width, flow_height,
                                   facecolor=self.colors[gas],
                                   alpha=0.3,
                                   edgecolor='none')
                    ax.add_patch(rect)
        
        # Set plot limits and remove axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title
        title = f'Greenhouse Gas Emissions Flow ({self.year})\nSources → Sectors'
        ax.set_title(title, fontsize=11, fontweight='bold', pad=20)
        
        # Add total emissions
        total_text = f'Total: {aggregated["total_emissions"]:,.0f} MT CO₂e'
        ax.text(0.5, 0.95, total_text,
               ha='center', va='top',
               fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3',
                        facecolor='white',
                        edgecolor='black',
                        alpha=0.8))