"""
Chord diagram visualization for energy transitions
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

class ChordDiagram:
    """Create chord diagram for energy source transitions"""
    
    def __init__(self, data: pd.DataFrame, config: Dict[str, Any]):
        self.data = data
        self.config = config
        self.setup_colors()
    
    def setup_colors(self):
        """Setup colors for energy sources"""
        colors_config = self.config.get('colors', {})
        self.colors = {
            'Coal': '#333333',
            'Oil': '#8B0000',
            'Gas': '#FF8C00',
            'Nuclear': '#FFD700',
            'Hydro': '#1E90FF',
            'Wind': '#32CD32',
            'Solar': '#FF4500',
            'Other Renewables': '#9370DB'
        }
    
    def plot(self, ax: plt.Axes):
        """Plot simplified chord diagram"""
        
        # Get unique energy sources
        if 'source' in self.data.columns:
            sources = self.data['source'].unique()
        else:
            sources = list(self.colors.keys())
        
        # Create a circular layout
        n_sources = len(sources)
        angles = np.linspace(0, 2 * np.pi, n_sources, endpoint=False)
        
        # Create circle
        circle = plt.Circle((0, 0), 1, fill=False, edgecolor='gray', linewidth=1)
        ax.add_patch(circle)
        
        # Add source nodes
        for i, (source, angle) in enumerate(zip(sources, angles)):
            # Calculate position
            x = np.cos(angle) * 0.9
            y = np.sin(angle) * 0.9
            
            # Add colored circle for source
            color = self.colors.get(source, '#999999')
            node = plt.Circle((x, y), 0.08, color=color, alpha=0.8, edgecolor='black')
            ax.add_patch(node)
            
            # Add label
            label_angle = angle * 180 / np.pi
            if label_angle > 90 and label_angle < 270:
                ha = 'right'
                label_angle -= 180
            else:
                ha = 'left'
            
            label_x = np.cos(angle) * 1.05
            label_y = np.sin(angle) * 1.05
            
            ax.text(label_x, label_y, source,
                   ha=ha, va='center',
                   fontsize=8, fontweight='bold',
                   rotation=label_angle,
                   rotation_mode='anchor')
        
        # Add chords (simplified connections)
        if n_sources > 1:
            # Create some sample connections (in real use, this would be based on data)
            connections = []
            for i in range(n_sources):
                for j in range(i + 1, n_sources):
                    # Random connection strength
                    strength = np.random.random() * 0.3
                    if strength > 0.1:
                        connections.append((i, j, strength))
            
            # Draw connections
            for i, j, strength in connections:
                angle_i = angles[i]
                angle_j = angles[j]
                
                # Create Bezier curve for chord
                control_angle = (angle_i + angle_j) / 2
                control_radius = 0.5  # How curved the chord is
                
                # Control point
                cx = np.cos(control_angle) * control_radius
                cy = np.sin(control_angle) * control_radius
                
                # Start and end points
                x1 = np.cos(angle_i) * 0.82
                y1 = np.sin(angle_i) * 0.82
                x2 = np.cos(angle_j) * 0.82
                y2 = np.sin(angle_j) * 0.82
                
                # Draw curve
                from matplotlib.path import Path
                from matplotlib.patches import PathPatch
                
                vertices = [(x1, y1),
                           (cx, cy),
                           (x2, y2)]
                
                codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                
                path = Path(vertices, codes)
                patch = PathPatch(path, 
                                facecolor='none',
                                edgecolor='gray',
                                linewidth=strength * 10,
                                alpha=0.5)
                ax.add_patch(patch)
        
        # Configure plot
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title
        ax.set_title('Energy Source Transitions\nChord Diagram', 
                    fontsize=12, fontweight='bold', pad=20)
        
        # Add legend
        from matplotlib.patches import Patch
        
        if len(sources) <= 8:
            legend_elements = []
            for source in sources[:8]:  # Limit to 8 for readability
                color = self.colors.get(source, '#999999')
                legend_elements.append(Patch(facecolor=color, label=source, alpha=0.8))
            
            ax.legend(handles=legend_elements, 
                     loc='upper left', 
                     fontsize=7,
                     bbox_to_anchor=(1.05, 1),
                     borderaxespad=0.)