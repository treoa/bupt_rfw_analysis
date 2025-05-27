import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

from PIL import Image
from rich.console import Console
from rich.logging import RichHandler
from plotly.subplots import make_subplots
from typing import List, Tuple, Any, Dict, Union


console = Console()

CUSTOM_PALETTE = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']


class AdvancedDatasetAnalyzer:
    """
    State-of-the-art visualization suite for comprehensive dataset analysis.
    Features modern aesthetics, interactive plots, and publication-ready graphics.
    """
    
    def __init__(self, console: Console, race_colors: Dict[str, str]):
        """Initialize analyzer with premium styling configurations."""
        self.console = console
        self.RACE_COLORS = race_colors
        self.setup_aesthetic_styling()
        
    def setup_aesthetic_styling(self):
        """Configure beautiful styling for all visualizations."""
        # Set modern seaborn style
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.2)
        
        # Custom matplotlib configuration for publication quality
        plt.rcParams.update({
            'figure.figsize': (14, 10),
            'font.family': 'sans-serif',
            'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'axes.titleweight': 'bold',
            'axes.labelweight': 'bold',
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 12,
            'legend.title_fontsize': 13,
            'figure.titlesize': 18,
            'figure.titleweight': 'bold',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.8,
            'lines.linewidth': 2.5,
            'patch.linewidth': 0.5,
            'xtick.major.width': 1.5,
            'ytick.major.width': 1.5,
            'axes.linewidth': 1.5
        })
        
        # Set custom color palette
        sns.set_palette(CUSTOM_PALETTE)
    
    def calculate_comprehensive_statistics(self, data: List[float]) -> Dict[str, float]:
        """Calculate extensive statistical measures for dataset analysis."""
        if not data:
            return {stat: 0 for stat in ['mean', 'median', 'std', 'variance', 'min', 'max', 
                                       'q25', 'q75', 'iqr', 'skewness', 'kurtosis', 'cv']}
        
        data_array = np.array(data)
        return {
            'mean': np.mean(data_array),
            'median': np.median(data_array),
            'std': np.std(data_array),
            'variance': np.var(data_array),
            'min': np.min(data_array),
            'max': np.max(data_array),
            'q25': np.percentile(data_array, 25),
            'q75': np.percentile(data_array, 75),
            'iqr': np.percentile(data_array, 75) - np.percentile(data_array, 25),
            'skewness': float(pd.Series(data).skew()),
            'kurtosis': float(pd.Series(data).kurtosis()),
            'cv': np.std(data_array) / np.mean(data_array) if np.mean(data_array) != 0 else 0
        }
    
    def create_images_per_identity_masterpiece(self, identities_df: pd.DataFrame, output_dir: str):
        """Create stunning visualizations for images per identity distribution."""
        self.console.print("[bold magenta]🎨 Creating images per identity masterpiece...[/bold magenta]")
        
        # Enhanced statistical analysis
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Elegant Box Plot with enhanced styling
        ax1 = fig.add_subplot(gs[0, 0])
        box_plot = sns.boxplot(
            data=identities_df, x='race', y='num_images', 
            palette=self.RACE_COLORS, ax=ax1,
            linewidth=2.5, fliersize=8, notch=True
        )
        ax1.set_title('📦 Distribution Box Plot\nImages per Identity by Race', 
                     fontweight='bold', pad=20)
        ax1.set_xlabel('Race', fontweight='bold')
        ax1.set_ylabel('Number of Images', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add mean markers
        for i, race in enumerate(identities_df['race'].unique()):
            race_data = identities_df[identities_df['race'] == race]['num_images']
            mean_val = race_data.mean()
            ax1.scatter(i, mean_val, color='red', s=100, marker='D', 
                       zorder=10, label='Mean' if i == 0 else "")
        ax1.legend()
        
        # 2. Sophisticated Violin Plot with inner quartiles
        ax2 = fig.add_subplot(gs[0, 1])
        violin_plot = sns.violinplot(
            data=identities_df, x='race', y='num_images',
            palette=self.RACE_COLORS, ax=ax2, inner='quart',
            linewidth=2
        )
        ax2.set_title('🎻 Violin Plot with Quartiles\nProbability Density Distribution', 
                     fontweight='bold', pad=20)
        ax2.set_xlabel('Race', fontweight='bold')
        ax2.set_ylabel('Number of Images', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Modern Swarm Plot with size variation
        ax3 = fig.add_subplot(gs[0, 2])
        sns.swarmplot(
            data=identities_df, x='race', y='num_images',
            palette=self.RACE_COLORS, ax=ax3, size=6, alpha=0.8
        )
        ax3.set_title('🐝 Swarm Plot\nIndividual Data Points Distribution', 
                     fontweight='bold', pad=20)
        ax3.set_xlabel('Race', fontweight='bold')
        ax3.set_ylabel('Number of Images', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Gradient Histogram with KDE overlay
        ax4 = fig.add_subplot(gs[1, :2])
        for i, race in enumerate(identities_df['race'].unique()):
            race_data = identities_df[identities_df['race'] == race]['num_images']
            ax4.hist(race_data, alpha=0.6, label=race, bins=25, 
                    color=self.RACE_COLORS[race], density=True, edgecolor='white')
            sns.kdeplot(data=race_data, ax=ax4, color=self.RACE_COLORS[race], 
                       linewidth=3, alpha=0.8)
        
        ax4.set_title('📊 Enhanced Histogram with KDE\nDistribution Density Analysis', 
                     fontweight='bold', pad=20)
        ax4.set_xlabel('Number of Images per Identity', fontweight='bold')
        ax4.set_ylabel('Density', fontweight='bold')
        ax4.legend(title='Race', title_fontsize=13, fontsize=11)
        ax4.grid(True, alpha=0.3)
        
        # 5. Ridgeline Plot (Joy Plot) simulation
        ax5 = fig.add_subplot(gs[1, 2])
        for i, race in enumerate(identities_df['race'].unique()):
            race_data = identities_df[identities_df['race'] == race]['num_images']
            density = sns.kdeplot(data=race_data, ax=ax5, color=self.RACE_COLORS[race], 
                                 fill=True, alpha=0.7, linewidth=2)
            
        ax5.set_title('🏔️ Ridge Plot Style\nStacked Density Distributions', 
                     fontweight='bold', pad=20)
        ax5.set_xlabel('Number of Images', fontweight='bold')
        ax5.set_ylabel('Density', fontweight='bold')
        ax5.legend(identities_df['race'].unique(), title='Race')
        
        # 6. Statistical Summary Heatmap
        ax6 = fig.add_subplot(gs[2, :])
        stats_data = []
        for race in identities_df['race'].unique():
            race_data = identities_df[identities_df['race'] == race]['num_images'].tolist()
            stats = self.calculate_comprehensive_statistics(race_data)
            stats['Race'] = race
            stats_data.append(stats)
        
        stats_df = pd.DataFrame(stats_data).set_index('Race')
        stats_for_heatmap = stats_df[['mean', 'median', 'std', 'min', 'max', 'q25', 'q75', 'skewness']]
        
        sns.heatmap(stats_for_heatmap.T, annot=True, fmt='.2f', cmap='RdYlBu_r',
                   ax=ax6, cbar_kws={'label': 'Statistical Values'},
                   linewidths=1, square=False)
        ax6.set_title('📈 Comprehensive Statistical Heatmap\nAll Key Metrics by Race', 
                     fontweight='bold', pad=20)
        ax6.set_xlabel('Race', fontweight='bold')
        ax6.set_ylabel('Statistical Measures', fontweight='bold')
        
        plt.suptitle('🎭 Images per Identity: Complete Statistical Analysis', 
                    fontsize=24, fontweight='bold', y=0.98)
        
        plt.savefig(f"{output_dir}/images_per_identity_masterpiece.png", 
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        # Create interactive Plotly version
        self.create_interactive_images_per_identity(identities_df, output_dir)
    
    def create_interactive_images_per_identity(self, identities_df: pd.DataFrame, output_dir: str):
        """Generate interactive Plotly visualization for images per identity."""
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "📦 Interactive Box Plot", "🎻 Interactive Violin Plot",
                "📊 Interactive Histogram", "📈 Statistical Comparison"
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Box plot
        for race in identities_df['race'].unique():
            race_data = identities_df[identities_df['race'] == race]['num_images']
            fig.add_trace(
                go.Box(y=race_data, name=race, marker_color=self.RACE_COLORS[race],
                      boxmean='sd', notched=True),
                row=1, col=1
            )
        
        # Violin plot
        for race in identities_df['race'].unique():
            race_data = identities_df[identities_df['race'] == race]['num_images']
            fig.add_trace(
                go.Violin(y=race_data, name=race, fillcolor=self.RACE_COLORS[race],
                         opacity=0.7, meanline_visible=True, box_visible=True),
                row=1, col=2
            )
        
        # Histogram
        for race in identities_df['race'].unique():
            race_data = identities_df[identities_df['race'] == race]['num_images']
            fig.add_trace(
                go.Histogram(x=race_data, name=race, marker_color=self.RACE_COLORS[race],
                           opacity=0.7, nbinsx=20),
                row=2, col=1
            )
        
        # Statistical comparison
        stats_data = []
        for race in identities_df['race'].unique():
            race_data = identities_df[identities_df['race'] == race]['num_images'].tolist()
            stats = self.calculate_comprehensive_statistics(race_data)
            stats_data.append([race, stats['mean'], stats['median'], stats['std']])
        
        stats_df = pd.DataFrame(stats_data, columns=['Race', 'Mean', 'Median', 'Std'])
        
        fig.add_trace(
            go.Bar(x=stats_df['Race'], y=stats_df['Mean'], 
                  name='Mean', marker_color='lightblue'),
            row=2, col=2
        )
        fig.add_trace(
            go.Bar(x=stats_df['Race'], y=stats_df['Median'], 
                  name='Median', marker_color='lightcoral'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="🎭 Interactive Images per Identity Analysis",
            title_font_size=24,
            showlegend=True,
            height=800,
            template="plotly_white"
        )
        
        # Save interactive plot
        fig.write_html(f"{output_dir}/interactive_images_per_identity.html")
    
    def create_image_dimensions_masterpiece(self, images_df: pd.DataFrame, output_dir: str):
        """Create breathtaking visualizations for image dimensions analysis."""
        self.console.print("[bold cyan]🖼️ Creating image dimensions masterpiece...[/bold cyan]")
        
        # Create comprehensive dimension analysis
        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.25)
        
        # WIDTH ANALYSIS
        # 1. Elegant Width Box Plot
        ax1 = fig.add_subplot(gs[0, 0])
        sns.boxplot(data=images_df, x='race', y='image_width', 
                   palette=self.RACE_COLORS, ax=ax1, notch=True, linewidth=2)
        ax1.set_title('📐 Width Box Plot\nDistribution by Race', fontweight='bold', pad=15)
        ax1.set_xlabel('Race', fontweight='bold')
        ax1.set_ylabel('Width (pixels)', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Width Violin Plot with inner points
        ax2 = fig.add_subplot(gs[0, 1])
        sns.violinplot(data=images_df, x='race', y='image_width',
                      palette=self.RACE_COLORS, ax=ax2, inner='point', linewidth=2)
        ax2.set_title('🎭 Width Violin Plot\nDensity Distribution', fontweight='bold', pad=15)
        ax2.set_xlabel('Race', fontweight='bold')
        ax2.set_ylabel('Width (pixels)', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # HEIGHT ANALYSIS
        # 3. Height Box Plot
        ax3 = fig.add_subplot(gs[0, 2])
        sns.boxplot(data=images_df, x='race', y='image_height',
                   palette=self.RACE_COLORS, ax=ax3, notch=True, linewidth=2)
        ax3.set_title('📏 Height Box Plot\nDistribution by Race', fontweight='bold', pad=15)
        ax3.set_xlabel('Race', fontweight='bold')
        ax3.set_ylabel('Height (pixels)', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Height Violin Plot
        ax4 = fig.add_subplot(gs[0, 3])
        sns.violinplot(data=images_df, x='race', y='image_height',
                      palette=self.RACE_COLORS, ax=ax4, inner='point', linewidth=2)
        ax4.set_title('🎪 Height Violin Plot\nDensity Distribution', fontweight='bold', pad=15)
        ax4.set_xlabel('Race', fontweight='bold')
        ax4.set_ylabel('Height (pixels)', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Width Distribution with KDE
        ax5 = fig.add_subplot(gs[1, :2])
        for race in images_df['race'].unique():
            race_data = images_df[images_df['race'] == race]['image_width']
            ax5.hist(race_data, alpha=0.6, label=race, bins=30,
                    color=self.RACE_COLORS[race], density=True, edgecolor='white')
            sns.kdeplot(data=race_data, ax=ax5, color=self.RACE_COLORS[race],
                       linewidth=3, alpha=0.9)
        
        ax5.set_title('📊 Width Distribution Analysis\nHistogram with KDE Overlay', 
                     fontweight='bold', pad=20)
        ax5.set_xlabel('Image Width (pixels)', fontweight='bold')
        ax5.set_ylabel('Density', fontweight='bold')
        ax5.legend(title='Race', fontsize=11)
        ax5.grid(True, alpha=0.3)
        
        # 6. Height Distribution with KDE
        ax6 = fig.add_subplot(gs[1, 2:])
        for race in images_df['race'].unique():
            race_data = images_df[images_df['race'] == race]['image_height']
            ax6.hist(race_data, alpha=0.6, label=race, bins=30,
                    color=self.RACE_COLORS[race], density=True, edgecolor='white')
            sns.kdeplot(data=race_data, ax=ax6, color=self.RACE_COLORS[race],
                       linewidth=3, alpha=0.9)
        
        ax6.set_title('📊 Height Distribution Analysis\nHistogram with KDE Overlay', 
                     fontweight='bold', pad=20)
        ax6.set_xlabel('Image Height (pixels)', fontweight='bold')
        ax6.set_ylabel('Density', fontweight='bold')
        ax6.legend(title='Race', fontsize=11)
        ax6.grid(True, alpha=0.3)
        
        # 7. Spectacular 2D Density Plot (Width vs Height)
        ax7 = fig.add_subplot(gs[2, :2])
        for race in images_df['race'].unique():
            race_data = images_df[images_df['race'] == race]
            ax7.scatter(race_data['image_width'], race_data['image_height'],
                       alpha=0.6, s=25, color=self.RACE_COLORS[race], label=race,
                       edgecolors='white', linewidth=0.5)
        
        ax7.set_title('🎯 Dimensions Correlation Scatter\nWidth vs Height by Race', 
                     fontweight='bold', pad=20)
        ax7.set_xlabel('Image Width (pixels)', fontweight='bold')
        ax7.set_ylabel('Image Height (pixels)', fontweight='bold')
        ax7.legend(title='Race', fontsize=11)
        ax7.grid(True, alpha=0.3)
        
        # 8. Aspect Ratio Analysis
        ax8 = fig.add_subplot(gs[2, 2:])
        images_df['aspect_ratio'] = images_df['image_width'] / images_df['image_height']
        
        sns.boxplot(data=images_df, x='race', y='aspect_ratio',
                   palette=self.RACE_COLORS, ax=ax8, notch=True, linewidth=2)
        ax8.set_title('⚖️ Aspect Ratio Analysis\nWidth/Height Ratio by Race', 
                     fontweight='bold', pad=20)
        ax8.set_xlabel('Race', fontweight='bold')
        ax8.set_ylabel('Aspect Ratio (W/H)', fontweight='bold')
        ax8.tick_params(axis='x', rotation=45)
        ax8.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Square (1:1)')
        ax8.legend()
        
        plt.suptitle('🖼️ Image Dimensions: Complete Statistical Analysis', 
                    fontsize=26, fontweight='bold', y=0.98)
        
        plt.savefig(f"{output_dir}/image_dimensions_masterpiece.png", 
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        # Create interactive Plotly version
        self.create_interactive_dimensions_analysis(images_df, output_dir)
    
    def create_interactive_dimensions_analysis(self, images_df: pd.DataFrame, output_dir: str):
        """Generate spectacular interactive dimension analysis."""
        
        # Calculate aspect ratio
        images_df['aspect_ratio'] = images_df['image_width'] / images_df['image_height']
        
        # Create main interactive figure
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                "📐 Width Distribution", "📏 Height Distribution", "🎯 Dimensions Scatter",
                "⚖️ Aspect Ratios", "📊 Dimension Heatmap", "🔍 Statistical Overview"
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Width distribution
        for race in images_df['race'].unique():
            race_data = images_df[images_df['race'] == race]
            fig.add_trace(
                go.Histogram(x=race_data['image_width'], name=f'{race} Width',
                           marker_color=self.RACE_COLORS[race], opacity=0.7, nbinsx=25),
                row=1, col=1
            )
        
        # Height distribution
        for race in images_df['race'].unique():
            race_data = images_df[images_df['race'] == race]
            fig.add_trace(
                go.Histogram(x=race_data['image_height'], name=f'{race} Height',
                           marker_color=self.RACE_COLORS[race], opacity=0.7, nbinsx=25),
                row=1, col=2
            )
        
        # Scatter plot
        for race in images_df['race'].unique():
            race_data = images_df[images_df['race'] == race]
            fig.add_trace(
                go.Scatter(x=race_data['image_width'], y=race_data['image_height'],
                          mode='markers', name=f'{race} Dimensions',
                          marker=dict(color=self.RACE_COLORS[race], size=6, opacity=0.6)),
                row=1, col=3
            )
        
        # Aspect ratio box plots
        for race in images_df['race'].unique():
            race_data = images_df[images_df['race'] == race]
            fig.add_trace(
                go.Box(y=race_data['aspect_ratio'], name=race,
                      marker_color=self.RACE_COLORS[race], boxmean='sd'),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title_text="🖼️ Interactive Image Dimensions Analysis",
            title_font_size=24,
            showlegend=True,
            height=1000,
            template="plotly_white"
        )
        
        fig.write_html(f"{output_dir}/interactive_dimensions_analysis.html")
        
    def create_gender_analysis_masterpiece(self, identities_df: pd.DataFrame, images_df: pd.DataFrame, output_dir: str):
        """Create comprehensive gender analysis visualizations."""
        self.console.print("[bold purple]👫 Creating gender analysis masterpiece...[/bold purple]")
        
        # Create comprehensive gender analysis
        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
        
        # 1. Gender Distribution by Race - Stacked Bar Chart
        ax1 = fig.add_subplot(gs[0, :2])
        gender_race_counts = identities_df.groupby(['race', 'gender']).size().unstack(fill_value=0)
        
        # Create stacked bar chart with beautiful colors
        gender_colors = {'Male': '#4A90E2', 'Female': '#F5A623', 'Unknown': '#D0021B'}
        gender_race_counts.plot(kind='bar', stacked=True, ax=ax1, 
                               color=[gender_colors.get(col, '#7ED321') for col in gender_race_counts.columns],
                               edgecolor='white', linewidth=2)
        ax1.set_title('👥 Gender Distribution by Race\nStacked Bar Chart', fontweight='bold', pad=20)
        ax1.set_xlabel('Race', fontweight='bold')
        ax1.set_ylabel('Number of Identities', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(title='Gender', title_fontsize=12, fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for container in ax1.containers:
            ax1.bar_label(container, label_type='center', fontweight='bold')
        
        # 2. Gender Proportions by Race - Donut Charts
        races = identities_df['race'].unique()
        for i, race in enumerate(races[:4]):  # Limit to 4 races for layout
            ax = fig.add_subplot(gs[0, 2 + (i % 2)])
            if i >= 2:
                ax = fig.add_subplot(gs[1, (i % 2)])
                
            race_gender_data = identities_df[identities_df['race'] == race]['gender'].value_counts()
            
            # Create donut chart
            wedges, texts, autotexts = ax.pie(race_gender_data.values, 
                                             labels=race_gender_data.index,
                                             colors=[gender_colors.get(gender, '#7ED321') for gender in race_gender_data.index],
                                             autopct='%1.1f%%', startangle=90,
                                             wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
                                             textprops={'fontweight': 'bold'})
            
            # Add center circle for donut effect
            centre_circle = plt.Circle((0,0), 0.70, fc='white', linewidth=2, edgecolor='gray')
            ax.add_artist(centre_circle)
            
            ax.set_title(f'🎯 {race}\nGender Distribution', fontweight='bold', pad=15)
        
        # 3. Gender vs Images per Identity Analysis
        ax3 = fig.add_subplot(gs[1, 2:])
        
        # Box plot showing images per identity by gender and race
        sns.boxplot(data=identities_df, x='race', y='num_images', hue='gender',
                   palette=gender_colors, ax=ax3, notch=True, linewidth=2)
        ax3.set_title('📦 Images per Identity by Gender and Race\nBox Plot Comparison', 
                     fontweight='bold', pad=20)
        ax3.set_xlabel('Race', fontweight='bold')
        ax3.set_ylabel('Number of Images per Identity', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(title='Gender', title_fontsize=12, fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # 4. Gender Distribution Heatmap
        ax4 = fig.add_subplot(gs[2, :2])
        
        # Create cross-tabulation for heatmap
        gender_crosstab = pd.crosstab(identities_df['race'], identities_df['gender'], normalize='index') * 100
        
        sns.heatmap(gender_crosstab, annot=True, fmt='.1f', cmap='RdYlBu_r',
                   ax=ax4, cbar_kws={'label': 'Percentage (%)'},
                   linewidths=2, square=False)
        ax4.set_title('🔥 Gender Distribution Heatmap\nPercentage by Race', 
                     fontweight='bold', pad=20)
        ax4.set_xlabel('Gender', fontweight='bold')
        ax4.set_ylabel('Race', fontweight='bold')
        
        # 5. Gender Statistics Summary
        ax5 = fig.add_subplot(gs[2, 2:])
        
        # Calculate comprehensive gender statistics
        gender_stats = []
        for race in identities_df['race'].unique():
            race_data = identities_df[identities_df['race'] == race]
            total_identities = len(race_data)
            
            for gender in ['Male', 'Female', 'Unknown']:
                count = len(race_data[race_data['gender'] == gender])
                percentage = (count / total_identities * 100) if total_identities > 0 else 0
                avg_images = race_data[race_data['gender'] == gender]['num_images'].mean() if count > 0 else 0
                
                gender_stats.append({
                    'Race': race,
                    'Gender': gender,
                    'Count': count,
                    'Percentage': percentage,
                    'Avg_Images': avg_images
                })
        
        gender_stats_df = pd.DataFrame(gender_stats)
        
        # Create grouped bar chart for statistics
        x_pos = np.arange(len(identities_df['race'].unique()))
        width = 0.25
        
        male_data = gender_stats_df[gender_stats_df['Gender'] == 'Male']
        female_data = gender_stats_df[gender_stats_df['Gender'] == 'Female']
        unknown_data = gender_stats_df[gender_stats_df['Gender'] == 'Unknown']
        
        ax5.bar(x_pos - width, male_data['Percentage'], width, 
               label='Male %', color=gender_colors.get('Male', '#4A90E2'), alpha=0.8)
        ax5.bar(x_pos, female_data['Percentage'], width,
               label='Female %', color=gender_colors.get('Female', '#F5A623'), alpha=0.8)
        ax5.bar(x_pos + width, unknown_data['Percentage'], width,
               label='Unknown %', color=gender_colors.get('Unknown', '#D0021B'), alpha=0.8)
        
        ax5.set_title('📊 Gender Distribution Comparison\nPercentage by Race', 
                     fontweight='bold', pad=20)
        ax5.set_xlabel('Race', fontweight='bold')
        ax5.set_ylabel('Percentage (%)', fontweight='bold')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(identities_df['race'].unique(), rotation=45, ha='right')
        ax5.legend(title='Gender', title_fontsize=12, fontsize=11)
        ax5.grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for i, (race, male_pct, female_pct, unknown_pct) in enumerate(zip(
            identities_df['race'].unique(),
            male_data['Percentage'],
            female_data['Percentage'], 
            unknown_data['Percentage'])):
            
            if male_pct > 2:  # Only show label if bar is big enough
                ax5.text(i - width, male_pct + 0.5, f'{male_pct:.1f}%', 
                        ha='center', va='bottom', fontweight='bold', fontsize=9)
            if female_pct > 2:
                ax5.text(i, female_pct + 0.5, f'{female_pct:.1f}%', 
                        ha='center', va='bottom', fontweight='bold', fontsize=9)
            if unknown_pct > 2:
                ax5.text(i + width, unknown_pct + 0.5, f'{unknown_pct:.1f}%', 
                        ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.suptitle('👫 Gender Analysis: Comprehensive Statistical Overview', 
                    fontsize=26, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gender_analysis_masterpiece.png", 
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        # Create interactive gender analysis
        self.create_interactive_gender_analysis(identities_df, images_df, output_dir)
    
    def create_interactive_gender_analysis(self, identities_df: pd.DataFrame, images_df: pd.DataFrame, output_dir: str):
        """Generate interactive Plotly gender analysis."""
        
        # Create subplot figure for interactive gender analysis
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                "👥 Gender by Race", "🎯 Gender Proportions", "📦 Images by Gender & Race",
                "🔥 Gender Heatmap", "📊 Gender Statistics", "👫 Gender Trends"
            ),
            specs=[[{"type": "bar"}, {"type": "pie"}, {"type": "box"}],
                   [{"type": "heatmap"}, {"type": "bar"}, {"type": "violin"}]]
        )
        
        # Color mapping for consistency
        gender_colors = {'Male': '#4A90E2', 'Female': '#F5A623', 'Unknown': '#D0021B'}
        
        # 1. Stacked bar chart
        gender_race_counts = identities_df.groupby(['race', 'gender']).size().unstack(fill_value=0)
        
        for gender in gender_race_counts.columns:
            fig.add_trace(
                go.Bar(x=gender_race_counts.index, y=gender_race_counts[gender],
                      name=gender, marker_color=gender_colors.get(gender, '#7ED321'),
                      text=gender_race_counts[gender], textposition='inside'),
                row=1, col=1
            )
        
        # 2. Overall gender pie chart
        gender_totals = identities_df['gender'].value_counts()
        fig.add_trace(
            go.Pie(labels=gender_totals.index, values=gender_totals.values,
                  marker_colors=[gender_colors.get(g, '#7ED321') for g in gender_totals.index],
                  textinfo='label+percent', textfont_size=12),
            row=1, col=2
        )
        
        # 3. Box plot for images per identity by gender
        for gender in identities_df['gender'].unique():
            gender_data = identities_df[identities_df['gender'] == gender]
            fig.add_trace(
                go.Box(y=gender_data['num_images'], name=gender,
                      marker_color=gender_colors.get(gender, '#7ED321'),
                      boxmean='sd'),
                row=1, col=3
            )
        
        # 4. Gender distribution heatmap data
        gender_crosstab = pd.crosstab(identities_df['race'], identities_df['gender'])
        
        fig.add_trace(
            go.Heatmap(z=gender_crosstab.values,
                      x=gender_crosstab.columns,
                      y=gender_crosstab.index,
                      colorscale='RdYlBu_r',
                      showscale=True,
                      text=gender_crosstab.values,
                      texttemplate="%{text}",
                      textfont={"size": 12}),
            row=2, col=1
        )
        
        # 5. Gender statistics by race
        gender_stats = []
        for race in identities_df['race'].unique():
            race_data = identities_df[identities_df['race'] == race]
            for gender in race_data['gender'].unique():
                count = len(race_data[race_data['gender'] == gender])
                gender_stats.append({'Race': race, 'Gender': gender, 'Count': count})
        
        stats_df = pd.DataFrame(gender_stats)
        
        for gender in stats_df['Gender'].unique():
            gender_stats_filtered = stats_df[stats_df['Gender'] == gender]
            fig.add_trace(
                go.Bar(x=gender_stats_filtered['Race'], y=gender_stats_filtered['Count'],
                      name=f'{gender} Count', marker_color=gender_colors.get(gender, '#7ED321'),
                      text=gender_stats_filtered['Count'], textposition='outside'),
                row=2, col=2
            )
        
        # 6. Violin plot for images per identity distribution by gender
        for gender in identities_df['gender'].unique():
            gender_data = identities_df[identities_df['gender'] == gender]
            fig.add_trace(
                go.Violin(y=gender_data['num_images'], name=gender,
                         fillcolor=gender_colors.get(gender, '#7ED321'),
                         opacity=0.7, meanline_visible=True, box_visible=True),
                row=2, col=3
            )
        
        # Update layout
        fig.update_layout(
            title_text="👫 Interactive Gender Analysis Dashboard",
            title_font_size=24,
            showlegend=True,
            height=1000,
            template="plotly_white",
            barmode='stack'  # For stacked bars in first subplot
        )
        
        # Update subplot titles and axes
        fig.update_xaxes(title_text="Race", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Images per Identity", row=1, col=3)
        fig.update_xaxes(title_text="Race", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        fig.update_yaxes(title_text="Images per Identity", row=2, col=3)
        
        # Save interactive plot
        fig.write_html(f"{output_dir}/interactive_gender_analysis.html")
    
    def create_statistical_dashboard(self, images_df: pd.DataFrame, identities_df: pd.DataFrame, output_dir: str):
        """Create a comprehensive statistical dashboard."""
        self.console.print("[bold yellow]📊 Creating statistical dashboard...[/bold yellow]")
        
        fig, axes = plt.subplots(2, 3, figsize=(22, 14))
        fig.suptitle('📊 Complete Dataset Statistical Dashboard', fontsize=24, fontweight='bold', y=0.98)
        
        # 1. Dataset Overview (Updated to include gender info)
        ax1 = axes[0, 0]
        
        # Calculate gender distribution for overview
        gender_counts = identities_df['gender'].value_counts()
        total_with_gender = gender_counts.get('Male', 0) + gender_counts.get('Female', 0)
        
        overview_data = {
            'Total Images': len(images_df),
            'Total Identities': len(identities_df),
            'Races': len(images_df['race'].unique()),
            'With Gender Info': total_with_gender,
            'Avg Images/Identity': images_df.groupby('identity_id').size().mean()
        }
        
        bars = ax1.bar(range(len(overview_data)), list(overview_data.values()), 
                      color=CUSTOM_PALETTE[:len(overview_data)], alpha=0.8, edgecolor='white', linewidth=2)
        ax1.set_xticks(range(len(overview_data)))
        ax1.set_xticklabels(list(overview_data.keys()), rotation=45, ha='right')
        ax1.set_title('📈 Dataset Overview with Gender Info', fontweight='bold', pad=15)
        ax1.set_ylabel('Count', fontweight='bold')
        
        # Add value labels on bars
        for bar, value in zip(bars, overview_data.values()):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Enhanced Race + Gender Distribution Pie Chart
        ax2 = axes[0, 1]
        
        # Create combined race-gender categories for more detailed pie chart
        identities_df['race_gender'] = identities_df['race'] + ' - ' + identities_df['gender']
        race_gender_counts = identities_df['race_gender'].value_counts()
        
        # Create a color map that combines race and gender colors
        colors_for_pie = []
        for category in race_gender_counts.index:
            race = category.split(' - ')[0]
            gender = category.split(' - ')[1]
            base_color = self.RACE_COLORS.get(race, '#7ED321')
            # Modify color based on gender
            if gender == 'Male':
                colors_for_pie.append(base_color)
            elif gender == 'Female':
                # Lighten the color for female
                colors_for_pie.append(base_color + '80')  # Add transparency
            else:  # Unknown
                colors_for_pie.append('#CCCCCC')
        
        wedges, texts, autotexts = ax2.pie(race_gender_counts.values[:8], # Limit to top 8 to avoid crowding
                                          labels=[label.replace(' - ', '\n') for label in race_gender_counts.index[:8]],
                                          colors=colors_for_pie[:8],
                                          autopct='%1.1f%%', startangle=90, 
                                          textprops={'fontweight': 'bold', 'fontsize': 9})
        ax2.set_title('🥧 Race-Gender Distribution\n(Top Categories)', fontweight='bold', pad=15)
        
        # 3. Dimension Statistics Heatmap
        ax3 = axes[0, 2]
        dim_stats = []
        for race in images_df['race'].unique():
            race_data = images_df[images_df['race'] == race]
            dim_stats.append([
                race_data['image_width'].mean(),
                race_data['image_height'].mean(),
                race_data['image_total_pixels'].mean() / 1000000  # Convert to megapixels
            ])
        
        dim_stats_df = pd.DataFrame(dim_stats, 
                                   columns=['Avg Width', 'Avg Height', 'Avg Megapixels'],
                                   index=images_df['race'].unique())
        
        sns.heatmap(dim_stats_df.T, annot=True, fmt='.1f', cmap='viridis',
                   ax=ax3, cbar_kws={'label': 'Pixel Values'}, linewidths=1)
        ax3.set_title('🔥 Dimension Statistics Heatmap', fontweight='bold', pad=15)
        ax3.set_xlabel('Race', fontweight='bold')
        ax3.set_ylabel('Metrics', fontweight='bold')
        
        # 4. Images per Identity Distribution
        ax4 = axes[1, 0]
        sns.violinplot(data=identities_df, x='race', y='num_images',
                      palette=self.RACE_COLORS, ax=ax4, inner='box', linewidth=2)
        ax4.set_title('🎻 Images per Identity Distribution', fontweight='bold', pad=15)
        ax4.set_xlabel('Race', fontweight='bold')
        ax4.set_ylabel('Number of Images', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Correlation Matrix
        ax5 = axes[1, 1]
        correlation_data = images_df[['image_width', 'image_height', 'image_total_pixels']].corr()
        mask = np.triu(np.ones_like(correlation_data, dtype=bool))
        sns.heatmap(correlation_data, mask=mask, annot=True, fmt='.3f', 
                   cmap='RdBu_r', center=0, ax=ax5, 
                   square=True, linewidths=2, cbar_kws={"shrink": .8})
        ax5.set_title('🔗 Dimension Correlation Matrix', fontweight='bold', pad=15)
        
        # 6. Quality Metrics Summary
        ax6 = axes[1, 2]
        quality_metrics = []
        for race in images_df['race'].unique():
            race_data = images_df[images_df['race'] == race]
            race_identities = identities_df[identities_df['race'] == race]
            
            quality_metrics.append({
                'Race': race,
                'Avg Resolution': np.sqrt(race_data['image_total_pixels']).mean(),
                'Std Resolution': np.sqrt(race_data['image_total_pixels']).std(),
                'Images/Identity': race_identities['num_images'].mean(),
                'Max Images/Identity': race_identities['num_images'].max()
            })
        
        quality_df = pd.DataFrame(quality_metrics).set_index('Race')
        
        # Create a radar-like representation
        x_pos = np.arange(len(quality_df.index))
        width = 0.2
        
        for i, col in enumerate(quality_df.columns):
            normalized_values = (quality_df[col] - quality_df[col].min()) / (quality_df[col].max() - quality_df[col].min())
            ax6.bar(x_pos + i*width, normalized_values, width, 
                   label=col, alpha=0.8, color=CUSTOM_PALETTE[i])
        
        ax6.set_title('🎯 Quality Metrics Summary\n(Normalized)', fontweight='bold', pad=15)
        ax6.set_xlabel('Race', fontweight='bold')
        ax6.set_ylabel('Normalized Score', fontweight='bold')
        ax6.set_xticks(x_pos + width * 1.5)
        ax6.set_xticklabels(quality_df.index, rotation=45, ha='right')
        ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/statistical_dashboard.png", 
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
