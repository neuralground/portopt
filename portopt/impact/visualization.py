"""Market impact visualization utilities."""

import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from ..utils.logging import setup_logging

class ImpactVisualizer:
    """Visualizes market impact analysis results."""
    
    def __init__(self):
        """Initialize impact visualizer."""
        self.logger = setup_logging(__name__)
        
    def plot_impact_decay(self, impact_data: Dict[str, np.ndarray],
                         title: str = "Impact Decay Analysis") -> Figure:
        """
        Plot impact decay over time.
        
        Args:
            impact_data: Dictionary with temporal impact data
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        times = np.arange(len(impact_data['temporary_impact']))
        ax.plot(times, impact_data['temporary_impact'], 
                label='Temporary Impact', linestyle='--')
        ax.plot(times, impact_data['permanent_impact'],
                label='Permanent Impact')
        ax.plot(times, impact_data['total_impact'],
                label='Total Impact', linewidth=2)
        
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Price Impact (%)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def plot_participation_analysis(self, participation_data: Dict[str, np.ndarray],
                                  bins: int = 20) -> Figure:
        """
        Plot participation rate analysis.
        
        Args:
            participation_data: Dictionary with participation data
            bins: Number of histogram bins
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Participation rate distribution
        ax1.hist(participation_data['rates'], bins=bins, 
                alpha=0.7, color='blue')
        ax1.axvline(participation_data['limit'], 
                   color='red', linestyle='--', 
                   label='Participation Limit')
        ax1.set_xlabel('Participation Rate (%)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Participation Rate Distribution')
        ax1.legend()
        
        # Cumulative volume
        ax2.plot(participation_data['cumulative_volume'], 
                label='Actual', linewidth=2)
        ax2.plot(participation_data['ideal_volume'],
                label='Ideal', linestyle='--')
        ax2.set_xlabel('Time (days)')
        ax2.set_ylabel('Cumulative Volume (%)')
        ax2.set_title('Trading Volume Profile')
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_cost_breakdown(self, cost_data: Dict[str, float],
                          title: str = "Trading Cost Breakdown") -> Figure:
        """
        Plot trading cost breakdown.
        
        Args:
            cost_data: Dictionary with cost components
            title: Plot title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart of cost components
        components = ['Spread Cost', 'Impact Cost', 'Timing Cost']
        values = [cost_data['spread_cost'], 
                 cost_data['impact_cost'],
                 cost_data['timing_cost']]
        ax1.pie(values, labels=components, autopct='%1.1f%%',
                colors=['lightblue', 'lightgreen', 'salmon'])
        ax1.set_title('Cost Component Breakdown')
        
        # Bar chart of costs by size bucket
        size_buckets = cost_data['size_buckets']
        bucket_costs = cost_data['bucket_costs']
        ax2.bar(size_buckets.keys(), bucket_costs.values())
        ax2.set_xlabel('Trade Size Bucket')
        ax2.set_ylabel('Cost (bps)')
        ax2.set_title('Costs by Trade Size')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_liquidation_profile(self, liquidation_data: Dict[str, np.ndarray],
                               trade_schedule: np.ndarray) -> Figure:
        """
        Plot liquidation profile and trading schedule.
        
        Args:
            liquidation_data: Dictionary with liquidation analysis
            trade_schedule: Planned trading schedule
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Remaining position over time
        times = np.arange(len(liquidation_data['remaining']))
        ax1.plot(times, liquidation_data['remaining'], 
                label='Actual', linewidth=2)
        ax1.plot(times, liquidation_data['ideal_remaining'],
                label='Ideal', linestyle='--')
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('Remaining Position (%)')
        ax1.set_title('Position Liquidation Profile')
        ax1.legend()
        ax1.grid(True)
        
        # Daily trading amounts
        ax2.bar(times, trade_schedule, alpha=0.7,
               label='Planned Trades')
        ax2.plot(times, liquidation_data['max_daily'],
                color='red', linestyle='--',
                label='Maximum Daily Capacity')
        ax2.set_xlabel('Time (days)')
        ax2.set_ylabel('Trading Amount')
        ax2.set_title('Daily Trading Schedule')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
    
    def create_impact_report(self, impact_results: Dict[str, Any],
                           output_file: str) -> None:
        """
        Create comprehensive impact analysis report.
        
        Args:
            impact_results: Dictionary with all impact analysis results
            output_file: Path to save the report
        """
        try:
            # Create decay plot
            fig1 = self.plot_impact_decay(impact_results['decay_analysis'])
            
            # Create participation analysis
            fig2 = self.plot_participation_analysis(
                impact_results['participation_analysis']
            )
            
            # Create cost breakdown
            fig3 = self.plot_cost_breakdown(impact_results['cost_analysis'])
            
            # Create liquidation profile
            fig4 = self.plot_liquidation_profile(
                impact_results['liquidation_analysis'],
                impact_results['trade_schedule']
            )
            
            # Save all plots to a multi-page PDF
            with PdfPages(output_file) as pdf:
                pdf.savefig(fig1)
                pdf.savefig(fig2)
                pdf.savefig(fig3)
                pdf.savefig(fig4)
                
            plt.close('all')
            self.logger.info(f"Impact analysis report saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating impact report: {str(e)}")
            raise

