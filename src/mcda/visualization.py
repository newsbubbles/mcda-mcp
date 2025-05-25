"""Visualization tools for MCDA results.

Provides functions to visualize MCDA results through various types of charts.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from mcda.models import MCDAResult, ComparisonResult


def plot_preferences(result: MCDAResult, title: Optional[str] = None, figsize: Tuple[int, int] = (10, 6)):
    """Plot preference scores of alternatives.
    
    Args:
        result: MCDAResult containing preferences and rankings
        title: Optional title for the chart (defaults to method name)
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get data
    preferences = result.preferences
    alternatives = [alt.name for alt in result.alternatives]
    rankings = result.rankings
    
    # Create sorted indices
    sorted_indices = np.argsort(rankings)
    
    # Sort preferences and alternatives by ranking
    sorted_prefs = [preferences[i] for i in sorted_indices]
    sorted_alts = [alternatives[i] for i in sorted_indices]
    
    # Create bar plot
    bars = ax.bar(sorted_alts, sorted_prefs)
    
    # Add ranking as text on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'Rank: {i+1}', ha='center', va='bottom')
    
    # Add labels
    ax.set_xlabel('Alternatives')
    ax.set_ylabel('Preference Score')
    ax.set_title(title or f"{result.method_name} Results")
    
    # Format y-axis as percentage if preferences are between 0 and 1
    if all(0 <= p <= 1 for p in preferences):
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    plt.tight_layout()
    return fig


def plot_ranking_comparison(results: Dict[str, MCDAResult], 
                           figsize: Tuple[int, int] = (12, 8),
                           alternatives_subset: Optional[List[str]] = None):
    """Plot comparison of rankings from different methods.
    
    Args:
        results: Dictionary mapping method names to MCDAResult objects
        figsize: Figure size as (width, height)
        alternatives_subset: Optional list of alternative IDs to include (None means all)
        
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique method names and all alternatives
    methods = list(results.keys())
    
    # Reference to first result to get all alternatives
    any_result = next(iter(results.values()))
    all_alternatives = [(alt.id, alt.name) for alt in any_result.alternatives]
    
    # Filter alternatives if subset specified
    if alternatives_subset is not None:
        alt_subset_ids = set(alternatives_subset)
        filtered_alternatives = [(alt_id, alt_name) for alt_id, alt_name in all_alternatives 
                                if alt_id in alt_subset_ids]
        if not filtered_alternatives:
            raise ValueError(f"None of the provided alternative IDs {alternatives_subset} were found")
        all_alternatives = filtered_alternatives
    
    # Create a mapping from alternative ID to index
    alt_id_to_idx = {alt[0]: i for i, alt in enumerate(all_alternatives)}
    
    # Extract rankings for each method
    rankings_data = []
    for method_name, result in results.items():
        # Map each alternative to its ranking
        alt_rankings = {}
        for i, alt in enumerate(result.alternatives):
            if alternatives_subset is None or alt.id in alternatives_subset:
                alt_rankings[alt.id] = result.rankings[i]
        
        # Append to data in the right order
        method_rankings = [alt_rankings.get(alt_id, np.nan) for alt_id, _ in all_alternatives]
        rankings_data.append(method_rankings)
    
    # Convert to numpy array for heatmap
    rankings_array = np.array(rankings_data)
    
    # Create heatmap
    sns.heatmap(rankings_array, annot=True, cmap="YlGnBu_r", fmt=".0f",
                xticklabels=[alt[1] for alt in all_alternatives],
                yticklabels=methods,
                cbar_kws={"label": "Ranking (lower is better)"})
    
    ax.set_xlabel('Alternatives')
    ax.set_ylabel('MCDA Methods')
    ax.set_title('Comparison of Rankings from Different MCDA Methods')
    
    plt.tight_layout()
    return fig


def plot_correlation_matrix(correlation_matrix: Dict[str, Dict[str, float]], 
                           figsize: Tuple[int, int] = (10, 8)):
    """Plot correlation matrix between methods.
    
    Args:
        correlation_matrix: Dictionary of dictionaries with correlation values
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract methods and create correlation array
    methods = list(correlation_matrix.keys())
    n_methods = len(methods)
    corr_array = np.zeros((n_methods, n_methods))
    
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            corr_array[i, j] = correlation_matrix[method1][method2]
    
    # Create heatmap
    sns.heatmap(corr_array, annot=True, cmap="coolwarm", vmin=-1, vmax=1,
                xticklabels=methods, yticklabels=methods,
                cbar_kws={"label": "Spearman's Rank Correlation"})
    
    ax.set_xlabel('MCDA Methods')
    ax.set_ylabel('MCDA Methods')
    ax.set_title('Correlation Between Method Rankings')
    
    plt.tight_layout()
    return fig


def plot_criteria_weights(criteria_names: List[str], weights: List[float], 
                         title: Optional[str] = None, 
                         figsize: Tuple[int, int] = (10, 6)):
    """Plot criteria weights.
    
    Args:
        criteria_names: List of criteria names
        weights: List of criteria weights
        title: Optional title for the chart
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by weight (descending)
    sorted_indices = np.argsort(weights)[::-1]
    sorted_weights = [weights[i] for i in sorted_indices]
    sorted_names = [criteria_names[i] for i in sorted_indices]
    
    # Create bar plot
    bars = ax.bar(sorted_names, sorted_weights)
    
    # Add weight as text on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    # Add labels
    ax.set_xlabel('Criteria')
    ax.set_ylabel('Weight')
    ax.set_title(title or "Criteria Weights")
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    plt.tight_layout()
    return fig