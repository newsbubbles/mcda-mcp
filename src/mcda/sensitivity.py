"""Sensitivity analysis tools for MCDA.

Provides functions to analyze the sensitivity of MCDA results to variations in input parameters.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import matplotlib.pyplot as plt

from mcda.models import DecisionMatrix, MCDAResult, WeightingResult
from mcda.base import MCDAMethod
from mcda.client import MCDAClient


class WeightSensitivityAnalysis:
    """Class for analyzing the sensitivity of MCDA results to changes in criteria weights."""
    
    def __init__(self, decision_matrix: DecisionMatrix, method: MCDAMethod):
        """Initialize with decision matrix and MCDA method.
        
        Args:
            decision_matrix: Decision matrix for analysis
            method: MCDA method to use for evaluation
        """
        self.decision_matrix = decision_matrix
        self.method = method
        
        # Get original weights and check they are valid
        original_weights = decision_matrix.get_criteria_weights()
        if original_weights is None:
            raise ValueError("Decision matrix must include weights for sensitivity analysis")
        self.original_weights = original_weights
        
        # Calculate original result
        self.original_result = method.evaluate(decision_matrix)
    
    def analyze_weight_perturbation(self, 
                                   criterion_index: int, 
                                   perturbation_range: List[float] = None,
                                   n_points: int = 10) -> Dict[str, Any]:
        """Analyze how changes to one criterion's weight affect rankings.
        
        Args:
            criterion_index: Index of the criterion to analyze
            perturbation_range: Range of perturbation as [min_factor, max_factor] 
                               (defaults to [0.5, 1.5])
            n_points: Number of points to evaluate
            
        Returns:
            Dictionary with perturbations, results, and changes in rankings
        """
        if criterion_index < 0 or criterion_index >= len(self.original_weights):
            raise ValueError(f"Criterion index {criterion_index} out of range")
            
        # Set default perturbation range if not provided
        if perturbation_range is None:
            perturbation_range = [0.5, 1.5]
        
        # Generate perturbation factors
        perturbation_factors = np.linspace(perturbation_range[0], perturbation_range[1], n_points)
        
        # Initialize results storage
        perturbed_weights = []
        results = []
        ranking_changes = []
        
        # Original rankings for comparison
        original_rankings = np.array(self.original_result.rankings)
        
        # Analyze each perturbation
        for factor in perturbation_factors:
            # Create perturbed weights
            weights = self.original_weights.copy()
            weights[criterion_index] *= factor
            
            # Normalize to sum to 1
            weights = weights / np.sum(weights)
            
            # Evaluate with the new weights
            result = self.method.evaluate(self.decision_matrix, weights)
            
            # Calculate changes in rankings
            new_rankings = np.array(result.rankings)
            changes = np.sum(new_rankings != original_rankings)
            
            # Store results
            perturbed_weights.append(weights.tolist())
            results.append(result)
            ranking_changes.append(changes)
        
        return {
            "criterion_index": criterion_index,
            "criterion_name": self.decision_matrix.criteria[criterion_index].name,
            "perturbation_factors": perturbation_factors.tolist(),
            "perturbed_weights": perturbed_weights,
            "results": results,
            "ranking_changes": ranking_changes
        }
    
    def analyze_all_weights(self, 
                          perturbation_range: List[float] = None, 
                          n_points: int = 5) -> Dict[int, Dict[str, Any]]:
        """Analyze sensitivity for all criteria weights.
        
        Args:
            perturbation_range: Range of perturbation as [min_factor, max_factor]
            n_points: Number of points to evaluate per criterion
            
        Returns:
            Dictionary mapping criterion indices to their sensitivity analysis results
        """
        results = {}
        
        for i in range(len(self.original_weights)):
            results[i] = self.analyze_weight_perturbation(i, perturbation_range, n_points)
        
        return results
    
    def plot_weight_sensitivity(self, 
                              analysis_result: Dict[str, Any],
                              figsize: Tuple[int, int] = (10, 6)):
        """Plot the results of weight sensitivity analysis.
        
        Args:
            analysis_result: Result from analyze_weight_perturbation
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure object
        """
        factors = analysis_result["perturbation_factors"]
        changes = analysis_result["ranking_changes"]
        criterion_name = analysis_result["criterion_name"]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot ranking changes
        ax1.plot(factors, changes, marker='o')
        ax1.set_xlabel('Weight Perturbation Factor')
        ax1.set_ylabel('Number of Ranking Changes')
        ax1.set_title(f'Sensitivity of Rankings to Weight of {criterion_name}')
        ax1.grid(True)
        
        # Plot alternative preferences for each perturbation
        for i, result in enumerate(analysis_result["results"]):
            # Only plot for a subset of factors to avoid clutter
            if i % max(1, len(factors) // 5) == 0:
                factor = factors[i]
                preferences = result.preferences
                ax2.plot(range(len(preferences)), preferences, 
                        marker='o', label=f'Factor: {factor:.2f}')
        
        ax2.set_xlabel('Alternative Index')
        ax2.set_ylabel('Preference Score')
        ax2.set_title(f'Preference Scores with Different {criterion_name} Weights')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_weight_sensitivity_heatmap(self, all_results: Dict[int, Dict[str, Any]],
                                      figsize: Tuple[int, int] = (12, 8)):
        """Plot a heatmap of ranking changes for all criteria.
        
        Args:
            all_results: Results from analyze_all_weights
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract data for the heatmap
        criteria = [self.decision_matrix.criteria[i].name for i in all_results.keys()]
        factors = all_results[0]["perturbation_factors"]  # Assuming all have the same factors
        
        # Create the data array
        data = np.zeros((len(criteria), len(factors)))
        for i, criterion_idx in enumerate(all_results.keys()):
            data[i, :] = all_results[criterion_idx]["ranking_changes"]
        
        # Create heatmap
        im = ax.imshow(data, cmap="YlOrRd")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Number of Ranking Changes", rotation=-90, va="bottom")
        
        # Set ticks and labels
        ax.set_yticks(np.arange(len(criteria)))
        ax.set_yticklabels(criteria)
        ax.set_xticks(np.arange(len(factors)))
        ax.set_xticklabels([f"{f:.2f}" for f in factors])
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add labels and title
        ax.set_xlabel("Perturbation Factor")
        ax.set_ylabel("Criterion")
        ax.set_title("Sensitivity of Rankings to Weight Perturbations")
        
        # Loop over data dimensions and create text annotations
        for i in range(len(criteria)):
            for j in range(len(factors)):
                text = ax.text(j, i, int(data[i, j]),
                              ha="center", va="center", color="black")
        
        fig.tight_layout()
        return fig


class ValueSensitivityAnalysis:
    """Class for analyzing the sensitivity of MCDA results to changes in decision matrix values."""
    
    def __init__(self, decision_matrix: DecisionMatrix, method: MCDAMethod):
        """Initialize with decision matrix and MCDA method.
        
        Args:
            decision_matrix: Decision matrix for analysis
            method: MCDA method to use for evaluation
        """
        self.decision_matrix = decision_matrix
        self.method = method
        
        # Get original values
        self.original_matrix = decision_matrix.to_numpy()
        
        # Calculate original result
        self.original_result = method.evaluate(decision_matrix)
    
    def analyze_value_perturbation(self, 
                                 alt_index: int, 
                                 crit_index: int,
                                 perturbation_range: List[float] = None,
                                 n_points: int = 10) -> Dict[str, Any]:
        """Analyze how changes to one value in the decision matrix affect rankings.
        
        Args:
            alt_index: Index of the alternative to analyze
            crit_index: Index of the criterion to analyze
            perturbation_range: Range of perturbation as [min_factor, max_factor]
                               (defaults to [0.5, 1.5])
            n_points: Number of points to evaluate
            
        Returns:
            Dictionary with perturbations, results, and changes in rankings
        """
        if alt_index < 0 or alt_index >= self.original_matrix.shape[0]:
            raise ValueError(f"Alternative index {alt_index} out of range")
        if crit_index < 0 or crit_index >= self.original_matrix.shape[1]:
            raise ValueError(f"Criterion index {crit_index} out of range")
            
        # Set default perturbation range if not provided
        if perturbation_range is None:
            perturbation_range = [0.5, 1.5]
        
        # Generate perturbation factors
        perturbation_factors = np.linspace(perturbation_range[0], perturbation_range[1], n_points)
        
        # Initialize results storage
        perturbed_values = []
        results = []
        ranking_changes = []
        
        # Original rankings for comparison
        original_rankings = np.array(self.original_result.rankings)
        
        # Original value to perturb
        original_value = self.original_matrix[alt_index, crit_index]
        
        # Analyze each perturbation
        for factor in perturbation_factors:
            # Create perturbed matrix
            perturbed_matrix = self.original_matrix.copy()
            perturbed_matrix[alt_index, crit_index] = original_value * factor
            
            # Update the decision matrix
            new_decision_matrix = DecisionMatrix(
                alternatives=self.decision_matrix.alternatives,
                criteria=self.decision_matrix.criteria,
                values=perturbed_matrix.tolist()
            )
            
            # Evaluate with the new decision matrix
            result = self.method.evaluate(new_decision_matrix)
            
            # Calculate changes in rankings
            new_rankings = np.array(result.rankings)
            changes = np.sum(new_rankings != original_rankings)
            
            # Store results
            perturbed_values.append(float(perturbed_matrix[alt_index, crit_index]))
            results.append(result)
            ranking_changes.append(changes)
        
        return {
            "alt_index": alt_index,
            "alt_name": self.decision_matrix.alternatives[alt_index].name,
            "crit_index": crit_index,
            "crit_name": self.decision_matrix.criteria[crit_index].name,
            "original_value": float(original_value),
            "perturbation_factors": perturbation_factors.tolist(),
            "perturbed_values": perturbed_values,
            "results": results,
            "ranking_changes": ranking_changes
        }
    
    def plot_value_sensitivity(self, 
                            analysis_result: Dict[str, Any],
                            figsize: Tuple[int, int] = (10, 6)):
        """Plot the results of value sensitivity analysis.
        
        Args:
            analysis_result: Result from analyze_value_perturbation
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure object
        """
        values = analysis_result["perturbed_values"]
        factors = analysis_result["perturbation_factors"]
        changes = analysis_result["ranking_changes"]
        alt_name = analysis_result["alt_name"]
        crit_name = analysis_result["crit_name"]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot ranking changes
        ax.plot(factors, changes, marker='o')
        
        # Add secondary x-axis with actual values
        ax2 = ax.twiny()
        ax2.plot(values, changes, alpha=0)  # Invisible plot just to set the x-range
        
        # Labels and title
        ax.set_xlabel('Perturbation Factor')
        ax.set_ylabel('Number of Ranking Changes')
        ax2.set_xlabel('Perturbed Value')
        ax.set_title(f'Sensitivity to {alt_name} - {crit_name} Value Change')
        
        ax.grid(True)
        
        plt.tight_layout()
        return fig