"""Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS) method.

Implements the TOPSIS method for multi-criteria decision analysis.
"""

from typing import Optional, Dict, Any, Type, Union
import numpy as np

from mcda.base import MCDAMethod, NormalizationMethod
from mcda.models import DecisionMatrix, MCDAResult
from mcda.normalization.vector import VectorNormalization


class TOPSIS(MCDAMethod):
    """TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) method.
    
    Steps:
    1. Normalize the decision matrix
    2. Calculate the weighted normalized decision matrix
    3. Determine the ideal and negative-ideal solutions
    4. Calculate separation measures from ideal and negative-ideal solutions
    5. Calculate the relative closeness to the ideal solution
    6. Rank the alternatives
    """
    
    def __init__(self, normalization_method: Optional[NormalizationMethod] = None):
        """Initialize TOPSIS method.
        
        Args:
            normalization_method: Method to use for normalization (defaults to VectorNormalization)
        """
        self._normalization = normalization_method or VectorNormalization()
    
    @property
    def name(self) -> str:
        return "TOPSIS"
    
    def evaluate(self,
                decision_matrix: DecisionMatrix,
                weights: Optional[np.ndarray] = None,
                **kwargs) -> MCDAResult:
        """Evaluate alternatives using TOPSIS method.
        
        Args:
            decision_matrix: The decision matrix containing alternatives and criteria
            weights: Optional criteria weights (if not included in decision matrix)
            
        Returns:
            MCDAResult containing preferences and rankings
        """
        # Get data from decision matrix
        matrix = decision_matrix.to_numpy()
        criteria_types = decision_matrix.get_types_as_coefficients()
        
        # Use provided weights or get from decision matrix
        if weights is None:
            weights = decision_matrix.get_criteria_weights()
            if weights is None:
                raise ValueError("Criteria weights must be provided or included in decision matrix")
        
        # Step 1: Normalize the decision matrix
        normalized_matrix = self._normalization.normalize(matrix, criteria_types)
        
        # Step 2: Calculate the weighted normalized decision matrix
        weighted_matrix = normalized_matrix * weights
        
        # Step 3: Determine the ideal and negative-ideal solutions
        ideal_solution = np.max(weighted_matrix, axis=0)
        negative_ideal_solution = np.min(weighted_matrix, axis=0)
        
        # Step 4: Calculate separation measures
        separation_ideal = np.sqrt(np.sum((weighted_matrix - ideal_solution) ** 2, axis=1))
        separation_negative = np.sqrt(np.sum((weighted_matrix - negative_ideal_solution) ** 2, axis=1))
        
        # Step 5: Calculate the relative closeness to the ideal solution
        relative_closeness = separation_negative / (separation_ideal + separation_negative)
        
        # Handle division by zero cases
        relative_closeness = np.nan_to_num(relative_closeness, nan=0.0)
        
        # Step 6: Rank the alternatives (larger relative closeness is better)
        # Using argsort of negative values to get descending order
        ranks = np.argsort(-relative_closeness)
        rankings = np.argsort(ranks) + 1  # Convert to 1-based rankings
        
        return MCDAResult(
            method_name=self.name,
            preferences=relative_closeness.tolist(),
            rankings=rankings.tolist(),
            alternatives=decision_matrix.alternatives,
            additional_data={
                "normalized_matrix": normalized_matrix.tolist(),
                "weighted_matrix": weighted_matrix.tolist(),
                "ideal_solution": ideal_solution.tolist(),
                "negative_ideal_solution": negative_ideal_solution.tolist(),
                "separation_ideal": separation_ideal.tolist(),
                "separation_negative": separation_negative.tolist(),
                "normalization_method": self._normalization.name
            }
        )