"""Manual weighting method.

Implements a weighting method that simply uses user-provided weights.
"""

from typing import List, Optional
import numpy as np

from mcda.base import WeightingMethod
from mcda.models import DecisionMatrix, WeightingResult, Criterion


class ManualWeighting(WeightingMethod):
    """Manual weighting method that uses user-provided weights."""
    
    @property
    def name(self) -> str:
        return "Manual Weighting"
    
    def calculate_weights(self,
                         decision_matrix: Optional[DecisionMatrix] = None,
                         criteria: Optional[List[Criterion]] = None,
                         weights: Optional[List[float]] = None,
                         **kwargs) -> WeightingResult:
        """Use user-provided weights as criteria weights.
        
        Args:
            decision_matrix: Optional decision matrix with criteria
            criteria: Optional list of criteria if decision matrix not provided
            weights: List of weights to use
            
        Returns:
            WeightingResult with the provided weights
        """
        if weights is None:
            if decision_matrix is not None:
                # Try to get weights from decision matrix
                matrix_weights = decision_matrix.get_criteria_weights()
                if matrix_weights is not None:
                    weights = matrix_weights.tolist()
                else:
                    raise ValueError("No weights provided and decision matrix does not contain weights")
            else:
                raise ValueError("Either weights or a decision matrix with weights must be provided")
        
        # Get criteria from decision matrix if not provided
        if criteria is None and decision_matrix is not None:
            criteria = decision_matrix.criteria
        elif criteria is None:
            raise ValueError("Either criteria or a decision matrix must be provided")
        
        # Normalize weights to sum to 1
        weights_sum = sum(weights)
        if weights_sum == 0:
            raise ValueError("Sum of weights cannot be zero")
        
        normalized_weights = [w / weights_sum for w in weights]
        
        return WeightingResult(
            method_name=self.name,
            weights=normalized_weights,
            criteria=criteria
        )