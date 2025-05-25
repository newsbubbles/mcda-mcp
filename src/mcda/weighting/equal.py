"""Equal weighting method.

Implements a weighting method that assigns equal weight to all criteria.
"""

from typing import List, Optional
import numpy as np

from mcda.base import WeightingMethod
from mcda.models import DecisionMatrix, WeightingResult, Criterion


class EqualWeighting(WeightingMethod):
    """Equal weighting method that assigns equal weight to all criteria."""
    
    @property
    def name(self) -> str:
        return "Equal Weighting"
    
    def calculate_weights(self,
                          decision_matrix: Optional[DecisionMatrix] = None,
                          criteria: Optional[List[Criterion]] = None,
                          **kwargs) -> WeightingResult:
        """Calculate equal weights for all criteria.
        
        Args:
            decision_matrix: Optional decision matrix with criteria
            criteria: Optional list of criteria if decision matrix not provided
            
        Returns:
            WeightingResult with equal weights
        """
        # Get criteria from decision matrix if not provided
        if criteria is None and decision_matrix is not None:
            criteria = decision_matrix.criteria
        elif criteria is None:
            raise ValueError("Either criteria or a decision matrix must be provided")
        
        n_criteria = len(criteria)
        if n_criteria == 0:
            raise ValueError("No criteria provided")
        
        # Equal weights sum to 1
        equal_weight = 1.0 / n_criteria
        weights = [equal_weight] * n_criteria
        
        return WeightingResult(
            method_name=self.name,
            weights=weights,
            criteria=criteria
        )