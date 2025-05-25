"""Weighted Sum Model (WSM) method.

Implements the WSM method for multi-criteria decision analysis.
"""

from typing import Optional
import numpy as np

from mcda.base import MCDAMethod, NormalizationMethod
from mcda.models import DecisionMatrix, MCDAResult
from mcda.normalization.linear_minmax import LinearMinMaxNormalization


class WSM(MCDAMethod):
    """WSM (Weighted Sum Model) method.
    
    Steps:
    1. Normalize the decision matrix
    2. Multiply each value by the corresponding criterion weight
    3. Calculate the sum of weighted normalized values for each alternative
    4. Rank the alternatives
    """
    
    def __init__(self, normalization_method: Optional[NormalizationMethod] = None):
        """Initialize WSM method.
        
        Args:
            normalization_method: Method to use for normalization (defaults to LinearMinMaxNormalization)
        """
        self._normalization = normalization_method or LinearMinMaxNormalization()
    
    @property
    def name(self) -> str:
        return "WSM"
    
    def evaluate(self,
                decision_matrix: DecisionMatrix,
                weights: Optional[np.ndarray] = None,
                **kwargs) -> MCDAResult:
        """Evaluate alternatives using WSM method.
        
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
        
        # Step 2 & 3: Multiply by weights and sum for each alternative
        wsm_scores = np.sum(normalized_matrix * weights, axis=1)
        
        # Step 4: Rank the alternatives (higher WSM score is better)
        ranks = np.argsort(-wsm_scores)  # Descending order
        rankings = np.argsort(ranks) + 1  # Convert to 1-based rankings
        
        return MCDAResult(
            method_name=self.name,
            preferences=wsm_scores.tolist(),
            rankings=rankings.tolist(),
            alternatives=decision_matrix.alternatives,
            additional_data={
                "normalized_matrix": normalized_matrix.tolist(),
                "normalization_method": self._normalization.name
            }
        )