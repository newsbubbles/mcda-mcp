"""Entropy weighting method.

Implements a weighting method based on Shannon's entropy concept to calculate objective weights.
"""

from typing import List, Optional, Dict, Any
import numpy as np

from mcda.base import WeightingMethod
from mcda.models import DecisionMatrix, WeightingResult, Criterion


class EntropyWeighting(WeightingMethod):
    """Entropy weighting method for MCDA.
    
    Calculates weights based on Shannon's entropy concept:
    - Values with high variability receive higher weights
    - Values with low variability receive lower weights
    """
    
    @property
    def name(self) -> str:
        return "Entropy Weighting"
    
    def calculate_weights(self,
                         decision_matrix: DecisionMatrix,
                         criteria: Optional[List[Criterion]] = None,
                         **kwargs) -> WeightingResult:
        """Calculate weights based on Shannon's entropy concept.
        
        Args:
            decision_matrix: Decision matrix with alternatives and criteria
            criteria: Optional list of criteria (ignored, uses decision_matrix.criteria)
            
        Returns:
            WeightingResult with entropy-based weights
        """
        if decision_matrix is None:
            raise ValueError("Decision matrix is required for entropy weighting")
            
        # Get data from decision matrix
        matrix = decision_matrix.to_numpy()
        criteria = decision_matrix.criteria
        criteria_types = decision_matrix.get_types_as_coefficients()
        
        # First, normalize the matrix using sum normalization
        normalized = np.zeros_like(matrix, dtype=float)
        for j in range(matrix.shape[1]):
            col = matrix[:, j]
            col_sum = np.sum(np.abs(col))  # Use absolute sum to handle negative values
            if col_sum == 0:  # Avoid division by zero
                normalized[:, j] = 0
            else:
                normalized[:, j] = col / col_sum
        
        # Calculate entropy for each criterion
        m = matrix.shape[0]  # Number of alternatives
        k = 1 / np.log(m)  # Normalization constant
        entropy = np.zeros(matrix.shape[1])
        
        for j in range(matrix.shape[1]):
            # Avoid log(0) for entropy calculation
            e_sum = 0
            for i in range(m):
                if normalized[i, j] > 0:  # Skip zero values to avoid log(0)
                    e_sum -= normalized[i, j] * np.log(normalized[i, j])
            entropy[j] = k * e_sum
        
        # Calculate degree of diversification
        diversification = 1 - entropy
        
        # Calculate weights
        div_sum = np.sum(diversification)
        if div_sum == 0:  # If diversification is zero, use equal weights
            weights = np.ones(matrix.shape[1]) / matrix.shape[1]
        else:
            weights = diversification / div_sum
        
        return WeightingResult(
            method_name=self.name,
            weights=weights.tolist(),
            criteria=criteria,
            additional_data={
                "entropy": entropy.tolist(),
                "diversification": diversification.tolist()
            }
        )