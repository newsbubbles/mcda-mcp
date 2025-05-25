"""Vector normalization method.

Implements normalization by dividing each value by the square root of the sum of squared values in its column.
"""

import numpy as np
from mcda.base import NormalizationMethod


class VectorNormalization(NormalizationMethod):
    """Vector normalization method for MCDA.
    
    For each column j of the decision matrix, normalize values using:
    r_ij = x_ij / sqrt(sum(x_ij^2))
    """
    
    @property
    def name(self) -> str:
        return "Vector Normalization"
    
    def normalize(self, matrix: np.ndarray, criteria_types: np.ndarray) -> np.ndarray:
        """Normalize the decision matrix using vector normalization.
        
        Args:
            matrix: Decision matrix as numpy array
            criteria_types: Array where 1 indicates benefit criterion, -1 indicates cost criterion
            
        Returns:
            Normalized decision matrix
        """
        # Convert cost criteria to benefit criteria (multiply by -1)
        # This ensures higher values are always better after normalization
        adjusted_matrix = matrix.copy()
        for j in range(matrix.shape[1]):
            if criteria_types[j] == -1:  # cost criterion
                adjusted_matrix[:, j] = -1 * matrix[:, j]
        
        # Perform vector normalization
        normalized = np.zeros_like(adjusted_matrix, dtype=float)
        for j in range(adjusted_matrix.shape[1]):
            col = adjusted_matrix[:, j]
            denominator = np.sqrt(np.sum(col ** 2))
            if denominator == 0:
                normalized[:, j] = 0  # Avoid division by zero
            else:
                normalized[:, j] = col / denominator
                
        return normalized