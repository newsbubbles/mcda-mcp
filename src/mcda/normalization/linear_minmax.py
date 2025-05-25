"""Linear Min-Max normalization method.

Implements normalization by scaling values to fall within the range [0,1].
"""

import numpy as np
from mcda.base import NormalizationMethod


class LinearMinMaxNormalization(NormalizationMethod):
    """Linear Min-Max normalization method for MCDA.
    
    For benefit criteria, normalize using:
    r_ij = (x_ij - min(x_j)) / (max(x_j) - min(x_j))
    
    For cost criteria, normalize using:
    r_ij = (max(x_j) - x_ij) / (max(x_j) - min(x_j))
    """
    
    @property
    def name(self) -> str:
        return "Linear Min-Max Normalization"
    
    def normalize(self, matrix: np.ndarray, criteria_types: np.ndarray) -> np.ndarray:
        """Normalize the decision matrix using linear min-max normalization.
        
        Args:
            matrix: Decision matrix as numpy array
            criteria_types: Array where 1 indicates benefit criterion, -1 indicates cost criterion
            
        Returns:
            Normalized decision matrix
        """
        normalized = np.zeros_like(matrix, dtype=float)
        
        for j in range(matrix.shape[1]):
            col = matrix[:, j]
            col_max = np.max(col)
            col_min = np.min(col)
            col_range = col_max - col_min
            
            if col_range == 0:  # All values are the same
                normalized[:, j] = 1 if criteria_types[j] == 1 else 0
            else:
                if criteria_types[j] == 1:  # Benefit criterion
                    normalized[:, j] = (col - col_min) / col_range
                else:  # Cost criterion
                    normalized[:, j] = (col_max - col) / col_range
                
        return normalized