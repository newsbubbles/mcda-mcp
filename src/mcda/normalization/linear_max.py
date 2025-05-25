"""Linear Max normalization method.

Implements normalization by dividing each value by the maximum value in its column.
"""

import numpy as np
from mcda.base import NormalizationMethod


class LinearMaxNormalization(NormalizationMethod):
    """Linear Max normalization method for MCDA.
    
    For benefit criteria, normalize using:
    r_ij = x_ij / max(x_j)
    
    For cost criteria, normalize using:
    r_ij = min(x_j) / x_ij
    """
    
    @property
    def name(self) -> str:
        return "Linear Max Normalization"
    
    def normalize(self, matrix: np.ndarray, criteria_types: np.ndarray) -> np.ndarray:
        """Normalize the decision matrix using linear max normalization.
        
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
            
            # Avoid division by zero
            if col_max == 0 and criteria_types[j] == 1:  # Benefit criterion with all zeros
                normalized[:, j] = 0
            elif col_min == 0 and criteria_types[j] == -1:  # Cost criterion with some zeros
                # Special handling for cost criteria with zero values
                # Assign 1 (best) to zeros and scale others appropriately
                for i in range(matrix.shape[0]):
                    if col[i] == 0:
                        normalized[i, j] = 1
                    else:
                        normalized[i, j] = col_min / col[i]
            else:
                if criteria_types[j] == 1:  # Benefit criterion
                    normalized[:, j] = col / col_max
                else:  # Cost criterion
                    normalized[:, j] = col_min / col
                
        return normalized