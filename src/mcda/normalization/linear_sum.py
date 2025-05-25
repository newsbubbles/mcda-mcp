"""Linear Sum normalization method.

Implements normalization by dividing each value by the sum of values in its column.
"""

import numpy as np
from mcda.base import NormalizationMethod


class LinearSumNormalization(NormalizationMethod):
    """Linear Sum normalization method for MCDA.
    
    For benefit criteria, normalize using:
    r_ij = x_ij / sum(x_j)
    
    For cost criteria, normalize using:
    r_ij = (1/x_ij) / sum(1/x_j)
    """
    
    @property
    def name(self) -> str:
        return "Linear Sum Normalization"
    
    def normalize(self, matrix: np.ndarray, criteria_types: np.ndarray) -> np.ndarray:
        """Normalize the decision matrix using linear sum normalization.
        
        Args:
            matrix: Decision matrix as numpy array
            criteria_types: Array where 1 indicates benefit criterion, -1 indicates cost criterion
            
        Returns:
            Normalized decision matrix
        """
        normalized = np.zeros_like(matrix, dtype=float)
        
        for j in range(matrix.shape[1]):
            col = matrix[:, j]
            
            if criteria_types[j] == 1:  # Benefit criterion
                col_sum = np.sum(col)
                if col_sum == 0:  # Avoid division by zero
                    normalized[:, j] = 0
                else:
                    normalized[:, j] = col / col_sum
            else:  # Cost criterion
                # For cost criteria, we first invert the values (1/x) and then normalize
                # But we need to handle potential zeros
                inv_col = np.zeros_like(col)
                for i in range(len(col)):
                    if col[i] == 0:  # Avoid division by zero
                        # Assign a very large number instead of infinity
                        inv_col[i] = 1e10  # This is a simplification
                    else:
                        inv_col[i] = 1 / col[i]
                
                inv_sum = np.sum(inv_col)
                if inv_sum == 0:  # Avoid division by zero
                    normalized[:, j] = 0
                else:
                    normalized[:, j] = inv_col / inv_sum
                
        return normalized