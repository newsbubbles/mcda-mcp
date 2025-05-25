"""Analytic Hierarchy Process (AHP) weighting method.

Implements the AHP method for deriving criteria weights from pairwise comparisons.
"""

from typing import List, Optional, Dict, Any, Union, Tuple
import numpy as np

from mcda.base import WeightingMethod
from mcda.models import DecisionMatrix, WeightingResult, Criterion


# Saaty's scale for consistency index
SAATY_RI = {
    1: 0.0,
    2: 0.0,
    3: 0.58,
    4: 0.9,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
}


class AHPWeighting(WeightingMethod):
    """Analytic Hierarchy Process (AHP) weighting method for MCDA.
    
    Calculates weights based on pairwise comparisons of criteria using the eigenvalue method.
    """
    
    @property
    def name(self) -> str:
        return "AHP Weighting"
    
    def calculate_weights(self,
                         decision_matrix: Optional[DecisionMatrix] = None,
                         criteria: Optional[List[Criterion]] = None,
                         comparison_matrix: Optional[Union[List[List[float]], np.ndarray]] = None,
                         **kwargs) -> WeightingResult:
        """Calculate criteria weights using AHP method.
        
        Args:
            decision_matrix: Optional decision matrix with criteria
            criteria: Optional list of criteria if decision matrix not provided
            comparison_matrix: Pairwise comparison matrix
            
        Returns:
            WeightingResult with AHP-derived weights and consistency information
        """
        if comparison_matrix is None:
            raise ValueError("Comparison matrix is required for AHP weighting")
            
        # Convert comparison matrix to numpy array if needed
        if not isinstance(comparison_matrix, np.ndarray):
            comparison_matrix = np.array(comparison_matrix)
            
        # Get criteria from decision matrix if not provided
        if criteria is None and decision_matrix is not None:
            criteria = decision_matrix.criteria
        elif criteria is None:
            # Create generic criteria for the number of columns in comparison matrix
            n_criteria = comparison_matrix.shape[0]
            criteria = [
                Criterion(id=f"C{i+1}", name=f"Criterion {i+1}", 
                         type="benefit")  # Default to benefit criteria
                for i in range(n_criteria)
            ]
        
        # Check if comparison matrix is square
        n = len(criteria)
        if comparison_matrix.shape != (n, n):
            raise ValueError(f"Comparison matrix must be a {n}x{n} square matrix")
            
        # Calculate weights using the eigenvalue method
        weights, consistency_ratio, consistency_info = self._eigenvalue_method(comparison_matrix)
        
        return WeightingResult(
            method_name=self.name,
            weights=weights.tolist(),
            criteria=criteria,
            additional_data={
                "consistency_ratio": float(consistency_ratio),
                "is_consistent": bool(consistency_ratio <= 0.1),
                **consistency_info
            }
        )
    
    def _eigenvalue_method(self, matrix: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Calculate weights using the eigenvalue method.
        
        Args:
            matrix: Pairwise comparison matrix
            
        Returns:
            Tuple containing:
            - Calculated weights
            - Consistency ratio
            - Dictionary with additional consistency information
        """
        n = matrix.shape[0]
        
        # Calculate eigenvalues and eigenvectors
        try:
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
        except np.linalg.LinAlgError:
            # Fallback to geometric mean method if eigenvalue method fails
            return self._geometric_mean_method(matrix)
        
        # Find the index of the maximum eigenvalue
        max_idx = np.argmax(eigenvalues.real)
        max_eigenvalue = eigenvalues[max_idx].real
        
        # Extract the corresponding eigenvector and normalize
        weights = eigenvectors[:, max_idx].real
        weights = weights / np.sum(weights)
        
        # Calculate consistency index (CI) and consistency ratio (CR)
        ci = (max_eigenvalue - n) / (n - 1) if n > 1 else 0
        ri = SAATY_RI.get(n, 1.5)  # Default to 1.5 for n > 10
        cr = ci / ri if ri > 0 else 0
        
        consistency_info = {
            "max_eigenvalue": float(max_eigenvalue),
            "consistency_index": float(ci)
        }
        
        return weights, cr, consistency_info
    
    def _geometric_mean_method(self, matrix: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Calculate weights using the geometric mean method as a fallback.
        
        Args:
            matrix: Pairwise comparison matrix
            
        Returns:
            Tuple containing:
            - Calculated weights
            - Consistency ratio
            - Dictionary with additional consistency information
        """
        n = matrix.shape[0]
        
        # Calculate geometric mean of each row
        geometric_means = np.zeros(n)
        for i in range(n):
            geometric_means[i] = np.prod(matrix[i, :]) ** (1/n)
        
        # Normalize to get weights
        weights = geometric_means / np.sum(geometric_means)
        
        # Calculate product of weight ratios and comparison values
        consistency_vector = np.zeros(n)
        for i in range(n):
            row_sum = 0
            for j in range(n):
                row_sum += matrix[i, j] * weights[j]
            consistency_vector[i] = row_sum / weights[i]
        
        # Estimate max eigenvalue as average of consistency vector
        max_eigenvalue = np.mean(consistency_vector)
        
        # Calculate consistency index (CI) and consistency ratio (CR)
        ci = (max_eigenvalue - n) / (n - 1) if n > 1 else 0
        ri = SAATY_RI.get(n, 1.5)  # Default to 1.5 for n > 10
        cr = ci / ri if ri > 0 else 0
        
        consistency_info = {
            "max_eigenvalue": float(max_eigenvalue),
            "consistency_index": float(ci),
            "method": "geometric_mean"  # Indicate we used the fallback method
        }
        
        return weights, cr, consistency_info