"""Analytic Hierarchy Process (AHP) method.

Implements the AHP method for multi-criteria decision analysis.
"""

from typing import Optional, Dict, Any, List, Union, Tuple
import numpy as np

from mcda.base import MCDAMethod
from mcda.models import DecisionMatrix, MCDAResult, Criterion, Alternative
from mcda.weighting.ahp import AHPWeighting, SAATY_RI


class AHP(MCDAMethod):
    """AHP (Analytic Hierarchy Process) method.
    
    Steps:
    1. Create pairwise comparison matrices for criteria and alternatives
    2. Calculate criteria weights using AHP eigenvector method
    3. Calculate alternative priorities for each criterion
    4. Calculate overall priorities and rank alternatives
    """
    
    @property
    def name(self) -> str:
        return "AHP"
    
    def evaluate(self,
                decision_matrix: Optional[DecisionMatrix] = None,
                weights: Optional[np.ndarray] = None,
                criteria_comparisons: Optional[np.ndarray] = None,
                alternative_comparisons: Optional[List[np.ndarray]] = None,
                **kwargs) -> MCDAResult:
        """Evaluate alternatives using AHP method.
        
        Args:
            decision_matrix: The decision matrix (optional if all comparisons are provided)
            weights: Optional pre-computed criteria weights
            criteria_comparisons: Pairwise comparison matrix for criteria
            alternative_comparisons: List of pairwise comparison matrices for alternatives
                                     (one matrix per criterion)
            
        Returns:
            MCDAResult containing preferences and rankings
        """
        # There are two ways to use AHP:
        # 1. With decision matrix - will convert to pairwise comparisons automatically
        # 2. With explicit pairwise comparisons for criteria and alternatives
        
        if decision_matrix is None and alternative_comparisons is None:
            raise ValueError("Either decision_matrix or alternative_comparisons must be provided")
        
        criteria = None
        alternatives = None
        
        # If decision matrix is provided, extract criteria and alternatives
        if decision_matrix is not None:
            criteria = decision_matrix.criteria
            alternatives = decision_matrix.alternatives
        
        # Step 1: Get or calculate criteria weights
        if weights is not None:
            criteria_weights = weights
        elif criteria_comparisons is not None:
            # Use AHPWeighting to calculate weights
            ahp_weighting = AHPWeighting()
            weighting_result = ahp_weighting.calculate_weights(criteria=criteria, 
                                                               comparison_matrix=criteria_comparisons)
            criteria_weights = np.array(weighting_result.weights)
            
            # If decision matrix provided but not criteria, get from weighting result
            if criteria is None:
                criteria = weighting_result.criteria
        else:
            raise ValueError("Either weights or criteria_comparisons must be provided")
            
        # Step 2: Calculate alternative priorities for each criterion
        n_criteria = len(criteria_weights)
        
        # If alternative_comparisons is provided, use them directly
        if alternative_comparisons is not None:
            if len(alternative_comparisons) != n_criteria:
                raise ValueError("Number of alternative comparison matrices must match number of criteria")
                
            # Check that all matrices have the same dimension (number of alternatives)
            n_alternatives = alternative_comparisons[0].shape[0]
            for i, comp_matrix in enumerate(alternative_comparisons):
                if comp_matrix.shape != (n_alternatives, n_alternatives):
                    raise ValueError(f"Alternative comparison matrix for criterion {i+1} has wrong dimensions")
            
            # If alternatives not provided, create generic ones
            if alternatives is None:
                alternatives = [
                    Alternative(id=f"A{i+1}", name=f"Alternative {i+1}") 
                    for i in range(n_alternatives)
                ]
                
            # Calculate priorities for each criterion using the eigenvalue method
            alt_priorities = np.zeros((n_alternatives, n_criteria))
            alt_consistency = np.zeros(n_criteria)
            
            for i, comp_matrix in enumerate(alternative_comparisons):
                priorities, cr, _ = self._calculate_priorities(comp_matrix)
                alt_priorities[:, i] = priorities
                alt_consistency[i] = cr
        
        # If decision matrix is provided, convert to priorities
        elif decision_matrix is not None:
            matrix = decision_matrix.to_numpy()
            criteria_types = decision_matrix.get_types_as_coefficients()
            n_alternatives = matrix.shape[0]
            
            # Convert values to priorities using a simple normalization
            alt_priorities = np.zeros((n_alternatives, n_criteria))
            
            for j in range(n_criteria):
                col = matrix[:, j].copy()
                
                # For cost criteria, invert the values (higher is better)
                if criteria_types[j] == -1:  # cost criterion
                    # Avoid division by zero
                    min_val = np.min(col[col > 0]) if np.any(col > 0) else 1e-10
                    col = np.where(col > 0, min_val / col, 0)
                
                # Normalize to sum to 1 (simple normalization for priorities)
                col_sum = np.sum(col)
                if col_sum > 0:
                    alt_priorities[:, j] = col / col_sum
                else:
                    # If sum is zero, assign equal priorities
                    alt_priorities[:, j] = 1.0 / n_alternatives
            
            # No consistency issues with direct conversion
            alt_consistency = np.zeros(n_criteria)
        
        else:
            raise ValueError("Either decision_matrix or alternative_comparisons must be provided")
            
        # Step 3: Calculate overall priorities
        overall_priorities = np.sum(alt_priorities * criteria_weights, axis=1)
        
        # Step 4: Rank the alternatives
        ranks = np.argsort(-overall_priorities)  # Descending order
        rankings = np.argsort(ranks) + 1  # Convert to 1-based rankings
        
        return MCDAResult(
            method_name=self.name,
            preferences=overall_priorities.tolist(),
            rankings=rankings.tolist(),
            alternatives=alternatives,
            additional_data={
                "criteria_weights": criteria_weights.tolist(),
                "alternative_priorities": alt_priorities.tolist(),
                "alternative_consistency": alt_consistency.tolist(),
                "max_consistency_ratio": float(np.max(alt_consistency))
            }
        )
    
    def _calculate_priorities(self, matrix: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Calculate priorities using the eigenvalue method.
        
        This is similar to the implementation in AHPWeighting, but kept separate for clarity.
        
        Args:
            matrix: Pairwise comparison matrix
            
        Returns:
            Tuple containing:
            - Calculated priorities
            - Consistency ratio
            - Dictionary with additional consistency information
        """
        n = matrix.shape[0]
        
        # Calculate eigenvalues and eigenvectors
        try:
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
        except np.linalg.LinAlgError:
            # Fallback to geometric mean method if eigenvalue method fails
            return self._geometric_mean_priorities(matrix)
        
        # Find the index of the maximum eigenvalue
        max_idx = np.argmax(eigenvalues.real)
        max_eigenvalue = eigenvalues[max_idx].real
        
        # Extract the corresponding eigenvector and normalize
        priorities = eigenvectors[:, max_idx].real
        priorities = priorities / np.sum(priorities)
        
        # Calculate consistency index (CI) and consistency ratio (CR)
        ci = (max_eigenvalue - n) / (n - 1) if n > 1 else 0
        ri = SAATY_RI.get(n, 1.5)  # Default to 1.5 for n > 10
        cr = ci / ri if ri > 0 else 0
        
        consistency_info = {
            "max_eigenvalue": float(max_eigenvalue),
            "consistency_index": float(ci)
        }
        
        return priorities, cr, consistency_info
    
    def _geometric_mean_priorities(self, matrix: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Calculate priorities using the geometric mean method as a fallback.
        
        Args:
            matrix: Pairwise comparison matrix
            
        Returns:
            Tuple containing:
            - Calculated priorities
            - Consistency ratio
            - Dictionary with additional consistency information
        """
        n = matrix.shape[0]
        
        # Calculate geometric mean of each row
        geometric_means = np.zeros(n)
        for i in range(n):
            geometric_means[i] = np.prod(matrix[i, :]) ** (1/n)
        
        # Normalize to get priorities
        priorities = geometric_means / np.sum(geometric_means)
        
        # Calculate product of weight ratios and comparison values
        consistency_vector = np.zeros(n)
        for i in range(n):
            row_sum = 0
            for j in range(n):
                row_sum += matrix[i, j] * priorities[j]
            consistency_vector[i] = row_sum / priorities[i]
        
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
        
        return priorities, cr, consistency_info