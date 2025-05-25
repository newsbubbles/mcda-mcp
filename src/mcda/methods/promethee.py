"""PROMETHEE method implementation.

Implements the PROMETHEE II (Preference Ranking Organization METHod for Enrichment of Evaluations) method for multi-criteria decision analysis.
"""

from typing import Optional, Dict, Any, List, Union, Callable, Tuple
import numpy as np
import enum

from mcda.base import MCDAMethod
from mcda.models import DecisionMatrix, MCDAResult, Alternative, Criterion, CriteriaType


class PreferenceFunction(enum.Enum):
    """Preference function types for PROMETHEE."""
    USUAL = "usual"
    U_SHAPE = "u-shape"
    V_SHAPE = "v-shape"
    LEVEL = "level"
    LINEAR = "linear"
    GAUSSIAN = "gaussian"


class PROMETHEE(MCDAMethod):
    """PROMETHEE II (Preference Ranking Organization METHod for Enrichment of Evaluations) method.
    
    Steps:
    1. Calculate preference values using preference functions
    2. Calculate aggregated preference indices
    3. Calculate positive and negative outranking flows
    4. Calculate net outranking flow and rank alternatives
    """
    
    @property
    def name(self) -> str:
        return "PROMETHEE II"
    
    def _usual_function(self, d: float, p: Optional[float] = None, q: Optional[float] = None) -> float:
        """Usual preference function.
        
        Args:
            d: Difference between alternatives
            p: Preference threshold (not used)
            q: Indifference threshold (not used)
            
        Returns:
            0 if d <= 0, 1 otherwise
        """
        return 0 if d <= 0 else 1
    
    def _u_shape_function(self, d: float, p: Optional[float] = None, q: float = 0) -> float:
        """U-Shape preference function.
        
        Args:
            d: Difference between alternatives
            p: Preference threshold (not used)
            q: Indifference threshold
            
        Returns:
            0 if d <= q, 1 otherwise
        """
        return 0 if d <= q else 1
    
    def _v_shape_function(self, d: float, p: float = 1, q: Optional[float] = None) -> float:
        """V-Shape preference function.
        
        Args:
            d: Difference between alternatives
            p: Preference threshold
            q: Indifference threshold (not used)
            
        Returns:
            0 if d <= 0, d/p if 0 < d <= p, 1 otherwise
        """
        if d <= 0:
            return 0
        elif d > p:
            return 1
        else:
            return d / p
    
    def _level_function(self, d: float, p: float = 1, q: float = 0) -> float:
        """Level preference function.
        
        Args:
            d: Difference between alternatives
            p: Preference threshold
            q: Indifference threshold
            
        Returns:
            0 if d <= q, 0.5 if q < d <= p, 1 otherwise
        """
        if d <= q:
            return 0
        elif d > p:
            return 1
        else:
            return 0.5
    
    def _linear_function(self, d: float, p: float = 1, q: float = 0) -> float:
        """Linear preference function.
        
        Args:
            d: Difference between alternatives
            p: Preference threshold
            q: Indifference threshold
            
        Returns:
            0 if d <= q, (d-q)/(p-q) if q < d <= p, 1 otherwise
        """
        if d <= q:
            return 0
        elif d > p:
            return 1
        else:
            return (d - q) / (p - q)
    
    def _gaussian_function(self, d: float, p: float = 1, q: Optional[float] = None) -> float:
        """Gaussian preference function.
        
        Args:
            d: Difference between alternatives
            p: Preference threshold (used as sigma)
            q: Indifference threshold (not used)
            
        Returns:
            0 if d <= 0, 1 - exp(-(d^2)/(2*p^2)) otherwise
        """
        if d <= 0:
            return 0
        else:
            return 1 - np.exp(-(d ** 2) / (2 * p ** 2))
    
    def _get_preference_function(self, function_type: PreferenceFunction) -> Callable[[float, Optional[float], Optional[float]], float]:
        """Get the appropriate preference function based on type."""
        functions = {
            PreferenceFunction.USUAL: self._usual_function,
            PreferenceFunction.U_SHAPE: self._u_shape_function,
            PreferenceFunction.V_SHAPE: self._v_shape_function,
            PreferenceFunction.LEVEL: self._level_function,
            PreferenceFunction.LINEAR: self._linear_function,
            PreferenceFunction.GAUSSIAN: self._gaussian_function
        }
        return functions[function_type]
    
    def evaluate(self,
                decision_matrix: DecisionMatrix,
                weights: Optional[np.ndarray] = None,
                preference_functions: Optional[List[PreferenceFunction]] = None,
                p_thresholds: Optional[List[float]] = None,
                q_thresholds: Optional[List[float]] = None,
                **kwargs) -> MCDAResult:
        """Evaluate alternatives using PROMETHEE II method.
        
        Args:
            decision_matrix: The decision matrix containing alternatives and criteria
            weights: Optional criteria weights (if not included in decision matrix)
            preference_functions: List of preference functions for each criterion
            p_thresholds: List of preference thresholds for each criterion
            q_thresholds: List of indifference thresholds for each criterion
            
        Returns:
            MCDAResult containing preferences (net flow) and rankings
        """
        # Get data from decision matrix
        matrix = decision_matrix.to_numpy()
        criteria_types = decision_matrix.get_types_as_coefficients()
        
        # Use provided weights or get from decision matrix
        if weights is None:
            weights = decision_matrix.get_criteria_weights()
            if weights is None:
                raise ValueError("Criteria weights must be provided or included in decision matrix")
        
        # Default preference functions and thresholds if not provided
        n_criteria = matrix.shape[1]
        
        if preference_functions is None:
            # Default to usual function for all criteria
            preference_functions = [PreferenceFunction.USUAL] * n_criteria
        elif len(preference_functions) != n_criteria:
            raise ValueError("Number of preference functions must match number of criteria")
            
        if p_thresholds is None:
            # Default p-threshold: 20% of the range of each criterion
            p_thresholds = []
            for j in range(n_criteria):
                col = matrix[:, j]
                col_range = np.max(col) - np.min(col)
                p_thresholds.append(0.2 * col_range if col_range > 0 else 1.0)
        elif len(p_thresholds) != n_criteria:
            raise ValueError("Number of p-thresholds must match number of criteria")
            
        if q_thresholds is None:
            # Default q-threshold: 5% of the range of each criterion
            q_thresholds = []
            for j in range(n_criteria):
                col = matrix[:, j]
                col_range = np.max(col) - np.min(col)
                q_thresholds.append(0.05 * col_range if col_range > 0 else 0.0)
        elif len(q_thresholds) != n_criteria:
            raise ValueError("Number of q-thresholds must match number of criteria")
            
        # Step 1: Calculate preference values
        n_alternatives = matrix.shape[0]
        preference_matrix = np.zeros((n_alternatives, n_alternatives, n_criteria))
        
        for i in range(n_alternatives):
            for j in range(n_alternatives):
                if i != j:  # Skip comparison with itself
                    for k in range(n_criteria):
                        # Calculate difference (adjusted for criteria type)
                        diff = (matrix[i, k] - matrix[j, k]) * criteria_types[k]
                        
                        # Apply preference function
                        pref_func = self._get_preference_function(preference_functions[k])
                        preference_matrix[i, j, k] = pref_func(diff, p_thresholds[k], q_thresholds[k])
        
        # Step 2: Calculate aggregated preference indices
        preference_indices = np.zeros((n_alternatives, n_alternatives))
        
        for i in range(n_alternatives):
            for j in range(n_alternatives):
                if i != j:
                    # Weighted sum of preference values
                    preference_indices[i, j] = np.sum(preference_matrix[i, j, :] * weights)
        
        # Step 3: Calculate positive and negative outranking flows
        positive_flow = np.sum(preference_indices, axis=1) / (n_alternatives - 1)  # Leaving flow
        negative_flow = np.sum(preference_indices, axis=0) / (n_alternatives - 1)  # Entering flow
        
        # Step 4: Calculate net outranking flow and rank alternatives
        net_flow = positive_flow - negative_flow
        
        # Rank the alternatives (higher net flow is better)
        ranks = np.argsort(-net_flow)  # Descending order
        rankings = np.argsort(ranks) + 1  # Convert to 1-based rankings
        
        return MCDAResult(
            method_name=self.name,
            preferences=net_flow.tolist(),
            rankings=rankings.tolist(),
            alternatives=decision_matrix.alternatives,
            additional_data={
                "positive_flow": positive_flow.tolist(),
                "negative_flow": negative_flow.tolist(),
                "preference_functions": [pf.value for pf in preference_functions],
                "p_thresholds": p_thresholds,
                "q_thresholds": q_thresholds
            }
        )