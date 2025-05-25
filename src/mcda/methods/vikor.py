"""VIKOR method implementation.

Implements the VIKOR (VIseKriterijumska Optimizacija I Kompromisno Resenje) method for multi-criteria decision analysis.
"""

from typing import Optional, Dict, Any, List, Union
import numpy as np

from mcda.base import MCDAMethod, NormalizationMethod
from mcda.models import DecisionMatrix, MCDAResult


class VIKOR(MCDAMethod):
    """VIKOR (VIseKriterijumska Optimizacija I Kompromisno Resenje) method.
    
    Steps:
    1. Determine the best and worst values for each criterion
    2. Calculate the utility and regret measures
    3. Calculate the VIKOR index
    4. Rank the alternatives
    """
    
    def __init__(self, v: float = 0.5):
        """Initialize VIKOR method.
        
        Args:
            v: Weight of strategy of maximum group utility (default: 0.5)
        """
        if not 0 <= v <= 1:
            raise ValueError("Parameter v must be between 0 and 1")
        self.v = v
    
    @property
    def name(self) -> str:
        return "VIKOR"
    
    def evaluate(self,
                decision_matrix: DecisionMatrix,
                weights: Optional[np.ndarray] = None,
                v: Optional[float] = None,
                **kwargs) -> MCDAResult:
        """Evaluate alternatives using VIKOR method.
        
        Args:
            decision_matrix: The decision matrix containing alternatives and criteria
            weights: Optional criteria weights (if not included in decision matrix)
            v: Weight of strategy of maximum group utility (overrides instance v)
            
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
        
        # Use provided v or instance v
        v_value = v if v is not None else self.v
        
        # Step 1: Determine the best and worst values for each criterion
        f_star = np.zeros(matrix.shape[1])
        f_minus = np.zeros(matrix.shape[1])
        
        for j in range(matrix.shape[1]):
            if criteria_types[j] == 1:  # Benefit criterion
                f_star[j] = np.max(matrix[:, j])
                f_minus[j] = np.min(matrix[:, j])
            else:  # Cost criterion
                f_star[j] = np.min(matrix[:, j])
                f_minus[j] = np.max(matrix[:, j])
        
        # Step 2: Calculate the utility and regret measures
        S = np.zeros(matrix.shape[0])  # Group utility
        R = np.zeros(matrix.shape[0])  # Individual regret
        
        for i in range(matrix.shape[0]):
            S_values = np.zeros(matrix.shape[1])
            
            for j in range(matrix.shape[1]):
                # Skip if the range is zero (best and worst are the same)
                if f_star[j] == f_minus[j]:
                    S_values[j] = 0
                else:
                    # Normalized measure
                    if criteria_types[j] == 1:  # Benefit criterion
                        S_values[j] = weights[j] * (f_star[j] - matrix[i, j]) / (f_star[j] - f_minus[j])
                    else:  # Cost criterion
                        S_values[j] = weights[j] * (matrix[i, j] - f_star[j]) / (f_minus[j] - f_star[j])
            
            S[i] = np.sum(S_values)
            R[i] = np.max(S_values)
        
        # Step 3: Calculate the VIKOR index
        S_star = np.min(S)
        S_minus = np.max(S)
        R_star = np.min(R)
        R_minus = np.max(R)
        
        Q = np.zeros(matrix.shape[0])  # VIKOR index
        
        # Avoid division by zero
        S_range = S_minus - S_star
        R_range = R_minus - R_star
        
        for i in range(matrix.shape[0]):
            term1 = 0 if S_range == 0 else (v_value * (S[i] - S_star) / S_range)
            term2 = 0 if R_range == 0 else ((1 - v_value) * (R[i] - R_star) / R_range)
            Q[i] = term1 + term2
        
        # Step 4: Rank the alternatives (smaller Q value is better)
        # Using argsort directly to get ascending order
        ranks = np.argsort(Q)
        rankings = np.argsort(ranks) + 1  # Convert to 1-based rankings
        
        # Convert Q to a preference score where higher is better (for consistency with other methods)
        # Simply invert the Q values and normalize to [0, 1]
        Q_range = np.max(Q) - np.min(Q)
        preferences = 1 - (Q - np.min(Q)) / Q_range if Q_range > 0 else np.ones_like(Q)
        
        return MCDAResult(
            method_name=self.name,
            preferences=preferences.tolist(),
            rankings=rankings.tolist(),
            alternatives=decision_matrix.alternatives,
            additional_data={
                "v": float(v_value),
                "S": S.tolist(),  # Group utility
                "R": R.tolist(),  # Individual regret
                "Q": Q.tolist(),  # VIKOR index (original - lower is better)
                "f_star": f_star.tolist(),  # Best values
                "f_minus": f_minus.tolist()  # Worst values
            }
        )