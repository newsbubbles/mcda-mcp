"""Weighted Aggregated Sum Product Assessment (WASPAS) method.

Implements the WASPAS method for multi-criteria decision analysis.
"""

from typing import Optional
import numpy as np

from mcda.base import MCDAMethod, NormalizationMethod
from mcda.models import DecisionMatrix, MCDAResult
from mcda.normalization.linear_minmax import LinearMinMaxNormalization
from mcda.methods.wsm import WSM
from mcda.methods.wpm import WPM


class WASPAS(MCDAMethod):
    """WASPAS (Weighted Aggregated Sum Product Assessment) method.
    
    WASPAS combines the WSM and WPM methods with a weighted average:
    WASPAS = lambda * WSM + (1 - lambda) * WPM
    where lambda is a parameter between 0 and 1.
    """
    
    def __init__(self, 
                 lambda_param: float = 0.5, 
                 normalization_method: Optional[NormalizationMethod] = None):
        """Initialize WASPAS method.
        
        Args:
            lambda_param: Weight between WSM and WPM (default: 0.5)
            normalization_method: Method to use for normalization (defaults to LinearMinMaxNormalization)
        """
        if not 0 <= lambda_param <= 1:
            raise ValueError("lambda_param must be between 0 and 1")
            
        self.lambda_param = lambda_param
        self._normalization = normalization_method or LinearMinMaxNormalization()
        self._wsm = WSM(normalization_method=self._normalization)
        self._wpm = WPM(normalization_method=self._normalization)
    
    @property
    def name(self) -> str:
        return "WASPAS"
    
    def evaluate(self,
                decision_matrix: DecisionMatrix,
                weights: Optional[np.ndarray] = None,
                lambda_param: Optional[float] = None,
                **kwargs) -> MCDAResult:
        """Evaluate alternatives using WASPAS method.
        
        Args:
            decision_matrix: The decision matrix containing alternatives and criteria
            weights: Optional criteria weights (if not included in decision matrix)
            lambda_param: Weight between WSM and WPM (overrides instance lambda_param)
            
        Returns:
            MCDAResult containing preferences and rankings
        """
        # Get WSM and WPM results
        wsm_result = self._wsm.evaluate(decision_matrix, weights, **kwargs)
        wpm_result = self._wpm.evaluate(decision_matrix, weights, **kwargs)
        
        # Get preference scores from results
        wsm_scores = np.array(wsm_result.preferences)
        wpm_scores = np.array(wpm_result.preferences)
        
        # Use provided lambda or instance lambda
        lambda_val = lambda_param if lambda_param is not None else self.lambda_param
        
        # Combine WSM and WPM scores using lambda
        waspas_scores = lambda_val * wsm_scores + (1 - lambda_val) * wpm_scores
        
        # Rank the alternatives (higher WASPAS score is better)
        ranks = np.argsort(-waspas_scores)  # Descending order
        rankings = np.argsort(ranks) + 1  # Convert to 1-based rankings
        
        return MCDAResult(
            method_name=self.name,
            preferences=waspas_scores.tolist(),
            rankings=rankings.tolist(),
            alternatives=decision_matrix.alternatives,
            additional_data={
                "lambda": float(lambda_val),
                "wsm_scores": wsm_scores.tolist(),
                "wpm_scores": wpm_scores.tolist(),
                "wsm_rankings": wsm_result.rankings,
                "wpm_rankings": wpm_result.rankings,
                "normalization_method": self._normalization.name
            }
        )