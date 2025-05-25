"""MCDA Client API.

This module provides a unified client interface for multi-criteria decision analysis methods.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any, Type, Tuple
from pydantic import BaseModel, Field

from mcda.base import MCDAMethod, WeightingMethod, NormalizationMethod
from mcda.models import (
    DecisionMatrix, MCDAResult, WeightingResult, Criterion, Alternative, CriteriaType
)

# Import MCDA methods
from mcda.methods import (
    TOPSIS, AHP, VIKOR, PROMETHEE, PROMETHEE1, PROMETHEE2, PROMETHEE3, PROMETHEE4, PROMETHEE5, PROMETHEE6,
    WSM, WPM, WASPAS
)

# Import weighting methods
from mcda.weighting import (
    ManualWeighting, AHPWeighting, EntropyWeighting, EqualWeighting
)

# Import normalization methods
from mcda.normalization import (
    VectorNormalization, LinearMinMaxNormalization, LinearMaxNormalization, LinearSumNormalization
)


class CreateAlternativeRequest(BaseModel):
    """Request to create an alternative."""
    id: str = Field(..., description="Unique identifier for the alternative")
    name: str = Field(..., description="Name of the alternative")
    description: Optional[str] = Field(None, description="Optional description of the alternative")


class CreateCriterionRequest(BaseModel):
    """Request to create a criterion."""
    id: str = Field(..., description="Unique identifier for the criterion")
    name: str = Field(..., description="Name of the criterion")
    description: Optional[str] = Field(None, description="Optional description of the criterion")
    type: CriteriaType = Field(..., description="Type of the criterion (benefit or cost)")
    weight: Optional[float] = Field(None, description="Weight of the criterion (if known)")


class CreateDecisionMatrixRequest(BaseModel):
    """Request to create a decision matrix."""
    alternatives: List[CreateAlternativeRequest] = Field(..., description="List of alternatives")
    criteria: List[CreateCriterionRequest] = Field(..., description="List of criteria")
    values: List[List[float]] = Field(..., description="Performance values for alternatives against criteria")


class EvaluateRequest(BaseModel):
    """Request to evaluate alternatives using an MCDA method."""
    decision_matrix: CreateDecisionMatrixRequest = Field(..., description="Decision matrix")
    method: str = Field(..., description="MCDA method to use")
    weights: Optional[List[float]] = Field(None, description="Optional criteria weights")
    method_params: Optional[Dict[str, Any]] = Field(None, description="Method-specific parameters")


class CompareMethodsRequest(BaseModel):
    """Request to compare multiple MCDA methods."""
    decision_matrix: CreateDecisionMatrixRequest = Field(..., description="Decision matrix")
    methods: List[str] = Field(..., description="List of MCDA methods to compare")
    weights: Optional[List[float]] = Field(None, description="Optional criteria weights")
    method_params: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="Method-specific parameters")


class CalculateWeightsRequest(BaseModel):
    """Request to calculate criteria weights."""
    method: str = Field(..., description="Weighting method to use")
    decision_matrix: Optional[CreateDecisionMatrixRequest] = Field(None, description="Decision matrix (required for some methods)")
    criteria: Optional[List[CreateCriterionRequest]] = Field(None, description="List of criteria (if decision matrix not provided)")
    method_params: Optional[Dict[str, Any]] = Field(None, description="Method-specific parameters")


class ComparisonResult(BaseModel):
    """Result of comparing multiple MCDA methods."""
    results: Dict[str, MCDAResult] = Field(..., description="Results from each method")
    correlation_matrix: Dict[str, Dict[str, float]] = Field(..., description="Correlation between rankings from different methods")
    agreement_rate: float = Field(..., description="Overall agreement rate between all methods")


class MCDAClient:
    """Client API for Multi-Criteria Decision Analysis."""
    
    def __init__(self):
        """Initialize the MCDA client with available methods."""
        # Available MCDA methods
        self._mcda_methods: Dict[str, Type[MCDAMethod]] = {
            "topsis": TOPSIS,
            "ahp": AHP,
            "vikor": VIKOR,
            "promethee": PROMETHEE,        # Alias for PROMETHEE II
            "promethee1": PROMETHEE1,
            "promethee2": PROMETHEE2,
            "promethee3": PROMETHEE3,
            "promethee4": PROMETHEE4,
            "promethee5": PROMETHEE5,
            "promethee6": PROMETHEE6,
            "wsm": WSM,
            "wpm": WPM,
            "waspas": WASPAS
        }
        
        # Available weighting methods
        self._weighting_methods: Dict[str, Type[WeightingMethod]] = {
            "manual": ManualWeighting,
            "ahp": AHPWeighting,
            "entropy": EntropyWeighting,
            "equal": EqualWeighting
        }
        
        # Available normalization methods
        self._normalization_methods: Dict[str, Type[NormalizationMethod]] = {
            "vector": VectorNormalization,
            "linear_minmax": LinearMinMaxNormalization,
            "linear_max": LinearMaxNormalization,
            "linear_sum": LinearSumNormalization
        }
    
    def _convert_decision_matrix(self, request: CreateDecisionMatrixRequest) -> DecisionMatrix:
        """Convert request to DecisionMatrix model."""
        alternatives = [Alternative(**alt.model_dump()) for alt in request.alternatives]
        criteria = [Criterion(**crit.model_dump()) for crit in request.criteria]
        
        return DecisionMatrix(
            alternatives=alternatives,
            criteria=criteria,
            values=request.values
        )
    
    def evaluate(self, request: EvaluateRequest) -> MCDAResult:
        """Evaluate alternatives using the specified MCDA method.
        
        Args:
            request: EvaluateRequest containing decision matrix and method information
            
        Returns:
            MCDAResult with preferences and rankings
            
        Raises:
            ValueError: If the specified method is not available
        """
        # Check if the method is available
        method_name = request.method.lower()
        if method_name not in self._mcda_methods:
            raise ValueError(f"MCDA method '{method_name}' is not available. Available methods: {list(self._mcda_methods.keys())}")
        
        # Create method instance
        method_class = self._mcda_methods[method_name]
        
        # Extract normalization method if specified
        norm_method = None
        if request.method_params and "normalization_method" in request.method_params:
            norm_method_name = request.method_params["normalization_method"]
            if norm_method_name in self._normalization_methods:
                norm_method = self._normalization_methods[norm_method_name]()
                # Remove from params to avoid duplicate arguments
                request.method_params.pop("normalization_method")
        
        # Initialize method with appropriate parameters
        method_params = request.method_params or {}
        if norm_method is not None and method_name in ["topsis", "wsm", "wpm", "waspas"]:
            method = method_class(normalization_method=norm_method, **method_params)
        else:
            method = method_class(**method_params)
        
        # Convert decision matrix
        decision_matrix = self._convert_decision_matrix(request.decision_matrix)
        
        # Convert weights if provided
        weights = np.array(request.weights) if request.weights is not None else None
        
        # Evaluate and return result
        return method.evaluate(decision_matrix, weights)
    
    def compare_methods(self, request: CompareMethodsRequest) -> ComparisonResult:
        """Compare multiple MCDA methods on the same decision problem.
        
        Args:
            request: CompareMethodsRequest containing decision matrix and methods to compare
            
        Returns:
            ComparisonResult with results from each method and comparison metrics
        """
        # Convert decision matrix once
        decision_matrix = self._convert_decision_matrix(request.decision_matrix)
        
        # Convert weights if provided
        weights = np.array(request.weights) if request.weights is not None else None
        
        # Evaluate with each method
        results: Dict[str, MCDAResult] = {}
        
        for method_name in request.methods:
            method_name_lower = method_name.lower()
            if method_name_lower not in self._mcda_methods:
                raise ValueError(f"MCDA method '{method_name}' is not available. Available methods: {list(self._mcda_methods.keys())}")
            
            # Get method-specific parameters
            method_params = {}
            if request.method_params and method_name in request.method_params:
                method_params = request.method_params[method_name]
            
            # Create evaluate request for this method
            eval_request = EvaluateRequest(
                decision_matrix=request.decision_matrix,
                method=method_name,
                weights=request.weights,
                method_params=method_params
            )
            
            # Evaluate and store result
            results[method_name] = self.evaluate(eval_request)
        
        # Calculate correlation between rankings
        correlation_matrix = self._calculate_correlation_matrix(results)
        
        # Calculate overall agreement rate
        agreement_rate = self._calculate_agreement_rate(correlation_matrix)
        
        return ComparisonResult(
            results=results,
            correlation_matrix=correlation_matrix,
            agreement_rate=agreement_rate
        )
    
    def calculate_weights(self, request: CalculateWeightsRequest) -> WeightingResult:
        """Calculate criteria weights using the specified weighting method.
        
        Args:
            request: CalculateWeightsRequest containing weighting method and parameters
            
        Returns:
            WeightingResult with calculated weights
            
        Raises:
            ValueError: If the specified method is not available
        """
        # Check if the method is available
        method_name = request.method.lower()
        if method_name not in self._weighting_methods:
            raise ValueError(f"Weighting method '{method_name}' is not available. Available methods: {list(self._weighting_methods.keys())}")
        
        # Create method instance
        method = self._weighting_methods[method_name]()
        
        # Convert decision matrix if provided
        decision_matrix = None
        if request.decision_matrix is not None:
            decision_matrix = self._convert_decision_matrix(request.decision_matrix)
        
        # Convert criteria if provided
        criteria = None
        if request.criteria is not None:
            criteria = [Criterion(**crit.model_dump()) for crit in request.criteria]
        
        # Extract method-specific parameters
        method_params = request.method_params or {}
        
        # Calculate weights and return result
        return method.calculate_weights(decision_matrix, criteria, **method_params)
    
    def _calculate_correlation_matrix(self, results: Dict[str, MCDAResult]) -> Dict[str, Dict[str, float]]:
        """Calculate correlation between rankings from different methods.
        
        Uses Spearman's rank correlation coefficient.
        """
        methods = list(results.keys())
        correlation_matrix = {m: {n: 0.0 for n in methods} for m in methods}
        
        for i, method1 in enumerate(methods):
            for method2 in methods:
                if method1 == method2:
                    correlation_matrix[method1][method2] = 1.0  # Perfect correlation with itself
                else:
                    # Extract rankings
                    rankings1 = np.array(results[method1].rankings)
                    rankings2 = np.array(results[method2].rankings)
                    
                    # Calculate Spearman correlation
                    n = len(rankings1)
                    d2 = np.sum((rankings1 - rankings2) ** 2)
                    spearman_corr = 1 - (6 * d2) / (n * (n**2 - 1))
                    
                    correlation_matrix[method1][method2] = float(spearman_corr)
        
        return correlation_matrix
    
    def _calculate_agreement_rate(self, correlation_matrix: Dict[str, Dict[str, float]]) -> float:
        """Calculate overall agreement rate as the average of all pairwise correlations."""
        methods = list(correlation_matrix.keys())
        n_methods = len(methods)
        
        if n_methods <= 1:
            return 1.0  # Perfect agreement with a single method
        
        # Calculate sum of all pairwise correlations (excluding self-correlations)
        correlation_sum = 0.0
        count = 0
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i != j:  # Exclude self-correlations
                    correlation_sum += correlation_matrix[method1][method2]
                    count += 1
        
        # Average correlation
        return correlation_sum / count if count > 0 else 1.0
        
    def get_available_methods(self) -> Dict[str, List[str]]:
        """Get lists of available MCDA, weighting, and normalization methods."""
        return {
            "mcda_methods": list(self._mcda_methods.keys()),
            "weighting_methods": list(self._weighting_methods.keys()),
            "normalization_methods": list(self._normalization_methods.keys())
        }