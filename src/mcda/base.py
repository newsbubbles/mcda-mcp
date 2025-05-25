"""Base classes for MCDA components.

This module contains abstract base classes for MCDA methods, weighting methods, and normalization techniques.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

from mcda.models import DecisionMatrix, MCDAResult, WeightingResult, Criterion


class MCDAMethod(ABC):
    """Abstract base class for MCDA methods."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the MCDA method."""
        pass
    
    @abstractmethod
    def evaluate(self, 
                decision_matrix: DecisionMatrix,
                weights: Optional[np.ndarray] = None,
                **kwargs) -> MCDAResult:
        """Evaluate alternatives using the MCDA method.
        
        Args:
            decision_matrix: The decision matrix containing alternatives and criteria
            weights: Optional numpy array of criteria weights (if not included in the decision matrix)
            **kwargs: Additional method-specific parameters
            
        Returns:
            MCDAResult containing preferences and rankings
        """
        pass


class WeightingMethod(ABC):
    """Abstract base class for weighting methods."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the weighting method."""
        pass
    
    @abstractmethod
    def calculate_weights(self,
                         decision_matrix: Optional[DecisionMatrix] = None,
                         criteria: Optional[List[Criterion]] = None,
                         **kwargs) -> WeightingResult:
        """Calculate criteria weights.
        
        Args:
            decision_matrix: Optional decision matrix (needed for some methods)
            criteria: Optional list of criteria (if decision matrix not provided)
            **kwargs: Additional method-specific parameters
            
        Returns:
            WeightingResult containing calculated weights
        """
        pass


class NormalizationMethod(ABC):
    """Abstract base class for normalization methods."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the normalization method."""
        pass
    
    @abstractmethod
    def normalize(self, 
                 matrix: np.ndarray,
                 criteria_types: np.ndarray) -> np.ndarray:
        """Normalize the decision matrix.
        
        Args:
            matrix: Decision matrix as numpy array
            criteria_types: Array where 1 indicates benefit criterion, -1 indicates cost criterion
            
        Returns:
            Normalized decision matrix
        """
        pass