"""Core data models and types for MCDA.

This module contains the base data structures used throughout the MCDA library.
"""

from enum import Enum
from typing import Dict, List, Optional, Union, Any
import numpy as np
from pydantic import BaseModel, Field


class CriteriaType(str, Enum):
    """Enum representing the type of criteria in MCDA."""
    BENEFIT = "benefit"  # Higher values are better
    COST = "cost"  # Lower values are better


class Alternative(BaseModel):
    """Model representing an alternative in the decision problem."""
    id: str = Field(..., description="Unique identifier for the alternative")
    name: str = Field(..., description="Name of the alternative")
    description: Optional[str] = Field(None, description="Optional description of the alternative")


class Criterion(BaseModel):
    """Model representing a criterion in the decision problem."""
    id: str = Field(..., description="Unique identifier for the criterion")
    name: str = Field(..., description="Name of the criterion")
    description: Optional[str] = Field(None, description="Optional description of the criterion")
    type: CriteriaType = Field(..., description="Type of the criterion (benefit or cost)")
    weight: Optional[float] = Field(None, description="Weight of the criterion (if known)")


class DecisionMatrix(BaseModel):
    """Model representing a decision matrix for MCDA."""
    alternatives: List[Alternative] = Field(..., description="List of alternatives")
    criteria: List[Criterion] = Field(..., description="List of criteria")
    values: List[List[float]] = Field(..., description="Performance values for alternatives against criteria")

    def to_numpy(self) -> np.ndarray:
        """Convert the decision matrix values to a numpy array."""
        return np.array(self.values)

    def get_criteria_types(self) -> List[CriteriaType]:
        """Get the types of criteria as a list."""
        return [criterion.type for criterion in self.criteria]
    
    def get_criteria_weights(self) -> Optional[np.ndarray]:
        """Get criteria weights as a numpy array if all weights are defined."""
        weights = [criterion.weight for criterion in self.criteria]
        if all(w is not None for w in weights):
            return np.array(weights)
        return None

    def get_types_as_coefficients(self) -> np.ndarray:
        """Convert criteria types to 1 (benefit) or -1 (cost) coefficients."""
        return np.array([1 if ctype == CriteriaType.BENEFIT else -1 for ctype in self.get_criteria_types()])


class MCDAResult(BaseModel):
    """Model representing the results of an MCDA method."""
    method_name: str = Field(..., description="Name of the MCDA method used")
    preferences: List[float] = Field(..., description="Preference values for alternatives")
    rankings: List[int] = Field(..., description="Rankings of alternatives (1-based)")
    alternatives: List[Alternative] = Field(..., description="List of alternatives")
    additional_data: Optional[Dict[str, Any]] = Field(None, description="Additional method-specific data")


class WeightingResult(BaseModel):
    """Model representing the results of a weighting method."""
    method_name: str = Field(..., description="Name of the weighting method used")
    weights: List[float] = Field(..., description="Weights for criteria")
    criteria: List[Criterion] = Field(..., description="List of criteria")
    additional_data: Optional[Dict[str, Any]] = Field(None, description="Additional method-specific data")
