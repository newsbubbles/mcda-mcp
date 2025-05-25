"""MCP Server for the Multi-Criteria Decision Analysis (MCDA) client.

This server provides tools for performing multi-criteria decision analysis through the
Model Context Protocol (MCP).
"""

import logging
import logging.handlers
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, List, Optional, Literal, Any, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator

from mcp.server.fastmcp import FastMCP, Context

from mcda.models import ( 
    DecisionMatrix, MCDAResult, WeightingResult, Criterion, Alternative, CriteriaType
)
from mcda.client import MCDAClient

# Set up logging
log_dir = os.path.join(os.path.dirname(__file__), "logs")
 os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "mcda_mcp.log")

logger = logging.getLogger("mcda_mcp")
logger.setLevel(logging.INFO)

# Create rotating file handler with automatic compression
handler = logging.handlers.RotatingFileHandler(
    log_file, maxBytes=10*1024*1024, backupCount=5
)
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(handler)

# Also log to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(console)

# Define BaseModels for requests and responses
class AlternativeModel(BaseModel):
    """Model for an alternative in the decision problem."""
    id: str = Field(..., description="Unique identifier for the alternative")
    name: str = Field(..., description="Name of the alternative")
    description: Optional[str] = Field(None, description="Optional description of the alternative")


class CriterionModel(BaseModel):
    """Model for a criterion in the decision problem."""
    id: str = Field(..., description="Unique identifier for the criterion")
    name: str = Field(..., description="Name of the criterion")
    description: Optional[str] = Field(None, description="Optional description of the criterion")
    type: Literal["benefit", "cost"] = Field(
        ..., description="Type of the criterion (benefit or cost)"
    )
    weight: Optional[float] = Field(None, description="Weight of the criterion (if known)")
    
    @field_validator('weight')
    def validate_weight(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Weight must be between 0 and 1")
        return v


class DecisionMatrixModel(BaseModel):
    """Model for a decision matrix."""
    alternatives: List[AlternativeModel] = Field(
        ..., description="List of alternatives"
    )
    criteria: List[CriterionModel] = Field(
        ..., description="List of criteria"
    )
    values: List[List[float]] = Field(
        ..., description="Performance values for alternatives against criteria"
    )
    
    @field_validator('values')
    def validate_values(cls, v, info):
        model_data = info.data
        if 'alternatives' in model_data and 'criteria' in model_data:
            n_alternatives = len(model_data['alternatives'])
            n_criteria = len(model_data['criteria'])
            
            # Check number of rows matches alternatives count
            if len(v) != n_alternatives:
                raise ValueError(f"Number of rows ({len(v)}) must match number of alternatives ({n_alternatives})")
            
            # Check each row has the correct number of columns
            for i, row in enumerate(v):
                if len(row) != n_criteria:
                    raise ValueError(f"Row {i} has {len(row)} values, but there are {n_criteria} criteria")
        
        return v


class EvaluateRequestModel(BaseModel):
    """Request model for evaluating alternatives using an MCDA method."""
    decision_matrix: DecisionMatrixModel = Field(
        ..., description="Decision matrix with alternatives, criteria, and values"
    )
    method: str = Field(
        ..., description="MCDA method to use (topsis, ahp, vikor, promethee, wsm, wpm, waspas)"
    )
    weights: Optional[List[float]] = Field(
        None, description="Optional criteria weights (overrides weights in decision matrix)"
    )
    method_params: Optional[Dict[str, Any]] = Field(
        None, description="Method-specific parameters"
    )


class CompareMethodsRequestModel(BaseModel):
    """Request model for comparing multiple MCDA methods."""
    decision_matrix: DecisionMatrixModel = Field(
        ..., description="Decision matrix with alternatives, criteria, and values"
    )
    methods: List[str] = Field(
        ..., description="List of MCDA methods to compare"
    )
    weights: Optional[List[float]] = Field(
        None, description="Optional criteria weights (overrides weights in decision matrix)"
    )
    method_params: Optional[Dict[str, Dict[str, Any]]] = Field(
        None, description="Method-specific parameters for each method"
    )


class CalculateWeightsRequestModel(BaseModel):
    """Request model for calculating criteria weights."""
    method: str = Field(
        ..., description="Weighting method to use (manual, ahp, entropy, equal)"
    )
    decision_matrix: Optional[DecisionMatrixModel] = Field(
        None, description="Decision matrix (required for some methods)"
    )
    criteria: Optional[List[CriterionModel]] = Field(
        None, description="List of criteria (if decision matrix not provided)"
    )
    method_params: Optional[Dict[str, Any]] = Field(
        None, description="Method-specific parameters"
    )


class AHPComparisonMatrixModel(BaseModel):
    """Model for AHP comparison matrix."""
    matrix: List[List[float]] = Field(
        ..., description="Pairwise comparison matrix"
    )
    
    @field_validator('matrix')
    def validate_matrix(cls, v):
        # Check it's square
        n = len(v)
        for row in v:
            if len(row) != n:
                raise ValueError("AHP comparison matrix must be square")
        
        # Check diagonal is all 1s
        for i in range(n):
            if v[i][i] != 1.0:
                raise ValueError("Diagonal elements in AHP comparison matrix must be 1.0")
        
        return v


class PerformanceScoreModel(BaseModel):
    """Performance score for an alternative."""
    alternative_id: str = Field(..., description="Alternative ID")
    alternative_name: str = Field(..., description="Alternative name")
    preference_score: float = Field(..., description="Preference score")
    rank: int = Field(..., description="Rank (1-based)")


class MCDAResultModel(BaseModel):
    """Result model for an MCDA evaluation."""
    method_name: str = Field(..., description="Name of the MCDA method used")
    scores: List[PerformanceScoreModel] = Field(..., description="Performance scores and rankings")
    additional_data: Optional[Dict[str, Any]] = Field(
        None, description="Additional method-specific data"
    )


class ComparisonResultModel(BaseModel):
    """Result model for method comparison."""
    results: Dict[str, MCDAResultModel] = Field(
        ..., description="Results from each method"
    )
    correlation_matrix: Dict[str, Dict[str, float]] = Field(
        ..., description="Correlation between rankings from different methods"
    )
    agreement_rate: float = Field(
        ..., description="Overall agreement rate between all methods"
    )


class WeightResultModel(BaseModel):
    """Result model for weight calculation."""
    method_name: str = Field(..., description="Name of the weighting method used")
    weights: List[Dict[str, Union[str, float]]] = Field(
        ..., description="Calculated weights with criteria information"
    )
    additional_data: Optional[Dict[str, Any]] = Field(
        None, description="Additional method-specific data"
    )


class AvailableMethodsModel(BaseModel):
    """Model for available methods in the MCDA client."""
    mcda_methods: List[str] = Field(..., description="Available MCDA methods")
    weighting_methods: List[str] = Field(..., description="Available weighting methods")
    normalization_methods: List[str] = Field(..., description="Available normalization methods")


class MCDAContext:
    """Context class for MCDA MCP server."""
    client: MCDAClient

    def __init__(self):
        self.client = MCDAClient()


@asynccontextmanager
async def mcda_lifespan(server: FastMCP) -> AsyncIterator[MCDAContext]:
    """Manage the lifecycle of the MCDA MCP server."""
    logger.info("Initializing MCDA MCP server...")
    context = MCDAContext()
    try:
        logger.info("MCDA MCP server started successfully")
        yield context
    finally:
        logger.info("Shutting down MCDA MCP server...")


# Create the MCP server with the lifespan manager
mcp = FastMCP("MCDA", lifespan=mcda_lifespan)


# Implement MCP tools that wrap the MCDA client functionality
@mcp.tool()
def get_available_methods(ctx: Context) -> AvailableMethodsModel:
    """Get all available MCDA methods, weighting methods, and normalization methods."""
    logger.info("Tool called: get_available_methods")
    try:
        mcda_client = ctx.request_context.lifespan_context.client
        methods = mcda_client.get_available_methods()
        return AvailableMethodsModel(**methods)
    except Exception as e:
        logger.error(f"Error in get_available_methods: {e}")
        raise


@mcp.tool()
def evaluate_mcda(request: EvaluateRequestModel, ctx: Context) -> MCDAResultModel:
    """Evaluate alternatives using the specified MCDA method."""
    logger.info(f"Tool called: evaluate_mcda with method {request.method}")
    try:
        mcda_client = ctx.request_context.lifespan_context.client
        
        # Convert request model to client model
        from mcda.client import (
            CreateAlternativeRequest, CreateCriterionRequest,
            CreateDecisionMatrixRequest, EvaluateRequest
        )
        
        # Convert the criterion type from string to enum
        criteria_requests = [
            CreateCriterionRequest(
                id=c.id,
                name=c.name,
                description=c.description,
                type=CriteriaType.BENEFIT if c.type == "benefit" else CriteriaType.COST,
                weight=c.weight
            ) for c in request.decision_matrix.criteria
        ]
        
        # Convert alternatives
        alt_requests = [
            CreateAlternativeRequest(
                id=a.id,
                name=a.name,
                description=a.description
            ) for a in request.decision_matrix.alternatives
        ]
        
        # Create decision matrix request
        dm_request = CreateDecisionMatrixRequest(
            alternatives=alt_requests,
            criteria=criteria_requests,
            values=request.decision_matrix.values
        )
        
        # Create evaluate request
        eval_request = EvaluateRequest(
            decision_matrix=dm_request,
            method=request.method,
            weights=request.weights,
            method_params=request.method_params
        )
        
        # Evaluate using the client
        result = mcda_client.evaluate(eval_request)
        
        # Convert result to response model
        scores = []
        for i, alt in enumerate(result.alternatives):
            scores.append(PerformanceScoreModel(
                alternative_id=alt.id,
                alternative_name=alt.name,
                preference_score=result.preferences[i],
                rank=result.rankings[i]
            ))
        
        # Return the result
        return MCDAResultModel(
            method_name=result.method_name,
            scores=scores,
            additional_data=result.additional_data
        )
    except Exception as e:
        logger.error(f"Error in evaluate_mcda: {e}")
        raise


@mcp.tool()
def compare_methods(request: CompareMethodsRequestModel, ctx: Context) -> ComparisonResultModel:
    """Compare multiple MCDA methods on the same decision problem."""
    logger.info(f"Tool called: compare_methods with methods {request.methods}")
    try:
        mcda_client = ctx.request_context.lifespan_context.client
        
        # Convert request model to client model
        from mcda.client import (
            CreateAlternativeRequest, CreateCriterionRequest,
            CreateDecisionMatrixRequest, CompareMethodsRequest
        )
        
        # Convert the criterion type from string to enum
        criteria_requests = [
            CreateCriterionRequest(
                id=c.id,
                name=c.name,
                description=c.description,
                type=CriteriaType.BENEFIT if c.type == "benefit" else CriteriaType.COST,
                weight=c.weight
            ) for c in request.decision_matrix.criteria
        ]
        
        # Convert alternatives
        alt_requests = [
            CreateAlternativeRequest(
                id=a.id,
                name=a.name,
                description=a.description
            ) for a in request.decision_matrix.alternatives
        ]
        
        # Create decision matrix request
        dm_request = CreateDecisionMatrixRequest(
            alternatives=alt_requests,
            criteria=criteria_requests,
            values=request.decision_matrix.values
        )
        
        # Create compare request
        compare_request = CompareMethodsRequest(
            decision_matrix=dm_request,
            methods=request.methods,
            weights=request.weights,
            method_params=request.method_params
        )
        
        # Perform comparison using the client
        result = mcda_client.compare_methods(compare_request)
        
        # Convert results to response models
        mcda_results = {}
        for method_name, res in result.results.items():
            scores = []
            for i, alt in enumerate(res.alternatives):
                scores.append(PerformanceScoreModel(
                    alternative_id=alt.id,
                    alternative_name=alt.name,
                    preference_score=res.preferences[i],
                    rank=res.rankings[i]
                ))
            
            mcda_results[method_name] = MCDAResultModel(
                method_name=res.method_name,
                scores=scores,
                additional_data=res.additional_data
            )
        
        # Return the comparison result
        return ComparisonResultModel(
            results=mcda_results,
            correlation_matrix=result.correlation_matrix,
            agreement_rate=result.agreement_rate
        )
    except Exception as e:
        logger.error(f"Error in compare_methods: {e}")
        raise


@mcp.tool()
def calculate_weights(request: CalculateWeightsRequestModel, ctx: Context) -> WeightResultModel:
    """Calculate weights for criteria using the specified weighting method."""
    logger.info(f"Tool called: calculate_weights with method {request.method}")
    try:
        mcda_client = ctx.request_context.lifespan_context.client
        
        # Convert request model to client model
        from mcda.client import (
            CreateAlternativeRequest, CreateCriterionRequest,
            CreateDecisionMatrixRequest, CalculateWeightsRequest
        )
        
        # Initialize variables that may be None
        dm_request = None
        criteria_list = None
        
        # Process decision matrix if provided
        if request.decision_matrix is not None:
            # Convert the criterion type from string to enum
            criteria_requests = [
                CreateCriterionRequest(
                    id=c.id,
                    name=c.name,
                    description=c.description,
                    type=CriteriaType.BENEFIT if c.type == "benefit" else CriteriaType.COST,
                    weight=c.weight
                ) for c in request.decision_matrix.criteria
            ]
            
            # Convert alternatives
            alt_requests = [
                CreateAlternativeRequest(
                    id=a.id,
                    name=a.name,
                    description=a.description
                ) for a in request.decision_matrix.alternatives
            ]
            
            # Create decision matrix request
            dm_request = CreateDecisionMatrixRequest(
                alternatives=alt_requests,
                criteria=criteria_requests,
                values=request.decision_matrix.values
            )
        
        # Process individual criteria if provided
        if request.criteria is not None:
            criteria_list = [
                CreateCriterionRequest(
                    id=c.id,
                    name=c.name,
                    description=c.description,
                    type=CriteriaType.BENEFIT if c.type == "benefit" else CriteriaType.COST,
                    weight=c.weight
                ) for c in request.criteria
            ]
        
        # Create weights request
        weights_request = CalculateWeightsRequest(
            method=request.method,
            decision_matrix=dm_request,
            criteria=criteria_list,
            method_params=request.method_params
        )
        
        # Calculate weights using the client
        result = mcda_client.calculate_weights(weights_request)
        
        # Format weights with criteria information
        weights_with_info = []
        for i, criterion in enumerate(result.criteria):
            weights_with_info.append({
                "id": criterion.id,
                "name": criterion.name,
                "weight": result.weights[i]
            })
        
        # Return the weights result
        return WeightResultModel(
            method_name=result.method_name,
            weights=weights_with_info,
            additional_data=result.additional_data
        )
    except Exception as e:
        logger.error(f"Error in calculate_weights: {e}")
        raise


# Additional specialized tools for common use cases
@mcp.tool()
def create_ahp_comparison_matrix(criteria_names: List[str]) -> AHPComparisonMatrixModel:
    """Create an empty AHP comparison matrix with diagonal filled with 1s."""
    logger.info(f"Tool called: create_ahp_comparison_matrix for {len(criteria_names)} criteria")
    try:
        n = len(criteria_names)
        if n < 2:
            raise ValueError("At least 2 criteria are needed for AHP comparison")
        
        # Create empty matrix with diagonals filled with 1s
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            matrix[i][i] = 1.0
        
        return AHPComparisonMatrixModel(matrix=matrix)
    except Exception as e:
        logger.error(f"Error in create_ahp_comparison_matrix: {e}")
        raise


@mcp.tool()
def validate_ahp_matrix(matrix: AHPComparisonMatrixModel) -> Dict[str, Any]:
    """Validate an AHP matrix for reciprocal values and check consistency ratio."""
    logger.info(f"Tool called: validate_ahp_matrix")
    try:
        n = len(matrix.matrix)
        np_matrix = np.array(matrix.matrix)
        
        # Check for reciprocal values
        is_reciprocal = True
        for i in range(n):
            for j in range(i+1, n):  # only upper triangle
                if abs(np_matrix[i][j] * np_matrix[j][i] - 1.0) > 1e-10:
                    is_reciprocal = False
        
        # Use AHPWeighting to check consistency
        from mcda.weighting.ahp import AHPWeighting
        ahp = AHPWeighting()
        
        # Generate generic criteria for the validation
        from mcda.models import Criterion
        criteria = [
            Criterion(id=f"C{i}", name=f"Criterion {i}", type=CriteriaType.BENEFIT) 
            for i in range(n)
        ]
        
        # calculate_weights will check consistency
        result = ahp.calculate_weights(
            criteria=criteria,
            comparison_matrix=np_matrix
        )
        
        return {
            "is_reciprocal": is_reciprocal,
            "is_consistent": result.additional_data.get("is_consistent", False),
            "consistency_ratio": result.additional_data.get("consistency_ratio", 1.0),
            "eigenvalues": result.additional_data.get("max_eigenvalue", None)
        }
    except Exception as e:
        logger.error(f"Error in validate_ahp_matrix: {e}")
        raise


def main():
    mcp.run()
    
if __name__ == "__main__":
    main()