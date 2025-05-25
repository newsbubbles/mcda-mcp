"""Example usage of the MCDA client API.

This script demonstrates how to use the MCDA client for a simple decision problem.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mcda.client import MCDAClient, CreateAlternativeRequest, CreateCriterionRequest, CreateDecisionMatrixRequest, EvaluateRequest, CompareMethodsRequest, CalculateWeightsRequest
from src.mcda.models import CriteriaType
from src.mcda.visualization import plot_preferences, plot_ranking_comparison, plot_correlation_matrix, plot_criteria_weights
from src.mcda.sensitivity import WeightSensitivityAnalysis, ValueSensitivityAnalysis

# Create MCDA client
client = MCDAClient()

# Print available methods
available_methods = client.get_available_methods()
print("Available MCDA methods:", available_methods["mcda_methods"])
print("Available weighting methods:", available_methods["weighting_methods"])
print("Available normalization methods:", available_methods["normalization_methods"])
print("\n------------------------------------\n")

# Define a simple decision problem: choosing a laptop

# Define alternatives
alternatives = [
    CreateAlternativeRequest(id="laptop1", name="Laptop 1", description="High-end gaming laptop"),
    CreateAlternativeRequest(id="laptop2", name="Laptop 2", description="Business ultrabook"),
    CreateAlternativeRequest(id="laptop3", name="Laptop 3", description="Budget laptop"),
    CreateAlternativeRequest(id="laptop4", name="Laptop 4", description="Mid-range all-purpose laptop"),
]

# Define criteria
criteria = [
    CreateCriterionRequest(id="price", name="Price", type=CriteriaType.COST),
    CreateCriterionRequest(id="performance", name="Performance", type=CriteriaType.BENEFIT),
    CreateCriterionRequest(id="battery", name="Battery Life", type=CriteriaType.BENEFIT),
    CreateCriterionRequest(id="weight", name="Weight", type=CriteriaType.COST),
    CreateCriterionRequest(id="storage", name="Storage", type=CriteriaType.BENEFIT),
]

# Performance values
# [Price, Performance, Battery, Weight, Storage]
values = [
    [1800, 90, 4, 2.5, 1000],  # Laptop 1
    [1500, 70, 10, 1.2, 512],   # Laptop 2
    [800, 50, 6, 2.0, 256],     # Laptop 3
    [1200, 75, 8, 1.8, 512],    # Laptop 4
]

# Create decision matrix
decision_matrix = CreateDecisionMatrixRequest(
    alternatives=alternatives,
    criteria=criteria,
    values=values
)

# Calculate weights using AHP method
print("Calculating weights using AHP method...")
# AHP pairwise comparison matrix for criteria
# [Price, Performance, Battery, Weight, Storage]
comparison_matrix = [
    [1,    1/3,  1/2,  2,    3],    # Price
    [3,    1,    2,    4,    5],    # Performance
    [2,    1/2,  1,    3,    4],    # Battery
    [1/2,  1/4,  1/3,  1,    2],    # Weight
    [1/3,  1/5,  1/4,  1/2,  1],    # Storage
]

weights_request = CalculateWeightsRequest(
    method="ahp",
    criteria=criteria,
    method_params={"comparison_matrix": comparison_matrix}
)

weights_result = client.calculate_weights(weights_request)
print("Weights calculated using AHP:")
for criterion, weight in zip(weights_result.criteria, weights_result.weights):
    print(f"{criterion.name}: {weight:.4f}")
print(f"Consistency check: {weights_result.additional_data}")
print("\n------------------------------------\n")

# Plot criteria weights
fig1 = plot_criteria_weights([criterion.name for criterion in criteria], weights_result.weights)
print("Criteria weights plotted.")

# Evaluate alternatives using different MCDA methods
print("Evaluating alternatives using different methods...")

comparison_request = CompareMethodsRequest(
    decision_matrix=decision_matrix,
    methods=["topsis", "vikor", "promethee", "wsm", "wpm", "waspas"],
    weights=weights_result.weights,
    method_params={
        "topsis": {"normalization_method": "vector"},
        "vikor": {"v": 0.5},
        "promethee": {},
        "wsm": {"normalization_method": "linear_minmax"},
        "wpm": {"normalization_method": "linear_minmax"},
        "waspas": {"lambda_param": 0.5, "normalization_method": "linear_minmax"}
    }
)

comparison_result = client.compare_methods(comparison_request)

print("Results of the comparison:")
for method_name, result in comparison_result.results.items():
    print(f"{method_name}:")
    for i, alt in enumerate(result.alternatives):
        print(f"  {alt.name}: Preference = {result.preferences[i]:.4f}, Rank = {result.rankings[i]}")
    print()
print(f"Agreement rate between methods: {comparison_result.agreement_rate:.4f}")
print("\n------------------------------------\n")

# Plot individual method results
fig2 = plot_preferences(comparison_result.results["topsis"], title="TOPSIS Results")
print("TOPSIS results plotted.")

# Plot ranking comparison
fig3 = plot_ranking_comparison(comparison_result.results)
print("Ranking comparison plotted.")

# Plot correlation matrix
fig4 = plot_correlation_matrix(comparison_result.correlation_matrix)
print("Correlation matrix plotted.")

# Perform sensitivity analysis on weights for TOPSIS method
print("\nPerforming sensitivity analysis on weights...")

# First, create a proper DecisionMatrix with the weights
for i, criterion in enumerate(criteria):
    criterion.weight = weights_result.weights[i]

from src.mcda.models import Alternative, Criterion, DecisionMatrix
from src.mcda.methods.topsis import TOPSIS

proper_alternatives = [Alternative(**alt.model_dump()) for alt in alternatives]
proper_criteria = [Criterion(**crit.model_dump()) for crit in criteria]
proper_matrix = DecisionMatrix(
    alternatives=proper_alternatives,
    criteria=proper_criteria,
    values=values
)

# Create sensitivity analysis
sensitivity = WeightSensitivityAnalysis(proper_matrix, TOPSIS())
performance_sensitivity = sensitivity.analyze_weight_perturbation(1)  # Performance criterion

# Plot sensitivity results
fig5 = sensitivity.plot_weight_sensitivity(performance_sensitivity)
print("Weight sensitivity analysis plotted.")

# Show all plots
plt.show()
