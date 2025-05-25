"""Example demonstrating the usage of all PROMETHEE methods.

This example creates a decision matrix and evaluates alternatives using all PROMETHEE methods (I-VI).
"""

import numpy as np
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from mcda.client import MCDAClient, CreateDecisionMatrixRequest, CreateAlternativeRequest, CreateCriterionRequest
from mcda.models import CriteriaType

# Create a decision problem for car selection
altA = CreateAlternativeRequest(id="A1", name="Car A", description="Compact sedan")
altB = CreateAlternativeRequest(id="A2", name="Car B", description="SUV")
altC = CreateAlternativeRequest(id="A3", name="Car C", description="Luxury sedan")
altD = CreateAlternativeRequest(id="A4", name="Car D", description="Electric vehicle")

# Define criteria
crit1 = CreateCriterionRequest(id="C1", name="Price", type=CriteriaType.COST, weight=0.25)
crit2 = CreateCriterionRequest(id="C2", name="Acceleration", type=CriteriaType.COST, weight=0.20)
crit3 = CreateCriterionRequest(id="C3", name="Fuel Economy", type=CriteriaType.BENEFIT, weight=0.30)
crit4 = CreateCriterionRequest(id="C4", name="Comfort", type=CriteriaType.BENEFIT, weight=0.15)
crit5 = CreateCriterionRequest(id="C5", name="Safety", type=CriteriaType.BENEFIT, weight=0.10)

# Create decision matrix with values
decision_matrix = CreateDecisionMatrixRequest(
    alternatives=[altA, altB, altC, altD],
    criteria=[crit1, crit2, crit3, crit4, crit5],
    values=[
        [25000, 9.5, 30, 7, 8],   # Car A
        [32000, 11.2, 22, 8, 9],  # Car B
        [42000, 7.8, 20, 9, 8],   # Car C
        [38000, 8.2, 120, 8, 9]   # Car D
    ]
)

# Initialize MCDA client
client = MCDAClient()

# Define preference function types for each criterion
preference_functions = ["linear", "v-shape", "u-shape", "usual", "gaussian"]

def main():
    print("\nPROMETHEE Methods Example\n" + "=" * 25)
    
    # PROMETHEE I
    print("\n1. PROMETHEE I - Partial Ranking")
    result1 = client.evaluate({
        "decision_matrix": decision_matrix,
        "method": "promethee1",
        "method_params": {
            "preference_functions": preference_functions
        }
    })
    
    # Display PROMETHEE I result
    print("\nPositive Flows:", {alt.name: flow for alt, flow in zip(result1.alternatives, result1.additional_data["positive_flows"])})
    print("Negative Flows:", {alt.name: flow for alt, flow in zip(result1.alternatives, result1.additional_data["negative_flows"])})
    print("\nPartial Ranking Matrix (1=better, 0=incomparable, -1=worse):")
    partial_ranking = result1.additional_data["partial_ranking"]
    for i, alt_i in enumerate(result1.alternatives):
        row = []
        for j, alt_j in enumerate(result1.alternatives):
            if i == j:
                row.append("-")
            else:
                row.append(partial_ranking[i][j])
        print(f"{alt_i.name}: {row}")
    
    # PROMETHEE II
    print("\n2. PROMETHEE II - Complete Ranking")
    result2 = client.evaluate({
        "decision_matrix": decision_matrix,
        "method": "promethee2",
        "method_params": {
            "preference_functions": preference_functions
        }
    })
    
    # Display PROMETHEE II result
    print("\nNet Flows:")
    for i, alt in enumerate(result2.alternatives):
        print(f"Rank {result2.rankings[i]}: {alt.name} (Net Flow: {result2.preferences[i]:.4f})")
    
    # PROMETHEE III
    print("\n3. PROMETHEE III - Ranking with Intervals")
    result3 = client.evaluate({
        "decision_matrix": decision_matrix,
        "method": "promethee3",
        "method_params": {
            "preference_functions": preference_functions,
            "alpha": 0.2  # Interval parameter
        }
    })
    
    # Display PROMETHEE III result
    print("\nIntervals (Net Flow, Lower Bound, Upper Bound):")
    for i, alt in enumerate(result3.alternatives):
        interval = result3.additional_data["intervals"][i]
        print(f"Rank {result3.rankings[i]}: {alt.name} ({interval[0]:.4f}, {interval[1]:.4f}, {interval[2]:.4f})")
    
    # PROMETHEE IV
    print("\n4. PROMETHEE IV - Normalized Flows")
    result4 = client.evaluate({
        "decision_matrix": decision_matrix,
        "method": "promethee4",
        "method_params": {
            "preference_functions": preference_functions
        }
    })
    
    # Display PROMETHEE IV result
    print("\nNormalized Flows:")
    for i, alt in enumerate(result4.alternatives):
        print(f"Rank {result4.rankings[i]}: {alt.name} (Normalized Flow: {result4.additional_data['normalized_flows'][i]:.4f})")
    
    # PROMETHEE V
    print("\n5. PROMETHEE V - Constrained Flows")
    # Define a constraint that price must be under 40000
    def price_constraint(values):
        return values[0] < 40000
    
    result5 = client.evaluate({
        "decision_matrix": decision_matrix,
        "method": "promethee5",
        "method_params": {
            "preference_functions": preference_functions,
            "constraints": [price_constraint]
        }
    })
    
    # Display PROMETHEE V result
    print("\nConstrained Flows (None means infeasible):")
    for i, alt in enumerate(result5.alternatives):
        feasible = result5.additional_data["feasible_alternatives"][i]
        flow = result5.additional_data["constrained_flows"][i]
        status = "FEASIBLE" if feasible else "INFEASIBLE"
        print(f"{alt.name} ({status}): {flow}")
    
    # PROMETHEE VI
    print("\n6. PROMETHEE VI - Flows with Weight Ranges")
    # Define weight ranges as (min, max) for each criterion
    weight_ranges = [
        (0.15, 0.35),  # Price (25% ± 10%)
        (0.10, 0.30),  # Acceleration (20% ± 10%)
        (0.20, 0.40),  # Fuel Economy (30% ± 10%)
        (0.05, 0.25),  # Comfort (15% ± 10%)
        (0.05, 0.15)   # Safety (10% ± 5%)
    ]
    
    result6 = client.evaluate({
        "decision_matrix": decision_matrix,
        "method": "promethee6",
        "method_params": {
            "preference_functions": preference_functions,
            "weight_ranges": weight_ranges,
            "iterations": 500  # Number of Monte Carlo iterations
        }
    })
    
    # Display PROMETHEE VI result
    print("\nRanking Stability (Min, Central, Max Flows):")
    for i, alt in enumerate(result6.alternatives):
        min_flow = result6.additional_data["min_flows"][i]
        central_flow = result6.additional_data["central_flows"][i]
        max_flow = result6.additional_data["max_flows"][i]
        spread = result6.additional_data["ranking_spread"][i]
        print(f"Rank {result6.rankings[i]}: {alt.name} (Min: {min_flow:.4f}, Central: {central_flow:.4f}, Max: {max_flow:.4f}, Spread: {spread:.4f})")
    
    print("\nComparison of Net Flows Across Methods:")
    comparison = {alt.name: {} for alt in result2.alternatives}
    
    for i, alt in enumerate(result2.alternatives):
        comparison[alt.name] = {
            "PROMETHEE I": result1.preferences[i],
            "PROMETHEE II": result2.preferences[i],
            "PROMETHEE III": result3.preferences[i],
            "PROMETHEE IV": result4.preferences[i],
            "PROMETHEE V": result5.preferences[i] if result5.additional_data["feasible_alternatives"][i] else "Infeasible",
            "PROMETHEE VI": result6.preferences[i]
        }
    
    # Print comparison table
    headers = ["Alternative"] + list(next(iter(comparison.values())).keys())
    row_format = "{:>12}" + "{:>15}" * len(headers[1:])
    print(row_format.format(*headers))
    
    for alt, flows in comparison.items():
        row = [alt]
        for method, flow in flows.items():
            if isinstance(flow, float):
                row.append(f"{flow:.4f}")
            else:
                row.append(flow)
        print(row_format.format(*row))


if __name__ == "__main__":
    main()