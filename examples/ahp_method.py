"""Example demonstrating the usage of the AHP method.

This example shows how to use AHP with either decision matrices or pairwise comparison matrices.
"""

import numpy as np
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from mcda.client import MCDAClient, CreateDecisionMatrixRequest, CreateAlternativeRequest, CreateCriterionRequest
from mcda.models import CriteriaType


def print_matrix(matrix, row_labels, col_labels):
    """Print a matrix with row and column labels."""
    # Print column headers
    header = "\t" + "\t".join(col_labels)
    print(header)
    
    # Print each row
    for i, row in enumerate(matrix):
        formatted_row = [f"{val:.2f}" for val in row]
        print(f"{row_labels[i]}\t{chr(9).join(formatted_row)}")


def main():
    # Initialize MCDA client
    client = MCDAClient()
    
    print("\nAHP Method Examples\n" + "=" * 25)
    
    print("\n1. AHP with Decision Matrix\n" + "-" * 30)
    
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
    
    # Use AHP with direct decision matrix
    result1 = client.evaluate({
        "decision_matrix": decision_matrix,
        "method": "ahp"
    })
    
    # Display AHP with direct decision matrix result
    print("\nAlternative Priorities:")
    for i, alt in enumerate(result1.alternatives):
        print(f"Rank {result1.rankings[i]}: {alt.name} (Priority: {result1.preferences[i]:.4f})")
    
    print("\nCriteria Weights:", {crit: weight for crit, weight in zip(result1.additional_data["criteria"], result1.additional_data["criteria_weights"])})
    
    print("\n2. AHP with Pairwise Comparison Matrices\n" + "-" * 30)
    
    # Define criteria comparison matrix (5x5)
    # Each value (i,j) indicates how much more important criterion i is compared to j on a scale of 1-9
    # where 1 = equal importance, 3 = moderate importance, 5 = strong importance, 7 = very strong, 9 = extreme importance
    criteria_comparisons = np.array([
        [1.0, 2.0, 0.5, 3.0, 2.0],  # Price compared to others
        [0.5, 1.0, 0.33, 2.0, 1.0],  # Acceleration compared to others
        [2.0, 3.0, 1.0, 4.0, 3.0],  # Fuel Economy compared to others
        [0.33, 0.5, 0.25, 1.0, 0.5],  # Comfort compared to others
        [0.5, 1.0, 0.33, 2.0, 1.0]   # Safety compared to others
    ])
    
    # Display the criteria comparison matrix
    criteria_names = [crit.name for crit in decision_matrix.criteria]
    print("\nCriteria Comparison Matrix:")
    print_matrix(criteria_comparisons, criteria_names, criteria_names)
    
    # Define alternative comparison matrices (4x4), one for each criterion
    # Each matrix compares how alternatives perform on a specific criterion
    alt_comp_price = np.array([
        [1.0, 2.0, 4.0, 3.0],  # Car A compared to others (Price)
        [0.5, 1.0, 3.0, 2.0],  # Car B compared to others (Price)
        [0.25, 0.33, 1.0, 0.5],  # Car C compared to others (Price)
        [0.33, 0.5, 2.0, 1.0]   # Car D compared to others (Price)
    ])
    
    alt_comp_acceleration = np.array([
        [1.0, 0.8, 0.5, 0.6],   # Car A compared to others (Acceleration)
        [1.25, 1.0, 0.4, 0.5],   # Car B compared to others (Acceleration)
        [2.0, 2.5, 1.0, 1.2],   # Car C compared to others (Acceleration)
        [1.67, 2.0, 0.83, 1.0]    # Car D compared to others (Acceleration)
    ])
    
    alt_comp_fuel = np.array([
        [1.0, 3.0, 2.0, 0.2],   # Car A compared to others (Fuel Economy)
        [0.33, 1.0, 1.5, 0.14],  # Car B compared to others (Fuel Economy)
        [0.5, 0.67, 1.0, 0.11],  # Car C compared to others (Fuel Economy)
        [5.0, 7.0, 9.0, 1.0]    # Car D compared to others (Fuel Economy)
    ])
    
    alt_comp_comfort = np.array([
        [1.0, 0.5, 0.33, 0.5],   # Car A compared to others (Comfort)
        [2.0, 1.0, 0.5, 1.0],   # Car B compared to others (Comfort)
        [3.0, 2.0, 1.0, 2.0],   # Car C compared to others (Comfort)
        [2.0, 1.0, 0.5, 1.0]    # Car D compared to others (Comfort)
    ])
    
    alt_comp_safety = np.array([
        [1.0, 0.5, 1.0, 0.5],   # Car A compared to others (Safety)
        [2.0, 1.0, 2.0, 1.0],   # Car B compared to others (Safety)
        [1.0, 0.5, 1.0, 0.5],   # Car C compared to others (Safety)
        [2.0, 1.0, 2.0, 1.0]    # Car D compared to others (Safety)
    ])
    
    # Combine all alternative comparison matrices
    alternative_comparisons = [
        alt_comp_price,
        alt_comp_acceleration,
        alt_comp_fuel,
        alt_comp_comfort,
        alt_comp_safety
    ]
    
    # Display one of the alternative comparison matrices as an example
    alt_names = [alt.name for alt in decision_matrix.alternatives]
    print("\nAlternative Comparison Matrix for Price Criterion:")
    print_matrix(alt_comp_price, alt_names, alt_names)
    
    # Use AHP with pairwise comparison matrices
    result2 = client.evaluate({
        "decision_matrix": decision_matrix,  # Needed for alternative/criteria information
        "method": "ahp",
        "method_params": {
            "criteria_comparisons": criteria_comparisons.tolist(),
            "alternative_comparisons": [matrix.tolist() for matrix in alternative_comparisons],
            "consistency_threshold": 0.20  # Higher threshold for demonstration purposes
        }
    })
    
    # Display AHP with pairwise comparison matrices result
    print("\nAlternative Priorities (from pairwise comparisons):")
    for i, alt in enumerate(result2.alternatives):
        print(f"Rank {result2.rankings[i]}: {alt.name} (Priority: {result2.preferences[i]:.4f})")
    
    print("\nCriteria Weights (from pairwise comparisons):", 
          {crit: weight for crit, weight in zip(result2.additional_data["criteria"], result2.additional_data["criteria_weights"])})
    
    # Display consistency information
    consistency_info = result2.additional_data["consistency_info"]
    print("\nConsistency Information:")
    print(f"Criteria Consistency Ratio: {consistency_info['criteria_consistency_ratio']:.4f}")
    print(f"Alternative Consistency Ratios: {[f'{cr:.4f}' for cr in consistency_info['alternative_consistency_ratios']]}")
    print(f"Maximum Consistency Ratio: {consistency_info['max_consistency_ratio']:.4f}")
    print(f"Is Consistent: {'Yes' if consistency_info['is_consistent'] else 'No'} (threshold: {consistency_info['consistency_threshold']})")
    
    print("\nComparison of AHP Results:")
    print("\nAlternative\tDecision Matrix\tPairwise Comp.\tRank Change")
    for i, alt in enumerate(result1.alternatives):
        rank1 = result1.rankings[i]
        rank2 = result2.rankings[i]
        rank_change = rank1 - rank2
        change_str = "0" if rank_change == 0 else f"+{rank_change}" if rank_change > 0 else f"{rank_change}"
        print(f"{alt.name}\t{result1.preferences[i]:.4f}\t{result2.preferences[i]:.4f}\t{change_str}")


if __name__ == "__main__":
    main()