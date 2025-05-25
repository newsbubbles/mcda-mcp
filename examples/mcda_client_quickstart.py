"""MCDA Client Quick Start Guide.

This example demonstrates the basic usage of the MCDAClient for various MCDA tasks.
"""

from typing import List, Dict, Any, Optional

# Import core components from MCDA client
from mcda.client import MCDAClient, CreateAlternativeRequest, CreateCriterionRequest, CreateDecisionMatrixRequest
from mcda.models import CriteriaType


def basic_example():
    """Basic example of using MCDA client to evaluate alternatives."""
    print("\n1. Basic TOPSIS Example\n" + "-" * 30)
    
    # Create MCDA client instance
    client = MCDAClient()
    
    # Define alternatives
    alternatives = [
        CreateAlternativeRequest(id="A1", name="Alternative 1", description="First option"),
        CreateAlternativeRequest(id="A2", name="Alternative 2", description="Second option"),
        CreateAlternativeRequest(id="A3", name="Alternative 3", description="Third option"),
    ]
    
    # Define criteria - specify type (BENEFIT = higher is better, COST = lower is better)
    criteria = [
        CreateCriterionRequest(id="C1", name="Cost", type=CriteriaType.COST, weight=0.4),
        CreateCriterionRequest(id="C2", name="Quality", type=CriteriaType.BENEFIT, weight=0.4),
        CreateCriterionRequest(id="C3", name="Delivery", type=CriteriaType.COST, weight=0.2),
    ]
    
    # Define performance values for each alternative on each criterion
    values = [
        [100, 8, 5],   # Alternative 1: Cost=100, Quality=8, Delivery=5
        [150, 9, 3],   # Alternative 2: Cost=150, Quality=9, Delivery=3
        [120, 7, 8],   # Alternative 3: Cost=120, Quality=7, Delivery=8
    ]
    
    # Create decision matrix combining alternatives, criteria, and values
    decision_matrix = CreateDecisionMatrixRequest(
        alternatives=alternatives,
        criteria=criteria,
        values=values
    )
    
    # Extract weights from criteria
    weights = [criterion.weight for criterion in criteria]
    print(f"Criteria weights: {weights}")
    
    # Evaluate alternatives using TOPSIS method
    result = client.evaluate({
        "decision_matrix": decision_matrix,
        "method": "topsis",
        "weights": weights,
        "method_params": {"normalization_method": "vector"}
    })
    
    # Display results
    print("\nTOPSIS Results:")
    for i, alt in enumerate(result.alternatives):
        print(f"{alt.name}: Score = {result.preferences[i]:.4f}, Rank = {result.rankings[i]}")
    
    return client, decision_matrix, weights


def compare_methods(client, decision_matrix, weights):
    """Compare multiple MCDA methods on the same decision problem."""
    print("\n2. Comparing Multiple MCDA Methods\n" + "-" * 30)
    
    # List of methods to compare
    methods = ["topsis", "ahp", "vikor", "promethee2", "wsm", "wpm"]
    
    # Method-specific parameters
    method_params = {
        "topsis": {"normalization_method": "vector"},
        "vikor": {"v": 0.5},  # Compromise parameter (0.5 = balanced)
        "promethee2": {"preference_functions": ["linear", "gaussian", "usual"]}
    }
    
    # Compare methods
    comparison_result = client.compare_methods({
        "decision_matrix": decision_matrix,
        "methods": methods,
        "weights": weights,
        "method_params": method_params
    })
    
    # Display results from each method
    print("Results from different MCDA methods:")
    for method in methods:
        result = comparison_result.results[method]
        print(f"\n{method.upper()} Rankings:")
        for i, alt in enumerate(result.alternatives):
            print(f"  {alt.name}: Rank = {result.rankings[i]}, Score = {result.preferences[i]:.4f}")
    
    # Display agreement between methods
    print(f"\nOverall agreement between methods: {comparison_result.agreement_rate:.4f}")
    print("(1.0 = perfect agreement, -1.0 = perfect disagreement)")
    
    # Display correlation between pair of methods
    print("\nCorrelation between TOPSIS and PROMETHEE II: ", 
          comparison_result.correlation_matrix["topsis"]["promethee2"])


def weighting_methods_example(client):
    """Demonstrate different weighting methods."""
    print("\n3. Using Different Weighting Methods\n" + "-" * 30)
    
    # Define criteria without weights
    criteria = [
        CreateCriterionRequest(id="C1", name="Criterion 1", type=CriteriaType.BENEFIT),
        CreateCriterionRequest(id="C2", name="Criterion 2", type=CriteriaType.COST),
        CreateCriterionRequest(id="C3", name="Criterion 3", type=CriteriaType.BENEFIT),
        CreateCriterionRequest(id="C4", name="Criterion 4", type=CriteriaType.BENEFIT),
    ]
    
    # Method 1: Equal weighting (simplest approach)
    equal_weights = client.calculate_weights({
        "method": "equal",
        "criteria": criteria
    })
    
    print("Equal weights:", equal_weights.weights)
    
    # Method 2: Manual weighting (direct specification)
    manual_weights = client.calculate_weights({
        "method": "manual",
        "criteria": criteria,
        "method_params": {
            "weights": [0.4, 0.3, 0.2, 0.1]  # Must sum to 1.0
        }
    })
    
    print("Manual weights:", manual_weights.weights)
    
    # Method 3: AHP weighting (pairwise comparisons)
    # Define pairwise comparison matrix: how much more important is criterion i than criterion j?
    # Scale: 1=equal, 3=moderate, 5=strong, 7=very strong, 9=extreme importance
    comparison_matrix = [
        [1, 3, 5, 7],    # Criterion 1 compared to others
        [1/3, 1, 3, 5],  # Criterion 2 compared to others
        [1/5, 1/3, 1, 3], # Criterion 3 compared to others
        [1/7, 1/5, 1/3, 1] # Criterion 4 compared to others
    ]
    
    ahp_weights = client.calculate_weights({
        "method": "ahp",
        "criteria": criteria,
        "method_params": {
            "comparison_matrix": comparison_matrix
        }
    })
    
    print("AHP weights:", ahp_weights.weights)
    print(f"Consistency ratio: {ahp_weights.additional_data['consistency_ratio']:.4f}")
    print(f"Is consistent: {ahp_weights.additional_data['is_consistent']}")


def promethee_example():
    """Demonstrate using the PROMETHEE family of methods."""
    print("\n4. Using PROMETHEE Methods\n" + "-" * 30)
    
    # Create MCDA client instance
    client = MCDAClient()
    
    # Define alternatives
    alternatives = [
        CreateAlternativeRequest(id="A1", name="Option A"),
        CreateAlternativeRequest(id="A2", name="Option B"),
        CreateAlternativeRequest(id="A3", name="Option C"),
        CreateAlternativeRequest(id="A4", name="Option D"),
    ]
    
    # Define criteria
    criteria = [
        CreateCriterionRequest(id="C1", name="Criterion 1", type=CriteriaType.BENEFIT, weight=0.25),
        CreateCriterionRequest(id="C2", name="Criterion 2", type=CriteriaType.COST, weight=0.25),
        CreateCriterionRequest(id="C3", name="Criterion 3", type=CriteriaType.BENEFIT, weight=0.25),
        CreateCriterionRequest(id="C4", name="Criterion 4", type=CriteriaType.BENEFIT, weight=0.25),
    ]
    
    # Define performance values
    values = [
        [8, 12, 7, 9],  # Option A
        [9, 10, 8, 7],  # Option B
        [7, 15, 9, 8],  # Option C
        [6, 8, 6, 10],  # Option D
    ]
    
    # Create decision matrix
    decision_matrix = CreateDecisionMatrixRequest(
        alternatives=alternatives,
        criteria=criteria,
        values=values
    )
    
    # Extract weights
    weights = [criterion.weight for criterion in criteria]
    
    # Preference function parameters
    preference_functions = ["linear", "v-shape", "gaussian", "usual"]
    p_thresholds = [2.0, 3.0, 1.0, None]  # Preference thresholds
    q_thresholds = [0.5, 1.0, None, None]  # Indifference thresholds
    
    # PROMETHEE I (partial ranking)
    promethee1_result = client.evaluate({
        "decision_matrix": decision_matrix,
        "method": "promethee1",
        "weights": weights,
        "method_params": {
            "preference_functions": preference_functions,
            "p_thresholds": p_thresholds,
            "q_thresholds": q_thresholds
        }
    })
    
    print("PROMETHEE I results (partial ranking):")
    print("Option\tPositive Flow\tNegative Flow\tNet Flow")
    for i, alt in enumerate(promethee1_result.alternatives):
        pos = promethee1_result.additional_data["positive_flows"][i]
        neg = promethee1_result.additional_data["negative_flows"][i]
        net = promethee1_result.preferences[i]
        print(f"{alt.name}\t{pos:.4f}\t\t{neg:.4f}\t\t{net:.4f}")
    
    # PROMETHEE II (complete ranking)
    promethee2_result = client.evaluate({
        "decision_matrix": decision_matrix,
        "method": "promethee2",
        "weights": weights,
        "method_params": {
            "preference_functions": preference_functions,
            "p_thresholds": p_thresholds,
            "q_thresholds": q_thresholds
        }
    })
    
    # PROMETHEE VI (stability analysis)
    # Define weight ranges as (min, max)
    weight_ranges = [(0.15, 0.35), (0.15, 0.35), (0.15, 0.35), (0.15, 0.35)]
    
    promethee6_result = client.evaluate({
        "decision_matrix": decision_matrix,
        "method": "promethee6",
        "weights": weights,
        "method_params": {
            "preference_functions": preference_functions,
            "p_thresholds": p_thresholds,
            "q_thresholds": q_thresholds,
            "weight_ranges": weight_ranges,
            "iterations": 200
        }
    })
    
    print("\nPROMETHEE VI results (stability analysis):")
    print("Option\tMin Flow\tCentral Flow\tMax Flow\tSpread")
    for i, alt in enumerate(promethee6_result.alternatives):
        min_flow = promethee6_result.additional_data["min_flows"][i]
        central = promethee6_result.additional_data["central_flows"][i]
        max_flow = promethee6_result.additional_data["max_flows"][i]
        spread = promethee6_result.additional_data["ranking_spread"][i]
        print(f"{alt.name}\t{min_flow:.4f}\t{central:.4f}\t{max_flow:.4f}\t{spread:.4f}")


def ahp_example():
    """Demonstrate using the AHP method with pairwise comparisons."""
    print("\n5. Using AHP with Pairwise Comparisons\n" + "-" * 30)
    
    # Create MCDA client instance
    client = MCDAClient()
    
    # Define alternatives
    alternatives = [
        CreateAlternativeRequest(id="A1", name="Product X"),
        CreateAlternativeRequest(id="A2", name="Product Y"),
        CreateAlternativeRequest(id="A3", name="Product Z"),
    ]
    
    # Define criteria
    criteria = [
        CreateCriterionRequest(id="C1", name="Price", type=CriteriaType.COST),
        CreateCriterionRequest(id="C2", name="Quality", type=CriteriaType.BENEFIT),
        CreateCriterionRequest(id="C3", name="Service", type=CriteriaType.BENEFIT),
    ]
    
    # Define dummy values (not used in this example)
    values = [
        [0, 0, 0],  # Dummy values for Product X
        [0, 0, 0],  # Dummy values for Product Y
        [0, 0, 0],  # Dummy values for Product Z
    ]
    
    # Create decision matrix (required for AHP to get alternatives and criteria info)
    decision_matrix = CreateDecisionMatrixRequest(
        alternatives=alternatives,
        criteria=criteria,
        values=values
    )
    
    # Define criteria pairwise comparison matrix
    # Scale: 1=equal, 3=moderate, 5=strong, 7=very strong, 9=extreme importance
    criteria_comparisons = [
        [1, 1/3, 1/5],  # Price compared to others
        [3, 1, 1/3],    # Quality compared to others
        [5, 3, 1]       # Service compared to others
    ]
    
    # Define alternative comparison matrices (one for each criterion)
    # How much better is alternative i than alternative j for criterion k?
    price_comparisons = [
        [1, 3, 5],    # Product X compared to others (Price)
        [1/3, 1, 2],   # Product Y compared to others (Price)
        [1/5, 1/2, 1]   # Product Z compared to others (Price)
    ]
    
    quality_comparisons = [
        [1, 1/3, 1/7],   # Product X compared to others (Quality)
        [3, 1, 1/3],     # Product Y compared to others (Quality)
        [7, 3, 1]        # Product Z compared to others (Quality)
    ]
    
    service_comparisons = [
        [1, 1/2, 1/3],   # Product X compared to others (Service)
        [2, 1, 1/2],     # Product Y compared to others (Service)
        [3, 2, 1]        # Product Z compared to others (Service)
    ]
    
    # Combine all alternative comparison matrices
    alternative_comparisons = [
        price_comparisons,
        quality_comparisons,
        service_comparisons
    ]
    
    # Run AHP with pairwise comparisons
    ahp_result = client.evaluate({
        "decision_matrix": decision_matrix,
        "method": "ahp",
        "method_params": {
            "criteria_comparisons": criteria_comparisons,
            "alternative_comparisons": alternative_comparisons,
            "consistency_threshold": 0.1  # Maximum acceptable consistency ratio
        }
    })
    
    # Display results
    print("AHP Results:")
    print("\nCriteria Weights:")
    for i, criterion in enumerate(criteria):
        print(f"{criterion.name}: {ahp_result.additional_data['criteria_weights'][i]:.4f}")
    
    print("\nAlternative Priorities:")
    for i, alt in enumerate(ahp_result.alternatives):
        priorities = ahp_result.additional_data["alternative_priorities"][i]
        print(f"{alt.name}: {', '.join([f'{p:.4f}' for p in priorities])}")
    
    print("\nOverall Priorities and Rankings:")
    for i, alt in enumerate(ahp_result.alternatives):
        print(f"{alt.name}: Priority = {ahp_result.preferences[i]:.4f}, Rank = {ahp_result.rankings[i]}")
    
    # Check consistency
    consistency_info = ahp_result.additional_data["consistency_info"]
    print(f"\nConsistency Check: {'Passed' if consistency_info['is_consistent'] else 'Failed'}")
    print(f"Max Consistency Ratio: {consistency_info['max_consistency_ratio']:.4f} (Threshold: {consistency_info['consistency_threshold']})")


def main():
    print("MCDA Client Quick Start Guide")
    print("=" * 30)
    
    # Example 1: Basic usage with TOPSIS
    client, decision_matrix, weights = basic_example()
    
    # Example 2: Compare multiple methods
    compare_methods(client, decision_matrix, weights)
    
    # Example 3: Demonstrate weighting methods
    weighting_methods_example(client)
    
    # Example 4: PROMETHEE methods
    promethee_example()
    
    # Example 5: AHP with pairwise comparisons
    ahp_example()
    
    print("\nAdditional Features:")
    print("- The MCDA client supports additional methods: VIKOR, WSM, WPM, WASPAS")
    print("- For more in-depth examples, check the examples/use_cases directory")
    print("- Review the documentation for advanced usage scenarios")


if __name__ == "__main__":
    main()
