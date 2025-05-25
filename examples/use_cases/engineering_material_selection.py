"""Engineering Material Selection Use Case.

This example demonstrates the use of MCDA methods for engineering material selection,
a classic multi-criteria decision problem in engineering design.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import pandas as pd
from tabulate import tabulate

from mcda.client import MCDAClient, CreateAlternativeRequest, CreateCriterionRequest, CreateDecisionMatrixRequest
from mcda.models import CriteriaType


def visualize_property_chart(decision_matrix, alternatives, criteria_indices, criteria_names, filename):
    """Create a 2D material property chart (Ashby diagram)."""
    # Get values for the two selected properties
    x_property = decision_matrix.values[:][criteria_indices[0]]
    y_property = decision_matrix.values[:][criteria_indices[1]]
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot each material as a point with labels
    for i, alt in enumerate(alternatives):
        x_val = decision_matrix.values[i][criteria_indices[0]]
        y_val = decision_matrix.values[i][criteria_indices[1]]
        plt.scatter(x_val, y_val, s=100, alpha=0.7)
        plt.annotate(alt.name, (x_val, y_val), fontsize=9, 
                     xytext=(5, 5), textcoords='offset points')
    
    # Adjust axes (optionally use log scale for material properties that span orders of magnitude)
    if max(x_property) / min(x_property) > 100:
        plt.xscale('log')
    if max(y_property) / min(y_property) > 100:
        plt.yscale('log')
    
    # Add labels and title
    plt.xlabel(criteria_names[0])
    plt.ylabel(criteria_names[1])
    plt.title(f"Material Selection Chart: {criteria_names[1]} vs {criteria_names[0]}")
    plt.grid(True, alpha=0.3)
    
    # Add performance index line if both criteria are of the same type or opposite types
    crit1_type = decision_matrix.criteria[criteria_indices[0]].type
    crit2_type = decision_matrix.criteria[criteria_indices[1]].type
    
    # Get axis limits for drawing lines
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    
    if crit1_type != crit2_type:  # Maximize one, minimize the other (e.g., strength/weight ratio)
        # Draw lines of equal performance index (y = const * x)
        for m in [0.1, 1, 10, 100]:
            if crit1_type == CriteriaType.BENEFIT:
                # If x is benefit and y is cost, we want to maximize x/y
                xs = np.linspace(x_min, x_max, 100)
                ys = m * xs
                plt.plot(xs, ys, 'k:', alpha=0.5)
            else:
                # If x is cost and y is benefit, we want to maximize y/x
                xs = np.linspace(x_min, x_max, 100)
                ys = m / xs
                plt.plot(xs, ys, 'k:', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"Material property chart saved as '{filename}'")


def main():
    # Initialize MCDAClient
    client = MCDAClient()
    
    print("\nEngineering Material Selection Case Study\n" + "=" * 40)
    
    # 1. Define the decision problem - selecting the best material for a mechanical component
    print("\n1. Setting up the material selection problem")
    
    # Define alternatives (materials)
    materials = [
        CreateAlternativeRequest(
            id="M1", 
            name="Aluminum Alloy 6061-T6", 
            description="General-purpose aluminum alloy with good mechanical properties"
        ),
        CreateAlternativeRequest(
            id="M2", 
            name="Carbon Steel AISI 1045", 
            description="Medium carbon steel with good strength and machinability"
        ),
        CreateAlternativeRequest(
            id="M3", 
            name="Stainless Steel 316", 
            description="Austenitic stainless steel with excellent corrosion resistance"
        ),
        CreateAlternativeRequest(
            id="M4", 
            name="Titanium Alloy Ti-6Al-4V", 
            description="High-strength titanium alloy with excellent strength-to-weight ratio"
        ),
        CreateAlternativeRequest(
            id="M5", 
            name="Magnesium Alloy AZ31B", 
            description="Lightweight magnesium alloy with moderate strength"
        ),
        CreateAlternativeRequest(
            id="M6", 
            name="Engineering Polymer PEEK", 
            description="High-performance thermoplastic with good mechanical properties"
        ),
        CreateAlternativeRequest(
            id="M7", 
            name="Fiber-Reinforced Composite", 
            description="Carbon fiber reinforced polymer with high specific strength"
        )
    ]
    
    # Define criteria for material evaluation
    density = CreateCriterionRequest(
        id="C1", 
        name="Density", 
        type=CriteriaType.COST,  # Lower density is better
        description="Material density in g/cm³"
    )
    
    yield_strength = CreateCriterionRequest(
        id="C2", 
        name="Yield Strength", 
        type=CriteriaType.BENEFIT,  # Higher strength is better
        description="Yield strength in MPa"
    )
    
    elastic_modulus = CreateCriterionRequest(
        id="C3", 
        name="Elastic Modulus", 
        type=CriteriaType.BENEFIT,  # Higher stiffness is better
        description="Young's modulus in GPa"
    )
    
    corrosion_resistance = CreateCriterionRequest(
        id="C4", 
        name="Corrosion Resistance", 
        type=CriteriaType.BENEFIT,  # Higher resistance is better
        description="Corrosion resistance rating (1-10)"
    )
    
    cost = CreateCriterionRequest(
        id="C5", 
        name="Cost", 
        type=CriteriaType.COST,  # Lower cost is better
        description="Relative cost ($/kg)"
    )
    
    machinability = CreateCriterionRequest(
        id="C6", 
        name="Machinability", 
        type=CriteriaType.BENEFIT,  # Higher machinability is better
        description="Machinability rating (1-10)"
    )
    
    thermal_expansion = CreateCriterionRequest(
        id="C7", 
        name="Thermal Expansion", 
        type=CriteriaType.COST,  # Lower thermal expansion is better
        description="Coefficient of thermal expansion (μm/m·K)"
    )
    
    # Create the decision matrix with material property values
    decision_matrix = CreateDecisionMatrixRequest(
        alternatives=materials,
        criteria=[density, yield_strength, elastic_modulus, corrosion_resistance, 
                  cost, machinability, thermal_expansion],
        values=[
            # Density, Yield Str, E-Modulus, Corrosion, Cost, Machinability, Thermal Exp
            [2.70, 276, 68.9, 5, 3.0, 8, 23.6],     # Aluminum 6061-T6
            [7.85, 505, 200.0, 3, 1.0, 7, 11.5],    # Carbon Steel 1045
            [8.00, 290, 193.0, 9, 4.5, 5, 16.0],    # Stainless Steel 316
            [4.43, 880, 113.8, 8, 22.0, 4, 8.6],    # Titanium Ti-6Al-4V
            [1.77, 200, 45.0, 4, 4.5, 8, 26.0],     # Magnesium AZ31B
            [1.32, 100, 3.6, 9, 45.0, 9, 47.0],     # PEEK
            [1.60, 600, 70.0, 9, 35.0, 2, 2.0]      # Carbon Fiber Composite
        ]
    )
    
    print("Decision matrix created with 7 materials and 7 properties")
    
    # 2. Application Requirements Analysis
    print("\n2. Analyzing application requirements")
    
    # Scenario: Material for an aerospace structural component
    print("\nApplication: Aerospace structural component")
    print("Key requirements:")
    print("- Low weight (density) is critical")
    print("- High strength-to-weight ratio")
    print("- Sufficient stiffness")
    print("- Good corrosion resistance")
    print("- Cost is important but secondary to performance")
    
    # Calculate derived properties
    values = np.array(decision_matrix.values)
    strength_to_weight = values[:, 1] / values[:, 0]  # Yield strength / density
    stiffness_to_weight = values[:, 2] / values[:, 0]  # Elastic modulus / density
    
    # Display derived properties
    print("\nDerived material properties:")
    print("Material\t\tStrength-to-Weight\tStiffness-to-Weight")
    for i, material in enumerate(materials):
        name = material.name[:20] + "..." if len(material.name) > 20 else material.name
        print(f"{name:<25}\t{strength_to_weight[i]:.1f}\t\t{stiffness_to_weight[i]:.1f}")
    
    # 3. Define weights based on application requirements
    print("\n3. Defining criteria weights for aerospace application")
    
    # AHP pairwise comparison for weight determination
    criteria_comparisons = [
        # Den, YS, EM, CR, Cost, Mach, TE
        [1.0, 1/2, 1/3, 2.0, 3.0, 5.0, 4.0],  # Density
        [2.0, 1.0, 1.0, 3.0, 4.0, 7.0, 5.0],  # Yield Strength
        [3.0, 1.0, 1.0, 3.0, 4.0, 7.0, 5.0],  # Elastic Modulus
        [1/2, 1/3, 1/3, 1.0, 2.0, 4.0, 3.0],  # Corrosion Resistance
        [1/3, 1/4, 1/4, 1/2, 1.0, 3.0, 2.0],  # Cost
        [1/5, 1/7, 1/7, 1/4, 1/3, 1.0, 1/2],  # Machinability
        [1/4, 1/5, 1/5, 1/3, 1/2, 2.0, 1.0]   # Thermal Expansion
    ]
    
    # Calculate weights using AHP
    weights_result = client.calculate_weights({
        "method": "ahp",
        "criteria": [density, yield_strength, elastic_modulus, corrosion_resistance, 
                     cost, machinability, thermal_expansion],
        "method_params": {
            "comparison_matrix": criteria_comparisons
        }
    })
    
    # Display the calculated weights
    criteria_names = ["Density", "Yield Strength", "Elastic Modulus", "Corrosion Resistance", 
                     "Cost", "Machinability", "Thermal Expansion"]
    print("\nCalculated criteria weights for aerospace application:")
    for i, name in enumerate(criteria_names):
        print(f"{name}: {weights_result.weights[i]:.3f}")
    
    # Check consistency of weights
    consistency_ratio = weights_result.additional_data["consistency_ratio"]
    is_consistent = weights_result.additional_data["is_consistent"]
    print(f"\nConsistency Ratio: {consistency_ratio:.3f} {'(Acceptable)' if is_consistent else '(Inconsistent)'}")
    
    # 4. Create material property charts (Ashby diagrams)
    print("\n4. Creating material property charts")
    
    # Strength vs Density chart (key for aerospace)
    visualize_property_chart(
        decision_matrix, materials, 
        [0, 1],  # Indices for density and yield strength
        ["Density (g/cm³)", "Yield Strength (MPa)"], 
        "strength_density_chart.png"
    )
    
    # Stiffness vs Density chart
    visualize_property_chart(
        decision_matrix, materials, 
        [0, 2],  # Indices for density and elastic modulus
        ["Density (g/cm³)", "Elastic Modulus (GPa)"], 
        "stiffness_density_chart.png"
    )
    
    # 5. Evaluate materials using different MCDA methods
    print("\n5. Evaluating materials using multiple MCDA methods")
    
    # Define method-specific parameters
    method_params = {
        "topsis": {
            "normalization_method": "vector"
        },
        "vikor": {
            "v": 0.5  # Compromise parameter (group utility vs. individual regret)
        },
        "promethee2": {
            "preference_functions": ["linear", "linear", "v-shape", "gaussian", 
                                    "linear", "u-shape", "linear"],
            "p_thresholds": [3.0, 300, 100, 3, 20, 3, 20],
            "q_thresholds": [0.5, 50, 20, 1, 3, 1, 5]
        }
    }
    
    # Methods to compare
    methods = ["topsis", "vikor", "ahp", "promethee2", "wsm", "wpm"]
    
    # Run the comparison
    comparison_result = client.compare_methods({
        "decision_matrix": decision_matrix,
        "methods": methods,
        "weights": weights_result.weights,
        "method_params": method_params
    })
    
    # Display results table
    print("\nRanking results across different methods:")
    print("\nMaterial\t\t" + "\t".join(methods))
    
    for i, material in enumerate(materials):
        name = material.name[:20] + "..." if len(material.name) > 20 else material.name
        rankings = [comparison_result.results[method].rankings[i] for method in methods]
        print(f"{name:<25}\t" + "\t".join([str(rank) for rank in rankings]))
    
    # Display method agreement
    print(f"\nOverall agreement between methods: {comparison_result.agreement_rate:.3f}")
    
    # 6. Detailed TOPSIS analysis
    print("\n6. Detailed TOPSIS analysis")
    
    # Get TOPSIS result
    topsis_result = comparison_result.results["topsis"]
    
    # Extract normalized values and distances
    if "normalized_matrix" in topsis_result.additional_data:
        norm_matrix = np.array(topsis_result.additional_data["normalized_matrix"])
        ideal = np.array(topsis_result.additional_data["ideal_solution"])
        anti_ideal = np.array(topsis_result.additional_data["anti_ideal_solution"])
        
        print("\nNormalized values and distances to ideal/anti-ideal solutions:")
        print("\nMaterial\t\t" + "\t".join([f"{name[:4]}" for name in criteria_names]) + 
              "\tD+\tD-\tScore")
        
        for i, material in enumerate(materials):
            name = material.name[:20] + "..." if len(material.name) > 20 else material.name
            norm_vals = "\t".join([f"{val:.2f}" for val in norm_matrix[i]])
            d_plus = topsis_result.additional_data["distances_to_ideal"][i]
            d_minus = topsis_result.additional_data["distances_to_anti_ideal"][i]
            score = topsis_result.preferences[i]
            print(f"{name:<25}\t{norm_vals}\t{d_plus:.2f}\t{d_minus:.2f}\t{score:.4f}")
    
    # 7. Sensitivity analysis - Effect of changing weights
    print("\n7. Sensitivity analysis - Effect of changing weights")
    
    # Create several different weight sets
    weight_scenarios = [
        {"name": "Current weights", "weights": weights_result.weights},
        {"name": "Equal weights", "weights": [1/7] * 7},
        {"name": "Lightweight focus", "weights": [0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05]},
        {"name": "Strength focus", "weights": [0.1, 0.4, 0.2, 0.1, 0.1, 0.05, 0.05]},
        {"name": "Cost focus", "weights": [0.1, 0.1, 0.1, 0.1, 0.5, 0.05, 0.05]}
    ]
    
    # Run TOPSIS for each weight scenario
    sensitivity_results = []
    
    for scenario in weight_scenarios:
        result = client.evaluate({
            "decision_matrix": decision_matrix,
            "method": "topsis",
            "weights": scenario["weights"],
            "method_params": method_params["topsis"]
        })
        sensitivity_results.append({
            "name": scenario["name"],
            "rankings": result.rankings,
            "preferences": result.preferences
        })
    
    # Display sensitivity analysis results
    print("\nRankings under different weight scenarios:")
    print("\nMaterial\t\t" + "\t".join([s["name"] for s in sensitivity_results]))
    
    for i, material in enumerate(materials):
        name = material.name[:20] + "..." if len(material.name) > 20 else material.name
        rankings = [s["rankings"][i] for s in sensitivity_results]
        print(f"{name:<25}\t" + "\t".join([str(rank) for rank in rankings]))
    
    # 8. Engineering performance indices analysis
    print("\n8. Engineering performance indices analysis")
    
    # Calculate common engineering performance indices
    values = np.array(decision_matrix.values)
    
    # Specific strength (strength-to-weight ratio)
    specific_strength = values[:, 1] / values[:, 0]  # Yield strength / density
    
    # Specific stiffness (stiffness-to-weight ratio) 
    specific_stiffness = values[:, 2] / values[:, 0]  # E modulus / density
    
    # Strength-to-cost ratio
    strength_cost_ratio = values[:, 1] / values[:, 4]  # Yield strength / cost
    
    # Performance indices dataframe
    indices_data = {
        "Material": [m.name for m in materials],
        "Specific Strength": specific_strength,
        "Rank": np.argsort(-specific_strength) + 1,
        "Specific Stiffness": specific_stiffness,
        "Rank.1": np.argsort(-specific_stiffness) + 1,
        "Strength-Cost Ratio": strength_cost_ratio,
        "Rank.2": np.argsort(-strength_cost_ratio) + 1
    }
    
    # Format as a table
    indices_table = tabulate(
        [[mat] + [f"{indices_data[col][i]:.1f}" if "." not in col else indices_data[col][i] 
                for col in indices_data.keys() if col != "Material"] 
         for i, mat in enumerate(indices_data["Material"])],
        headers=["Material", "Spec. Strength", "Rank", "Spec. Stiffness", "Rank", "Str/Cost", "Rank"],
        tablefmt="grid"
    )
    
    print(f"\nEngineering Performance Indices:\n{indices_table}")
    
    # 9. Final recommendation
    print("\n9. Final material recommendation for aerospace component")
    
    # Identify the most consistent top performers
    rankings = np.zeros((len(materials), len(methods)))
    for i, method in enumerate(methods):
        result = comparison_result.results[method]
        for j in range(len(materials)):
            rankings[j, i] = result.rankings[j]
    
    # Calculate average rank and consistency for each material
    avg_ranks = np.mean(rankings, axis=1)
    rank_std = np.std(rankings, axis=1)
    
    # Create final recommendation table
    recommendation_data = {
        "Material": [m.name for m in materials],
        "Avg Rank": avg_ranks,
        "Rank Std Dev": rank_std,
        "Times Ranked #1": [np.sum(rankings[i] == 1) for i in range(len(materials))],
        "Spec. Strength": specific_strength,
        "Spec. Stiffness": specific_stiffness
    }
    
    # Sort by average rank
    sorted_indices = np.argsort(avg_ranks)
    recommendation_data = {k: [recommendation_data[k][i] for i in sorted_indices] 
                          for k in recommendation_data.keys()}
    
    # Format as a table
    recommendation_table = tabulate(
        [[recommendation_data["Material"][i][:30]] + 
         [f"{recommendation_data['Avg Rank'][i]:.2f}", 
          f"{recommendation_data['Rank Std Dev'][i]:.2f}", 
          recommendation_data['Times Ranked #1'][i],
          f"{recommendation_data['Spec. Strength'][i]:.1f}",
          f"{recommendation_data['Spec. Stiffness'][i]:.1f}"]
         for i in range(len(materials))],
        headers=["Material", "Avg Rank", "Std Dev", "#1 Counts", "Spec. Str", "Spec. Stiff"],
        tablefmt="grid"
    )
    
    print(f"\nFinal Material Ranking:\n{recommendation_table}")
    
    # Determine the best material
    best_idx = np.argmin(avg_ranks)
    best_material = materials[best_idx]
    
    # Final recommendation
    print(f"\nRecommended material: {best_material.name}")
    print(f"Key advantages:")
    
    # List key advantages based on performance
    if specific_strength[best_idx] > np.mean(specific_strength):
        print(f"- Excellent specific strength: {specific_strength[best_idx]:.1f} (Avg: {np.mean(specific_strength):.1f})")
    
    if specific_stiffness[best_idx] > np.mean(specific_stiffness):
        print(f"- High specific stiffness: {specific_stiffness[best_idx]:.1f} (Avg: {np.mean(specific_stiffness):.1f})")
    
    if values[best_idx, 3] > np.mean(values[:, 3]):  # Corrosion resistance
        print(f"- Good corrosion resistance: {values[best_idx, 3]:.1f}/10 (Avg: {np.mean(values[:, 3]):.1f}/10)")
    
    if values[best_idx, 4] < np.mean(values[:, 4]):  # Cost
        print(f"- Reasonable cost: ${values[best_idx, 4]:.1f}/kg (Avg: ${np.mean(values[:, 4]):.1f}/kg)")
    
    # Design considerations
    print("\nDesign considerations:")
    # Add design considerations specific to the best material
    if "Titanium" in best_material.name:
        print("- Special machining practices may be required due to titanium's poor thermal conductivity")
        print("- Excellent fatigue resistance makes it ideal for cyclic loading conditions")
        print("- Higher initial cost is offset by long service life and low maintenance")
    elif "Composite" in best_material.name:
        print("- Anisotropic properties require careful orientation of fibers in the direction of loading")
        print("- Special joining techniques needed when connecting to metal parts")
        print("- Low thermal expansion minimizes thermal stress issues")
    elif "Aluminum" in best_material.name:
        print("- Excellent machinability reduces manufacturing costs")
        print("- Consider protective coatings in corrosive environments")
        print("- Higher thermal expansion may need compensation in thermal cycling applications")
    else:
        print("- Material-specific design implications should be considered")
        print("- Consult material property datasheets for detailed specifications")
        print("- Consider manufacturing processes compatible with this material")


if __name__ == "__main__":
    main()
