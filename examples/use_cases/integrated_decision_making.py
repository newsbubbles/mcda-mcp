"""Integrated Decision Making Use Case.

This example demonstrates the integration of multiple MCDA methods for a comprehensive
decision-making process in a product selection scenario.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
import pandas as pd
from tabulate import tabulate

from mcda.client import MCDAClient, CreateAlternativeRequest, CreateCriterionRequest, CreateDecisionMatrixRequest
from mcda.models import CriteriaType


def visualize_correlation_heatmap(comparison_result, filename):
    """Visualize correlation between different MCDA methods."""
    methods = list(comparison_result.correlation_matrix.keys())
    
    # Create correlation matrix
    corr_data = []
    for method1 in methods:
        row = []
        for method2 in methods:
            row.append(comparison_result.correlation_matrix[method1][method2])
        corr_data.append(row)
    
    corr_df = pd.DataFrame(corr_data, index=methods, columns=methods)
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", vmin=-1, vmax=1, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Between MCDA Methods", fontsize=15)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"Correlation heatmap saved as '{filename}'")


def visualize_rankings_comparison(comparison_result, alternatives, filename):
    """Visualize rankings from different MCDA methods."""
    methods = list(comparison_result.results.keys())
    alt_names = [alt.name for alt in alternatives]
    
    # Create rankings matrix
    rankings = np.zeros((len(methods), len(alt_names)))
    for i, method in enumerate(methods):
        result = comparison_result.results[method]
        for j, alt in enumerate(result.alternatives):
            rankings[i, j] = result.rankings[j]  # 1-based rankings
    
    # Create DataFrame for plotting
    rankings_df = pd.DataFrame(rankings, index=methods, columns=alt_names)
    
    # Plot rankings (lower is better)
    plt.figure(figsize=(12, 6))
    ax = rankings_df.plot(kind="bar", colormap="viridis")
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3)
    
    plt.title("Rankings Comparison Across MCDA Methods", fontsize=15)
    plt.xlabel("MCDA Method")
    plt.ylabel("Ranking (lower is better)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Alternatives")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"Rankings comparison chart saved as '{filename}'")


def visualize_sensitivity(weights_results, topsis_results, alternatives, criteria_names, filename):
    """Visualize sensitivity of rankings to weight changes."""
    # Extract data
    weight_variations = [result["description"] for result in weights_results]
    alt_names = [alt.name for alt in alternatives]
    
    # Create rankings matrix
    rankings = np.zeros((len(weight_variations), len(alt_names)))
    for i, result in enumerate(topsis_results):
        for j, alt in enumerate(result.alternatives):
            rankings[i, j] = result.rankings[j]  # 1-based rankings
    
    # Create DataFrame for plotting
    rankings_df = pd.DataFrame(rankings, index=weight_variations, columns=alt_names)
    
    # Plot rankings
    plt.figure(figsize=(12, 8))
    ax = rankings_df.plot(kind="bar", colormap="tab10")
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3)
    
    plt.title("Sensitivity Analysis: Rankings with Different Criteria Weights", fontsize=15)
    plt.xlabel("Weight Variations")
    plt.ylabel("Ranking (lower is better)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Alternatives")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"Sensitivity analysis chart saved as '{filename}'")


def main():
    # Initialize MCDAClient
    client = MCDAClient()
    
    print("\nIntegrated Decision Making Case Study\n" + "=" * 40)
    
    # 1. Define the decision problem - product selection
    print("\n1. Setting up the product selection problem")
    
    # Define alternatives (products)
    product_A = CreateAlternativeRequest(
        id="P1", 
        name="Product A", 
        description="High-end model with advanced features"
    )
    
    product_B = CreateAlternativeRequest(
        id="P2", 
        name="Product B", 
        description="Mid-range model with balanced specs"
    )
    
    product_C = CreateAlternativeRequest(
        id="P3", 
        name="Product C", 
        description="Budget model with basic functionality"
    )
    
    product_D = CreateAlternativeRequest(
        id="P4", 
        name="Product D", 
        description="Specialized model for specific use cases"
    )
    
    # Define criteria for product evaluation
    performance = CreateCriterionRequest(
        id="C1", 
        name="Performance", 
        type=CriteriaType.BENEFIT, 
        description="Overall performance score (1-100)"
    )
    
    cost = CreateCriterionRequest(
        id="C2", 
        name="Cost", 
        type=CriteriaType.COST, 
        description="Total cost including maintenance (thousands $)"
    )
    
    reliability = CreateCriterionRequest(
        id="C3", 
        name="Reliability", 
        type=CriteriaType.BENEFIT, 
        description="Reliability rating based on tests (1-10)"
    )
    
    usability = CreateCriterionRequest(
        id="C4", 
        name="Usability", 
        type=CriteriaType.BENEFIT, 
        description="Usability and user experience score (1-10)"
    )
    
    support = CreateCriterionRequest(
        id="C5", 
        name="Support", 
        type=CriteriaType.BENEFIT, 
        description="Customer support quality (1-10)"
    )
    
    # Create the decision matrix with performance values
    decision_matrix = CreateDecisionMatrixRequest(
        alternatives=[product_A, product_B, product_C, product_D],
        criteria=[performance, cost, reliability, usability, support],
        values=[
            # Performance, Cost, Reliability, Usability, Support
            [95, 85, 9.2, 8.5, 9.0],    # Product A
            [80, 60, 8.7, 9.0, 8.5],    # Product B
            [65, 40, 7.8, 8.2, 7.0],    # Product C
            [85, 75, 8.5, 7.5, 8.0],    # Product D
        ]
    )
    
    criteria_names = ["Performance", "Cost", "Reliability", "Usability", "Support"]
    alternatives = [product_A, product_B, product_C, product_D]
    
    print("Decision matrix created with 4 products and 5 criteria")
    
    # 2. Calculate weights using multiple methods
    print("\n2. Calculating criteria weights using different methods")
    
    # 2.1 Equal weights
    equal_weights = client.calculate_weights({
        "method": "equal",
        "criteria": [performance, cost, reliability, usability, support]
    })
    
    # 2.2 AHP weights through pairwise comparisons
    # Define criteria pairwise comparison matrix
    criteria_comparisons = [
        # Performance, Cost, Reliability, Usability, Support
        [1.0, 2.0, 3.0, 4.0, 5.0],  # Performance
        [1/2, 1.0, 2.0, 3.0, 4.0],  # Cost
        [1/3, 1/2, 1.0, 2.0, 3.0],  # Reliability
        [1/4, 1/3, 1/2, 1.0, 2.0],  # Usability
        [1/5, 1/4, 1/3, 1/2, 1.0]   # Support
    ]
    
    ahp_weights = client.calculate_weights({
        "method": "ahp",
        "criteria": [performance, cost, reliability, usability, support],
        "method_params": {
            "comparison_matrix": criteria_comparisons
        }
    })
    
    # 2.3 Entropy weights (objective weights based on data dispersion)
    entropy_weights = client.calculate_weights({
        "method": "entropy",
        "decision_matrix": decision_matrix
    })
    
    # 2.4 Custom weights from stakeholder input
    custom_weights = client.calculate_weights({
        "method": "manual",
        "criteria": [performance, cost, reliability, usability, support],
        "method_params": {
            "weights": [0.30, 0.25, 0.20, 0.15, 0.10]
        }
    })
    
    # Display all calculated weights
    print("\nWeights calculated using different methods:")
    print("\nCriterion\tEqual\tAHP\tEntropy\tCustom")
    
    for i, criterion in enumerate(criteria_names):
        print(f"{criterion}\t{equal_weights.weights[i]:.3f}\t{ahp_weights.weights[i]:.3f}\t{entropy_weights.weights[i]:.3f}\t{custom_weights.weights[i]:.3f}")
    
    # 3. Evaluate products using multiple MCDA methods with custom weights
    print("\n3. Evaluating products using multiple MCDA methods with custom weights")
    
    # Define method-specific parameters
    method_params = {
        "topsis": {
            "normalization_method": "vector"
        },
        "vikor": {
            "v": 0.5  # Compromise parameter
        },
        "ahp": {
            "criteria_comparisons": criteria_comparisons
        },
        "promethee2": {
            "preference_functions": ["gaussian", "linear", "v-shape", "usual", "u-shape"],
            "p_thresholds": [15, 20, 1.0, 1.0, 1.5],  # Preference thresholds
            "q_thresholds": [5, 5, 0.3, 0.2, 0.5]      # Indifference thresholds
        }
    }
    
    # Methods to compare
    methods = ["topsis", "vikor", "promethee2", "ahp", "wsm", "wpm"]
    
    # Run the comparison with custom weights
    comparison_result = client.compare_methods({
        "decision_matrix": decision_matrix,
        "methods": methods,
        "weights": custom_weights.weights,
        "method_params": method_params
    })
    
    # Display comprehensive results table
    print("\nComprehensive results across methods:")
    print("\nProduct\t" + "\t".join([f"{m.upper()} Rank" for m in methods]) + "\tMean Rank")
    
    # Calculate average rank for each alternative
    mean_ranks = {}
    for alt in alternatives:
        ranks = []
        for method in methods:
            result = comparison_result.results[method]
            for i, res_alt in enumerate(result.alternatives):
                if res_alt.id == alt.id:
                    ranks.append(result.rankings[i])
                    break
        mean_rank = sum(ranks) / len(ranks)
        mean_ranks[alt.id] = mean_rank
        rank_str = "\t".join([str(rank) for rank in ranks])
        print(f"{alt.name}\t{rank_str}\t{mean_rank:.2f}")
    
    # Display method correlation and agreement
    print(f"\nOverall agreement between methods: {comparison_result.agreement_rate:.3f}")
    
    print("\nMethod correlation matrix:")
    corr_table = [["Method"] + methods]
    for method1 in methods:
        row = [method1.upper()]
        for method2 in methods:
            corr = comparison_result.correlation_matrix[method1][method2]
            row.append(f"{corr:.2f}")
        corr_table.append(row)
    
    print(tabulate(corr_table[1:], headers=corr_table[0], tablefmt="grid"))
    
    # Visualize method correlation
    visualize_correlation_heatmap(comparison_result, "method_correlation.png")
    
    # Visualize rankings comparison
    visualize_rankings_comparison(comparison_result, alternatives, "rankings_comparison.png")
    
    # 4. Perform sensitivity analysis - how do rankings change with different weight sets?
    print("\n4. Performing sensitivity analysis with different weight sets")
    
    # Store different weight sets with descriptions
    weights_sets = [
        {"weights": equal_weights.weights, "description": "Equal"}, 
        {"weights": ahp_weights.weights, "description": "AHP"}, 
        {"weights": entropy_weights.weights, "description": "Entropy"}, 
        {"weights": custom_weights.weights, "description": "Custom"},
        # Additional weight variations to test sensitivity
        {"weights": [0.5, 0.2, 0.1, 0.1, 0.1], "description": "High Performance"},
        {"weights": [0.1, 0.5, 0.1, 0.1, 0.2], "description": "Cost Conscious"},
        {"weights": [0.1, 0.1, 0.5, 0.2, 0.1], "description": "Reliability Focus"}
    ]
    
    # Run TOPSIS with each weight set (TOPSIS chosen for sensitivity analysis due to its simplicity)
    topsis_results = []
    
    for weight_set in weights_sets:
        result = client.evaluate({
            "decision_matrix": decision_matrix,
            "method": "topsis",
            "weights": weight_set["weights"],
            "method_params": method_params["topsis"]
        })
        topsis_results.append(result)
    
    # Display sensitivity analysis results
    print("\nSensitivity analysis results (TOPSIS rankings with different weight sets):")
    print("\nProduct\t" + "\t".join([w["description"] for w in weights_sets]))
    
    for alt in alternatives:
        ranks = []
        for result in topsis_results:
            for i, res_alt in enumerate(result.alternatives):
                if res_alt.id == alt.id:
                    ranks.append(result.rankings[i])
                    break
        rank_str = "\t".join([str(rank) for rank in ranks])
        print(f"{alt.name}\t{rank_str}")
    
    # Visualize sensitivity analysis
    visualize_sensitivity(weights_sets, topsis_results, alternatives, criteria_names, "sensitivity_analysis.png")
    
    # 5. Perform deep dive analysis of top two alternatives
    print("\n5. Performing deep dive analysis of top alternatives")
    
    # Identify top two alternatives based on average ranking
    top_two = sorted(mean_ranks.items(), key=lambda x: x[1])[:2]
    top_two_ids = [item[0] for item in top_two]
    top_two_alts = [alt for alt in alternatives if alt.id in top_two_ids]
    
    # Get original performance values
    product_values = {}
    for i, alt in enumerate(alternatives):
        product_values[alt.id] = decision_matrix.values[i]
    
    # Create comparative radar chart data
    radar_data = []
    for alt in top_two_alts:
        # Normalize values to 0-1 scale for radar chart
        values = product_values[alt.id]
        normalized_values = []
        
        for i, val in enumerate(values):
            criterion_values = [product_values[a.id][i] for a in alternatives]
            min_val = min(criterion_values)
            max_val = max(criterion_values)
            
            if max_val > min_val:
                if criteria[i].type == CriteriaType.BENEFIT:
                    norm_val = (val - min_val) / (max_val - min_val)
                else:  # COST criterion
                    norm_val = (max_val - val) / (max_val - min_val)
            else:
                norm_val = 0.5  # Default if all values are the same
                
            normalized_values.append(norm_val)
            
        radar_data.append((alt.name, normalized_values))
    
    # Create radar chart (headless mode, saving to file)
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    # Set up the angle for each criterion
    angles = np.linspace(0, 2*np.pi, len(criteria_names), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Plot each product
    for name, values in radar_data:
        values += values[:1]  # Close the loop for plotting
        ax.plot(angles, values, linewidth=2, label=name)
        ax.fill(angles, values, alpha=0.1)
    
    # Set the labels for each criterion
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(criteria_names)
    
    # Add legend and title
    plt.legend(loc='upper right')
    plt.title("Comparison of Top Two Products")
    
    # Save the radar chart
    plt.tight_layout()
    plt.savefig("top_alternatives_comparison.png")
    plt.close()
    
    print(f"\nTop two alternatives based on average ranking:\n1. {top_two_alts[0].name}\n2. {top_two_alts[1].name}")
    print("\nDetailed performance comparison of top two alternatives:")
    print("\nCriterion\t" + "\t".join([alt.name for alt in top_two_alts]))
    
    for i, criterion in enumerate(criteria_names):
        values = [product_values[alt.id][i] for alt in top_two_alts]
        print(f"{criterion}\t{values[0]}\t{values[1]}")
    
    print("\nRadar chart comparison saved as 'top_alternatives_comparison.png'")
    
    # 6. Analyze PROMETHEE results in detail
    print("\n6. Analyzing PROMETHEE II results in detail")
    
    # Get PROMETHEE II result
    promethee_result = comparison_result.results["promethee2"]
    
    # Extract flows
    pos_flows = promethee_result.additional_data["positive_flows"]
    neg_flows = promethee_result.additional_data["negative_flows"]
    net_flows = promethee_result.preferences
    
    # Display detailed PROMETHEE results
    print("\nPROMETHEE II flow analysis:")
    print("\nProduct\tPositive Flow\tNegative Flow\tNet Flow\tRank")
    
    for i, alt in enumerate(alternatives):
        print(f"{alt.name}\t{pos_flows[i]:.3f}\t{neg_flows[i]:.3f}\t{net_flows[i]:.3f}\t{promethee_result.rankings[i]}")
    
    # 7. Final integrated recommendation
    print("\n7. Final integrated recommendation")
    
    # Determine the most consistently top-ranked product
    # Count number of methods ranking each product as #1
    top_ranked_counts = {alt.name: 0 for alt in alternatives}
    for method in methods:
        result = comparison_result.results[method]
        for i, alt in enumerate(result.alternatives):
            if result.rankings[i] == 1:  # Rank 1 = top
                top_ranked_counts[alt.name] += 1
    
    # Find best product based on most #1 rankings and average rank
    best_product = max(top_ranked_counts.items(), key=lambda x: x[1])[0]
    best_average = min([(alt.name, mean_ranks[alt.id]) for alt in alternatives], key=lambda x: x[1])[0]
    
    print(f"\nProduct rankings summary:")
    for alt in alternatives:
        print(f"{alt.name}:\n - Average rank: {mean_ranks[alt.id]:.2f}\n - Times ranked #1: {top_ranked_counts[alt.name]} out of {len(methods)} methods")
    
    print(f"\nBased on comprehensive multi-criteria analysis:")
    
    if best_product == best_average:
        print(f"The clear recommendation is {best_product}. It achieves the best average rank ({mean_ranks[next(alt.id for alt in alternatives if alt.name == best_product)]:.2f}) and is ranked #1 by {top_ranked_counts[best_product]} out of {len(methods)} methods.")
    else:
        print(f"The primary recommendation is {best_average} with the best average rank ({mean_ranks[next(alt.id for alt in alternatives if alt.name == best_average)]:.2f}).")
        print(f"However, {best_product} is worth considering as it was ranked #1 by the most methods ({top_ranked_counts[best_product]} out of {len(methods)}).")
    
    # Sensitivity insight
    sensitivity_insights = []
    for i, alt in enumerate(alternatives):
        ranks = [result.rankings[i] for result in topsis_results]
        max_rank_diff = max(ranks) - min(ranks)
        if max_rank_diff > 1:  # Significant rank change
            sensitivity_insights.append(f"{alt.name}: Rankings vary from {min(ranks)} to {max(ranks)} depending on weights")
    
    if sensitivity_insights:
        print("\nSensitivity insights:")
        for insight in sensitivity_insights:
            print(f"- {insight}")
    
    # Final recommendation with context
    print("\nFinal recommendation summary:")
    best_alt = next(alt for alt in alternatives if alt.name == best_average or alt.name == best_product)
    best_alt_values = next(vals for i, vals in enumerate(decision_matrix.values) if alternatives[i].id == best_alt.id)
    
    print(f"- {best_alt.name} is recommended based on integrated analysis across multiple methods")
    print(f"- Key strengths: ", end="")
    
    # Identify strengths based on normalized criterion values
    strengths = []
    for i, val in enumerate(best_alt_values):
        criterion_values = [product_values[a.id][i] for a in alternatives]
        min_val = min(criterion_values)
        max_val = max(criterion_values)
        
        if criteria[i].type == CriteriaType.BENEFIT and val > np.mean(criterion_values):
            strengths.append(criteria_names[i])
        elif criteria[i].type == CriteriaType.COST and val < np.mean(criterion_values):
            strengths.append(criteria_names[i])
    
    print(", ".join(strengths))
    print(f"- Decision confidence: {'High' if comparison_result.agreement_rate > 0.7 else 'Medium' if comparison_result.agreement_rate > 0.4 else 'Low'}")
    print(f"- Method agreement rate: {comparison_result.agreement_rate:.2f} (correlation between different methods)")


if __name__ == "__main__":
    main()
