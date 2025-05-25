"""Supplier Selection Use Case.

This example demonstrates the use of MCDA methods for supplier selection - a common business decision problem.
"""

import numpy as np
import matplotlib.pyplot as plt
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd

from mcda.client import MCDAClient, CreateAlternativeRequest, CreateCriterionRequest, CreateDecisionMatrixRequest
from mcda.models import CriteriaType


def visualize_supplier_comparison(result, alternatives):
    """Create a radar chart comparing the suppliers across criteria."""
    # Extract data for radar chart
    criteria_names = ["Cost", "Quality", "Delivery", "Service", "Financial Stability"]
    alt_names = [alt.name for alt in alternatives]
    
    # Get alternative priorities for each criterion
    alt_priorities = np.array(result.additional_data["alternative_priorities"])
    
    # Create figure and polar axis
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Number of criteria
    N = len(criteria_names)
    
    # Angle of each axis
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Plot each supplier
    for i, alt_name in enumerate(alt_names):
        values = alt_priorities[i, :].tolist()
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=alt_name)
        ax.fill(angles, values, alpha=0.1)
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(criteria_names)
    
    # Draw y-axis labels (0 to 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"])
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title("Supplier Comparison Across Criteria", size=15, y=1.1)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig("supplier_comparison.png")
    plt.close()
    
    print("Radar chart saved as 'supplier_comparison.png'")


def visualize_method_comparison(comparison_result):
    """Create a bar chart comparing rankings from different methods."""
    # Extract data from comparison result
    methods = list(comparison_result.results.keys())
    alternatives = comparison_result.results[methods[0]].alternatives
    alt_names = [alt.name for alt in alternatives]
    
    # Create a dictionary to store rankings for each alternative across methods
    rankings = {alt_name: [] for alt_name in alt_names}
    
    for method in methods:
        result = comparison_result.results[method]
        for i, alt in enumerate(result.alternatives):
            rankings[alt.name].append(result.rankings[i])
    
    # Convert to pandas DataFrame for easier plotting
    df = pd.DataFrame(rankings, index=methods)
    
    # Create bar chart
    ax = df.plot(kind='bar', figsize=(12, 6), width=0.8)
    
    # Customize plot
    ax.set_xlabel('MCDA Method')
    ax.set_ylabel('Ranking (lower is better)')
    ax.set_title('Supplier Rankings Across Different MCDA Methods')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add ranking values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f')
    
    # Add horizontal line for mean ranking
    for alt_name in alt_names:
        mean_rank = np.mean(rankings[alt_name])
        plt.axhline(y=mean_rank, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig("method_comparison.png")
    plt.close()
    
    print("Method comparison chart saved as 'method_comparison.png'")


def main():
    # Initialize MCDAClient
    client = MCDAClient()
    
    print("\nSupplier Selection Case Study\n" + "=" * 30)
    
    # 1. Define the decision problem - selecting the best supplier
    print("\n1. Setting up the decision problem for supplier selection")
    
    # Define alternatives (suppliers)
    supplierA = CreateAlternativeRequest(
        id="S1", 
        name="Supplier A", 
        description="Local supplier with long-term relationship"
    )
    
    supplierB = CreateAlternativeRequest(
        id="S2", 
        name="Supplier B", 
        description="Large international supplier with competitive prices"
    )
    
    supplierC = CreateAlternativeRequest(
        id="S3", 
        name="Supplier C", 
        description="Specialized supplier with high-quality materials"
    )
    
    supplierD = CreateAlternativeRequest(
        id="S4", 
        name="Supplier D", 
        description="New supplier with innovative offerings"
    )
    
    # Define criteria for supplier evaluation
    cost = CreateCriterionRequest(
        id="C1", 
        name="Cost", 
        type=CriteriaType.COST, 
        description="Unit price and total cost of ownership"
    )
    
    quality = CreateCriterionRequest(
        id="C2", 
        name="Quality", 
        type=CriteriaType.BENEFIT, 
        description="Product quality and defect rate"
    )
    
    delivery = CreateCriterionRequest(
        id="C3", 
        name="Delivery", 
        type=CriteriaType.COST, 
        description="Delivery time and reliability"
    )
    
    service = CreateCriterionRequest(
        id="C4", 
        name="Service", 
        type=CriteriaType.BENEFIT, 
        description="Customer service and support"
    )
    
    financial = CreateCriterionRequest(
        id="C5", 
        name="Financial Stability", 
        type=CriteriaType.BENEFIT, 
        description="Financial health and stability of supplier"
    )
    
    # Create the decision matrix with performance values
    decision_matrix = CreateDecisionMatrixRequest(
        alternatives=[supplierA, supplierB, supplierC, supplierD],
        criteria=[cost, quality, delivery, service, financial],
        values=[
            # Cost, Quality, Delivery, Service, Financial
            [75, 80, 8, 85, 75],    # Supplier A
            [60, 75, 12, 70, 90],   # Supplier B
            [90, 95, 10, 80, 70],   # Supplier C
            [80, 85, 15, 90, 60]    # Supplier D
        ]
    )
    
    print("Decision matrix created with 4 suppliers and 5 criteria")
    
    # 2. Calculate criteria weights using AHP pairwise comparisons
    print("\n2. Calculating criteria weights using AHP pairwise comparisons")
    
    # Define pairwise comparison matrix for criteria
    # Scale: 1 = equal, 3 = moderate, 5 = strong, 7 = very strong, 9 = extreme importance
    criteria_comparisons = [
        [1.0, 3.0, 5.0, 2.0, 4.0],  # Cost compared to others
        [1/3, 1.0, 2.0, 1/2, 2.0],  # Quality compared to others
        [1/5, 1/2, 1.0, 1/3, 1/2],  # Delivery compared to others
        [1/2, 2.0, 3.0, 1.0, 2.0],  # Service compared to others
        [1/4, 1/2, 2.0, 1/2, 1.0]   # Financial stability compared to others
    ]
    
    # Calculate weights using AHP
    weights_result = client.calculate_weights({
        "method": "ahp",
        "criteria": [cost, quality, delivery, service, financial],
        "method_params": {
            "comparison_matrix": criteria_comparisons
        }
    })
    
    # Display the calculated weights
    criterion_names = ["Cost", "Quality", "Delivery", "Service", "Financial Stability"]
    print("\nCalculated criteria weights:")
    for i, name in enumerate(criterion_names):
        print(f"{name}: {weights_result.weights[i]:.3f}")
    
    # Check consistency ratio
    cr = weights_result.additional_data["consistency_ratio"]
    is_consistent = weights_result.additional_data["is_consistent"]
    print(f"\nConsistency Ratio: {cr:.3f} {'(Acceptable)' if is_consistent else '(Inconsistent)'}")
    
    # 3. Evaluate suppliers using different MCDA methods
    print("\n3. Evaluating suppliers using multiple MCDA methods")
    
    # Define method-specific parameters
    method_params = {
        "ahp": {
            "criteria_comparisons": criteria_comparisons
        },
        "topsis": {
            "normalization_method": "vector"
        },
        "vikor": {
            "v": 0.5  # Balance between group utility and individual regret
        },
        "promethee2": {
            "preference_functions": ["linear", "gaussian", "usual", "v-shape", "u-shape"]
        },
        "promethee6": {
            "preference_functions": ["linear", "gaussian", "usual", "v-shape", "u-shape"],
            "iterations": 200,
            "weight_ranges": [
                (max(0.2, weights_result.weights[0]-0.1), min(0.5, weights_result.weights[0]+0.1)),
                (max(0.1, weights_result.weights[1]-0.1), min(0.3, weights_result.weights[1]+0.1)),
                (max(0.05, weights_result.weights[2]-0.05), min(0.2, weights_result.weights[2]+0.05)),
                (max(0.1, weights_result.weights[3]-0.1), min(0.3, weights_result.weights[3]+0.1)),
                (max(0.05, weights_result.weights[4]-0.05), min(0.25, weights_result.weights[4]+0.05))
            ]
        }
    }
    
    # Compare multiple methods
    methods_to_compare = ["ahp", "topsis", "vikor", "promethee2", "wpm", "wsm"]
    
    comparison_result = client.compare_methods({
        "decision_matrix": decision_matrix,
        "methods": methods_to_compare,
        "weights": weights_result.weights,
        "method_params": method_params
    })
    
    # Display comparison results
    print("\nRanking results across different methods:")
    print("\nSupplier\t" + "\t".join(methods_to_compare))
    
    for i, supplier in enumerate([supplierA, supplierB, supplierC, supplierD]):
        rankings = [comparison_result.results[method].rankings[i] for method in methods_to_compare]
        print(f"{supplier.name}\t\t" + "\t".join([str(rank) for rank in rankings]))
    
    # Display agreement between methods
    print(f"\nOverall agreement between methods: {comparison_result.agreement_rate:.3f}")
    
    # Generate correlation matrix heat map
    print("\nCorrelation between methods:")
    for method1 in methods_to_compare:
        for method2 in methods_to_compare:
            corr = comparison_result.correlation_matrix[method1][method2]
            if method1 != method2:
                print(f"{method1} vs {method2}: {corr:.3f}")
    
    # 4. Perform detailed AHP analysis
    print("\n4. Performing detailed AHP analysis")
    
    # Define pairwise comparison matrices for each supplier on each criterion
    cost_comparisons = [
        [1.0, 1/3, 2.0, 1.5],  # Supplier A vs others (Cost)
        [3.0, 1.0, 4.0, 3.0],   # Supplier B vs others (Cost)
        [1/2, 1/4, 1.0, 1/2],   # Supplier C vs others (Cost)
        [2/3, 1/3, 2.0, 1.0]    # Supplier D vs others (Cost)
    ]
    
    quality_comparisons = [
        [1.0, 2.0, 1/4, 1/2],   # Supplier A vs others (Quality)
        [1/2, 1.0, 1/5, 1/3],   # Supplier B vs others (Quality)
        [4.0, 5.0, 1.0, 3.0],   # Supplier C vs others (Quality)
        [2.0, 3.0, 1/3, 1.0]    # Supplier D vs others (Quality)
    ]
    
    delivery_comparisons = [
        [1.0, 2.0, 1.5, 3.0],   # Supplier A vs others (Delivery)
        [1/2, 1.0, 1/2, 2.0],   # Supplier B vs others (Delivery)
        [2/3, 2.0, 1.0, 2.0],   # Supplier C vs others (Delivery)
        [1/3, 1/2, 1/2, 1.0]    # Supplier D vs others (Delivery)
    ]
    
    service_comparisons = [
        [1.0, 3.0, 2.0, 1/2],   # Supplier A vs others (Service)
        [1/3, 1.0, 1/2, 1/5],   # Supplier B vs others (Service)
        [1/2, 2.0, 1.0, 1/3],   # Supplier C vs others (Service)
        [2.0, 5.0, 3.0, 1.0]    # Supplier D vs others (Service)
    ]
    
    financial_comparisons = [
        [1.0, 1/3, 2.0, 3.0],   # Supplier A vs others (Financial)
        [3.0, 1.0, 4.0, 5.0],   # Supplier B vs others (Financial)
        [1/2, 1/4, 1.0, 2.0],   # Supplier C vs others (Financial)
        [1/3, 1/5, 1/2, 1.0]    # Supplier D vs others (Financial)
    ]
    
    # Combine all comparison matrices
    alternative_comparisons = [
        cost_comparisons,
        quality_comparisons,
        delivery_comparisons,
        service_comparisons,
        financial_comparisons
    ]
    
    # Perform AHP analysis
    ahp_result = client.evaluate({
        "decision_matrix": decision_matrix,
        "method": "ahp",
        "method_params": {
            "criteria_comparisons": criteria_comparisons,
            "alternative_comparisons": alternative_comparisons
        }
    })
    
    # Extract and display results
    print("\nAHP Results:")
    for i, supplier in enumerate([supplierA, supplierB, supplierC, supplierD]):
        print(f"Rank {ahp_result.rankings[i]}: {supplier.name} (Score: {ahp_result.preferences[i]:.4f})")
    
    # 5. Conduct sensitivity analysis with PROMETHEE VI
    print("\n5. Conducting sensitivity analysis using PROMETHEE VI")
    
    # Use PROMETHEE VI to evaluate weight stability
    promethee6_result = client.evaluate({
        "decision_matrix": decision_matrix,
        "method": "promethee6",
        "weights": weights_result.weights,
        "method_params": method_params["promethee6"]
    })
    
    # Display stability results
    print("\nRobustness analysis results:")
    print("\nSupplier\tMin Flow\tCentral Flow\tMax Flow\tRank Stability")
    
    suppliers = [supplierA, supplierB, supplierC, supplierD]
    for i, supplier in enumerate(suppliers):
        min_flow = promethee6_result.additional_data["min_flows"][i]
        central_flow = promethee6_result.additional_data["central_flows"][i]
        max_flow = promethee6_result.additional_data["max_flows"][i]
        spread = promethee6_result.additional_data["ranking_spread"][i]
        stability = "High" if spread < 0.2 else "Medium" if spread < 0.4 else "Low"
        
        print(f"{supplier.name}\t{min_flow:.4f}\t{central_flow:.4f}\t{max_flow:.4f}\t{stability} (Spread: {spread:.4f})")
    
    # 6. Visualize results
    print("\n6. Visualizing results")
    
    # Create supplier comparison radar chart
    visualize_supplier_comparison(ahp_result, suppliers)
    
    # Create method comparison chart
    visualize_method_comparison(comparison_result)
    
    # 7. Final recommendation
    print("\n7. Final recommendation")
    
    # Find the most consistent top performer
    # Count how many times each supplier is ranked 1st
    first_rank_counts = {supplier.name: 0 for supplier in suppliers}
    for method in methods_to_compare:
        result = comparison_result.results[method]
        for i, supplier in enumerate(suppliers):
            if result.rankings[i] == 1:  # Ranked 1st
                first_rank_counts[supplier.name] += 1
    
    # Find average rank for each supplier
    avg_ranks = {supplier.name: 0 for supplier in suppliers}
    for supplier in suppliers:
        ranks = []
        for method in methods_to_compare:
            result = comparison_result.results[method]
            for i, s in enumerate(result.alternatives):
                if s.name == supplier.name:
                    ranks.append(result.rankings[i])
        avg_ranks[supplier.name] = sum(ranks) / len(ranks)
    
    # Display ranking summary
    print("\nTop ranked supplier by each method:")
    for method in methods_to_compare:
        result = comparison_result.results[method]
        for i, rank in enumerate(result.rankings):
            if rank == 1:
                print(f"{method.upper()}: {result.alternatives[i].name}")
    
    print("\nNumber of times each supplier ranked 1st:")
    for supplier, count in first_rank_counts.items():
        print(f"{supplier}: {count} out of {len(methods_to_compare)} methods")
    
    print("\nAverage rank across all methods:")
    for supplier, avg in sorted(avg_ranks.items(), key=lambda x: x[1]):
        print(f"{supplier}: {avg:.2f}")
    
    # Final recommendation
    best_supplier = min(avg_ranks.items(), key=lambda x: x[1])[0]
    print(f"\nBased on comprehensive analysis across multiple MCDA methods, {best_supplier} is recommended as the best supplier.")
    print("This supplier demonstrates the best overall performance considering all criteria and analysis methods.")


if __name__ == "__main__":
    main()
