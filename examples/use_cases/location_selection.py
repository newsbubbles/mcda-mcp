"""Location Selection Use Case.

This example demonstrates the use of PROMETHEE methods for facility location selection.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Callable
import pandas as pd
from tabulate import tabulate

from mcda.client import MCDAClient, CreateAlternativeRequest, CreateCriterionRequest, CreateDecisionMatrixRequest
from mcda.models import CriteriaType
from mcda.methods.promethee import PreferenceFunction


def visualize_flows(results, alternatives, title, filename):
    """Visualize positive and negative flows for PROMETHEE I."""
    alt_names = [alt.name for alt in alternatives]
    positive_flows = results.additional_data["positive_flows"]
    negative_flows = results.additional_data["negative_flows"]
    
    # Invert negative flows for visualization
    negative_flows_inv = [-f for f in negative_flows]
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create x-axis positions
    x = np.arange(len(alt_names))
    width = 0.35
    
    # Plot bars
    ax.bar(x - width/2, positive_flows, width, label="Positive Flow (Higher is better)", color="green", alpha=0.7)
    ax.bar(x + width/2, negative_flows_inv, width, label="Negative Flow (Higher is worse)", color="red", alpha=0.7)
    
    # Add labels and legend
    ax.set_xlabel("Locations")
    ax.set_ylabel("Flow Value")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(alt_names)
    ax.legend()
    
    # Add gridlines
    ax.grid(True, linestyle="--", alpha=0.7)
    
    # Add value labels on bars
    for i, v in enumerate(positive_flows):
        ax.text(i - width/2, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
    
    for i, v in enumerate(negative_flows_inv):
        ax.text(i + width/2, v + 0.01, f"{-negative_flows[i]:.3f}", ha="center", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"Flow visualization saved as '{filename}'")


def visualize_intervals(results, alternatives, filename):
    """Visualize intervals from PROMETHEE III."""
    alt_names = [alt.name for alt in alternatives]
    intervals = results.additional_data["intervals"]
    
    # Extract interval data
    centers = [interval[0] for interval in intervals]
    lower_bounds = [interval[1] for interval in intervals]
    upper_bounds = [interval[2] for interval in intervals]
    
    # Create error bars (lower and upper differences from center)
    lower_errors = [centers[i] - lower_bounds[i] for i in range(len(centers))]
    upper_errors = [upper_bounds[i] - centers[i] for i in range(len(centers))]
    errors = [lower_errors, upper_errors]
    
    # Sort by center value (descending)
    sorted_indices = np.argsort(-np.array(centers))
    sorted_alt_names = [alt_names[i] for i in sorted_indices]
    sorted_centers = [centers[i] for i in sorted_indices]
    sorted_errors = [[errors[0][i] for i in sorted_indices], [errors[1][i] for i in sorted_indices]]
    
    # Plot horizontal error bars
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(sorted_alt_names))
    ax.errorbar(
        x=sorted_centers,
        y=y_pos,
        xerr=sorted_errors,
        fmt="o",
        capsize=5,
        capthick=2,
        elinewidth=2,
        markersize=8
    )
    
    # Add interval values as text
    for i, (center, lb, ub) in enumerate(zip(sorted_centers, [lower_bounds[j] for j in sorted_indices], [upper_bounds[j] for j in sorted_indices])):
        ax.text(center, y_pos[i] + 0.2, f"[{lb:.3f}, {center:.3f}, {ub:.3f}]", ha="center")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_alt_names)
    ax.set_xlabel("Net Flow with Intervals")
    ax.set_title("PROMETHEE III Intervals (Lower Bound, Net Flow, Upper Bound)")
    ax.grid(axis="x", linestyle="--")
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"Interval visualization saved as '{filename}'")


def visualize_normalized_flows(results, alternatives, filename):
    """Visualize normalized flows from PROMETHEE IV."""
    alt_names = [alt.name for alt in alternatives]
    normalized_flows = results.additional_data["normalized_flows"]
    
    # Create a color map based on flow values
    colors = plt.cm.viridis(np.array(normalized_flows))
    
    # Sort by normalized flow
    sorted_indices = np.argsort(-np.array(normalized_flows))
    sorted_alt_names = [alt_names[i] for i in sorted_indices]
    sorted_flows = [normalized_flows[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(sorted_alt_names, sorted_flows, height=0.6, color=colors[sorted_indices])
    
    # Add value labels
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{sorted_flows[i]:.3f}", va="center", fontweight="bold")
    
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Normalized Net Flow (Higher is better)")
    ax.set_title("PROMETHEE IV Normalized Flows")
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"Normalized flow visualization saved as '{filename}'")


def visualize_constrained_flows(results, alternatives, filename):
    """Visualize constrained flows from PROMETHEE V."""
    alt_names = [alt.name for alt in alternatives]
    constrained_flows = results.additional_data["constrained_flows"]
    feasible = results.additional_data["feasible_alternatives"]
    
    # Create a DataFrame for better visualization
    df = pd.DataFrame({
        "Location": alt_names,
        "Net Flow": results.preferences,
        "Constrained": constrained_flows,
        "Feasible": feasible
    })
    
    # Sort by net flow
    df = df.sort_values("Net Flow", ascending=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot feasible alternatives
    feasible_df = df[df["Feasible"] == True]
    infeasible_df = df[df["Feasible"] == False]
    
    # Plot bars
    ax.barh(feasible_df["Location"], feasible_df["Net Flow"], height=0.6, color="green", alpha=0.7, label="Feasible")
    ax.barh(infeasible_df["Location"], infeasible_df["Net Flow"], height=0.6, color="red", alpha=0.4, label="Infeasible")
    
    # Add value labels
    for i, row in df.iterrows():
        status = "Feasible" if row["Feasible"] else "Infeasible"
        ax.text(row["Net Flow"] + 0.01, i, f"{row['Net Flow']:.3f} ({status})", va="center")
    
    ax.set_xlabel("Net Flow (Higher is better)")
    ax.set_title("PROMETHEE V Constrained Analysis")
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"Constrained flow visualization saved as '{filename}'")


def visualize_stability(results, alternatives, filename):
    """Visualize flow stability from PROMETHEE VI."""
    alt_names = [alt.name for alt in alternatives]
    min_flows = results.additional_data["min_flows"]
    central_flows = results.additional_data["central_flows"]
    max_flows = results.additional_data["max_flows"]
    spread = results.additional_data["ranking_spread"]
    
    # Create a DataFrame for easier sorting and plotting
    df = pd.DataFrame({
        "Location": alt_names,
        "Min": min_flows,
        "Central": central_flows,
        "Max": max_flows,
        "Spread": spread
    })
    
    # Sort by central flow
    df = df.sort_values("Central", ascending=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot bars with error bars
    y_pos = np.arange(len(df))
    ax.errorbar(
        x=df["Central"],
        y=y_pos,
        xerr=[df["Central"] - df["Min"], df["Max"] - df["Central"]],
        fmt="o",
        capsize=5,
        capthick=2,
        elinewidth=2,
        markersize=8
    )
    
    # Add spread labels
    for i, (idx, row) in enumerate(df.iterrows()):
        stability = "High" if row["Spread"] < 0.2 else "Medium" if row["Spread"] < 0.4 else "Low"
        ax.text(row["Central"], y_pos[i] + 0.2, 
                f"[{row['Min']:.3f}, {row['Central']:.3f}, {row['Max']:.3f}] (Stability: {stability})", 
                ha="center")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["Location"])
    ax.set_xlabel("Net Flow Range")
    ax.set_title("PROMETHEE VI Stability Analysis")
    ax.grid(axis="x", linestyle="--")
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"Stability analysis visualization saved as '{filename}'")


def main():
    # Initialize MCDAClient
    client = MCDAClient()
    
    print("\nFacility Location Selection Case Study\n" + "=" * 40)
    
    # 1. Define the decision problem - selecting the best location for a new facility
    print("\n1. Setting up the location selection problem")
    
    # Define alternatives (potential locations)
    location_A = CreateAlternativeRequest(
        id="A1", 
        name="Downtown", 
        description="Central business district location"
    )
    
    location_B = CreateAlternativeRequest(
        id="A2", 
        name="Industrial Zone", 
        description="Industrial park with good infrastructure"
    )
    
    location_C = CreateAlternativeRequest(
        id="A3", 
        name="Suburban Area", 
        description="Developing suburban area with residential proximity"
    )
    
    location_D = CreateAlternativeRequest(
        id="A4", 
        name="Technology Park", 
        description="Modern technology park with research facilities"
    )
    
    location_E = CreateAlternativeRequest(
        id="A5", 
        name="Rural District", 
        description="Low-cost area with development potential"
    )
    
    # Define criteria for location evaluation
    cost = CreateCriterionRequest(
        id="C1", 
        name="Cost", 
        type=CriteriaType.COST, 
        description="Land and construction costs (millions $)",
        weight=0.25
    )
    
    transportation = CreateCriterionRequest(
        id="C2", 
        name="Transportation", 
        type=CriteriaType.BENEFIT, 
        description="Accessibility via transportation networks (score 1-10)",
        weight=0.20
    )
    
    workforce = CreateCriterionRequest(
        id="C3", 
        name="Workforce", 
        type=CriteriaType.BENEFIT, 
        description="Availability of qualified workforce (score 1-10)",
        weight=0.20
    )
    
    expansion = CreateCriterionRequest(
        id="C4", 
        name="Expansion", 
        type=CriteriaType.BENEFIT, 
        description="Possibilities for future expansion (score 1-10)",
        weight=0.15
    )
    
    incentives = CreateCriterionRequest(
        id="C5", 
        name="Incentives", 
        type=CriteriaType.BENEFIT, 
        description="Government incentives and tax benefits (score 1-10)",
        weight=0.10
    )
    
    environmental = CreateCriterionRequest(
        id="C6", 
        name="Environmental", 
        type=CriteriaType.BENEFIT, 
        description="Environmental impact assessment (score 1-10)",
        weight=0.10
    )
    
    # Create the decision matrix with performance values
    decision_matrix = CreateDecisionMatrixRequest(
        alternatives=[location_A, location_B, location_C, location_D, location_E],
        criteria=[cost, transportation, workforce, expansion, incentives, environmental],
        values=[
            # Cost($M), Transport(1-10), Workforce(1-10), Expansion(1-10), Incentives(1-10), Environmental(1-10)
            [8.5, 9, 8, 4, 5, 6],     # Downtown
            [5.2, 7, 7, 8, 7, 5],     # Industrial Zone
            [6.8, 6, 6, 7, 6, 7],     # Suburban Area
            [7.5, 8, 9, 6, 8, 8],     # Technology Park
            [3.5, 4, 4, 9, 9, 7]      # Rural District
        ]
    )
    
    print("Decision matrix created with 5 potential locations and 6 criteria")
    
    # Extract the weights from the criteria
    weights = [crit.weight for crit in [cost, transportation, workforce, expansion, incentives, environmental]]
    print(f"\nCriteria weights: {weights}")
    
    # 2. Define preference functions for each criterion
    print("\n2. Defining preference functions for the PROMETHEE methods")
    
    # Define preference functions for each criterion
    preference_functions = [
        "linear",     # Cost - Linear function with gradual preference increase
        "v-shape",    # Transportation - V-shape for clear preference thresholds
        "gaussian",   # Workforce - Gaussian for smooth transition of preference
        "u-shape",    # Expansion - U-shape for binary preference after a threshold
        "level",      # Incentives - Level function with intermediate preference zone
        "usual"       # Environmental - Usual function for strict preference
    ]
    
    # Define preference thresholds for each criterion
    p_thresholds = [2.0, 3.0, 2.0, 3.0, 2.0, 2.0]  # Preference thresholds
    q_thresholds = [0.5, 1.0, 0.5, 1.0, 0.5, 0.0]  # Indifference thresholds
    
    # Output the preference function settings
    criteria_names = ["Cost", "Transportation", "Workforce", "Expansion", "Incentives", "Environmental"]
    print("\nPreference functions for each criterion:")
    for i, name in enumerate(criteria_names):
        print(f"{name}: {preference_functions[i]} function (p={p_thresholds[i]}, q={q_thresholds[i]})")
    
    # 3. Apply PROMETHEE I for partial ranking
    print("\n3. Applying PROMETHEE I for partial ranking analysis")
    
    # Run PROMETHEE I
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
    
    # Extract and display PROMETHEE I results
    alternatives = [location_A, location_B, location_C, location_D, location_E]
    alt_names = [alt.name for alt in alternatives]
    
    print("\nPROMETHEE I Results:")
    print("Location\tPositive Flow\tNegative Flow\tNet Flow")
    
    for i, alt in enumerate(alternatives):
        pos_flow = promethee1_result.additional_data["positive_flows"][i]
        neg_flow = promethee1_result.additional_data["negative_flows"][i]
        net_flow = promethee1_result.preferences[i]
        print(f"{alt.name}\t{pos_flow:.3f}\t\t{neg_flow:.3f}\t\t{net_flow:.3f}")
    
    # Extract the partial ranking matrix
    partial_ranking = promethee1_result.additional_data["partial_ranking"]
    
    # Display the partial ranking relationships
    print("\nPartial ranking relationships:")
    print("A outranks B: A has better positive AND negative flows or one is better and one equal")
    print("A incomparable to B: A has better positive flow but worse negative flow (or vice versa)")
    print("\nOutranking relationships matrix (1:outranks, 0:incomparable, -1:outranked by):")
    
    # Create a readable table for the partial ranking matrix
    table_data = [[""]+alt_names]
    for i, row in enumerate(partial_ranking):
        table_row = [alt_names[i]]
        for j, val in enumerate(row):
            if i == j:
                table_row.append("-")
            elif val == 1:
                table_row.append("outranks")
            elif val == -1:
                table_row.append("outranked by")
            else:
                table_row.append("incomparable")
        table_data.append(table_row)
    
    print(tabulate(table_data[1:], headers=table_data[0], tablefmt="grid"))
    
    # Visualize the flows
    visualize_flows(promethee1_result, alternatives, "PROMETHEE I Positive and Negative Flows", "promethee1_flows.png")
    
    # 4. Apply PROMETHEE II for complete ranking
    print("\n4. Applying PROMETHEE II for complete ranking")
    
    # Run PROMETHEE II
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
    
    # Display PROMETHEE II results
    print("\nPROMETHEE II Complete Ranking:")
    
    # Sort the alternatives by ranking
    rank_order = np.argsort([promethee2_result.rankings[i] for i in range(len(alternatives))])
    
    for idx in rank_order:
        alt = alternatives[idx]
        rank = promethee2_result.rankings[idx]
        net_flow = promethee2_result.preferences[idx]
        print(f"{rank}. {alt.name} (Net Flow: {net_flow:.3f})")
    
    # 5. Apply PROMETHEE III for interval-based ranking
    print("\n5. Applying PROMETHEE III for interval-based ranking")
    
    # Run PROMETHEE III with alpha parameter
    promethee3_result = client.evaluate({
        "decision_matrix": decision_matrix,
        "method": "promethee3",
        "weights": weights,
        "method_params": {
            "preference_functions": preference_functions,
            "p_thresholds": p_thresholds,
            "q_thresholds": q_thresholds,
            "alpha": 0.15  # Interval parameter
        }
    })
    
    # Display PROMETHEE III results
    print("\nPROMETHEE III Interval-Based Ranking:")
    print("Location\tNet Flow\tLower Bound\tUpper Bound")
    
    intervals = promethee3_result.additional_data["intervals"]
    std_dev = promethee3_result.additional_data["std_dev"]
    alpha = promethee3_result.additional_data["alpha"]
    
    print(f"Standard deviation of flows: {std_dev:.3f}, Alpha parameter: {alpha}")
    print(f"Interval width: ±{alpha * std_dev:.3f}")
    
    for i, alt in enumerate(alternatives):
        interval = intervals[i]
        print(f"{alt.name}\t{interval[0]:.3f}\t{interval[1]:.3f}\t{interval[2]:.3f}")
    
    # Determine indifference relationships based on overlapping intervals
    print("\nIndifference relationships based on overlapping intervals:")
    for i, alt_i in enumerate(alternatives):
        for j, alt_j in enumerate(alternatives):
            if i < j:  # Only check each pair once
                interval_i = intervals[i]
                interval_j = intervals[j]
                
                # Check for overlap
                if (interval_i[1] <= interval_j[2] and interval_i[2] >= interval_j[1]):
                    print(f"{alt_i.name} and {alt_j.name} are indifferent (overlapping intervals)")
    
    # Visualize the intervals
    visualize_intervals(promethee3_result, alternatives, "promethee3_intervals.png")
    
    # 6. Apply PROMETHEE IV for normalized flows
    print("\n6. Applying PROMETHEE IV for normalized flow analysis")
    
    # Run PROMETHEE IV
    promethee4_result = client.evaluate({
        "decision_matrix": decision_matrix,
        "method": "promethee4",
        "weights": weights,
        "method_params": {
            "preference_functions": preference_functions,
            "p_thresholds": p_thresholds,
            "q_thresholds": q_thresholds
        }
    })
    
    # Display PROMETHEE IV results
    print("\nPROMETHEE IV Normalized Flows:")
    print("Location\tNet Flow\tNormalized Flow")
    
    norm_flows = promethee4_result.additional_data["normalized_flows"]
    
    for i, alt in enumerate(alternatives):
        net_flow = promethee4_result.preferences[i]
        norm_flow = norm_flows[i]
        print(f"{alt.name}\t{net_flow:.3f}\t{norm_flow:.3f}")
    
    # Visualize normalized flows
    visualize_normalized_flows(promethee4_result, alternatives, "promethee4_normalized.png")
    
    # 7. Apply PROMETHEE V with constraints
    print("\n7. Applying PROMETHEE V with feasibility constraints")
    
    # Define constraints
    # Example: Location must have cost < 7.0M and transportation score > 5
    def cost_transport_constraint(values):
        cost_value = values[0]  # Cost is first criterion
        transport_value = values[1]  # Transportation is second criterion
        return cost_value < 7.0 and transport_value > 5
    
    # Example: Location must have good environmental score and expansion possibility
    def env_expansion_constraint(values):
        expansion_value = values[3]  # Expansion is fourth criterion
        env_value = values[5]  # Environmental is sixth criterion
        return expansion_value > 6 and env_value > 6
    
    # Run PROMETHEE V with constraints
    promethee5_result = client.evaluate({
        "decision_matrix": decision_matrix,
        "method": "promethee5",
        "weights": weights,
        "method_params": {
            "preference_functions": preference_functions,
            "p_thresholds": p_thresholds,
            "q_thresholds": q_thresholds,
            "constraints": [cost_transport_constraint, env_expansion_constraint]
        }
    })
    
    # Display PROMETHEE V results
    print("\nPROMETHEE V Results with Constraints:")
    print("Applied constraints:")
    print("1. Location must have cost < 7.0M and transportation score > 5")
    print("2. Location must have expansion possibility > 6 and environmental score > 6")
    
    print("\nLocation\tFeasible\tNet Flow\tConstrained Flow")
    
    constrained_flows = promethee5_result.additional_data["constrained_flows"]
    feasible_alternatives = promethee5_result.additional_data["feasible_alternatives"]
    
    for i, alt in enumerate(alternatives):
        feasible = "Yes" if feasible_alternatives[i] else "No"
        net_flow = promethee5_result.preferences[i]
        c_flow = constrained_flows[i] if constrained_flows[i] is not None else "N/A"
        print(f"{alt.name}\t{feasible}\t\t{net_flow:.3f}\t{c_flow}")
    
    # Display only feasible alternatives in rank order
    print("\nFeasible alternatives ranking:")
    feasible_indices = [i for i, feasible in enumerate(feasible_alternatives) if feasible]
    feasible_tuples = [(alternatives[i].name, promethee5_result.preferences[i]) for i in feasible_indices]
    feasible_tuples.sort(key=lambda x: x[1], reverse=True)  # Sort by net flow
    
    for rank, (name, flow) in enumerate(feasible_tuples, 1):
        print(f"{rank}. {name} (Net Flow: {flow:.3f})")
    
    # Visualize constrained flows
    visualize_constrained_flows(promethee5_result, alternatives, "promethee5_constrained.png")
    
    # 8. Apply PROMETHEE VI for stability analysis
    print("\n8. Applying PROMETHEE VI for stability analysis")
    
    # Define weight ranges for each criterion (±40% of original weights)
    original_weights = weights
    weight_ranges = []
    
    for w in original_weights:
        min_w = max(0.01, w * 0.6)  # Ensure minimum weight is positive
        max_w = min(0.5, w * 1.4)   # Ensure maximum weight doesn't exceed 0.5
        weight_ranges.append((min_w, max_w))
    
    # Run PROMETHEE VI
    promethee6_result = client.evaluate({
        "decision_matrix": decision_matrix,
        "method": "promethee6",
        "weights": weights,
        "method_params": {
            "preference_functions": preference_functions,
            "p_thresholds": p_thresholds,
            "q_thresholds": q_thresholds,
            "weight_ranges": weight_ranges,
            "iterations": 500  # Number of Monte Carlo iterations
        }
    })
    
    # Display PROMETHEE VI results
    print("\nPROMETHEE VI Stability Analysis:")
    print("Weight ranges (±40% of original weights):")
    
    for i, name in enumerate(criteria_names):
        print(f"{name}: [{weight_ranges[i][0]:.3f}, {weight_ranges[i][1]:.3f}] (Original: {original_weights[i]:.3f})")
    
    print("\nLocation\tMin Flow\tCentral Flow\tMax Flow\tSpread\tRank Stability")
    
    min_flows = promethee6_result.additional_data["min_flows"]
    central_flows = promethee6_result.additional_data["central_flows"]
    max_flows = promethee6_result.additional_data["max_flows"]
    spread = promethee6_result.additional_data["ranking_spread"]
    
    for i, alt in enumerate(alternatives):
        stability = "High" if spread[i] < 0.2 else "Medium" if spread[i] < 0.4 else "Low"
        print(f"{alt.name}\t{min_flows[i]:.3f}\t{central_flows[i]:.3f}\t{max_flows[i]:.3f}\t{spread[i]:.3f}\t{stability}")
    
    # Visualize stability analysis
    visualize_stability(promethee6_result, alternatives, "promethee6_stability.png")
    
    # 9. Compare results across all PROMETHEE methods
    print("\n9. Comparing results across all PROMETHEE methods")
    
    # Create a comparison table
    methods = ["PROMETHEE I", "PROMETHEE II", "PROMETHEE III", "PROMETHEE IV", "PROMETHEE V", "PROMETHEE VI"]
    results = [promethee1_result, promethee2_result, promethee3_result, promethee4_result, promethee5_result, promethee6_result]
    
    comparison_data = [["Location"] + methods]
    
    # For each alternative, collect the net flow from each method
    for i, alt in enumerate(alternatives):
        row = [alt.name]
        
        # For PROMETHEE I, use net flow
        row.append(f"{promethee1_result.preferences[i]:.3f}")
        
        # For PROMETHEE II, use net flow
        row.append(f"{promethee2_result.preferences[i]:.3f}")
        
        # For PROMETHEE III, use net flow with interval width
        interval = promethee3_result.additional_data["intervals"][i]
        row.append(f"{interval[0]:.3f} [{interval[1]:.3f}, {interval[2]:.3f}]")
        
        # For PROMETHEE IV, use normalized flow
        row.append(f"{promethee4_result.additional_data['normalized_flows'][i]:.3f}")
        
        # For PROMETHEE V, use constrained flow or "Infeasible"
        if promethee5_result.additional_data["feasible_alternatives"][i]:
            row.append(f"{promethee5_result.additional_data['constrained_flows'][i]:.3f}")
        else:
            row.append("Infeasible")
        
        # For PROMETHEE VI, use central flow and range
        row.append(f"{central_flows[i]:.3f} [{min_flows[i]:.3f}, {max_flows[i]:.3f}]")
        
        comparison_data.append(row)
    
    # Print comparison table
    print("\nFlow values across methods:")
    print(tabulate(comparison_data[1:], headers=comparison_data[0], tablefmt="grid"))
    
    # 10. Final recommendation
    print("\n10. Final recommendation")
    
    # Count how many times each alternative is ranked first
    top_ranked = {alt.name: 0 for alt in alternatives}
    feasible_alts = set(alternatives[i].name for i, feasible in enumerate(feasible_alternatives) if feasible)
    
    for result in [promethee2_result, promethee4_result, promethee6_result]:
        for i, rank in enumerate(result.rankings):
            if rank == 1 and alternatives[i].name in feasible_alts:
                top_ranked[alternatives[i].name] += 1
    
    # Find the most stable high-performing alternative
    stable_scores = {}
    for i, alt in enumerate(alternatives):
        if alt.name in feasible_alts:
            # Combine average flow and normalized stability (1-spread)
            avg_flow = (promethee2_result.preferences[i] + 
                       promethee4_result.additional_data["normalized_flows"][i] + 
                       central_flows[i]) / 3
            stability = 1 - (spread[i] / max(spread))  # Normalize and invert spread (higher is more stable)
            stable_scores[alt.name] = 0.7 * avg_flow + 0.3 * stability
    
    # Get the recommendation
    if stable_scores:
        recommended = max(stable_scores.items(), key=lambda x: x[1])[0]
        
        print("\nBased on comprehensive analysis using all PROMETHEE methods:")
        print(f"1. The recommended location is: {recommended}")
        
        # Add details about the recommended location
        rec_idx = [i for i, alt in enumerate(alternatives) if alt.name == recommended][0]
        print(f"   - Net flow: {promethee2_result.preferences[rec_idx]:.3f}")
        print(f"   - Stability: {spread[rec_idx]:.3f} ({promethee6_result.additional_data['min_flows'][rec_idx]:.3f} to {promethee6_result.additional_data['max_flows'][rec_idx]:.3f})")
        print("   - Meets all feasibility constraints")
        
        # Rank all feasible locations
        print("\n2. Ranking of all feasible locations:")
        for rank, (name, score) in enumerate(sorted(stable_scores.items(), key=lambda x: x[1], reverse=True), 1):
            loc_idx = [i for i, alt in enumerate(alternatives) if alt.name == name][0]
            stability = "High" if spread[loc_idx] < 0.2 else "Medium" if spread[loc_idx] < 0.4 else "Low"
            print(f"{rank}. {name} - Score: {score:.3f}, Stability: {stability}")
            
        # Identify key strengths of top location
        print("\n3. Key advantages of the recommended location:")
        alt_priorities = np.array(promethee2_result.additional_data["alternative_priorities"])
        rec_priorities = alt_priorities[rec_idx]
        
        # Find criteria where the recommended location excels
        for i, priority in enumerate(rec_priorities):
            avg_others = np.mean([alt_priorities[j][i] for j in range(len(alternatives)) if j != rec_idx])
            if priority > avg_others * 1.2:  # 20% better than average
                print(f"   - Strong performance in {criteria_names[i]}: {priority:.3f} (vs. average {avg_others:.3f})")
    else:
        print("No locations meet all the feasibility constraints. Consider relaxing some constraints.")


if __name__ == "__main__":
    main()
