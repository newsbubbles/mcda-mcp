"""Portfolio Optimization Use Case.

This example demonstrates the use of MCDA methods for portfolio optimization - selecting the best mix of investments.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import pandas as pd

from mcda.client import MCDAClient, CreateAlternativeRequest, CreateCriterionRequest, CreateDecisionMatrixRequest
from mcda.models import CriteriaType


def visualize_portfolio_comparison(result):
    """Create a bar chart comparing the investment options across criteria."""
    # Extract data
    alternatives = result.alternatives
    alt_names = [alt.name for alt in alternatives]
    criteria_names = ["Expected Return", "Risk (Volatility)", "Liquidity", "Time Horizon", "ESG Score"] 
    
    # Get alternative priorities for each criterion from AHP result
    alt_priorities = np.array(result.additional_data["alternative_priorities"])
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame(alt_priorities, index=alt_names, columns=criteria_names)
    
    # Plot stacked bar chart
    ax = df.plot(kind='bar', stacked=False, figsize=(12, 6), width=0.8)
    
    # Customize plot
    ax.set_xlabel('Investment Option')
    ax.set_ylabel('Score')
    ax.set_title('Investment Options Comparison Across Criteria')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("portfolio_comparison.png")
    plt.close()
    
    print("\nPortfolio comparison chart saved as 'portfolio_comparison.png'")
    
    # Create pie chart for optimal allocation
    preferences = result.preferences
    normalized_prefs = np.array(preferences) / sum(preferences)
    
    plt.figure(figsize=(10, 8))
    plt.pie(normalized_prefs, labels=alt_names, autopct='%1.1f%%', startangle=90, shadow=True)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Recommended Portfolio Allocation')
    plt.savefig("portfolio_allocation.png")
    plt.close()
    
    print("Portfolio allocation pie chart saved as 'portfolio_allocation.png'")


def main():
    # Initialize MCDAClient
    client = MCDAClient()
    
    print("\nPortfolio Optimization Case Study\n" + "=" * 30)
    
    # 1. Define the decision problem - selecting the best investment portfolio
    print("\n1. Setting up the portfolio optimization problem")
    
    # Define alternatives (investment options)
    stock_fund = CreateAlternativeRequest(
        id="A1", 
        name="Stock Fund", 
        description="Diversified equity fund with focus on growth stocks"
    )
    
    bond_fund = CreateAlternativeRequest(
        id="A2", 
        name="Bond Fund", 
        description="Government and corporate bond fund with regular income"
    )
    
    real_estate = CreateAlternativeRequest(
        id="A3", 
        name="Real Estate", 
        description="REIT fund investing in commercial and residential properties"
    )
    
    crypto = CreateAlternativeRequest(
        id="A4", 
        name="Cryptocurrency", 
        description="Basket of major cryptocurrencies"
    )
    
    gold = CreateAlternativeRequest(
        id="A5", 
        name="Gold ETF", 
        description="Exchange-traded fund tracking gold prices"
    )
    
    # Define criteria for investment evaluation
    returns = CreateCriterionRequest(
        id="C1", 
        name="Expected Return", 
        type=CriteriaType.BENEFIT, 
        description="Projected annual return percentage"
    )
    
    risk = CreateCriterionRequest(
        id="C2", 
        name="Risk (Volatility)", 
        type=CriteriaType.COST, 
        description="Standard deviation of historical returns"
    )
    
    liquidity = CreateCriterionRequest(
        id="C3", 
        name="Liquidity", 
        type=CriteriaType.BENEFIT, 
        description="Ease of converting to cash without significant loss of value"
    )
    
    time_horizon = CreateCriterionRequest(
        id="C4", 
        name="Time Horizon", 
        type=CriteriaType.COST, 
        description="Recommended minimum investment period in years"
    )
    
    esg = CreateCriterionRequest(
        id="C5", 
        name="ESG Score", 
        type=CriteriaType.BENEFIT, 
        description="Environmental, Social, and Governance rating"
    )
    
    # Create the decision matrix with performance values
    decision_matrix = CreateDecisionMatrixRequest(
        alternatives=[stock_fund, bond_fund, real_estate, crypto, gold],
        criteria=[returns, risk, liquidity, time_horizon, esg],
        values=[
            # Return(%), Risk(%), Liquidity(1-10), Time Horizon(years), ESG Score(1-10)
            [12.0, 18.0, 9, 5, 7],    # Stock Fund
            [5.0, 5.0, 8, 3, 6],      # Bond Fund
            [8.0, 12.0, 3, 7, 5],      # Real Estate
            [25.0, 45.0, 7, 4, 3],     # Cryptocurrency
            [6.0, 15.0, 9, 2, 4]       # Gold ETF
        ]
    )
    
    print("Decision matrix created with 5 investment options and 5 criteria")
    
    # 2. Calculate weights using the entropy method (objective weighting)
    print("\n2. Calculating objective criteria weights using entropy method")
    
    # Calculate entropy weights
    entropy_weights = client.calculate_weights({
        "method": "entropy",
        "decision_matrix": decision_matrix
    })
    
    # Display entropy weights
    print("\nEntropy-based weights:")
    criteria_names = ["Expected Return", "Risk (Volatility)", "Liquidity", "Time Horizon", "ESG Score"]
    for i, name in enumerate(criteria_names):
        print(f"{name}: {entropy_weights.weights[i]:.3f}")
    
    # 3. Adjust weights based on investor preferences (manual adjustment)
    print("\n3. Adjusting weights based on investor preferences")
    
    # Define investor profile: Balanced Growth
    investor_weights = [0.30, 0.30, 0.15, 0.10, 0.15]  # Custom weights reflecting a balanced growth strategy
    
    print("\nInvestor preference weights (Balanced Growth profile):")
    for i, name in enumerate(criteria_names):
        print(f"{name}: {investor_weights[i]:.3f}")
    
    # 4. Evaluate investment options using multiple MCDA methods
    print("\n4. Evaluating investment options using different MCDA methods")
    
    # Define method-specific parameters
    method_params = {
        "topsis": {
            "normalization_method": "linear_minmax"
        },
        "vikor": {
            "v": 0.5  # Balance between group utility and individual regret
        },
        "promethee2": {
            "preference_functions": ["linear", "v-shape", "linear", "linear", "gaussian"]
        }
    }
    
    # Compare multiple methods
    methods = ["topsis", "vikor", "promethee2", "ahp", "wsm", "wpm"]
    
    comparison_result = client.compare_methods({
        "decision_matrix": decision_matrix,
        "methods": methods,
        "weights": investor_weights,
        "method_params": method_params
    })
    
    # Display comparison results
    print("\nRanking results across different methods:")
    print("\nInvestment\t" + "\t".join(methods))
    
    alternatives = [stock_fund, bond_fund, real_estate, crypto, gold]
    for i, alt in enumerate(alternatives):
        rankings = [comparison_result.results[method].rankings[i] for method in methods]
        print(f"{alt.name}\t" + "\t".join([str(rank) for rank in rankings]))
    
    # Display agreement between methods
    print(f"\nOverall agreement between methods: {comparison_result.agreement_rate:.3f}")
    
    # 5. Analyze specific methods in detail
    print("\n5. Detailed TOPSIS analysis for portfolio optimization")
    
    # Get TOPSIS result
    topsis_result = client.evaluate({
        "decision_matrix": decision_matrix,
        "method": "topsis",
        "weights": investor_weights,
        "method_params": method_params["topsis"]
    })
    
    # Display TOPSIS scores and rankings
    print("\nTOPSIS results (closeness to ideal solution):")
    for i, alt in enumerate(alternatives):
        print(f"Rank {topsis_result.rankings[i]}: {alt.name} (Score: {topsis_result.preferences[i]:.4f})")
    
    # Display additional TOPSIS data
    if "ideal_solution" in topsis_result.additional_data:
        print("\nIdeal solution values:")
        for i, name in enumerate(criteria_names):
            print(f"{name}: {topsis_result.additional_data['ideal_solution'][i]:.4f}")
    
    if "anti_ideal_solution" in topsis_result.additional_data:
        print("\nAnti-ideal solution values:")
        for i, name in enumerate(criteria_names):
            print(f"{name}: {topsis_result.additional_data['anti_ideal_solution'][i]:.4f}")
    
    # 6. Perform stability analysis with PROMETHEE
    print("\n6. Performing stability analysis with PROMETHEE VI")
    
    # Define weight ranges for sensitivity analysis (±30% of defined weights)
    weight_ranges = [
        (max(0.1, investor_weights[0]*0.7), min(0.5, investor_weights[0]*1.3)),  # Return
        (max(0.1, investor_weights[1]*0.7), min(0.5, investor_weights[1]*1.3)),  # Risk
        (max(0.05, investor_weights[2]*0.7), min(0.3, investor_weights[2]*1.3)),  # Liquidity
        (max(0.05, investor_weights[3]*0.7), min(0.2, investor_weights[3]*1.3)),  # Time Horizon
        (max(0.05, investor_weights[4]*0.7), min(0.3, investor_weights[4]*1.3))   # ESG
    ]
    
    # Run PROMETHEE VI
    promethee6_result = client.evaluate({
        "decision_matrix": decision_matrix,
        "method": "promethee6",
        "weights": investor_weights,
        "method_params": {
            "preference_functions": ["linear", "v-shape", "linear", "linear", "gaussian"],
            "weight_ranges": weight_ranges,
            "iterations": 500
        }
    })
    
    # Display PROMETHEE VI results
    print("\nStability analysis results:")
    print("\nInvestment\tMin Flow\tCentral Flow\tMax Flow\tRank Stability")
    
    for i, alt in enumerate(alternatives):
        min_flow = promethee6_result.additional_data["min_flows"][i]
        central_flow = promethee6_result.additional_data["central_flows"][i]
        max_flow = promethee6_result.additional_data["max_flows"][i]
        spread = promethee6_result.additional_data["ranking_spread"][i]
        stability = "High" if spread < 0.2 else "Medium" if spread < 0.4 else "Low"
        
        print(f"{alt.name}\t{min_flow:.4f}\t{central_flow:.4f}\t{max_flow:.4f}\t{stability}")
    
    # 7. Determine optimal portfolio allocation using AHP results
    print("\n7. Determining optimal portfolio allocation")
    
    # Use AHP for the final portfolio allocation
    ahp_result = client.evaluate({
        "decision_matrix": decision_matrix,
        "method": "ahp",
        "weights": investor_weights
    })
    
    # Calculate portfolio allocation percentages
    preferences = ahp_result.preferences
    total_preference = sum(preferences)
    allocation_percentages = [(pref / total_preference) * 100 for pref in preferences]
    
    # Display recommended allocation
    print("\nRecommended portfolio allocation:")
    for i, alt in enumerate(alternatives):
        print(f"{alt.name}: {allocation_percentages[i]:.1f}%")
    
    # 8. Visualize portfolio comparison and allocation
    print("\n8. Visualizing portfolio comparison and allocation")
    visualize_portfolio_comparison(ahp_result)
    
    # 9. Final recommendation
    print("\n9. Final investment recommendation")
    print("\nBased on comprehensive analysis across multiple MCDA methods, the recommended balanced growth portfolio allocation is:")
    for i, alt in enumerate(alternatives):
        print(f"- {alt.name}: {allocation_percentages[i]:.1f}%")
    
    # Find best and worst performing investments
    avg_ranks = []
    for i in range(len(alternatives)):
        ranks = [comparison_result.results[method].rankings[i] for method in methods]
        avg_rank = sum(ranks) / len(ranks)
        avg_ranks.append((alternatives[i].name, avg_rank))
    
    best_investment = min(avg_ranks, key=lambda x: x[1])[0]
    worst_investment = max(avg_ranks, key=lambda x: x[1])[0]
    
    print(f"\n• {best_investment} appears to be the strongest single investment option across all evaluation criteria and methods.")
    print(f"• {worst_investment} shows the poorest overall performance and should have limited allocation.")
    print("• For the optimal risk-adjusted return, follow the above percentage allocation.")
    print("• Consider rebalancing the portfolio quarterly to maintain the target allocation.")


if __name__ == "__main__":
    main()
