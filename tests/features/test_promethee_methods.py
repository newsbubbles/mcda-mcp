"""Tests for PROMETHEE methods.

This module contains tests for all PROMETHEE methods (I-VI).
"""

import unittest
import numpy as np
from mcda.client import MCDAClient, CreateAlternativeRequest, CreateCriterionRequest, CreateDecisionMatrixRequest
from mcda.models import CriteriaType
from mcda.methods.promethee import PreferenceFunction, PROMETHEE1, PROMETHEE2, PROMETHEE3, PROMETHEE4, PROMETHEE5, PROMETHEE6


class TestPROMETHEEMethods(unittest.TestCase):
    """Test case for PROMETHEE methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = MCDAClient()
        
        # Create a decision problem for car selection
        self.altA = CreateAlternativeRequest(id="A1", name="Car A", description="Compact sedan")
        self.altB = CreateAlternativeRequest(id="A2", name="Car B", description="SUV")
        self.altC = CreateAlternativeRequest(id="A3", name="Car C", description="Luxury sedan")
        self.altD = CreateAlternativeRequest(id="A4", name="Car D", description="Electric vehicle")
        
        # Define criteria
        self.crit1 = CreateCriterionRequest(id="C1", name="Price", type=CriteriaType.COST, weight=0.25)
        self.crit2 = CreateCriterionRequest(id="C2", name="Acceleration", type=CriteriaType.COST, weight=0.20)
        self.crit3 = CreateCriterionRequest(id="C3", name="Fuel Economy", type=CriteriaType.BENEFIT, weight=0.30)
        self.crit4 = CreateCriterionRequest(id="C4", name="Comfort", type=CriteriaType.BENEFIT, weight=0.15)
        self.crit5 = CreateCriterionRequest(id="C5", name="Safety", type=CriteriaType.BENEFIT, weight=0.10)
        
        # Create decision matrix with values
        self.decision_matrix = CreateDecisionMatrixRequest(
            alternatives=[self.altA, self.altB, self.altC, self.altD],
            criteria=[self.crit1, self.crit2, self.crit3, self.crit4, self.crit5],
            values=[
                [25000, 9.5, 30, 7, 8],   # Car A
                [32000, 11.2, 22, 8, 9],  # Car B
                [42000, 7.8, 20, 9, 8],   # Car C
                [38000, 8.2, 120, 8, 9]   # Car D
            ]
        )
        
        # Define preference function types for each criterion
        self.preference_functions = ["linear", "v-shape", "u-shape", "usual", "gaussian"]

    def test_promethee1(self):
        """Test PROMETHEE I method."""
        # Evaluate using the client
        result = self.client.evaluate({
            "decision_matrix": self.decision_matrix,
            "method": "promethee1",
            "method_params": {
                "preference_functions": self.preference_functions
            }
        })
        
        # Verify the result
        self.assertEqual(result.method_name, "PROMETHEE I")
        self.assertEqual(len(result.preferences), 4)  # 4 alternatives
        self.assertEqual(len(result.rankings), 4)  # 4 rankings
        
        # Check that additional data contains positive and negative flows
        self.assertIn("positive_flows", result.additional_data)
        self.assertIn("negative_flows", result.additional_data)
        self.assertIn("partial_ranking", result.additional_data)
        
        # Check partial ranking shape
        self.assertEqual(len(result.additional_data["partial_ranking"]), 4)
        self.assertEqual(len(result.additional_data["partial_ranking"][0]), 4)

    def test_promethee2(self):
        """Test PROMETHEE II method."""
        # Evaluate using the client
        result = self.client.evaluate({
            "decision_matrix": self.decision_matrix,
            "method": "promethee2",
            "method_params": {
                "preference_functions": self.preference_functions
            }
        })
        
        # Verify the result
        self.assertEqual(result.method_name, "PROMETHEE II")
        self.assertEqual(len(result.preferences), 4)  # 4 alternatives
        self.assertEqual(len(result.rankings), 4)  # 4 rankings
        
        # Check that additional data contains positive and negative flows
        self.assertIn("positive_flows", result.additional_data)
        self.assertIn("negative_flows", result.additional_data)
        
        # Check sum of rankings (1+2+3+4 = 10)
        self.assertEqual(sum(result.rankings), 10)

    def test_promethee3(self):
        """Test PROMETHEE III method."""
        # Evaluate using the client
        result = self.client.evaluate({
            "decision_matrix": self.decision_matrix,
            "method": "promethee3",
            "method_params": {
                "preference_functions": self.preference_functions,
                "alpha": 0.2  # Interval parameter
            }
        })
        
        # Verify the result
        self.assertEqual(result.method_name, "PROMETHEE III")
        self.assertEqual(len(result.preferences), 4)  # 4 alternatives
        self.assertEqual(len(result.rankings), 4)  # 4 rankings
        
        # Check that additional data contains intervals
        self.assertIn("intervals", result.additional_data)
        self.assertEqual(len(result.additional_data["intervals"]), 4)
        
        # Check interval format (net_flow, lower_bound, upper_bound)
        for interval in result.additional_data["intervals"]:
            self.assertEqual(len(interval), 3)
            self.assertLessEqual(interval[1], interval[0])  # lower_bound <= net_flow
            self.assertGreaterEqual(interval[2], interval[0])  # upper_bound >= net_flow

    def test_promethee4(self):
        """Test PROMETHEE IV method."""
        # Evaluate using the client
        result = self.client.evaluate({
            "decision_matrix": self.decision_matrix,
            "method": "promethee4",
            "method_params": {
                "preference_functions": self.preference_functions
            }
        })
        
        # Verify the result
        self.assertEqual(result.method_name, "PROMETHEE IV")
        self.assertEqual(len(result.preferences), 4)  # 4 alternatives
        self.assertEqual(len(result.rankings), 4)  # 4 rankings
        
        # Check that additional data contains normalized flows
        self.assertIn("normalized_flows", result.additional_data)
        self.assertEqual(len(result.additional_data["normalized_flows"]), 4)
        
        # Check normalized flows range [0, 1]
        for flow in result.additional_data["normalized_flows"]:
            self.assertGreaterEqual(flow, 0)
            self.assertLessEqual(flow, 1)

    def test_promethee5(self):
        """Test PROMETHEE V method."""
        # Define a constraint function that price must be under 40000
        def price_constraint(values):
            return values[0] < 40000
        
        # Evaluate using the client
        result = self.client.evaluate({
            "decision_matrix": self.decision_matrix,
            "method": "promethee5",
            "method_params": {
                "preference_functions": self.preference_functions,
                "constraints": [price_constraint]
            }
        })
        
        # Verify the result
        self.assertEqual(result.method_name, "PROMETHEE V")
        self.assertEqual(len(result.preferences), 4)  # 4 alternatives
        self.assertEqual(len(result.rankings), 4)  # 4 rankings
        
        # Check that additional data contains constrained flows
        self.assertIn("constrained_flows", result.additional_data)
        self.assertIn("feasible_alternatives", result.additional_data)
        
        # Verify constraint application: Car C (42000) should be infeasible
        self.assertFalse(result.additional_data["feasible_alternatives"][2])  # Car C
        self.assertIsNone(result.additional_data["constrained_flows"][2])

    def test_promethee6(self):
        """Test PROMETHEE VI method."""
        # Define weight ranges as (min, max) for each criterion
        weight_ranges = [
            (0.15, 0.35),  # Price (25% ± 10%)
            (0.10, 0.30),  # Acceleration (20% ± 10%)
            (0.20, 0.40),  # Fuel Economy (30% ± 10%)
            (0.05, 0.25),  # Comfort (15% ± 10%)
            (0.05, 0.15)   # Safety (10% ± 5%)
        ]
        
        # Evaluate using the client
        result = self.client.evaluate({
            "decision_matrix": self.decision_matrix,
            "method": "promethee6",
            "method_params": {
                "preference_functions": self.preference_functions,
                "weight_ranges": weight_ranges,
                "iterations": 100  # Number of Monte Carlo iterations
            }
        })
        
        # Verify the result
        self.assertEqual(result.method_name, "PROMETHEE VI")
        self.assertEqual(len(result.preferences), 4)  # 4 alternatives
        self.assertEqual(len(result.rankings), 4)  # 4 rankings
        
        # Check that additional data contains min, central, max flows
        self.assertIn("min_flows", result.additional_data)
        self.assertIn("central_flows", result.additional_data)
        self.assertIn("max_flows", result.additional_data)
        self.assertIn("ranking_spread", result.additional_data)
        
        # Check flow ranges: min_flow <= central_flow <= max_flow
        for i in range(4):
            self.assertLessEqual(result.additional_data["min_flows"][i], result.additional_data["central_flows"][i])
            self.assertLessEqual(result.additional_data["central_flows"][i], result.additional_data["max_flows"][i])


if __name__ == '__main__':
    unittest.main()
