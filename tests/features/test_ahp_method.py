"""Tests for AHP method.

This module contains tests for the AHP (Analytic Hierarchy Process) method.
"""

import unittest
import numpy as np
from mcda.client import MCDAClient, CreateAlternativeRequest, CreateCriterionRequest, CreateDecisionMatrixRequest
from mcda.models import CriteriaType
from mcda.methods.ahp import AHP


class TestAHPMethod(unittest.TestCase):
    """Test case for AHP method."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = MCDAClient()
        
        # Create a decision problem for car selection
        self.altA = CreateAlternativeRequest(id="A1", name="Car A", description="Compact sedan")
        self.altB = CreateAlternativeRequest(id="A2", name="Car B", description="SUV")
        self.altC = CreateAlternativeRequest(id="A3", name="Car C", description="Luxury sedan")
        
        # Define criteria
        self.crit1 = CreateCriterionRequest(id="C1", name="Price", type=CriteriaType.COST, weight=0.4)
        self.crit2 = CreateCriterionRequest(id="C2", name="Performance", type=CriteriaType.BENEFIT, weight=0.3)
        self.crit3 = CreateCriterionRequest(id="C3", name="Fuel Economy", type=CriteriaType.BENEFIT, weight=0.3)
        
        # Create decision matrix with values
        self.decision_matrix = CreateDecisionMatrixRequest(
            alternatives=[self.altA, self.altB, self.altC],
            criteria=[self.crit1, self.crit2, self.crit3],
            values=[
                [25000, 7, 30],  # Car A
                [32000, 8, 22],  # Car B
                [42000, 9, 20],  # Car C
            ]
        )
        
        # Define criteria comparison matrix (3x3)
        self.criteria_comparisons = [
            [1.0, 2.0, 3.0],  # Price compared to others
            [0.5, 1.0, 1.0],  # Performance compared to others
            [0.33, 1.0, 1.0]  # Fuel Economy compared to others
        ]
        
        # Define alternative comparison matrices (3x3), one for each criterion
        self.alt_comp_price = [
            [1.0, 2.0, 4.0],  # Car A compared to others (Price)
            [0.5, 1.0, 3.0],  # Car B compared to others (Price)
            [0.25, 0.33, 1.0]  # Car C compared to others (Price)
        ]
        
        self.alt_comp_performance = [
            [1.0, 0.5, 0.33],  # Car A compared to others (Performance)
            [2.0, 1.0, 0.5],   # Car B compared to others (Performance)
            [3.0, 2.0, 1.0]    # Car C compared to others (Performance)
        ]
        
        self.alt_comp_fuel = [
            [1.0, 3.0, 4.0],   # Car A compared to others (Fuel Economy)
            [0.33, 1.0, 1.5],  # Car B compared to others (Fuel Economy)
            [0.25, 0.67, 1.0]  # Car C compared to others (Fuel Economy)
        ]
        
        # Combine all alternative comparison matrices
        self.alternative_comparisons = [
            self.alt_comp_price,
            self.alt_comp_performance,
            self.alt_comp_fuel
        ]

    def test_ahp_with_decision_matrix(self):
        """Test AHP with direct decision matrix."""
        # Evaluate using the client
        result = self.client.evaluate({
            "decision_matrix": self.decision_matrix,
            "method": "ahp"
        })
        
        # Verify the result
        self.assertEqual(result.method_name, "AHP")
        self.assertEqual(len(result.preferences), 3)  # 3 alternatives
        self.assertEqual(len(result.rankings), 3)  # 3 rankings
        
        # Check that additional data contains weights and priorities
        self.assertIn("criteria_weights", result.additional_data)
        self.assertIn("alternative_priorities", result.additional_data)
        self.assertIn("consistency_info", result.additional_data)
        
        # Verify weights sum to approximately 1.0
        weights_sum = sum(result.additional_data["criteria_weights"])
        self.assertAlmostEqual(weights_sum, 1.0, places=6)
        
        # Verify rankings sum to 6 (1+2+3)
        self.assertEqual(sum(result.rankings), 6)

    def test_ahp_with_pairwise_comparisons(self):
        """Test AHP with pairwise comparison matrices."""
        # Evaluate using the client
        result = self.client.evaluate({
            "decision_matrix": self.decision_matrix,  # Still needed for alternative/criteria info
            "method": "ahp",
            "method_params": {
                "criteria_comparisons": self.criteria_comparisons,
                "alternative_comparisons": self.alternative_comparisons,
                "consistency_threshold": 0.15  # Higher threshold for testing
            }
        })
        
        # Verify the result
        self.assertEqual(result.method_name, "AHP")
        self.assertEqual(len(result.preferences), 3)  # 3 alternatives
        self.assertEqual(len(result.rankings), 3)  # 3 rankings
        
        # Check that additional data contains weights and priorities
        self.assertIn("criteria_weights", result.additional_data)
        self.assertIn("alternative_priorities", result.additional_data)
        self.assertIn("consistency_info", result.additional_data)
        
        # Verify consistency information
        consistency_info = result.additional_data["consistency_info"]
        self.assertIn("criteria_consistency_ratio", consistency_info)
        self.assertIn("alternative_consistency_ratios", consistency_info)
        self.assertIn("max_consistency_ratio", consistency_info)
        self.assertIn("is_consistent", consistency_info)
        
        # Verify weights sum to approximately 1.0
        weights_sum = sum(result.additional_data["criteria_weights"])
        self.assertAlmostEqual(weights_sum, 1.0, places=6)
        
        # Check alternative priorities shape
        alt_priorities = np.array(result.additional_data["alternative_priorities"])
        self.assertEqual(alt_priorities.shape, (3, 3))  # 3 alternatives x 3 criteria

    def test_ahp_consistency_check(self):
        """Test AHP consistency checking."""
        # Create an intentionally inconsistent comparison matrix
        inconsistent_criteria = [
            [1.0, 5.0, 9.0],
            [0.2, 1.0, 9.0],  # Inconsistent: if 1>2 (5) and 2>3 (9), then 1>3 should be much higher than 9
            [0.11, 0.11, 1.0]
        ]
        
        # Evaluate using the client with strict consistency threshold
        result = self.client.evaluate({
            "decision_matrix": self.decision_matrix,
            "method": "ahp",
            "method_params": {
                "criteria_comparisons": inconsistent_criteria,
                "alternative_comparisons": self.alternative_comparisons,
                "consistency_threshold": 0.05  # Strict threshold
            }
        })
        
        # Check that the method detects inconsistency
        self.assertFalse(result.additional_data["consistency_info"]["is_consistent"])
        self.assertGreater(result.additional_data["consistency_info"]["criteria_consistency_ratio"], 0.05)


if __name__ == '__main__':
    unittest.main()
