"""Tests for weighting methods.

This module contains tests for different weighting methods.
"""

import unittest
import numpy as np
from mcda.client import MCDAClient, CreateAlternativeRequest, CreateCriterionRequest, CreateDecisionMatrixRequest, CalculateWeightsRequest
from mcda.models import CriteriaType


class TestWeightingMethods(unittest.TestCase):
    """Test case for weighting methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = MCDAClient()
        
        # Create a decision matrix for testing entropy weighting
        self.altA = CreateAlternativeRequest(id="A1", name="Alternative A")
        self.altB = CreateAlternativeRequest(id="A2", name="Alternative B")
        self.altC = CreateAlternativeRequest(id="A3", name="Alternative C")
        self.altD = CreateAlternativeRequest(id="A4", name="Alternative D")
        
        # Define criteria without weights (to be calculated)
        self.crit1 = CreateCriterionRequest(id="C1", name="Criterion 1", type=CriteriaType.BENEFIT)
        self.crit2 = CreateCriterionRequest(id="C2", name="Criterion 2", type=CriteriaType.COST)
        self.crit3 = CreateCriterionRequest(id="C3", name="Criterion 3", type=CriteriaType.BENEFIT)
        
        # Create decision matrix with values
        self.decision_matrix = CreateDecisionMatrixRequest(
            alternatives=[self.altA, self.altB, self.altC, self.altD],
            criteria=[self.crit1, self.crit2, self.crit3],
            values=[
                [8, 5, 7],  # Alternative A
                [6, 3, 9],  # Alternative B
                [9, 7, 6],  # Alternative C
                [7, 4, 8],  # Alternative D
            ]
        )
        
        # Define criteria for AHP weighting
        self.ahp_criteria = [
            CreateCriterionRequest(id="C1", name="Criterion 1", type=CriteriaType.BENEFIT),
            CreateCriterionRequest(id="C2", name="Criterion 2", type=CriteriaType.COST),
            CreateCriterionRequest(id="C3", name="Criterion 3", type=CriteriaType.BENEFIT),
        ]
        
        # Define pairwise comparison matrix for AHP
        self.comparison_matrix = [
            [1.0, 3.0, 2.0],  # Criterion 1 compared to others
            [1/3, 1.0, 1.0],  # Criterion 2 compared to others
            [1/2, 1.0, 1.0],  # Criterion 3 compared to others
        ]

    def test_equal_weighting(self):
        """Test equal weighting method."""
        # Calculate weights using equal weighting
        result = self.client.calculate_weights(CalculateWeightsRequest(
            method="equal",
            criteria=self.ahp_criteria
        ))
        
        # Verify the result
        self.assertEqual(result.method_name, "Equal Weighting")
        self.assertEqual(len(result.weights), 3)  # 3 criteria
        
        # Check that all weights are equal and sum to 1.0
        self.assertAlmostEqual(result.weights[0], 1.0/3, places=6)
        self.assertAlmostEqual(sum(result.weights), 1.0, places=6)

    def test_entropy_weighting(self):
        """Test entropy weighting method."""
        # Calculate weights using entropy weighting
        result = self.client.calculate_weights(CalculateWeightsRequest(
            method="entropy",
            decision_matrix=self.decision_matrix
        ))
        
        # Verify the result
        self.assertEqual(result.method_name, "Entropy Weighting")
        self.assertEqual(len(result.weights), 3)  # 3 criteria
        
        # Check that weights sum to 1.0
        self.assertAlmostEqual(sum(result.weights), 1.0, places=6)
        
        # Check that all weights are between 0 and 1
        for weight in result.weights:
            self.assertGreaterEqual(weight, 0.0)
            self.assertLessEqual(weight, 1.0)

    def test_ahp_weighting(self):
        """Test AHP weighting method."""
        # Calculate weights using AHP weighting
        result = self.client.calculate_weights(CalculateWeightsRequest(
            method="ahp",
            criteria=self.ahp_criteria,
            method_params={
                "comparison_matrix": self.comparison_matrix
            }
        ))
        
        # Verify the result
        self.assertEqual(result.method_name, "AHP Weighting")
        self.assertEqual(len(result.weights), 3)  # 3 criteria
        
        # Check that weights sum to 1.0
        self.assertAlmostEqual(sum(result.weights), 1.0, places=6)
        
        # Check that all weights are between 0 and 1
        for weight in result.weights:
            self.assertGreaterEqual(weight, 0.0)
            self.assertLessEqual(weight, 1.0)
        
        # Check that additional data contains consistency information
        self.assertIn("consistency_ratio", result.additional_data)
        self.assertIn("is_consistent", result.additional_data)

    def test_manual_weighting(self):
        """Test manual weighting method."""
        # Define manual weights
        manual_weights = [0.4, 0.35, 0.25]
        
        # Calculate weights using manual weighting
        result = self.client.calculate_weights(CalculateWeightsRequest(
            method="manual",
            criteria=self.ahp_criteria,
            method_params={
                "weights": manual_weights
            }
        ))
        
        # Verify the result
        self.assertEqual(result.method_name, "Manual Weighting")
        self.assertEqual(len(result.weights), 3)  # 3 criteria
        
        # Check that weights match the input
        for i, weight in enumerate(result.weights):
            self.assertAlmostEqual(weight, manual_weights[i], places=6)
        
        # Check that weights sum to 1.0
        self.assertAlmostEqual(sum(result.weights), 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
