"""Tests for comparing multiple MCDA methods.

This module contains tests for comparing the results of different MCDA methods.
"""

import unittest
import numpy as np
from mcda.client import MCDAClient, CreateAlternativeRequest, CreateCriterionRequest, CreateDecisionMatrixRequest, CompareMethodsRequest
from mcda.models import CriteriaType


class TestMethodComparison(unittest.TestCase):
    """Test case for comparing multiple MCDA methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = MCDAClient()
        
        # Create a decision problem for supplier selection
        self.altA = CreateAlternativeRequest(id="A1", name="Supplier A")
        self.altB = CreateAlternativeRequest(id="A2", name="Supplier B")
        self.altC = CreateAlternativeRequest(id="A3", name="Supplier C")
        self.altD = CreateAlternativeRequest(id="A4", name="Supplier D")
        
        # Define criteria
        self.crit1 = CreateCriterionRequest(id="C1", name="Cost", type=CriteriaType.COST, weight=0.35)
        self.crit2 = CreateCriterionRequest(id="C2", name="Quality", type=CriteriaType.BENEFIT, weight=0.25)
        self.crit3 = CreateCriterionRequest(id="C3", name="Delivery Time", type=CriteriaType.COST, weight=0.20)
        self.crit4 = CreateCriterionRequest(id="C4", name="Service", type=CriteriaType.BENEFIT, weight=0.20)
        
        # Create decision matrix with values
        self.decision_matrix = CreateDecisionMatrixRequest(
            alternatives=[self.altA, self.altB, self.altC, self.altD],
            criteria=[self.crit1, self.crit2, self.crit3, self.crit4],
            values=[
                [110, 90, 12, 70],  # Supplier A
                [130, 95, 9, 80],   # Supplier B
                [120, 85, 11, 85],  # Supplier C
                [105, 80, 14, 75],  # Supplier D
            ]
        )
        
        # List of methods to compare
        self.methods = ["topsis", "ahp", "vikor", "promethee2", "wsm", "wpm"]
        
        # Method-specific parameters
        self.method_params = {
            "topsis": {
                "normalization_method": "vector"
            },
            "vikor": {
                "v": 0.5  # Consensus parameter
            },
            "promethee2": {
                "preference_functions": ["linear", "u-shape", "v-shape", "usual"]
            }
        }

    def test_compare_methods(self):
        """Test comparison of multiple MCDA methods."""
        # Create comparison request
        compare_request = CompareMethodsRequest(
            decision_matrix=self.decision_matrix,
            methods=self.methods,
            method_params=self.method_params
        )
        
        # Perform comparison
        result = self.client.compare_methods(compare_request)
        
        # Verify the result
        self.assertEqual(len(result.results), len(self.methods))  # Results for all methods
        
        # Check that all methods are present in the results
        for method in self.methods:
            self.assertIn(method, result.results)
        
        # Check correlation matrix dimensions and properties
        for method1 in self.methods:
            self.assertIn(method1, result.correlation_matrix)
            for method2 in self.methods:
                self.assertIn(method2, result.correlation_matrix[method1])
                # Correlation with self should be 1.0
                if method1 == method2:
                    self.assertEqual(result.correlation_matrix[method1][method2], 1.0)
                # Correlation should be between -1.0 and 1.0
                else:
                    self.assertGreaterEqual(result.correlation_matrix[method1][method2], -1.0)
                    self.assertLessEqual(result.correlation_matrix[method1][method2], 1.0)
        
        # Check agreement rate
        self.assertGreaterEqual(result.agreement_rate, -1.0)
        self.assertLessEqual(result.agreement_rate, 1.0)

    def test_compare_specific_methods(self):
        """Test comparison of specific method subsets."""
        # Compare only TOPSIS, AHP, and PROMETHEE2
        methods_subset = ["topsis", "ahp", "promethee2"]
        
        # Create comparison request
        compare_request = CompareMethodsRequest(
            decision_matrix=self.decision_matrix,
            methods=methods_subset,
            method_params=self.method_params
        )
        
        # Perform comparison
        result = self.client.compare_methods(compare_request)
        
        # Verify the result
        self.assertEqual(len(result.results), len(methods_subset))  # Results for selected methods only
        
        # Check that all selected methods are present in the results
        for method in methods_subset:
            self.assertIn(method, result.results)
        
        # Check that other methods are not in the results
        for method in set(self.methods) - set(methods_subset):
            self.assertNotIn(method, result.results)


if __name__ == '__main__':
    unittest.main()
