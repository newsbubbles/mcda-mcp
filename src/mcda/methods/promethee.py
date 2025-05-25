"""PROMETHEE methods implementation.

Implements the PROMETHEE (Preference Ranking Organization METHod for Enrichment of Evaluations) family of methods 
for multi-criteria decision analysis, including PROMETHEE I, II, III, IV, V, and VI.
"""

from typing import Optional, Dict, Any, List, Union, Callable, Tuple, Literal
import numpy as np
import enum

from mcda.base import MCDAMethod
from mcda.models import DecisionMatrix, MCDAResult, Alternative, Criterion, CriteriaType


class PreferenceFunction(enum.Enum):
    """Preference function types for PROMETHEE."""
    USUAL = "usual"
    U_SHAPE = "u-shape"
    V_SHAPE = "v-shape"
    LEVEL = "level"
    LINEAR = "linear"
    GAUSSIAN = "gaussian"


class PROMETHEEBase(MCDAMethod):
    """Base PROMETHEE (Preference Ranking Organization METHod for Enrichment of Evaluations) method.
    
    Common implementation shared by all PROMETHEE methods.
    """
    
    def _usual_function(self, d: float, p: Optional[float] = None, q: Optional[float] = None) -> float:
        """Usual preference function.
        
        Args:
            d: Difference between alternatives
            p: Preference threshold (not used)
            q: Indifference threshold (not used)
            
        Returns:
            0 if d <= 0, 1 otherwise
        """
        return 0 if d <= 0 else 1
    
    def _u_shape_function(self, d: float, p: Optional[float] = None, q: float = 0) -> float:
        """U-Shape preference function.
        
        Args:
            d: Difference between alternatives
            p: Preference threshold (not used)
            q: Indifference threshold
            
        Returns:
            0 if d <= q, 1 otherwise
        """
        return 0 if d <= q else 1
    
    def _v_shape_function(self, d: float, p: float = 1, q: Optional[float] = None) -> float:
        """V-Shape preference function.
        
        Args:
            d: Difference between alternatives
            p: Preference threshold
            q: Indifference threshold (not used)
            
        Returns:
            0 if d <= 0, d/p if 0 < d <= p, 1 otherwise
        """
        if d <= 0:
            return 0
        elif d > p:
            return 1
        else:
            return d / p
    
    def _level_function(self, d: float, p: float = 1, q: float = 0) -> float:
        """Level preference function.
        
        Args:
            d: Difference between alternatives
            p: Preference threshold
            q: Indifference threshold
            
        Returns:
            0 if d <= q, 0.5 if q < d <= p, 1 otherwise
        """
        if d <= q:
            return 0
        elif d > p:
            return 1
        else:
            return 0.5
    
    def _linear_function(self, d: float, p: float = 1, q: float = 0) -> float:
        """Linear preference function.
        
        Args:
            d: Difference between alternatives
            p: Preference threshold
            q: Indifference threshold
            
        Returns:
            0 if d <= q, (d-q)/(p-q) if q < d <= p, 1 otherwise
        """
        if d <= q:
            return 0
        elif d > p:
            return 1
        else:
            return (d - q) / (p - q)
    
    def _gaussian_function(self, d: float, p: float = 1, q: Optional[float] = None) -> float:
        """Gaussian preference function.
        
        Args:
            d: Difference between alternatives
            p: Preference threshold (used as sigma)
            q: Indifference threshold (not used)
            
        Returns:
            0 if d <= 0, 1 - exp(-(d^2)/(2*p^2)) otherwise
        """
        if d <= 0:
            return 0
        else:
            return 1 - np.exp(-(d ** 2) / (2 * p ** 2))
    
    def _get_preference_function(self, function_type: PreferenceFunction) -> Callable[[float, Optional[float], Optional[float]], float]:
        """Get the appropriate preference function based on type."""
        functions = {
            PreferenceFunction.USUAL: self._usual_function,
            PreferenceFunction.U_SHAPE: self._u_shape_function,
            PreferenceFunction.V_SHAPE: self._v_shape_function,
            PreferenceFunction.LEVEL: self._level_function,
            PreferenceFunction.LINEAR: self._linear_function,
            PreferenceFunction.GAUSSIAN: self._gaussian_function
        }
        return functions[function_type]
    
    def _calculate_flows(self,
                        decision_matrix: DecisionMatrix,
                        weights: Optional[np.ndarray] = None,
                        preference_functions: Optional[List[PreferenceFunction]] = None,
                        p_thresholds: Optional[List[float]] = None,
                        q_thresholds: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate positive, negative and net flows for PROMETHEE methods.
        
        Args:
            decision_matrix: The decision matrix containing alternatives and criteria
            weights: Optional criteria weights (if not included in decision matrix)
            preference_functions: List of preference functions for each criterion
            p_thresholds: List of preference thresholds for each criterion
            q_thresholds: List of indifference thresholds for each criterion
            
        Returns:
            Tuple containing:
            - positive_flow: Numpy array of positive (leaving) flows
            - negative_flow: Numpy array of negative (entering) flows
            - preference_indices: Numpy array of preference indices
        """
        # Get data from decision matrix
        matrix = decision_matrix.to_numpy()
        criteria_types = decision_matrix.get_types_as_coefficients()
        
        # Use provided weights or get from decision matrix
        if weights is None:
            weights = decision_matrix.get_criteria_weights()
            if weights is None:
                raise ValueError("Criteria weights must be provided or included in decision matrix")
        
        # Default preference functions and thresholds if not provided
        n_criteria = matrix.shape[1]
        
        if preference_functions is None:
            # Default to usual function for all criteria
            preference_functions = [PreferenceFunction.USUAL] * n_criteria
        elif len(preference_functions) != n_criteria:
            raise ValueError("Number of preference functions must match number of criteria")
            
        if p_thresholds is None:
            # Default p-threshold: 20% of the range of each criterion
            p_thresholds = []
            for j in range(n_criteria):
                col = matrix[:, j]
                col_range = np.max(col) - np.min(col)
                p_thresholds.append(0.2 * col_range if col_range > 0 else 1.0)
        elif len(p_thresholds) != n_criteria:
            raise ValueError("Number of p-thresholds must match number of criteria")
            
        if q_thresholds is None:
            # Default q-threshold: 5% of the range of each criterion
            q_thresholds = []
            for j in range(n_criteria):
                col = matrix[:, j]
                col_range = np.max(col) - np.min(col)
                q_thresholds.append(0.05 * col_range if col_range > 0 else 0.0)
        elif len(q_thresholds) != n_criteria:
            raise ValueError("Number of q-thresholds must match number of criteria")
            
        # Step 1: Calculate preference values
        n_alternatives = matrix.shape[0]
        preference_matrix = np.zeros((n_alternatives, n_alternatives, n_criteria))
        
        for i in range(n_alternatives):
            for j in range(n_alternatives):
                if i != j:  # Skip comparison with itself
                    for k in range(n_criteria):
                        # Calculate difference (adjusted for criteria type)
                        diff = (matrix[i, k] - matrix[j, k]) * criteria_types[k]
                        
                        # Apply preference function
                        pref_func = self._get_preference_function(preference_functions[k])
                        preference_matrix[i, j, k] = pref_func(diff, p_thresholds[k], q_thresholds[k])
        
        # Step 2: Calculate aggregated preference indices
        preference_indices = np.zeros((n_alternatives, n_alternatives))
        
        for i in range(n_alternatives):
            for j in range(n_alternatives):
                if i != j:
                    # Weighted sum of preference values
                    preference_indices[i, j] = np.sum(preference_matrix[i, j, :] * weights)
        
        # Step 3: Calculate positive and negative outranking flows
        positive_flow = np.sum(preference_indices, axis=1) / (n_alternatives - 1)  # Leaving flow
        negative_flow = np.sum(preference_indices, axis=0) / (n_alternatives - 1)  # Entering flow
        
        return positive_flow, negative_flow, preference_indices


class PROMETHEE1(PROMETHEEBase):
    """PROMETHEE I method.
    
    Computes partial ranking using positive (entering) and negative (leaving) flows.
    """
    
    @property
    def name(self) -> str:
        return "PROMETHEE I"
    
    def evaluate(self,
                 decision_matrix: DecisionMatrix,
                 weights: Optional[np.ndarray] = None,
                 preference_functions: Optional[List[PreferenceFunction]] = None,
                 p_thresholds: Optional[List[float]] = None,
                 q_thresholds: Optional[List[float]] = None,
                 **kwargs) -> MCDAResult:
        """Evaluate alternatives using PROMETHEE I method.
        
        Args:
            decision_matrix: The decision matrix containing alternatives and criteria
            weights: Optional criteria weights (if not included in decision matrix)
            preference_functions: List of preference functions for each criterion
            p_thresholds: List of preference thresholds for each criterion
            q_thresholds: List of indifference thresholds for each criterion
            
        Returns:
            MCDAResult containing preferences and rankings
        """
        positive_flow, negative_flow, preference_indices = self._calculate_flows(
            decision_matrix=decision_matrix, 
            weights=weights,
            preference_functions=preference_functions,
            p_thresholds=p_thresholds,
            q_thresholds=q_thresholds
        )
        
        n_alternatives = len(decision_matrix.alternatives)
        
        # In PROMETHEE I, we don't have a single ranking because it's a partial ranking
        # We'll represent this as a preference graph through pairwise comparisons
        # Each element will be -1 (worse), 0 (incomparable), or 1 (better)
        partial_ranking = np.zeros((n_alternatives, n_alternatives), dtype=int)
        
        for i in range(n_alternatives):
            for j in range(n_alternatives):
                if i != j:
                    # Alternative i outranks j if both flows are better or one is better and the other equal
                    if (positive_flow[i] > positive_flow[j] and negative_flow[i] < negative_flow[j]) or \
                       (positive_flow[i] > positive_flow[j] and negative_flow[i] == negative_flow[j]) or \
                       (positive_flow[i] == positive_flow[j] and negative_flow[i] < negative_flow[j]):
                        partial_ranking[i, j] = 1
                    # Alternatives are incomparable if the flows give contradictory results
                    elif (positive_flow[i] > positive_flow[j] and negative_flow[i] > negative_flow[j]) or \
                         (positive_flow[i] < positive_flow[j] and negative_flow[i] < negative_flow[j]):
                        partial_ranking[i, j] = 0
                    # Otherwise, j outranks i
                    else:
                        partial_ranking[i, j] = -1
        
        # For consistency with the MCDAResult model, we need to provide some kind of ranking
        # We'll use the net flow as a simplified preference score
        net_flow = positive_flow - negative_flow
        ranks = np.argsort(-net_flow)  # Descending order
        rankings = np.argsort(ranks) + 1  # Convert to 1-based rankings
        
        return MCDAResult(
            method_name=self.name,
            preferences=net_flow.tolist(),
            rankings=rankings.tolist(),
            alternatives=decision_matrix.alternatives,
            additional_data={
                "positive_flows": positive_flow.tolist(),
                "negative_flows": negative_flow.tolist(),
                "partial_ranking": partial_ranking.tolist(),
                "preference_functions": [pf.value for pf in preference_functions] if preference_functions else None,
                "p_thresholds": p_thresholds,
                "q_thresholds": q_thresholds
            }
        )


class PROMETHEE2(PROMETHEEBase):
    """PROMETHEE II method.
    
    Computes complete ranking using net flows (positive - negative).
    """
    
    @property
    def name(self) -> str:
        return "PROMETHEE II"
    
    def evaluate(self,
                 decision_matrix: DecisionMatrix,
                 weights: Optional[np.ndarray] = None,
                 preference_functions: Optional[List[PreferenceFunction]] = None,
                 p_thresholds: Optional[List[float]] = None,
                 q_thresholds: Optional[List[float]] = None,
                 **kwargs) -> MCDAResult:
        """Evaluate alternatives using PROMETHEE II method.
        
        Args:
            decision_matrix: The decision matrix containing alternatives and criteria
            weights: Optional criteria weights (if not included in decision matrix)
            preference_functions: List of preference functions for each criterion
            p_thresholds: List of preference thresholds for each criterion
            q_thresholds: List of indifference thresholds for each criterion
            
        Returns:
            MCDAResult containing preferences (net flow) and rankings
        """
        positive_flow, negative_flow, _ = self._calculate_flows(
            decision_matrix=decision_matrix, 
            weights=weights,
            preference_functions=preference_functions,
            p_thresholds=p_thresholds,
            q_thresholds=q_thresholds
        )
        
        # Calculate net flow
        net_flow = positive_flow - negative_flow
        
        # Rank the alternatives (higher net flow is better)
        ranks = np.argsort(-net_flow)  # Descending order
        rankings = np.argsort(ranks) + 1  # Convert to 1-based rankings
        
        return MCDAResult(
            method_name=self.name,
            preferences=net_flow.tolist(),
            rankings=rankings.tolist(),
            alternatives=decision_matrix.alternatives,
            additional_data={
                "positive_flows": positive_flow.tolist(),
                "negative_flows": negative_flow.tolist(),
                "preference_functions": [pf.value for pf in preference_functions] if preference_functions else None,
                "p_thresholds": p_thresholds,
                "q_thresholds": q_thresholds
            }
        )


class PROMETHEE3(PROMETHEEBase):
    """PROMETHEE III method.
    
    Computes ranking with intervals to account for uncertainty, using net flows and an alpha parameter.
    """
    
    @property
    def name(self) -> str:
        return "PROMETHEE III"
    
    def evaluate(self,
                 decision_matrix: DecisionMatrix,
                 weights: Optional[np.ndarray] = None,
                 preference_functions: Optional[List[PreferenceFunction]] = None,
                 p_thresholds: Optional[List[float]] = None,
                 q_thresholds: Optional[List[float]] = None,
                 alpha: float = 0.15,
                 **kwargs) -> MCDAResult:
        """Evaluate alternatives using PROMETHEE III method.
        
        Args:
            decision_matrix: The decision matrix containing alternatives and criteria
            weights: Optional criteria weights (if not included in decision matrix)
            preference_functions: List of preference functions for each criterion
            p_thresholds: List of preference thresholds for each criterion
            q_thresholds: List of indifference thresholds for each criterion
            alpha: Interval parameter (default 0.15)
            
        Returns:
            MCDAResult containing preferences and rankings
        """
        positive_flow, negative_flow, _ = self._calculate_flows(
            decision_matrix=decision_matrix, 
            weights=weights,
            preference_functions=preference_functions,
            p_thresholds=p_thresholds,
            q_thresholds=q_thresholds
        )
        
        # Calculate net flow
        net_flow = positive_flow - negative_flow
        n_alternatives = len(net_flow)
        
        # Calculate standard deviation of net flow
        std_dev = np.std(net_flow) if n_alternatives > 1 else 0
        
        # Calculate interval for each alternative
        intervals = []
        for i in range(n_alternatives):
            lower_bound = net_flow[i] - alpha * std_dev
            upper_bound = net_flow[i] + alpha * std_dev
            intervals.append((float(net_flow[i]), float(lower_bound), float(upper_bound)))
        
        # In PROMETHEE III, two alternatives are indifferent if their intervals overlap
        # Create a ranking based on the interval midpoints
        ranks = np.argsort(-net_flow)  # Descending order based on net flow
        rankings = np.argsort(ranks) + 1  # Convert to 1-based rankings
        
        return MCDAResult(
            method_name=self.name,
            preferences=net_flow.tolist(),
            rankings=rankings.tolist(),
            alternatives=decision_matrix.alternatives,
            additional_data={
                "positive_flows": positive_flow.tolist(),
                "negative_flows": negative_flow.tolist(),
                "intervals": intervals,
                "alpha": alpha,
                "std_dev": float(std_dev),
                "preference_functions": [pf.value for pf in preference_functions] if preference_functions else None,
                "p_thresholds": p_thresholds,
                "q_thresholds": q_thresholds
            }
        )


class PROMETHEE4(PROMETHEEBase):
    """PROMETHEE IV method.
    
    Computes normalized net flows for a continuous case (simplified for discrete alternatives).
    """
    
    @property
    def name(self) -> str:
        return "PROMETHEE IV"
    
    def evaluate(self,
                 decision_matrix: DecisionMatrix,
                 weights: Optional[np.ndarray] = None,
                 preference_functions: Optional[List[PreferenceFunction]] = None,
                 p_thresholds: Optional[List[float]] = None,
                 q_thresholds: Optional[List[float]] = None,
                 **kwargs) -> MCDAResult:
        """Evaluate alternatives using PROMETHEE IV method.
        
        Args:
            decision_matrix: The decision matrix containing alternatives and criteria
            weights: Optional criteria weights (if not included in decision matrix)
            preference_functions: List of preference functions for each criterion
            p_thresholds: List of preference thresholds for each criterion
            q_thresholds: List of indifference thresholds for each criterion
            
        Returns:
            MCDAResult containing preferences and rankings
        """
        positive_flow, negative_flow, _ = self._calculate_flows(
            decision_matrix=decision_matrix, 
            weights=weights,
            preference_functions=preference_functions,
            p_thresholds=p_thresholds,
            q_thresholds=q_thresholds
        )
        
        # Calculate net flow
        net_flow = positive_flow - negative_flow
        
        # Normalize net flow to [0, 1] range
        min_flow = np.min(net_flow)
        max_flow = np.max(net_flow)
        
        if max_flow > min_flow:
            normalized_flows = (net_flow - min_flow) / (max_flow - min_flow)
        else:
            normalized_flows = np.ones_like(net_flow) * 0.5  # Equal ranking if all flows are the same
        
        # Rank the alternatives (higher net flow is better)
        ranks = np.argsort(-normalized_flows)  # Descending order
        rankings = np.argsort(ranks) + 1  # Convert to 1-based rankings
        
        return MCDAResult(
            method_name=self.name,
            preferences=net_flow.tolist(),  # Original net flow for consistency
            rankings=rankings.tolist(),
            alternatives=decision_matrix.alternatives,
            additional_data={
                "positive_flows": positive_flow.tolist(),
                "negative_flows": negative_flow.tolist(),
                "normalized_flows": normalized_flows.tolist(),
                "preference_functions": [pf.value for pf in preference_functions] if preference_functions else None,
                "p_thresholds": p_thresholds,
                "q_thresholds": q_thresholds
            }
        )


class PROMETHEE5(PROMETHEEBase):
    """PROMETHEE V method.
    
    Computes net flows with constraints to filter feasible alternatives.
    """
    
    @property
    def name(self) -> str:
        return "PROMETHEE V"
    
    def evaluate(self,
                 decision_matrix: DecisionMatrix,
                 weights: Optional[np.ndarray] = None,
                 preference_functions: Optional[List[PreferenceFunction]] = None,
                 p_thresholds: Optional[List[float]] = None,
                 q_thresholds: Optional[List[float]] = None,
                 constraints: Optional[List[Callable[[List[float]], bool]]] = None,
                 **kwargs) -> MCDAResult:
        """Evaluate alternatives using PROMETHEE V method.
        
        Args:
            decision_matrix: The decision matrix containing alternatives and criteria
            weights: Optional criteria weights (if not included in decision matrix)
            preference_functions: List of preference functions for each criterion
            p_thresholds: List of preference thresholds for each criterion
            q_thresholds: List of indifference thresholds for each criterion
            constraints: List of constraint functions that take criteria values and return boolean
            
        Returns:
            MCDAResult containing preferences and rankings
        """
        positive_flow, negative_flow, _ = self._calculate_flows(
            decision_matrix=decision_matrix, 
            weights=weights,
            preference_functions=preference_functions,
            p_thresholds=p_thresholds,
            q_thresholds=q_thresholds
        )
        
        # Calculate net flow
        net_flow = positive_flow - negative_flow
        n_alternatives = len(net_flow)
        matrix = decision_matrix.to_numpy()
        
        # Apply constraints if provided
        constrained_flows = net_flow.copy()
        if constraints is not None:
            for i in range(n_alternatives):
                alternative_values = matrix[i, :].tolist()
                
                # Check if the alternative satisfies all constraints
                if not all(constraint(alternative_values) for constraint in constraints):
                    constrained_flows[i] = None
        
        # Rank only the feasible alternatives (higher net flow is better)
        feasible_mask = np.array([flow is not None for flow in constrained_flows])
        
        if np.any(feasible_mask):
            feasible_flows = np.array([flow if flow is not None else -np.inf for flow in constrained_flows])
            ranks = np.zeros(n_alternatives, dtype=int)
            
            # Rank feasible alternatives
            feasible_indices = np.where(feasible_mask)[0]
            feasible_ranks = np.argsort(-feasible_flows[feasible_indices])
            
            # Assign ranks to feasible alternatives
            for i, idx in enumerate(feasible_indices[feasible_ranks]):
                ranks[idx] = i + 1
            
            # Assign a default large rank to infeasible alternatives
            infeasible_indices = np.where(~feasible_mask)[0]
            ranks[infeasible_indices] = len(feasible_indices) + 1
            
            # Convert to 1-based rankings
            rankings = ranks.tolist()
        else:
            # If no feasible alternatives, all have the same rank
            rankings = [n_alternatives] * n_alternatives
        
        # Convert None to NaN for JSON serialization
        constrained_flows_list = [float(flow) if flow is not None else None for flow in constrained_flows]
        
        return MCDAResult(
            method_name=self.name,
            preferences=net_flow.tolist(),  # Original net flow for consistency
            rankings=rankings,
            alternatives=decision_matrix.alternatives,
            additional_data={
                "positive_flows": positive_flow.tolist(),
                "negative_flows": negative_flow.tolist(),
                "constrained_flows": constrained_flows_list,
                "feasible_alternatives": feasible_mask.tolist(),
                "preference_functions": [pf.value for pf in preference_functions] if preference_functions else None,
                "p_thresholds": p_thresholds,
                "q_thresholds": q_thresholds
            }
        )


class PROMETHEE6(PROMETHEEBase):
    """PROMETHEE VI method.
    
    Computes min, central, and max flows to model hesitation in decision-making.
    """
    
    @property
    def name(self) -> str:
        return "PROMETHEE VI"
    
    def evaluate(self,
                 decision_matrix: DecisionMatrix,
                 weights: Optional[np.ndarray] = None,
                 preference_functions: Optional[List[PreferenceFunction]] = None,
                 p_thresholds: Optional[List[float]] = None,
                 q_thresholds: Optional[List[float]] = None,
                 weight_ranges: Optional[List[Tuple[float, float]]] = None,
                 iterations: int = 100,
                 **kwargs) -> MCDAResult:
        """Evaluate alternatives using PROMETHEE VI method.
        
        Args:
            decision_matrix: The decision matrix containing alternatives and criteria
            weights: Optional criteria weights (if not included in decision matrix)
            preference_functions: List of preference functions for each criterion
            p_thresholds: List of preference thresholds for each criterion
            q_thresholds: List of indifference thresholds for each criterion
            weight_ranges: List of (min, max) weight ranges for each criterion
            iterations: Number of iterations for Monte Carlo simulation
            
        Returns:
            MCDAResult containing preferences and rankings
        """
        # Get data from decision matrix
        matrix = decision_matrix.to_numpy()
        n_alternatives = matrix.shape[0]
        n_criteria = matrix.shape[1]
        
        # Use provided weights or get from decision matrix
        if weights is None:
            weights = decision_matrix.get_criteria_weights()
            if weights is None:
                raise ValueError("Criteria weights must be provided or included in decision matrix")
        
        # Process weight ranges
        if weight_ranges is None:
            # Default: ±30% of the central weight
            weight_ranges = [(max(0, w * 0.7), min(1, w * 1.3)) for w in weights]
        elif len(weight_ranges) != n_criteria:
            raise ValueError("Number of weight ranges must match number of criteria")
        
        # Generate random weights for each iteration
        all_net_flows = np.zeros((n_alternatives, iterations))
        
        for i in range(iterations):
            # Generate random weights within the specified ranges
            random_weights = np.array([np.random.uniform(low, high) for low, high in weight_ranges])
            
            # Normalize weights to sum to 1
            random_weights = random_weights / np.sum(random_weights)
            
            # Calculate flows with random weights
            positive_flow, negative_flow, _ = self._calculate_flows(
                decision_matrix=decision_matrix, 
                weights=random_weights,
                preference_functions=preference_functions,
                p_thresholds=p_thresholds,
                q_thresholds=q_thresholds
            )
            
            # Store the net flow for this iteration
            all_net_flows[:, i] = positive_flow - negative_flow
        
        # Calculate min, central, and max flows for each alternative
        min_flows = np.min(all_net_flows, axis=1)
        central_flows = np.mean(all_net_flows, axis=1)
        max_flows = np.max(all_net_flows, axis=1)
        
        # Rank based on central flows (higher is better)
        ranks = np.argsort(-central_flows)  # Descending order
        rankings = np.argsort(ranks) + 1  # Convert to 1-based rankings
        
        # Calculate stability of ranking
        ranking_spread = max_flows - min_flows
        
        return MCDAResult(
            method_name=self.name,
            preferences=central_flows.tolist(),  # Central flow as the main preference
            rankings=rankings.tolist(),
            alternatives=decision_matrix.alternatives,
            additional_data={
                "min_flows": min_flows.tolist(),
                "central_flows": central_flows.tolist(),
                "max_flows": max_flows.tolist(),
                "ranking_spread": ranking_spread.tolist(),
                "iterations": iterations,
                "weight_ranges": weight_ranges,
                "preference_functions": [pf.value for pf in preference_functions] if preference_functions else None,
                "p_thresholds": p_thresholds,
                "q_thresholds": q_thresholds
            }
        )


# Alias for backward compatibility
PROMETHEE = PROMETHEE2