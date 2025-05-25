# Multi-Criteria Decision Analysis (MCDA) Library

This library provides tools for performing multi-criteria decision analysis. It includes a comprehensive set of MCDA methods, weighting techniques, and normalization approaches.

## Components

1. **MCDA Client** - Core library with implementation of MCDA methods
2. **MCP Server** - Integration with Model Context Protocol (MCP) for LLM use
3. **MCDA Agent** - AI assistant for guided decision analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mcda_project.git
cd mcda_project

# Install dependencies
pip install -r requirements.txt
```

## MCDA Methods Included

- **TOPSIS** (Technique for Order of Preference by Similarity to Ideal Solution)
- **AHP** (Analytic Hierarchy Process)
- **VIKOR** (VIseKriterijumska Optimizacija I Kompromisno Resenje)
- **PROMETHEE** (Preference Ranking Organization METHod for Enrichment of Evaluations)
  - PROMETHEE I - Partial ranking using positive and negative flows
  - PROMETHEE II - Complete ranking using net flows
  - PROMETHEE III - Ranking with intervals to account for uncertainty
  - PROMETHEE IV - Normalized net flows for continuous case
  - PROMETHEE V - Net flows with constraints to filter feasible alternatives
  - PROMETHEE VI - Min, central, and max flows to model hesitation in decision-making
- **WSM** (Weighted Sum Model)
- **WPM** (Weighted Product Model)
- **WASPAS** (Weighted Aggregated Sum Product Assessment)

## Weighting Methods

- **Manual Weighting** - User-provided weights
- **AHP Weighting** - Derive weights using AHP pairwise comparisons
- **Entropy Weighting** - Objective weights based on information entropy
- **Equal Weighting** - All criteria receive equal importance

## Normalization Methods

- **Vector Normalization**
- **Linear Min-Max Normalization**
- **Linear Max Normalization**
- **Linear Sum Normalization**

## Usage

### Using the MCDA Client Directly

```python
from mcda.client import MCDAClient, CreateAlternativeRequest, CreateCriterionRequest, CreateDecisionMatrixRequest, EvaluateRequest
from mcda.models import CriteriaType

# Create MCDA client
client = MCDAClient()

# Define alternatives
alternatives = [
    CreateAlternativeRequest(id="alt1", name="Alternative 1"),
    CreateAlternativeRequest(id="alt2", name="Alternative 2"),
    CreateAlternativeRequest(id="alt3", name="Alternative 3"),
]

# Define criteria
criteria = [
    CreateCriterionRequest(id="crit1", name="Cost", type=CriteriaType.COST),
    CreateCriterionRequest(id="crit2", name="Quality", type=CriteriaType.BENEFIT),
    CreateCriterionRequest(id="crit3", name="Delivery Time", type=CriteriaType.COST),
]

# Define performance values
values = [
    [100, 8, 5],   # Alternative 1
    [150, 9, 3],   # Alternative 2
    [120, 7, 8],   # Alternative 3
]

# Create decision matrix
decision_matrix = CreateDecisionMatrixRequest(
    alternatives=alternatives,
    criteria=criteria,
    values=values
)

# Define weights
weights = [0.4, 0.4, 0.2]

# Create evaluation request
evaluate_request = EvaluateRequest(
    decision_matrix=decision_matrix,
    method="topsis",
    weights=weights
)

# Evaluate alternatives
result = client.evaluate(evaluate_request)

# Print results
for i, alt in enumerate(result.alternatives):
    print(f"{alt.name}: Score = {result.preferences[i]:.4f}, Rank = {result.rankings[i]}")
```

### Using the PROMETHEE Methods

```python
# Use PROMETHEE I for partial ranking
result = client.evaluate({
    "decision_matrix": decision_matrix,
    "method": "promethee1",
    "method_params": {
        "preference_functions": ["linear", "usual", "v-shape"]
    }
})

# Use PROMETHEE VI to analyze stability with weight ranges
weight_ranges = [(0.3, 0.5), (0.3, 0.5), (0.1, 0.3)]  # min-max ranges
result = client.evaluate({
    "decision_matrix": decision_matrix,
    "method": "promethee6",
    "weights": weights,
    "method_params": {
        "preference_functions": ["linear", "usual", "v-shape"],
        "weight_ranges": weight_ranges
    }
})
```

### Using AHP with Pairwise Comparisons

```python
# Define pairwise comparison matrix for criteria
criteria_comparisons = [
    [1, 3, 5],   # Criterion 1 compared to others
    [1/3, 1, 2], # Criterion 2 compared to others
    [1/5, 1/2, 1] # Criterion 3 compared to others
]

# Define pairwise comparison matrices for each alternative (one matrix per criterion)
alternative_comparisons = [
    [  # For Criterion 1
        [1, 2, 3],
        [1/2, 1, 2],
        [1/3, 1/2, 1]
    ],
    [  # For Criterion 2
        [1, 1/2, 2],
        [2, 1, 3],
        [1/2, 1/3, 1]
    ],
    [  # For Criterion 3
        [1, 3, 1/2],
        [1/3, 1, 1/4],
        [2, 4, 1]
    ]
]

# Use AHP with pairwise comparisons
result = client.evaluate({
    "decision_matrix": decision_matrix,  # Still needed for alternative/criteria info
    "method": "ahp",
    "method_params": {
        "criteria_comparisons": criteria_comparisons,
        "alternative_comparisons": alternative_comparisons,
        "consistency_threshold": 0.1  # Maximum acceptable consistency ratio
    }
})
```

### Running the MCP Server

The MCP server allows the MCDA functionality to be accessed by LLMs like Claude.

```bash
# Run the MCP server directly
python src/mcp_server.py

# Or run it through the agent interface
python agent.py
```

### Using the Agent

The MCDA Agent provides an interactive interface for using MCDA functionality through an AI assistant.

```bash
# Set up environment variables (only needed for OpenRouter integration)
export OPENROUTER_API_KEY=your_api_key_here

# Run the agent
python agent.py

# Specify a different model (optional)
python agent.py --model anthropic/claude-3.7-haiku
```

## Example Agent Interactions

```
> I need to choose a new laptop for software development. I have three options: a gaming laptop for $1800, a business ultrabook for $1500, and a mid-range laptop for $1200. I care about processor speed, battery life, weight, and storage capacity.

[Agent helps set up the decision matrix, weights, and analyzes the options]

> Compare the results using different MCDA methods

[Agent compares various methods and explains differences in rankings]

> Help me calculate weights using the AHP method

[Agent guides through pairwise comparisons and shows calculated weights]

> Can you analyze this decision using PROMETHEE VI to account for my uncertainty in the weights?

[Agent sets up weight ranges and shows how rankings might change with different weight combinations]
```

## Development

### Project Structure

```
mcda_project/
├── src/                      # Source code
│   ├── mcda/                 # MCDA library
│   │   ├── methods/          # MCDA methods
│   │   ├── weighting/        # Weighting methods
│   │   ├── normalization/    # Normalization methods
│   │   ├── client.py         # Main client interface
│   │   ├── base.py           # Base classes and interfaces
│   │   └── models.py         # Data models
│   └── mcp_server.py         # MCP server implementation
├── agents/                   # Agent prompts
│   └── MCDAAgent.md          # MCDA Agent system prompt
├── tests/                    # Tests
├── examples/                 # Example usage scripts
├── docs/                     # Documentation
├── agent.py                  # Agent script
└── README.md                 # This file
```

### Running Tests

```bash
# Run unit tests
python -m unittest discover tests
```

## License

MIT
