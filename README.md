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
