# MCDA System Usage Guide

This document provides instructions for using the Multi-Criteria Decision Analysis (MCDA) system, including the client library, MCP server, and agent.

## Quick Start

### 1. Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Using the MCDA Agent

The easiest way to get started is by running the MCDA Agent:

```bash
# Set up environment variables (if using OpenRouter)
export OPENROUTER_API_KEY=your_api_key_here

# Run the agent
python agent.py
```

You'll get a command-line interface where you can interact with the AI assistant. The assistant will use the MCDA tools to help with decision analysis.

### 3. Running the MCP Server Directly

If you want to run the MCP server without the agent:

```bash
python src/mcp_server.py
```

This will start the server in stdio mode, ready to communicate with any MCP client.

## MCDA Workflow Examples

### Basic Decision Analysis

Here's how to perform a basic MCDA analysis with the agent:

1. **Define the problem**: Describe your decision problem to the agent.

```
> I need to choose between three laptops for work. I'm considering price, performance, battery life, and weight.
```

2. **Build the decision matrix**: The agent will guide you through creating the decision matrix with alternatives and criteria.

3. **Specify criteria types and weights**: Indicate which criteria are benefit or cost, and how important each is.

4. **Get the results**: The agent will run the analysis and present the results.

### Advanced Analysis with Method Comparison

For a more comprehensive analysis:

```
> Compare the laptop options using multiple MCDA methods and show me if there's agreement in the rankings.
```

The agent will use the `compare_methods` tool to apply multiple methods and analyze their agreement rate.

### AHP Weight Calculation

To calculate weights using the Analytic Hierarchy Process:

```
> Help me determine the weights for my criteria using AHP pairwise comparison.
```

The agent will guide you through the process of pairwise comparisons and calculate the weights.

## Direct API Usage

If you prefer to use the MCDA library programmatically, you can import and use the client directly:

```python
from mcda.client import MCDAClient

# Create client
client = MCDAClient()

# Get available methods
methods = client.get_available_methods()
print(methods)

# Use the client for MCDA analysis
# (See the example in README.md for detailed usage)
```

## MCP Server Tools Reference

The MCP server exposes these tools:

1. **get_available_methods** - Returns all available MCDA methods, weighting methods, and normalization methods.

2. **evaluate_mcda** - Evaluates alternatives using a specified MCDA method.
   - Required input: decision matrix, method name
   - Optional: weights, method parameters

3. **compare_methods** - Compares results from multiple MCDA methods.
   - Required input: decision matrix, list of methods to compare
   - Optional: weights, method parameters

4. **calculate_weights** - Calculates criteria weights using a specified method.
   - Required input: weighting method name
   - Optional: decision matrix or criteria list, method parameters

5. **create_ahp_comparison_matrix** - Creates an empty AHP comparison matrix with diagonal filled with 1s.
   - Required input: criteria names

6. **validate_ahp_matrix** - Validates an AHP matrix for reciprocal values and consistency.
   - Required input: comparison matrix

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're running from the project root directory.

2. **Agent connection issues**: Check that you have set the required environment variables:
   ```bash
   export OPENROUTER_API_KEY=your_api_key_here
   ```

3. **Matrix validation errors**: Ensure your decision matrix has the correct dimensions and that criteria weights sum to 1.

4. **AHP consistency issues**: If your AHP matrix has a high consistency ratio, revise your pairwise comparisons.

### Logs

Check the logs for detailed error information:

```
src/logs/mcda_mcp.log
```

## Advanced Configuration

### Customizing the Agent

You can modify the agent's system prompt by editing:

```
agents/MCDAAgent.md
```

### Changing the Model

Specify a different model for the agent:

```bash
python agent.py --model anthropic/claude-3.7-haiku
```

### Environment Variables

The system checks for these environment variables:

- `OPENROUTER_API_KEY` - Required for the agent's LLM access
- `LOGFIRE_API_KEY` - Optional for enhanced logging
- Any variables starting with `MCDA_` or `NUMPY_` are forwarded to the MCP server
