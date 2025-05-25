# Multi-Criteria Decision Analysis (MCDA) Client Project

## Project Overview

This project aims to create a comprehensive MCDA client, MCP server, and agent for performing multi-criteria decision analysis. The implementation will allow users to evaluate and rank alternatives based on multiple criteria, weighting schemes, and normalization methods.

## Requirements

### Core MCDA Methods

1. **TOPSIS** (Technique for Order of Preference by Similarity to Ideal Solution)
2. **AHP** (Analytic Hierarchy Process)
3. **VIKOR** (VIseKriterijumska Optimizacija I Kompromisno Resenje)
4. **PROMETHEE II** (Preference Ranking Organization METHod for Enrichment of Evaluations II)
5. **WSM** (Weighted Sum Model)
6. **WPM** (Weighted Product Model)
7. **WASPAS** (Weighted Aggregated Sum Product Assessment)

### Weighting Methods

1. **Manual Weighting** - User-provided weights
2. **AHP Weighting** - Derive weights using AHP pairwise comparisons
3. **Entropy Weighting** - Objective weights based on information entropy
4. **Equal Weighting** - All criteria receive equal importance

### Normalization Methods

1. **Vector Normalization**
2. **Linear Normalization (Min-Max)**
3. **Linear Normalization (Max)**
4. **Linear Normalization (Sum)**

### Features

1. Input/output for decision matrices
2. Customizable criteria types (benefit/cost)
3. Result visualization
4. Sensitivity analysis
5. Comparison of multiple methods

## Project Structure

```
mcda_project/
├── src/
│   ├── mcda/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── methods/
│   │   │   ├── __init__.py
│   │   │   ├── topsis.py
│   │   │   ├── ahp.py
│   │   │   ├── vikor.py
│   │   │   ├── promethee.py
│   │   │   ├── wsm.py
│   │   │   ├── wpm.py
│   │   │   └── waspas.py
│   │   ├── weighting/
│   │   │   ├── __init__.py
│   │   │   ├── manual.py
│   │   │   ├── ahp.py
│   │   │   ├── entropy.py
│   │   │   └── equal.py
│   │   ├── normalization/
│   │   │   ├── __init__.py
│   │   │   ├── vector.py
│   │   │   ├── linear_minmax.py
│   │   │   ├── linear_max.py
│   │   │   └── linear_sum.py
│   │   ├── visualization.py
│   │   └── sensitivity.py
│   └── mcp_server.py
├── tests/
├── examples/
└── docs/
```

## Implementation Plan

1. Create the base data structures and interfaces
2. Implement normalization methods
3. Implement weighting methods
4. Implement core MCDA methods
5. Build visualization and sensitivity analysis components
6. Create the integrated client API
7. Develop the MCP server based on the client
8. Implement an agent for end-to-end testing

## Progress

- [ ] Set up project structure
- [ ] Implement data structures and base interfaces
- [ ] Implement normalization methods
- [ ] Implement weighting methods
- [ ] Implement MCDA methods
- [ ] Implement visualization and sensitivity analysis
- [ ] Create client API
- [ ] Create MCP server
- [ ] Create test agent

