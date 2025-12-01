# Replicating Martingales Project

## Overview
This project implements **Replicating Martingales** for the pricing and risk management of financial derivatives. It focuses on comparing the efficiency and accuracy of Replicating Martingales against standard **Nested Monte Carlo (Nested MC)** and **Least Squares Monte Carlo (LSMC)** methods.

The project supports pricing and risk analysis for:
- **European Call Options**
- **Variable Annuities** (with GMDB - Guaranteed Minimum Death Benefit)

## Features
- **Scenario Generation**: Geometric Brownian Motion (GBM) for asset prices and Lee-Carter model for mortality rates.
- **Pricing Models**:
  - **Nested Monte Carlo**: The benchmark for accuracy, though computationally expensive.
  - **Least Squares Monte Carlo (LSMC)**: A proxy model using polynomial basis functions.
  - **Replicating Martingale**: A machine learning-based approach (using Neural Networks or Polynomials) to learn the hedging strategy directly.
- **Risk Analysis**: Calculation of Mean, Standard Deviation, and 95% Value at Risk (VaR) over a specified risk horizon.

## Project Structure
```
.
├── config/                 # Configuration files for experiments
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Source code
│   ├── data/               # Scenario generation logic
│   ├── models/             # Pricing and risk models (NestedMC, LSMC, ReplicatingMartingale)
│   └── products/           # Financial product definitions (EuropeanCall, VariableAnnuity)
├── run_experiments.py      # Script to run pricing experiments
├── run_risk_analysis.py    # Script to run risk analysis comparisons
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd replicating-martingales-project
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Pricing Experiments
To estimate the price of a financial product using the configured parameters:

```bash
python run_experiments.py --config config/european_call_T5_d3.yml
```
or
```bash
python run_experiments.py --config config/variable_annuity_T40_d5.yml
```

### 2. Risk Analysis
To perform a comparative risk analysis (Nested MC vs. LSMC vs. Replicating Martingale):

```bash
python run_risk_analysis.py --config config/variable_annuity_T40_d5.yml
```

### Configuration
Experiments are defined using YAML files in the `config/` directory. You can modify these files to change:
- **Product parameters** (strike, maturity, fees, etc.)
- **Simulation settings** (number of scenarios, time steps, risk-free rate)
- **Model parameters** (neural network architecture, polynomial degree)

## Dependencies
- `numpy`
- `scipy`
- `pandas`
- `scikit-learn`
- `tensorflow`
- `matplotlib`
- `seaborn`
- `pymanopt`
- `pyyaml`
