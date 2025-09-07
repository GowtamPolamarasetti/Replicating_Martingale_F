import argparse
import yaml
import numpy as np

from src.data.scenario_generator import ScenarioGenerator
from src.products.variable_annuity import VariableAnnuity
from src.models.nested_mc import NestedMC
from src.models.lsmc import LSMC
from src.models.replicating_martingale import ReplicatingMartingale

def main(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"--- Starting Risk Analysis: {config['experiment_name']} ---")

    product = VariableAnnuity(config)
    mc_params = config['nested_mc_parameters']
    risk_horizon = mc_params['risk_horizon_years']
    n_outer = mc_params['n_outer_scenarios']
    n_inner = mc_params['n_inner_scenarios']

    print("\nGenerating a large set of scenarios for model training...")
    training_generator = ScenarioGenerator(config)
    training_scenarios = training_generator.generate_scenarios()
    print("Scenario generation complete.")

    lsmc_model = LSMC(config, product)
    lsmc_model.fit(training_scenarios, risk_horizon_years=risk_horizon)

    martingale_model = ReplicatingMartingale(config, product)
    martingale_model.fit(training_scenarios, risk_horizon_years=risk_horizon)

    nmc_lsmc_proxy = NestedMC(config, product, proxy_model=lsmc_model)
    lsmc_proxy_values = nmc_lsmc_proxy.run(risk_horizon_years=risk_horizon, n_outer=n_outer)

    nmc_martingale_proxy = NestedMC(config, product, proxy_model=martingale_model)
    martingale_proxy_values = nmc_martingale_proxy.run(risk_horizon_years=risk_horizon, n_outer=n_outer)
    
    nmc_full = NestedMC(config, product)
    full_values = nmc_full.run(risk_horizon_years=risk_horizon, n_outer=n_outer, n_inner=n_inner)

    print("\n--- Final Risk Analysis Results ---")
    print(f"Risk Horizon: {risk_horizon} year(s)")
    
    metrics = {}
    models = {
        "Full Nested MC": full_values,
        "LSMC Proxy": lsmc_proxy_values,
        "Martingale Proxy": martingale_proxy_values
    }

    for name, values in models.items():
        metrics[name] = {
            "Mean": np.mean(values),
            "Std Dev": np.std(values),
            "95% VaR": -np.percentile(values, 5)
        }

    print("\nMetric          | Full Nested MC  | LSMC Proxy      | Martingale Proxy")
    print("----------------|-----------------|-----------------|-----------------")
    print(f"Mean Value      | {metrics['Full Nested MC']['Mean']:15.2f} | {metrics['LSMC Proxy']['Mean']:15.2f} | {metrics['Martingale Proxy']['Mean']:15.2f}")
    print(f"Std. Deviation  | {metrics['Full Nested MC']['Std Dev']:15.2f} | {metrics['LSMC Proxy']['Std Dev']:15.2f} | {metrics['Martingale Proxy']['Std Dev']:15.2f}")
    print(f"95% VaR (Loss)  | {metrics['Full Nested MC']['95% VaR']:15.2f} | {metrics['LSMC Proxy']['95% VaR']:15.2f} | {metrics['Martingale Proxy']['95% VaR']:15.2f}")
    print("--------------------------------------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a risk analysis experiment.")
    parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file.')
    args = parser.parse_args()
    main(args.config)
