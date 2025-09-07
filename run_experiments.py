import argparse
import yaml
import numpy as np
import time

from src.data.scenario_generator import ScenarioGenerator
from src.products.european_option import EuropeanCall
from src.products.variable_annuity import VariableAnnuity

def main(config_path: str):
 
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"--- Starting Experiment: {config['experiment_name']} ---")

    generator = ScenarioGenerator(config)


    product_type = config['product']['type']
    if product_type == 'EuropeanCall':
        product = EuropeanCall(config)
    elif product_type == 'VariableAnnuity':
        product = VariableAnnuity(config)
    else:
        raise ValueError(f"Unknown product type: {product_type}")


    start_time = time.time()
    scenarios = generator.generate_scenarios()
    

    cashflows = product.get_cashflows(scenarios)

    rf = config['simulation']['risk_free_rate']
    T = config['simulation']['time_horizon_years']
    n_steps = config['simulation']['n_steps']
    time_points = np.linspace(0, T, n_steps + 1)
    discount_factors = np.exp(-rf * time_points)
    
    present_values = np.sum(cashflows * discount_factors, axis=1)


    price = np.mean(present_values)
    
    end_time = time.time()
    
    print("\n--- Results ---")
    print(f"Product Type: {product_type}")
    print(f"Estimated Price: {price:.4f}")
    print(f"Standard Error: {np.std(present_values) / np.sqrt(len(present_values)):.4f}")
    print(f"Execution Time: {end_time - start_time:.2f} seconds")
    print("-----------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a financial product pricing experiment.")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the experiment configuration YAML file.'
    )
    args = parser.parse_args()
    main(args.config)