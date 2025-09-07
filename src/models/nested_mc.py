import numpy as np
import time
from src.models.base_model import BaseModel
from src.data.scenario_generator import ScenarioGenerator

class NestedMC(BaseModel):
    def __init__(self, config, product, proxy_model=None):
        super().__init__(config, product)
        self.proxy_model = proxy_model

    def run(self, risk_horizon_years: float, n_outer: int, n_inner: int = None):
        print(f"\n--- Running Nested Simulation ---")
        if self.proxy_model:
            print("Mode: Using Proxy Model")
        else:
            print("Mode: Full Brute-Force Simulation")
        
        start_time = time.time()

        outer_config = self.config.copy()
        outer_config['simulation']['n_scenarios'] = n_outer
        
        total_steps = self.config['simulation']['n_steps']
        total_horizon = self.config['simulation']['time_horizon_years']
        risk_horizon_steps = int(total_steps * risk_horizon_years / total_horizon)

        outer_generator = ScenarioGenerator(outer_config)
        outer_scenarios = outer_generator.generate_scenarios(end_step=risk_horizon_steps)
        horizon_states = outer_scenarios['asset_paths'][:, -1, :]

        portfolio_values_at_horizon = []
        inner_config = self.config.copy()
        if n_inner:
            inner_config['simulation']['n_scenarios'] = n_inner
        
        for i in range(n_outer):
            start_prices = horizon_states[i, :]
            
            if self.proxy_model:
                value = self.proxy_model.predict(start_prices.reshape(1, -1))[0]
            else:
                if n_inner is None:
                    raise ValueError("n_inner must be provided for full nMC.")
                
                inner_generator = ScenarioGenerator(inner_config)
                inner_scenarios = inner_generator.generate_scenarios(start_prices=start_prices, start_step=risk_horizon_steps)
                cashflows = self.product.get_cashflows(inner_scenarios)
                rf = self.config['simulation']['risk_free_rate']
                full_time_points = np.linspace(0, total_horizon, total_steps + 1)
                relevant_cashflows = cashflows[:, risk_horizon_steps:]
                relevant_time_points = full_time_points[risk_horizon_steps:]
                discount_factors = np.exp(-rf * (relevant_time_points - risk_horizon_years))
                present_values = np.sum(relevant_cashflows * discount_factors, axis=1)
                value = np.mean(present_values)

            portfolio_values_at_horizon.append(value)
            
            if (i + 1) % 25 == 0 or i == n_outer - 1:
                print(f"  - Completed outer path {i + 1}/{n_outer}")
        
        end_time = time.time()
        print(f"Nested simulation finished in {end_time - start_time:.2f} seconds.")
        return np.array(portfolio_values_at_horizon)
