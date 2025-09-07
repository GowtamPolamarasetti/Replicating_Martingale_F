import numpy as np
import pandas as pd
from src.products.base_product import BaseProduct

class VariableAnnuity(BaseProduct):

    def __init__(self, config: dict):
        super().__init__(config)
        self.initial_investment = self.config['product']['initial_investment']
        self.annual_fee_rate = self.config['product']['annual_fee_rate']
        self.start_age = self.config['product']['policyholder_start_age']
        # self.n_scenarios = self.config['simulation']['n_scenarios'] # <<< REMOVED THIS LINE
        self.n_steps = self.config['simulation']['n_steps']
        self.dt = self.config['simulation']['time_horizon_years'] / self.n_steps

    def get_cashflows(self, scenarios: dict) -> np.ndarray:

        asset_paths = scenarios['asset_paths'] # Shape: (n_scenarios, n_steps + 1, n_dims)
        q_xt = scenarios['mortality_probs'] # Shape: (n_scenarios, n_steps)

  
        n_scenarios = asset_paths.shape[0]


        account_value = np.zeros((n_scenarios, self.n_steps + 1))
        account_value[:, 0] = self.initial_investment
        
        guaranteed_value = self.initial_investment

       
        denominator = asset_paths[:, :-1, 0]

        safe_denominator = np.where(denominator == 0, 1, denominator)
        fund_returns = asset_paths[:, 1:, 0] / safe_denominator - 1
        fund_returns = np.where(denominator == 0, 0, fund_returns)
        
        cashflows = np.zeros((n_scenarios, self.n_steps + 1))
        
        p_survival = np.ones(n_scenarios)

        for t in range(self.n_steps):
            account_value[:, t+1] = account_value[:, t] * (1 + fund_returns[:, t]) * (1 - self.annual_fee_rate * self.dt)

            net_amount_at_risk = np.maximum(guaranteed_value - account_value[:, t+1], 0)

            prob_death_this_period = p_survival * q_xt[:, t]
            
            cashflows[:, t+1] = -net_amount_at_risk * prob_death_this_period
            
            p_survival *= (1 - q_xt[:, t])

        return cashflows