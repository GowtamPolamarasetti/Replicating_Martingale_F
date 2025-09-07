import numpy as np
import pandas as pd
from scipy.linalg import svd

class ScenarioGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.sim_config = config['simulation']
        self.gbm_config = config.get('gbm_parameters')
        self.lc_config = config.get('lee_carter_parameters')

        self.n_scenarios = self.sim_config['n_scenarios']
        self.T = self.sim_config['time_horizon_years']
        self.n_steps = self.sim_config['n_steps']
        self.dt = self.T / self.n_steps
        self.risk_free_rate = self.sim_config['risk_free_rate']


    def _simulate_gbm_paths(self, start_prices=None, start_step=0, end_step=None) -> np.ndarray:

        if end_step is None:
            end_step = self.n_steps

        n_dims = self.gbm_config['n_dims']
        if start_prices is None:
            s0 = np.array(self.gbm_config['initial_prices'])
        else:
            s0 = np.array(start_prices)
            if s0.ndim == 1:
                s0 = np.tile(s0, (self.n_scenarios, 1))

        sigma = np.array(self.gbm_config['volatilities'])
        corr_matrix = np.array(self.gbm_config['correlation_matrix'])
        L = np.linalg.cholesky(corr_matrix)

        sim_steps = end_step - start_step
        if sim_steps <= 0:
            paths = np.zeros((self.n_scenarios, end_step + 1, n_dims))
            if start_step < paths.shape[1]:
                 paths[:, start_step, :] = s0
            return paths

        Z = np.random.standard_normal((self.n_scenarios, sim_steps, n_dims))
        correlated_Z = Z @ L.T
        drift = (self.risk_free_rate - 0.5 * sigma**2) * self.dt
        diffusion = sigma * np.sqrt(self.dt) * correlated_Z

        paths = np.zeros((self.n_scenarios, self.n_steps + 1, n_dims))
        paths[:, start_step, :] = s0

        for t in range(1, sim_steps + 1):
            current_step = start_step + t
            prev_step = start_step + t - 1
            paths[:, current_step, :] = paths[:, prev_step, :] * np.exp(drift + diffusion[:, t-1, :])

        return paths[:, :end_step + 1, :]

    def _fit_and_forecast_lee_carter(self) -> np.ndarray:
        try:
            ages = np.arange(self.config['product']['policyholder_start_age'], self.config['product']['policyholder_start_age'] + self.n_steps)
            forecast_kt = np.random.normal(loc=-0.02, scale=0.01, size=(self.n_scenarios, self.n_steps))
            base_qxt = 0.01 * np.exp(0.08 * (ages - self.config['product']['policyholder_start_age']))
            mortality_probs = base_qxt * np.exp(forecast_kt)
            return np.clip(mortality_probs, 0.0001, 0.99)
        except Exception:
            ages = np.arange(self.config['product']['policyholder_start_age'], self.config['product']['policyholder_start_age'] + self.n_steps)
            kt_forecast = np.random.normal(loc=-0.02, scale=0.01, size=(self.n_scenarios, self.n_steps))
            base_qxt = 0.01 * np.exp(0.08 * (ages - self.config['product']['policyholder_start_age']))
            mortality_probs = base_qxt * np.exp(kt_forecast)
            return np.clip(mortality_probs, 0.0001, 0.99)

    def generate_scenarios(self, start_prices=None, start_step=0, end_step=None) -> dict:
        scenarios = {}
        if self.gbm_config:
            scenarios['asset_paths'] = self._simulate_gbm_paths(start_prices, start_step, end_step)

        if self.config['product']['type'] == 'VariableAnnuity':
            # Note: This assumes mortality is independent of path and can be generated once.
            all_mortality_probs = self._fit_and_forecast_lee_carter()
            scenarios['mortality_probs'] = all_mortality_probs
        
        return scenarios