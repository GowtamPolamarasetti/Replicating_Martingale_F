import numpy as np
from src.products.base_product import BaseProduct

class EuropeanCall(BaseProduct):

    def __init__(self, config: dict):
        super().__init__(config)
        self.strike_price = self.config['product']['strike_price']
        self.asset_index = self.config['product']['asset_index']
        self.n_steps = self.config['simulation']['n_steps']

    def get_cashflows(self, scenarios: dict) -> np.ndarray:

        asset_paths = scenarios['asset_paths']
        terminal_prices = asset_paths[:, -1, self.asset_index]

        payoffs = np.maximum(terminal_prices - self.strike_price, 0)

  
        cashflows = np.zeros((asset_paths.shape[0], self.n_steps + 1))
        cashflows[:, -1] = payoffs

        return cashflows