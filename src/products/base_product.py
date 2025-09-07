from abc import ABC, abstractmethod
import numpy as np

class BaseProduct(ABC):

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def get_cashflows(self, scenarios: dict) -> np.ndarray:
        """
        Calculates the cash flows for the product for each scenario.

        Args:
            scenarios (dict): A dictionary containing simulated paths of risk factors.
                              e.g., {'asset_paths': array, 'mortality_rates': array}

        Returns:
            np.ndarray: A numpy array of cash flows of shape (n_scenarios, n_steps).
        """
        pass