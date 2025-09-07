import numpy as np
from src.models.base_model import BaseModel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import tensorflow as tf

class LSMC(BaseModel):
    
    def __init__(self, config, product):
        super().__init__(config, product)
        self.lsmc_params = config['lsmc_parameters']
        self.basis_type = self.lsmc_params['basis_type']
        self.model = self._build_model()

    def _build_model(self):
        
        if self.basis_type == 'polynomial':
            degree = self.lsmc_params['degree']
            return Pipeline([
                ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                ('reg', LinearRegression())
            ])
        elif self.basis_type == 'poly_ldr':
            degree = self.lsmc_params['degree']
            n_components = self.lsmc_params['n_components']
            return Pipeline([
                ('pca', PCA(n_components=n_components)),
                ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                ('reg', LinearRegression())
            ])
        elif self.basis_type == 'nn':
            n_dims = self.config['gbm_parameters']['n_dims']
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(n_dims,)),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            return model
        else:
            raise ValueError(f"Unknown basis type: {self.basis_type}")

    def fit(self, scenarios: dict, risk_horizon_years: float):

        print(f"\n--- Fitting LSMC Model ---")
        print(f"Basis type: {self.basis_type}")

     
        total_steps = self.config['simulation']['n_steps']
        total_horizon = self.config['simulation']['time_horizon_years']
        risk_horizon_steps = int(total_steps * risk_horizon_years / total_horizon)
        X = scenarios['asset_paths'][:, risk_horizon_steps, :]

        cashflows = self.product.get_cashflows(scenarios)
        
        terminal_cashflows = np.sum(cashflows[:, risk_horizon_steps:], axis=1)

        rf = self.config['simulation']['risk_free_rate']
        discount_factor = np.exp(-rf * (total_horizon - risk_horizon_years))
        y = terminal_cashflows * discount_factor
        
        if self.basis_type == 'nn':
            self.model.fit(X, y, epochs=25, batch_size=512, verbose=0)
        else:
            self.model.fit(X, y)
        print("LSMC fitting complete.")

    def predict(self, states: np.ndarray) -> np.ndarray:
        return self.model.predict(states).flatten()

    def run(self):
        print("LSMC model is a component. Use .fit() and .predict() methods.")
        pass