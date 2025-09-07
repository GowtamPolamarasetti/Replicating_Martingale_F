import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
import pymanopt
from pymanopt.manifolds import Stiefel, Product
from pymanopt.optimizers import TrustRegions

from src.models.base_model import BaseModel
from sklearn.preprocessing import PolynomialFeatures

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class ReplicatingMartingale(BaseModel):
    def __init__(self, config, product):
        super().__init__(config, product)
        self.params = config['replicating_martingale_parameters']
        self.basis_type = self.params['basis_type']
        self.theta = None

    def _fit_nn(self, X_train: np.ndarray, y_train: np.ndarray):
        nn_params = self.params['nn_params']
        n_features = X_train.shape[1]
        hidden_units = nn_params['hidden_units']
        shapes = [
            (n_features, hidden_units), (hidden_units,),
            (hidden_units, 1), (1,)
        ]
        n_params = sum(np.prod(s) for s in shapes)
        theta0 = np.random.randn(n_params) * 0.1

        @tf.function
        def predict_nn(X, theta_flat):
            idx = 0
            unpacked_theta = []
            for shape in shapes:
                size = np.prod(shape)
                param = tf.reshape(theta_flat[idx:idx + size], shape)
                unpacked_theta.append(tf.cast(param, tf.float32))
                idx += size
            W1, b1, W2, b2 = unpacked_theta
            hidden = tf.nn.relu(X @ W1 + b1)
            return hidden @ W2 + b2

        def objective_function(theta_flat):
            X_tf = tf.constant(X_train, dtype=tf.float32)
            y_tf = tf.constant(y_train, dtype=tf.float32)
            theta_tf = tf.constant(theta_flat)
            with tf.GradientTape() as tape:
                tape.watch(theta_tf)
                y_pred = predict_nn(X_tf, theta_tf)
                loss = tf.reduce_mean((y_tf - tf.squeeze(y_pred))**2)
            grads = tape.gradient(loss, theta_tf)
            return loss.numpy().astype(np.float64), grads.numpy().astype(np.float64)

        print("Optimizing Neural Network with L-BFGS-B...")
        res = minimize(
            fun=objective_function,
            x0=theta0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': nn_params['optimizer_max_iter'], 'disp': True}
        )
        print("Optimization complete.")
        return res.x

    def _fit_poly_ldr(self, X_train: np.ndarray, y_train: np.ndarray):
        poly_params = self.params['poly_ldr_params']
        D = X_train.shape[1]
        k = poly_params['n_components']
        degree = poly_params['degree']
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        poly.fit(np.zeros((1, k)))
        n_poly_features = len(poly.get_feature_names_out())
        stiefel_manifold = Stiefel(D, k)
        euclidean_manifold = pymanopt.manifolds.Euclidean(n_poly_features, 1)
        manifold = Product([stiefel_manifold, euclidean_manifold])
        X_tf = tf.constant(X_train, dtype=tf.float32)
        y_tf = tf.constant(y_train, dtype=tf.float32)

        @pymanopt.function.tensorflow(manifold)
        def cost(A, beta):
            X_proj = X_tf @ A
            poly_features = []
            for d in range(degree + 1):
                if d == 0:
                    poly_features.append(tf.ones([X_proj.shape[0], 1], dtype=tf.float32))
                else:
                    poly_features.append(X_proj**d)
            poly_features_tf = tf.concat(poly_features, axis=1)
            y_pred = poly_features_tf @ beta
            return tf.reduce_mean((y_tf - tf.squeeze(y_pred))**2)

        problem = pymanopt.Problem(manifold=manifold, cost=cost)
        optimizer = TrustRegions(verbosity=2)
        print("Optimizing Polynomial-LDR with Riemannian Trust Regions...")
        optimal_theta = optimizer.run(problem)
        print("Optimization complete.")
        return {'A': optimal_theta[0], 'beta': optimal_theta[1]}

    def fit(self, scenarios: dict, risk_horizon_years: float):
        print(f"\n--- Fitting Replicating Martingale Model (Basis: {self.basis_type}) ---")
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
            self.theta = self._fit_nn(X, y.reshape(-1, 1))
        elif self.basis_type == 'poly_ldr':
            self.theta = self._fit_poly_ldr(X, y.reshape(-1, 1))

    def predict(self, states: np.ndarray) -> np.ndarray:
        if self.theta is None:
            raise RuntimeError("Model has not been fitted yet.")
        X_tf = tf.constant(states, dtype=tf.float32)
        if self.basis_type == 'nn':
            nn_params = self.params['nn_params']
            n_features = states.shape[1]
            hidden_units = nn_params['hidden_units']
            shapes = [
                (n_features, hidden_units), (hidden_units,),
                (hidden_units, 1), (1,)
            ]
            idx, unpacked_theta = 0, []
            for shape in shapes:
                size = np.prod(shape)
                param = tf.reshape(self.theta[idx:idx + size], shape)
                unpacked_theta.append(tf.cast(param, tf.float32))
                idx += size
            W1, b1, W2, b2 = unpacked_theta
            hidden = tf.nn.relu(X_tf @ W1 + b1)
            predictions = hidden @ W2 + b2
            scalar_or_array = tf.squeeze(predictions).numpy()
            return np.atleast_1d(scalar_or_array)
        elif self.basis_type == 'poly_ldr':
            A = tf.constant(self.theta['A'], dtype=tf.float32)
            beta = tf.constant(self.theta['beta'], dtype=tf.float32)
            degree = self.params['poly_ldr_params']['degree']
            X_proj = X_tf @ A
            poly_features_list = []
            for d in range(degree + 1):
                if d == 0:
                    poly_features_list.append(tf.ones([X_proj.shape[0], 1], dtype=tf.float32))
                else:
                    poly_features_list.append(X_proj**d)
            poly_features = tf.concat(poly_features_list, axis=1)
            predictions = poly_features @ beta
            scalar_or_array = tf.squeeze(predictions).numpy()
            return np.atleast_1d(scalar_or_array)

    def run(self):
        print(f"ReplicatingMartingale model (basis: {self.basis_type}) is a component.")
        print("To use, instantiate and call .fit(scenarios), then use it as a proxy model.")
        pass
