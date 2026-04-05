# causalstream_paper.py
# CausalStream: A General Architecture for Real-Time Causal Inference on Data Streams
# Omar El quammah, Cui Weijun, Ouahiba Ouchkhi, Kristina Darbinian
# Nanjing University of Information Science and Technology
#
# This script implements the single-node CausalStream prototype evaluated in the paper.
# Dataset: FreshRetailNet-50K (350,000 retail transactions, 30-day window)
# Treatment: discount > 0 (binary)
# Outcome: sale_amount

import os
import time
import argparse
import numpy as np
import pandas as pd
from collections import deque
from sklearn.linear_model import SGDClassifier, SGDRegressor

# =============================================================================
# CORE ESTIMATOR
# =============================================================================

class WelfordVariance:
    """
    Online variance estimation via Welford's algorithm.
    Maintains O(1) memory — no historical influence functions stored.
    Equations (12)-(13) in the paper.
    """
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def variance(self):
        return self.M2 / (self.n - 1) if self.n > 1 else 0.0

    def std(self):
        return np.sqrt(self.variance())


class IncrementalDREstimator:
    """
    Incremental doubly robust ATE estimator with O(1) per-event complexity.
    Updates treatment effect estimate using the influence function at each event.
    Equations (5)-(8) in the paper.
    """
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.tau_hat = 0.0       # Running ATE estimate
        self.n = 0               # Events processed
        self.welford = WelfordVariance()
        self.ate_history = []

    def update(self, Y_obs, T, pi, mu0, mu1):
        """
        Update ATE estimate with one new observation.

        Parameters
        ----------
        Y_obs : float  Observed outcome
        T     : int    Treatment indicator (0 or 1)
        pi    : float  Propensity score P(T=1|X)
        mu0   : float  Predicted outcome under control
        mu1   : float  Predicted outcome under treatment
        """
        pi = np.clip(pi, self.epsilon, 1.0 - self.epsilon)

        # Influence function (Equation 8)
        psi = (
            T * (Y_obs - mu1) / pi
            - (1 - T) * (Y_obs - mu0) / (1 - pi)
            + mu1 - mu0
        )

        # Incremental update (Equation 7)
        self.n += 1
        gamma = 1.0 / self.n
        prev_tau = self.tau_hat
        self.tau_hat = prev_tau + gamma * (psi - prev_tau)

        # Welford variance update (Equations 12-13)
        self.welford.update(psi)
        self.ate_history.append(self.tau_hat)

        return self.tau_hat

    def confidence_interval(self, alpha=0.05):
        """
        Asymptotically valid Wald-type confidence interval (Equation 14).
        Returns (lower, upper) tuple.
        """
        if self.n < 2:
            return (float('-inf'), float('inf'))
        z = 1.96  # 95% CI
        se = self.welford.std() / np.sqrt(self.n)
        return (self.tau_hat - z * se, self.tau_hat + z * se)

    def std_error(self):
        if self.n < 2:
            return float('inf')
        return self.welford.std() / np.sqrt(self.n)


class PropensityService:
    """
    Online propensity score estimation via SGD logistic regression.
    Updates P(T=1|X) incrementally as new observations arrive.
    Equations (9)-(10) in the paper.
    """
    def __init__(self, learning_rate=0.01):
        self.model = SGDClassifier(
            loss='log_loss',
            learning_rate='invscaling',
            eta0=learning_rate,
            power_t=1.0,
            warm_start=True,
            random_state=42
        )
        self.n_seen = 0
        self.classes = np.array([0, 1])

    def update_and_predict(self, X, T):
        """Update model with new observation and return P(T=1|X)."""
        X = np.array(X).reshape(1, -1)
        T = np.array([T])
        if self.n_seen == 0:
            self.model.partial_fit(X, T, classes=self.classes)
        else:
            self.model.partial_fit(X, T)
        self.n_seen += 1
        return self.model.predict_proba(X)[0][1]

    def predict(self, X):
        if self.n_seen == 0:
            return 0.5
        return self.model.predict_proba(np.array(X).reshape(1, -1))[0][1]


class OutcomeService:
    """
    Online outcome model estimation via SGD regression.
    Maintains separate models for treated (a=1) and control (a=0) units.
    Equations (11) in the paper.
    """
    def __init__(self, learning_rate=0.01):
        self.models = {
            0: SGDRegressor(learning_rate='invscaling', eta0=learning_rate,
                            power_t=1.0, warm_start=True, random_state=42),
            1: SGDRegressor(learning_rate='invscaling', eta0=learning_rate,
                            power_t=1.0, warm_start=True, random_state=42)
        }
        self.n_seen = {0: 0, 1: 0}

    def update_and_predict(self, X, T, Y):
        """Update outcome model for treatment arm T and return mu0, mu1."""
        X = np.array(X).reshape(1, -1)
        self.models[T].partial_fit(X, [Y])
        self.n_seen[T] += 1
        mu0 = self.models[0].predict(X)[0] if self.n_seen[0] > 0 else Y
        mu1 = self.models[1].predict(X)[0] if self.n_seen[1] > 0 else Y
        return mu0, mu1


class WindowService:
    """
    Sliding window management with circular buffer.
    Maintains O(W) memory — no full history retention.
    Equations (15)-(16) in the paper.
    """
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.psi_sum = 0.0
        self.psi_buffer = deque(maxlen=window_size)

    def update(self, event, psi):
        """Add new event and influence function value to window."""
        self.buffer.append(event)
        if len(self.psi_buffer) == self.window_size:
            self.psi_sum -= self.psi_buffer[0]
        self.psi_buffer.append(psi)
        self.psi_sum += psi

    def windowed_ate(self):
        """Windowed ATE estimate (Equation 15)."""
        n = len(self.psi_buffer)
        return self.psi_sum / n if n > 0 else 0.0

    def size(self):
        return len(self.buffer)


# =============================================================================
# CAUSALSTREAM PIPELINE
# =============================================================================

class CausalStream:
    """
    CausalStream: complete streaming causal inference pipeline.

    Coordinates PropensityService, OutcomeService, WindowService, and
    IncrementalDREstimator to produce real-time ATE estimates with
    O(1) per-event complexity.
    """
    def __init__(self, window_size=1000, warmup=10000):
        self.window_size = window_size
        self.warmup = warmup
        self.propensity_svc = PropensityService()
        self.outcome_svc = OutcomeService()
        self.window_svc = WindowService(window_size)
        self.dr_estimator = IncrementalDREstimator()
        self.n_processed = 0

    def process_event(self, Y_obs, T, X):
        """
        Process one streaming event.

        Parameters
        ----------
        Y_obs : float  Observed outcome (sale_amount)
        T     : int    Treatment indicator (1 if discount > 0)
        X     : list   Feature vector (covariates)

        Returns
        -------
        ate   : float  Current ATE estimate
        ci    : tuple  95% confidence interval (lower, upper)
        """
        # Update propensity and outcome models
        pi = self.propensity_svc.update_and_predict(X, T)
        mu0, mu1 = self.outcome_svc.update_and_predict(X, T, Y_obs)

        # Update DR estimator
        ate = self.dr_estimator.update(Y_obs, T, pi, mu0, mu1)

        # Compute influence function for windowing
        pi_c = np.clip(pi, 1e-8, 1.0 - 1e-8)
        psi = (
            T * (Y_obs - mu1) / pi_c
            - (1 - T) * (Y_obs - mu0) / (1 - pi_c)
            + mu1 - mu0
        )
        self.window_svc.update({'Y': Y_obs, 'T': T, 'X': X}, psi)

        self.n_processed += 1
        ci = self.dr_estimator.confidence_interval()
        return ate, ci

    def get_results(self):
        return {
            'ate': self.dr_estimator.tau_hat,
            'std_error': self.dr_estimator.std_error(),
            'ci': self.dr_estimator.confidence_interval(),
            'windowed_ate': self.window_svc.windowed_ate(),
            'n_processed': self.n_processed
        }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(data_path, window_size=1000, warmup=10000):
    """
    Main experiment: evaluate CausalStream on FreshRetailNet-50K.
    Reproduces the results in Table 2 and Table 3 of the paper.
    """
    print("CausalStream: Real-Time Causal Inference on Data Streams")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading dataset from: {data_path}")
    if data_path.endswith('.xlsx') or data_path.endswith('.xls'):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)

    print(f"Loaded {len(df):,} transactions")
    print(f"Columns: {list(df.columns)}")

    # Define treatment and outcome
    # Treatment: discount > 0 (binary)
    # Outcome: sale_amount
    df['T'] = (df['discount'] > 0).astype(int)

    # Feature columns (covariates)
    feature_cols = [c for c in df.columns if c not in
                    ['sale_amount', 'discount', 'T', 'product_id', 'store_id',
                     'transaction_id', 'date', 'timestamp']]
    print(f"Treatment rate: {df['T'].mean():.3f}")
    print(f"Features used: {feature_cols}")

    # Initialize CausalStream
    cs = CausalStream(window_size=window_size, warmup=warmup)

    # Process stream
    print(f"\nProcessing {len(df):,} events...")
    print(f"Window size: {window_size:,}")
    t_start = time.time()

    log_interval = max(len(df) // 10, 1)

    for i, row in df.iterrows():
        Y = row['sale_amount']
        T = int(row['T'])
        X = [row[c] for c in feature_cols] if feature_cols else [0.0]

        ate, ci = cs.process_event(Y, T, X)

        if i % log_interval == 0:
            elapsed = time.time() - t_start
            throughput = cs.n_processed / elapsed if elapsed > 0 else 0
            print(f"  [{i:>7,}] ATE={ate:+.4f}  CI=({ci[0]:+.3f}, {ci[1]:+.3f})  "
                  f"Throughput={throughput:,.0f} ev/sec")

    t_end = time.time()
    elapsed = t_end - t_start

    # Final results
    results = cs.get_results()
    throughput = cs.n_processed / elapsed

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Events processed     : {results['n_processed']:,}")
    print(f"Wall-clock time      : {elapsed:.1f} seconds")
    print(f"Throughput           : {throughput:,.0f} events/sec")
    print(f"ATE estimate         : {results['ate']:+.4f}")
    print(f"Std error            : {results['std_error']:.4f}")
    print(f"95% CI               : ({results['ci'][0]:+.3f}, {results['ci'][1]:+.3f})")
    print(f"Windowed ATE (W={window_size:,}) : {results['windowed_ate']:+.4f}")

    return results, cs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CausalStream: Real-Time Causal Inference on Data Streams')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to FreshRetailNet-50K dataset (.xlsx or .csv)')
    parser.add_argument('--window_size', type=int, default=1000,
                        help='Sliding window size (default: 1000)')
    parser.add_argument('--warmup', type=int, default=10000,
                        help='Warm-up events before reporting (default: 10000)')
    args = parser.parse_args()

    run_experiment(args.data, args.window_size, args.warmup)
