# experiments/robustness.py
# CausalStream: Additional Robustness Analysis
# Section 5.4.4 of the paper
#
# Three experiments:
#   1. High-dimensional sparse data (100 covariates, 90% zeros)
#   2. Non-stationary adaptation (true ATE shifts mid-stream)
#   3. Robustness to confounding strength (0.1 to 5.0)
#
# Usage: python experiments/robustness.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier, SGDRegressor

np.random.seed(42)
EPS = 1e-8


# ── Core CausalStream runner ───────────────────────────────────

class WelfordVariance:
    def __init__(self):
        self.n = 0; self.mean = 0.0; self.M2 = 0.0
    def update(self, x):
        self.n += 1; delta = x - self.mean
        self.mean += delta / self.n; self.M2 += delta * (x - self.mean)
    def std(self):
        return np.sqrt(self.M2 / (self.n - 1)) if self.n > 1 else 0.0


def run_streaming(X, T, Y):
    """Run CausalStream and return (ate_history, final_ate, std_error)."""
    n = len(T)
    prop = SGDClassifier(loss='log_loss', learning_rate='invscaling',
                         eta0=0.01, power_t=1.0, random_state=42)
    out = {
        0: SGDRegressor(learning_rate='invscaling', eta0=0.01, power_t=1.0, random_state=42),
        1: SGDRegressor(learning_rate='invscaling', eta0=0.01, power_t=1.0, random_state=42)
    }
    classes = np.array([0, 1])
    n_seen = {0: 0, 1: 0}
    prop_n = 0
    welford = WelfordVariance()
    tau_hat = 0.0
    ate_history = []

    for i in range(n):
        x = X[i].reshape(1, -1)
        t = int(T[i]); y = Y[i]

        if prop_n == 0:
            prop.partial_fit(x, [t], classes=classes)
        else:
            prop.partial_fit(x, [t])
        prop_n += 1
        pi = np.clip(prop.predict_proba(x)[0][1], EPS, 1 - EPS)

        out[t].partial_fit(x, [y])
        n_seen[t] += 1
        mu0 = out[0].predict(x)[0] if n_seen[0] > 0 else y
        mu1 = out[1].predict(x)[0] if n_seen[1] > 0 else y

        psi = T[i] * (y - mu1) / pi - (1 - T[i]) * (y - mu0) / (1 - pi) + mu1 - mu0
        tau_hat += (psi - tau_hat) / (i + 1)
        welford.update(psi)
        ate_history.append(tau_hat)

    se = welford.std() / np.sqrt(n)
    return ate_history, tau_hat, se


def run_windowed(X, T, Y, window=500):
    """Run CausalStream with sliding window — for non-stationary experiment."""
    from collections import deque
    n = len(T)
    prop = SGDClassifier(loss='log_loss', learning_rate='invscaling',
                         eta0=0.01, power_t=1.0, random_state=42)
    out = {
        0: SGDRegressor(learning_rate='invscaling', eta0=0.01, power_t=1.0, random_state=42),
        1: SGDRegressor(learning_rate='invscaling', eta0=0.01, power_t=1.0, random_state=42)
    }
    classes = np.array([0, 1])
    n_seen = {0: 0, 1: 0}
    prop_n = 0
    psi_buf = deque(maxlen=window)
    ate_history = []

    for i in range(n):
        x = X[i].reshape(1, -1)
        t = int(T[i]); y = Y[i]

        if prop_n == 0:
            prop.partial_fit(x, [t], classes=classes)
        else:
            prop.partial_fit(x, [t])
        prop_n += 1
        pi = np.clip(prop.predict_proba(x)[0][1], EPS, 1 - EPS)

        out[t].partial_fit(x, [y])
        n_seen[t] += 1
        mu0 = out[0].predict(x)[0] if n_seen[0] > 0 else y
        mu1 = out[1].predict(x)[0] if n_seen[1] > 0 else y

        psi = T[i] * (y - mu1) / pi - (1 - T[i]) * (y - mu0) / (1 - pi) + mu1 - mu0
        psi_buf.append(psi)
        ate_history.append(np.mean(psi_buf))

    return ate_history


# ── Experiment 1: High-dimensional sparse data ─────────────────

def experiment_highdim():
    print("\nExperiment 1: High-Dimensional Sparse Data")
    print("-" * 50)
    n = 10000
    d = 100
    sparsity = 0.90
    TRUE_ATE = -2.5

    # 90% of features are zero
    X = np.zeros((n, d))
    active_cols = int(d * (1 - sparsity))
    X[:, :active_cols] = np.random.normal(0, 1, (n, active_cols))
    np.random.shuffle(X.T)

    logit = -0.5 + 0.5 * X[:, 0]
    propensity = 1.0 / (1.0 + np.exp(-logit))
    T = np.random.binomial(1, propensity, n)
    eps = np.random.normal(0, 2, n)
    Y = 10 + 2 * X[:, 0] + TRUE_ATE * T + eps

    import time
    t0 = time.time()
    _, ate, se = run_streaming(X, T, Y)
    elapsed = time.time() - t0

    bias = abs(ate - TRUE_ATE)
    throughput = n / elapsed

    print(f"  Covariates           : {d} (sparsity={sparsity*100:.0f}%)")
    print(f"  True ATE             : {TRUE_ATE}")
    print(f"  Estimated ATE        : {ate:+.4f}")
    print(f"  Bias                 : {bias:.4f}")
    print(f"  Throughput           : {throughput:,.0f} events/sec")
    return ate, bias


# ── Experiment 2: Non-stationary adaptation ────────────────────

def experiment_nonstationary():
    print("\nExperiment 2: Non-Stationary Adaptation")
    print("-" * 50)
    n = 10000
    shift_point = 5000
    tau1, tau2 = -2.5, -1.25

    X = np.random.normal(0, 1, n).reshape(-1, 1)
    propensity = 1.0 / (1.0 + np.exp(-(-0.5 + 0.5 * X[:, 0])))
    T = np.random.binomial(1, propensity, n)
    eps = np.random.normal(0, 2, n)
    tau_true = np.where(np.arange(n) < shift_point, tau1, tau2)
    Y = 10 + 2 * X[:, 0] + tau_true * T + eps

    ate_full = run_windowed(X, T, Y, window=n)[:]
    ate_win  = run_windowed(X, T, Y, window=500)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    t_idx = np.arange(n)
    ax.plot(t_idx, ate_full, color='steelblue', linewidth=1.5,
            label='Full-history estimator', alpha=0.8)
    ax.plot(t_idx, ate_win, color='darkorange', linewidth=1.5,
            label='Sliding window (W=500)', alpha=0.8)
    ax.axvline(shift_point, color='red', linestyle='--', linewidth=1.5,
               label=f'ATE shift at t={shift_point}')
    ax.axhline(tau1, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    ax.axhline(tau2, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    ax.text(100, tau1 + 0.05, f'τ = {tau1}', fontsize=10, color='gray')
    ax.text(shift_point + 100, tau2 + 0.05, f'τ = {tau2}', fontsize=10, color='gray')
    ax.set_xlabel('Event index (t)', fontsize=12)
    ax.set_ylabel('ATE estimate', fontsize=12)
    ax.set_title('Non-Stationary Adaptation: Sliding Window vs Full-History', fontsize=12)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('nonstationary_adaptation.png', dpi=150, bbox_inches='tight')
    print("  Figure saved to nonstationary_adaptation.png")

    # Adaptation speed
    post_shift = ate_win[shift_point:]
    adapt_idx = next((i for i, v in enumerate(post_shift) if abs(v - tau2) < 0.3), None)
    if adapt_idx:
        print(f"  Window adapter converges within ~{adapt_idx} events of shift")


# ── Experiment 3: Confounding strength ────────────────────────

def experiment_confounding():
    print("\nExperiment 3: Robustness to Confounding Strength")
    print("-" * 50)
    strengths = [0.1, 0.5, 1.0, 2.0, 5.0]
    n = 5000
    TRUE_ATE = -2.5

    print(f"  {'Strength':>10}  {'ATE Estimate':>14}  {'Bias':>8}  {'Std Error':>10}")
    print("  " + "-" * 46)

    for s in strengths:
        X = np.random.normal(0, 1, n).reshape(-1, 1)
        logit = -0.5 + s * X[:, 0]
        propensity = 1.0 / (1.0 + np.exp(-logit))
        T = np.random.binomial(1, propensity, n)
        eps = np.random.normal(0, 2, n)
        Y = 10 + 2 * X[:, 0] + TRUE_ATE * T + eps

        _, ate, se = run_streaming(X, T, Y)
        bias = abs(ate - TRUE_ATE)
        print(f"  {s:>10.1f}  {ate:>14.4f}  {bias:>8.4f}  {se:>10.4f}")


# ── Main ──────────────────────────────────────────────────────

def main():
    print("CausalStream: Additional Robustness Analysis")
    print("=" * 60)
    experiment_highdim()
    experiment_nonstationary()
    experiment_confounding()
    print("\nDone.")


if __name__ == '__main__':
    main()
