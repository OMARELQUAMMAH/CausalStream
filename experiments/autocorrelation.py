# experiments/autocorrelation.py
# CausalStream: Autocorrelation Analysis of Influence Function Sequence
# Section 5.4.3 of the paper
#
# Validates the martingale structure by showing the influence function
# sequence {psi_t} decorrelates rapidly despite continuous SGD updates.
# Repeats across 10 simulations x 4 confounding levels.
#
# Usage: python experiments/autocorrelation.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier, SGDRegressor

np.random.seed(42)

CONFOUNDING_LEVELS = {
    'Weak':       0.1,
    'Moderate':   0.5,
    'Strong':     1.0,
    'Very Strong': 2.0,
}
N_SIM    = 10
N_OBS    = 10000
TRUE_ATE = -2.5
MAX_LAG  = 20
REPORT_LAGS = [1, 5, 10, 20]


def generate_data(n, confounding_strength, tau=-2.5):
    X = np.random.normal(0, 1, n)
    logit = -0.5 + confounding_strength * X
    propensity = 1.0 / (1.0 + np.exp(-logit))
    T = np.random.binomial(1, propensity, n)
    eps = np.random.normal(0, 2, n)
    Y = 10 + 2 * X + tau * T + eps
    return X.reshape(-1, 1), T, Y


def compute_influence_functions(X, T, Y):
    """Run CausalStream and return the full sequence of influence functions."""
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
    eps = 1e-8
    psi_seq = []

    for i in range(n):
        x = X[i].reshape(1, -1)
        t = int(T[i])
        y = Y[i]

        if prop_n == 0:
            prop.partial_fit(x, [t], classes=classes)
        else:
            prop.partial_fit(x, [t])
        prop_n += 1
        pi = np.clip(prop.predict_proba(x)[0][1], eps, 1 - eps)

        out[t].partial_fit(x, [y])
        n_seen[t] += 1
        mu0 = out[0].predict(x)[0] if n_seen[0] > 0 else y
        mu1 = out[1].predict(x)[0] if n_seen[1] > 0 else y

        psi = T[i] * (y - mu1) / pi - (1 - T[i]) * (y - mu0) / (1 - pi) + mu1 - mu0
        psi_seq.append(psi)

    return np.array(psi_seq)


def acf(series, max_lag):
    """Compute autocorrelation function up to max_lag."""
    n = len(series)
    series = series - series.mean()
    var = np.var(series)
    if var == 0:
        return np.zeros(max_lag)
    acf_vals = []
    for lag in range(1, max_lag + 1):
        cov = np.mean(series[:n-lag] * series[lag:])
        acf_vals.append(cov / var)
    return np.array(acf_vals)


def main():
    print("Autocorrelation Analysis of Influence Function Sequence")
    print("=" * 60)
    print(f"Simulations per level : {N_SIM}")
    print(f"Observations          : {N_OBS}")
    print(f"Confounding levels    : {list(CONFOUNDING_LEVELS.keys())}")
    print()

    results = {}
    all_acf = {}

    for level_name, strength in CONFOUNDING_LEVELS.items():
        acf_matrix = []
        print(f"  Processing: {level_name} (strength={strength})...")

        for sim in range(N_SIM):
            X, T, Y = generate_data(N_OBS, strength)
            psi_seq = compute_influence_functions(X, T, Y)
            acf_vals = acf(psi_seq, MAX_LAG)
            acf_matrix.append(acf_vals)

        acf_matrix = np.array(acf_matrix)
        all_acf[level_name] = acf_matrix
        mean_acf = acf_matrix.mean(axis=0)
        std_acf  = acf_matrix.std(axis=0)

        results[level_name] = {
            'mean': mean_acf,
            'std':  std_acf
        }

    # Print Table 6
    print()
    print("Table 6: Mean autocorrelation of psi_t across simulations")
    print("=" * 75)
    header = f"{'Confounding':<15}" + "".join(
        f"  Lag {lag:>2}" for lag in REPORT_LAGS)
    print(header)
    print("-" * 75)

    for level_name in CONFOUNDING_LEVELS:
        mean_acf = results[level_name]['mean']
        std_acf  = results[level_name]['std']
        row = f"{level_name:<15}"
        for lag in REPORT_LAGS:
            m = mean_acf[lag - 1]
            s = std_acf[lag - 1]
            row += f"  {m:+.3f}±{s:.3f}"
        print(row)

    # Plot ACF for representative simulation
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    lags = np.arange(1, MAX_LAG + 1)

    for idx, (level_name, strength) in enumerate(CONFOUNDING_LEVELS.items()):
        ax = axes[idx]
        mean_acf = results[level_name]['mean']
        std_acf  = results[level_name]['std']

        ax.bar(lags, mean_acf, color='steelblue', alpha=0.7, label='Mean ACF')
        ax.fill_between(lags, mean_acf - std_acf, mean_acf + std_acf,
                        alpha=0.3, color='steelblue')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axhline(0.05,  color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(-0.05, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_title(f'{level_name} Confounding (strength={strength})', fontsize=11)
        ax.set_xlabel('Lag', fontsize=10)
        ax.set_ylabel('Autocorrelation', fontsize=10)
        ax.set_ylim(-0.3, 0.3)
        ax.set_xticks([1, 5, 10, 15, 20])

    plt.suptitle(f'ACF of Influence Function Sequence\n'
                 f'({N_SIM} simulations per level, n={N_OBS})', fontsize=13)
    plt.tight_layout()
    plt.savefig('autocorrelation.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved to autocorrelation.png")


if __name__ == '__main__':
    main()
