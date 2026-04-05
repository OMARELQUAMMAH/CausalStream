# experiments/baseline_comparison.py
# CausalStream: Comprehensive Baseline Comparison
# Section 5.4.5 of the paper
#
# Compares CausalStream against five methods on synthetic data
# with known ground truth (true ATE = -2.5):
#   - AIPW  (online augmented IPW)
#   - IPW   (online inverse propensity weighting)
#   - Causal Forest (approximate online via rolling window)
#   - BSTS  (Bayesian structural time series approximation)
#   - DML   (double machine learning, batch)
#
# Usage: python experiments/baseline_comparison.py

import time
import numpy as np
from sklearn.linear_model import SGDClassifier, SGDRegressor, LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor

np.random.seed(42)

N      = 10000
TRUE_ATE = -2.5
EPS    = 1e-8


def generate_data(n, tau=-2.5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = np.random.normal(0, 1, n).reshape(-1, 1)
    prop = 1.0 / (1.0 + np.exp(-(-0.5 + 0.5 * X[:, 0])))
    T = np.random.binomial(1, prop, n)
    Y = 10 + 2 * X[:, 0] + tau * T + np.random.normal(0, 2, n)
    return X, T, Y, prop


# ── CausalStream ───────────────────────────────────────────────

def run_causalstream(X, T, Y):
    n = len(T)
    prop_m = SGDClassifier(loss='log_loss', learning_rate='invscaling',
                           eta0=0.01, power_t=1.0, random_state=42)
    out_m = {
        0: SGDRegressor(learning_rate='invscaling', eta0=0.01, power_t=1.0, random_state=42),
        1: SGDRegressor(learning_rate='invscaling', eta0=0.01, power_t=1.0, random_state=42)
    }
    classes = np.array([0, 1])
    n_seen = {0: 0, 1: 0}; pn = 0
    tau_hat = 0.0
    t0 = time.perf_counter()

    for i in range(n):
        x = X[i].reshape(1, -1); t = int(T[i]); y = Y[i]
        if pn == 0: prop_m.partial_fit(x, [t], classes=classes)
        else:       prop_m.partial_fit(x, [t])
        pn += 1
        pi = np.clip(prop_m.predict_proba(x)[0][1], EPS, 1 - EPS)
        out_m[t].partial_fit(x, [y]); n_seen[t] += 1
        mu0 = out_m[0].predict(x)[0] if n_seen[0] > 0 else y
        mu1 = out_m[1].predict(x)[0] if n_seen[1] > 0 else y
        psi = T[i] * (y - mu1) / pi - (1-T[i]) * (y - mu0)/(1-pi) + mu1 - mu0
        tau_hat += (psi - tau_hat) / (i + 1)

    elapsed = time.perf_counter() - t0
    return tau_hat, elapsed / n * 1e6  # latency in microseconds


# ── AIPW (online) ──────────────────────────────────────────────

def run_aipw(X, T, Y):
    """Online AIPW — same influence function as CausalStream but no variance tracking."""
    n = len(T)
    prop_m = SGDClassifier(loss='log_loss', learning_rate='invscaling',
                           eta0=0.01, power_t=1.0, random_state=42)
    out_m = {
        0: SGDRegressor(learning_rate='invscaling', eta0=0.01, power_t=1.0, random_state=42),
        1: SGDRegressor(learning_rate='invscaling', eta0=0.01, power_t=1.0, random_state=42)
    }
    classes = np.array([0, 1])
    n_seen = {0: 0, 1: 0}; pn = 0
    psi_sum = 0.0
    t0 = time.perf_counter()

    for i in range(n):
        x = X[i].reshape(1, -1); t = int(T[i]); y = Y[i]
        if pn == 0: prop_m.partial_fit(x, [t], classes=classes)
        else:       prop_m.partial_fit(x, [t])
        pn += 1
        pi = np.clip(prop_m.predict_proba(x)[0][1], EPS, 1 - EPS)
        out_m[t].partial_fit(x, [y]); n_seen[t] += 1
        mu0 = out_m[0].predict(x)[0] if n_seen[0] > 0 else y
        mu1 = out_m[1].predict(x)[0] if n_seen[1] > 0 else y
        psi = T[i]*(y-mu1)/pi - (1-T[i])*(y-mu0)/(1-pi) + mu1 - mu0
        psi_sum += psi

    elapsed = time.perf_counter() - t0
    return psi_sum / n, elapsed / n * 1e6


# ── IPW (online) ───────────────────────────────────────────────

def run_ipw(X, T, Y):
    """Online IPW — no outcome model, propensity weighting only."""
    n = len(T)
    prop_m = SGDClassifier(loss='log_loss', learning_rate='invscaling',
                           eta0=0.01, power_t=1.0, random_state=42)
    classes = np.array([0, 1]); pn = 0
    num = 0.0; denom = 0.0
    t0 = time.perf_counter()

    for i in range(n):
        x = X[i].reshape(1, -1); t = int(T[i]); y = Y[i]
        if pn == 0: prop_m.partial_fit(x, [t], classes=classes)
        else:       prop_m.partial_fit(x, [t])
        pn += 1
        pi = np.clip(prop_m.predict_proba(x)[0][1], EPS, 1 - EPS)
        w = T[i]/pi - (1-T[i])/(1-pi)
        num += w * y; denom += abs(w)

    elapsed = time.perf_counter() - t0
    ate = num / denom if denom > 0 else 0.0
    return ate, elapsed / n * 1e6


# ── Causal Forest (approximate online) ────────────────────────

def run_causal_forest_approx(X, T, Y, retrain_every=1000):
    """
    Approximate online Causal Forest: periodic retraining on rolling window.
    Not natively online; retrains GBM every retrain_every observations.
    """
    from sklearn.ensemble import GradientBoostingRegressor as GBR
    n = len(T)
    ate_hat = 0.0
    t0 = time.perf_counter()
    XT = np.hstack([X, T.reshape(-1, 1)])

    for start in range(0, n, retrain_every):
        end = min(start + retrain_every, n)
        Xs = X[start:end]; Ts = T[start:end]; Ys = Y[start:end]
        if len(Xs) < 50:
            continue
        XT_s = XT[start:end]
        m = GBR(n_estimators=50, max_depth=3, random_state=42)
        m.fit(XT_s, Ys)
        X1 = np.hstack([Xs, np.ones((len(Xs), 1))])
        X0 = np.hstack([Xs, np.zeros((len(Xs), 1))])
        ate_hat = np.mean(m.predict(X1) - m.predict(X0))

    elapsed = time.perf_counter() - t0
    return ate_hat, elapsed / n * 1e6


# ── BSTS approximation ─────────────────────────────────────────

def run_bsts_approx(X, T, Y):
    """
    BSTS approximation using Bayesian linear regression with online updates.
    Full BSTS requires Stan/PyMC; this implements the core structural component.
    """
    n = len(T)
    # Bayesian linear regression with conjugate updates
    # Prior: beta ~ N(0, 10*I), sigma^2 = 4
    d = X.shape[1] + 2  # intercept + X + T
    prior_cov = 10.0 * np.eye(d)
    prior_mean = np.zeros(d)
    sigma2 = 4.0

    post_cov  = prior_cov.copy()
    post_mean = prior_mean.copy()
    t0 = time.perf_counter()

    for i in range(n):
        xi = np.hstack([[1.0], X[i], [T[i]]])
        # Bayesian linear regression update (conjugate)
        k = post_cov @ xi / (sigma2 + xi @ post_cov @ xi)
        post_mean = post_mean + k * (Y[i] - xi @ post_mean)
        post_cov  = post_cov  - np.outer(k, xi @ post_cov)

    elapsed = time.perf_counter() - t0
    ate = post_mean[-1]  # coefficient of T
    return ate, elapsed / n * 1e6


# ── DML (batch) ────────────────────────────────────────────────

def run_dml(X, T, Y):
    """Double machine learning (batch, cross-fitting with 2 folds)."""
    n = len(T)
    t0 = time.perf_counter()

    mid = n // 2
    # Fold 1: train on second half, predict on first
    prop1 = LogisticRegression(random_state=42).fit(X[mid:], T[mid:])
    out1  = Ridge().fit(X[mid:], Y[mid:])
    T_res1 = T[:mid] - prop1.predict_proba(X[:mid])[:, 1]
    Y_res1 = Y[:mid] - out1.predict(X[:mid])

    # Fold 2: train on first half, predict on second
    prop2 = LogisticRegression(random_state=42).fit(X[:mid], T[:mid])
    out2  = Ridge().fit(X[:mid], Y[:mid])
    T_res2 = T[mid:] - prop2.predict_proba(X[mid:])[:, 1]
    Y_res2 = Y[mid:] - out2.predict(X[mid:])

    T_res = np.concatenate([T_res1, T_res2])
    Y_res = np.concatenate([Y_res1, Y_res2])
    ate = np.dot(T_res, Y_res) / np.dot(T_res, T_res)

    elapsed = time.perf_counter() - t0
    return ate, elapsed / n * 1e6


# ── Main ──────────────────────────────────────────────────────

def main():
    print("Baseline Comparison")
    print("=" * 70)
    print(f"n = {N:,}  |  True ATE = {TRUE_ATE}")
    print()

    X, T, Y, _ = generate_data(N, TRUE_ATE)

    methods = {
        'CausalStream':   run_causalstream,
        'AIPW':           run_aipw,
        'IPW':            run_ipw,
        'Causal Forest':  run_causal_forest_approx,
        'BSTS':           run_bsts_approx,
        'DML':            run_dml,
    }

    print(f"{'Method':<18}  {'ATE':>10}  {'Bias':>8}  {'Latency (μs)':>14}")
    print("-" * 56)

    for name, fn in methods.items():
        try:
            ate, latency_us = fn(X, T, Y)
            bias = abs(ate - TRUE_ATE)
            print(f"{name:<18}  {ate:>10.4f}  {bias:>8.4f}  {latency_us:>14.2f}")
        except Exception as e:
            print(f"{name:<18}  ERROR: {e}")

    print()
    print("Note: Causal Forest is not natively online; periodic retraining used.")
    print("      DML is batch-only; latency reflects full-dataset processing.")


if __name__ == '__main__':
    main()
