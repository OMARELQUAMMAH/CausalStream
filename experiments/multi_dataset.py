# experiments/multi_dataset.py
# CausalStream: Multi-Dataset Validation with Ground Truth
# Section 5.4.6 of the paper
#
# Evaluates CausalStream on three datasets with known ground truth:
#   1. IHDP       — healthcare benchmark (Hill, 2011)
#   2. Online Shoppers — e-commerce (UCI, Sakar et al. 2019)
#   3. Synthetic Uplift — 1M samples (Zhao et al. 2022, Zenodo)
#
# Dataset paths must be provided via command line arguments.
#
# Usage:
#   python experiments/multi_dataset.py \
#       --ihdp     path/to/ihdp_data.csv \
#       --shoppers path/to/online_shoppers.csv \
#       --uplift   path/to/uplift_synthetic.csv

import time
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, SGDRegressor

EPS = 1e-8

# ── Core runner ────────────────────────────────────────────────

class WelfordVariance:
    def __init__(self):
        self.n = 0; self.mean = 0.0; self.M2 = 0.0
    def update(self, x):
        self.n += 1; delta = x - self.mean
        self.mean += delta / self.n; self.M2 += delta * (x - self.mean)
    def std(self):
        return np.sqrt(self.M2 / (self.n - 1)) if self.n > 1 else 0.0


def run_causalstream(X, T, Y):
    """
    Run CausalStream on a dataset. Returns (ate, std_error, throughput).
    X: (n, d) feature matrix
    T: (n,) binary treatment vector
    Y: (n,) outcome vector
    """
    n = len(T)
    prop = SGDClassifier(loss='log_loss', learning_rate='invscaling',
                         eta0=0.01, power_t=1.0, random_state=42)
    out = {
        0: SGDRegressor(learning_rate='invscaling', eta0=0.01, power_t=1.0, random_state=42),
        1: SGDRegressor(learning_rate='invscaling', eta0=0.01, power_t=1.0, random_state=42)
    }
    classes = np.array([0, 1])
    n_seen = {0: 0, 1: 0}; pn = 0
    welford = WelfordVariance()
    tau_hat = 0.0
    t0 = time.time()

    for i in range(n):
        x = X[i].reshape(1, -1); t = int(T[i]); y = float(Y[i])
        if pn == 0: prop.partial_fit(x, [t], classes=classes)
        else:       prop.partial_fit(x, [t])
        pn += 1
        pi = np.clip(prop.predict_proba(x)[0][1], EPS, 1 - EPS)
        out[t].partial_fit(x, [y]); n_seen[t] += 1
        mu0 = out[0].predict(x)[0] if n_seen[0] > 0 else y
        mu1 = out[1].predict(x)[0] if n_seen[1] > 0 else y
        psi = T[i]*(y-mu1)/pi - (1-T[i])*(y-mu0)/(1-pi) + mu1 - mu0
        tau_hat += (psi - tau_hat) / (i + 1)
        welford.update(psi)

    elapsed = time.time() - t0
    se = welford.std() / np.sqrt(n) if n > 1 else 0.0
    throughput = n / elapsed if elapsed > 0 else 0.0
    return tau_hat, se, throughput


# ── Dataset 1: IHDP ───────────────────────────────────────────

def run_ihdp(path):
    """
    IHDP benchmark (Hill, 2011).
    Expected columns: treatment, y_factual, x1..x25 (or similar)
    True ATE ≈ 4.0 (depends on simulation setting)
    Download: https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/IHDP
    """
    print("\n[1] IHDP Benchmark (Hill, 2011)")
    print("-" * 40)
    df = pd.read_csv(path)
    print(f"    Loaded {len(df):,} observations")

    # Try common column naming conventions
    if 'treatment' in df.columns:
        T_col = 'treatment'
    elif 'treat' in df.columns:
        T_col = 'treat'
    else:
        T_col = df.columns[0]

    if 'y_factual' in df.columns:
        Y_col = 'y_factual'
    elif 'outcome' in df.columns:
        Y_col = 'outcome'
    else:
        Y_col = df.columns[1]

    feature_cols = [c for c in df.columns if c not in [T_col, Y_col, 'mu0', 'mu1',
                                                         'y_cfactual', 'e', 'yf', 'ycf']]

    T = df[T_col].values
    Y = df[Y_col].values
    X = df[feature_cols].values.astype(float)

    ate, se, throughput = run_causalstream(X, T, Y)
    ci_lo, ci_hi = ate - 1.96*se, ate + 1.96*se

    print(f"    ATE estimate   : {ate:+.4f}")
    print(f"    Std error      : {se:.4f}")
    print(f"    95% CI         : ({ci_lo:+.3f}, {ci_hi:+.3f})")
    print(f"    Throughput     : {throughput:,.0f} events/sec")
    print(f"    Note: True ATE ≈ 4.0 (simulation A, Hill 2011)")


# ── Dataset 2: Online Shoppers ────────────────────────────────

def run_shoppers(path):
    """
    UCI Online Shoppers Purchasing Intention Dataset (Sakar et al., 2019).
    Treatment: Weekend (binary), Outcome: Revenue (binary)
    Download: https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset
    """
    print("\n[2] Online Shoppers (UCI, Sakar et al. 2019)")
    print("-" * 40)
    df = pd.read_csv(path)
    print(f"    Loaded {len(df):,} sessions")

    # Treatment: Weekend visit
    T = df['Weekend'].astype(int).values if 'Weekend' in df.columns else df['weekend'].astype(int).values
    # Outcome: Revenue
    Y = df['Revenue'].astype(int).values if 'Revenue' in df.columns else df['revenue'].astype(int).values

    # Features: numeric columns only
    skip_cols = ['Weekend', 'weekend', 'Revenue', 'revenue', 'Month', 'VisitorType']
    feature_cols = [c for c in df.columns if c not in skip_cols
                    and df[c].dtype in [np.float64, np.int64, float, int]]
    X = df[feature_cols].fillna(0).values.astype(float)

    ate, se, throughput = run_causalstream(X, T, Y)
    ci_lo, ci_hi = ate - 1.96*se, ate + 1.96*se
    naive_diff = Y[T == 1].mean() - Y[T == 0].mean()

    print(f"    ATE estimate   : {ate:+.4f}")
    print(f"    Naive diff     : {naive_diff:+.4f}")
    print(f"    95% CI         : ({ci_lo:+.3f}, {ci_hi:+.3f})")
    print(f"    Throughput     : {throughput:,.0f} events/sec")
    print(f"    Interpretation : Weekend visitors {ate*100:+.2f}% purchase probability")


# ── Dataset 3: Synthetic Uplift (1M) ─────────────────────────

def run_uplift(path):
    """
    Large-scale synthetic uplift dataset (Zhao et al., 2022).
    Known true ATE = 0.1097
    Download: https://zenodo.org/record/6342552
    """
    print("\n[3] Synthetic Uplift Dataset (Zhao et al., 2022) — 1M samples")
    print("-" * 40)

    # Read in chunks for memory efficiency
    chunks = []
    chunk_size = 100000
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f"    Loaded {len(df):,} samples")

    TRUE_ATE = 0.1097

    # Common column names in Zhao et al. dataset
    if 'treatment' in df.columns:
        T_col = 'treatment'
    elif 'treat' in df.columns:
        T_col = 'treat'
    else:
        T_col = df.columns[0]

    if 'outcome' in df.columns:
        Y_col = 'outcome'
    elif 'y' in df.columns:
        Y_col = 'y'
    else:
        Y_col = df.columns[1]

    feature_cols = [c for c in df.columns if c not in [T_col, Y_col, 'ite', 'uplift']]

    T = df[T_col].values
    Y = df[Y_col].values
    X = df[feature_cols].fillna(0).values.astype(float)

    ate, se, throughput = run_causalstream(X, T, Y)
    bias = abs(ate - TRUE_ATE)
    ci_lo, ci_hi = ate - 1.96*se, ate + 1.96*se
    relative_error = bias / abs(TRUE_ATE) * 100

    print(f"    True ATE       : {TRUE_ATE}")
    print(f"    ATE estimate   : {ate:+.4f}")
    print(f"    Bias           : {bias:.4f}  ({relative_error:.1f}% relative error)")
    print(f"    95% CI         : ({ci_lo:+.3f}, {ci_hi:+.3f})")
    print(f"    Throughput     : {throughput:,.0f} events/sec")


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CausalStream multi-dataset validation")
    parser.add_argument("--ihdp",     type=str, default=None,
                        help="Path to IHDP dataset (.csv)")
    parser.add_argument("--shoppers", type=str, default=None,
                        help="Path to UCI Online Shoppers dataset (.csv)")
    parser.add_argument("--uplift",   type=str, default=None,
                        help="Path to Zhao et al. 1M synthetic uplift dataset (.csv)")
    args = parser.parse_args()

    print("CausalStream: Multi-Dataset Validation")
    print("=" * 60)

    if not any([args.ihdp, args.shoppers, args.uplift]):
        print("No datasets provided. Please specify at least one dataset path.")
        print("Usage:")
        print("  python multi_dataset.py --ihdp path/to/ihdp.csv")
        print("  python multi_dataset.py --shoppers path/to/shoppers.csv")
        print("  python multi_dataset.py --uplift path/to/uplift.csv")
        return

    if args.ihdp:
        run_ihdp(args.ihdp)
    if args.shoppers:
        run_shoppers(args.shoppers)
    if args.uplift:
        run_uplift(args.uplift)

    print("\nDone.")


if __name__ == '__main__':
    main()
