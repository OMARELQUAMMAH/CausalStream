# worker.py
# CausalStream: Distributed Experiment - Causal Worker Node
# Omar El quammah, Cui Weijun, Ouahiba Ouchkhi, Kristina Darbinian
# Nanjing University of Information Science and Technology
#
# Runs on Machine 2 (worker node).
# Receives event micro-batches from the producer, executes the incremental
# doubly robust estimator in real time, and returns acknowledgements.
# Reproduces the physical two-node distributed experiment in Section 5.3.
#
# Usage:
#   python worker.py --port 5000

import socket
import json
import time
import argparse
import traceback
import numpy as np
from collections import deque
from sklearn.linear_model import SGDClassifier, SGDRegressor


# =============================================================================
# CAUSALSTREAM CORE (mirrors causalstream_paper.py)
# =============================================================================

class WelfordVariance:
    """Online variance estimation — O(1) memory, no history retained."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)

    def variance(self):
        return self.M2 / (self.n - 1) if self.n > 1 else 0.0

    def std(self):
        return np.sqrt(self.variance())


class IncrementalDREstimator:
    """Incremental doubly robust ATE estimator with O(1) per-event complexity."""
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.tau_hat = 0.0
        self.n = 0
        self.welford = WelfordVariance()
        self.ate_history = []

    def update(self, Y_obs, T, pi, mu0, mu1):
        pi = np.clip(pi, self.epsilon, 1.0 - self.epsilon)
        psi = (
            T * (Y_obs - mu1) / pi
            - (1 - T) * (Y_obs - mu0) / (1 - pi)
            + mu1 - mu0
        )
        self.n += 1
        self.tau_hat += (psi - self.tau_hat) / self.n
        self.welford.update(psi)
        self.ate_history.append(self.tau_hat)
        return self.tau_hat

    def confidence_interval(self, alpha=0.05):
        if self.n < 2:
            return (float('-inf'), float('inf'))
        se = self.welford.std() / np.sqrt(self.n)
        return (self.tau_hat - 1.96 * se, self.tau_hat + 1.96 * se)

    def std_error(self):
        if self.n < 2:
            return float('inf')
        return self.welford.std() / np.sqrt(self.n)


class PropensityService:
    """Online propensity score estimation via SGD logistic regression."""
    def __init__(self):
        self.model = SGDClassifier(
            loss='log_loss', learning_rate='invscaling',
            eta0=0.01, power_t=1.0, warm_start=True, random_state=42)
        self.n_seen = 0
        self.classes = np.array([0, 1])

    def update_and_predict(self, X, T):
        X = np.array(X).reshape(1, -1)
        if self.n_seen == 0:
            self.model.partial_fit(X, [T], classes=self.classes)
        else:
            self.model.partial_fit(X, [T])
        self.n_seen += 1
        return self.model.predict_proba(X)[0][1]


class OutcomeService:
    """Online outcome model estimation via SGD regression."""
    def __init__(self):
        self.models = {
            0: SGDRegressor(learning_rate='invscaling', eta0=0.01,
                            power_t=1.0, warm_start=True, random_state=42),
            1: SGDRegressor(learning_rate='invscaling', eta0=0.01,
                            power_t=1.0, warm_start=True, random_state=42)
        }
        self.n_seen = {0: 0, 1: 0}

    def update_and_predict(self, X, T, Y):
        X = np.array(X).reshape(1, -1)
        self.models[T].partial_fit(X, [Y])
        self.n_seen[T] += 1
        mu0 = self.models[0].predict(X)[0] if self.n_seen[0] > 0 else Y
        mu1 = self.models[1].predict(X)[0] if self.n_seen[1] > 0 else Y
        return mu0, mu1


class CausalStreamWorker:
    """
    CausalStream worker: receives events, runs incremental DR estimator,
    maintains O(1) state per event.
    """
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.propensity_svc = PropensityService()
        self.outcome_svc = OutcomeService()
        self.dr_estimator = IncrementalDREstimator()
        self.n_processed = 0
        self.peak_memory_mb = 0.0

    def process_event(self, Y_obs, T, X):
        T = int(T)
        pi = self.propensity_svc.update_and_predict(X, T)
        mu0, mu1 = self.outcome_svc.update_and_predict(X, T, Y_obs)
        ate = self.dr_estimator.update(Y_obs, T, pi, mu0, mu1)
        self.n_processed += 1
        return ate

    def process_batch(self, batch):
        """Process a list of events, return latest ATE."""
        ate = 0.0
        for event in batch:
            ate = self.process_event(
                event['Y_obs'], event['T'], event['features'])
        return ate

    def get_results(self):
        try:
            import psutil, os
            proc = psutil.Process(os.getpid())
            mem_mb = proc.memory_info().rss / 1024 / 1024
            self.peak_memory_mb = max(self.peak_memory_mb, mem_mb)
        except ImportError:
            mem_mb = 0.0
        return {
            'ate':           self.dr_estimator.tau_hat,
            'std_error':     self.dr_estimator.std_error(),
            'ci':            self.dr_estimator.confidence_interval(),
            'n_processed':   self.n_processed,
            'peak_memory_mb': self.peak_memory_mb
        }


# =============================================================================
# WORKER SERVER
# =============================================================================

def run_worker(port, window_size):
    """
    Listen for producer connections, process event batches,
    send acknowledgements, and report final ATE.
    """
    worker = CausalStreamWorker(window_size=window_size)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', port))
    server.listen(1)

    print(f"[Worker] Listening on port {port} ...")
    print(f"[Worker] Window size: {window_size:,}")
    conn, addr = server.accept()
    print(f"[Worker] Producer connected from {addr}\n")

    start_time = time.time()
    buffer = b""
    batches_received = 0
    log_interval = 10000

    try:
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            buffer += chunk

            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()

                if line == b"DONE":
                    print("[Worker] Received DONE signal from producer.")
                    break

                try:
                    batch = json.loads(line.decode())
                    worker.process_batch(batch)
                    batches_received += 1

                    # Send acknowledgement
                    conn.sendall(b"ACK\n")

                    if worker.n_processed % log_interval == 0:
                        elapsed = time.time() - start_time
                        throughput = worker.n_processed / elapsed if elapsed > 0 else 0
                        ci = worker.dr_estimator.confidence_interval()
                        print(f"[Worker] Processed {worker.n_processed:,} events | "
                              f"ATE={worker.dr_estimator.tau_hat:+.4f} "
                              f"CI=({ci[0]:+.3f},{ci[1]:+.3f}) | "
                              f"Throughput={throughput:,.0f} ev/sec")

                except json.JSONDecodeError as e:
                    print(f"[Worker] JSON decode error: {e}")
                    conn.sendall(b"ACK\n")

            if line == b"DONE":
                break

    except Exception as e:
        print(f"[Worker] Error: {e}")
        traceback.print_exc()
    finally:
        conn.close()
        server.close()

    elapsed = time.time() - start_time
    results = worker.get_results()
    throughput = results['n_processed'] / elapsed if elapsed > 0 else 0

    print("\n" + "=" * 60)
    print("WORKER FINAL RESULTS")
    print("=" * 60)
    print(f"  Events processed       : {results['n_processed']:,}")
    print(f"  Batches received       : {batches_received:,}")
    print(f"  Total time             : {elapsed:.2f} seconds")
    print(f"  Throughput             : {throughput:,.1f} events/sec")
    print(f"  ATE estimate           : {results['ate']:+.4f}")
    print(f"  Std error              : {results['std_error']:.4f}")
    print(f"  95% CI                 : ({results['ci'][0]:+.3f}, {results['ci'][1]:+.3f})")
    if results['peak_memory_mb'] > 0:
        print(f"  Peak memory            : {results['peak_memory_mb']:.1f} MB")
    print("=" * 60)

    import json as json_mod
    metrics = {
        'n_processed':    results['n_processed'],
        'total_time_sec': round(elapsed, 2),
        'throughput_eps': round(throughput, 1),
        'ate':            round(results['ate'], 6),
        'std_error':      round(results['std_error'], 6),
        'ci_lower':       round(results['ci'][0], 4),
        'ci_upper':       round(results['ci'][1], 4),
        'peak_memory_mb': round(results['peak_memory_mb'], 1)
    }
    with open("worker_metrics.json", "w") as f:
        json_mod.dump(metrics, f, indent=2)
    print("[Worker] Metrics saved to worker_metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CausalStream distributed experiment - worker node")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to listen on (default: 5000)")
    parser.add_argument("--window_size", type=int, default=1000,
                        help="Sliding window size (default: 1000)")
    args = parser.parse_args()

    run_worker(args.port, args.window_size)
