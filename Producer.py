# producer.py
# CausalStream: Distributed Experiment - Event Producer
# Omar El quammah, Cui Weijun, Ouahiba Ouchkhi, Kristina Darbinian
# Nanjing University of Information Science and Technology
#
# Runs on Machine 1 (producer node).
# Streams FreshRetailNet-50K events to the worker node in micro-batches.
# Reproduces the physical two-node distributed experiment in Section 5.3.
#
# Usage:
#   python producer.py --data path/to/dataset.xlsx --host <worker_ip> --port 5000

import socket
import json
import time
import argparse
import pandas as pd
import numpy as np


def load_dataset(path):
    print("[Producer] Loading dataset...")
    if path.endswith('.xlsx') or path.endswith('.xls'):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    print(f"[Producer] Loaded {len(df):,} rows")
    return df


def prepare_event(row):
    """Convert a dataset row into a streaming event payload."""
    return {
        "Y_obs":      float(row['sale_amount']),
        "T":          float(1 if row['discount'] > 0 else 0),
        "product_id": str(row['product_id']),
        "features": [
            float(row['avg_temperature']),
            float(row['avg_humidity']),
            float(row['holiday_flag']),
        ]
    }


def run_producer(data_path, worker_ip, worker_port, batch_size):
    df = load_dataset(data_path)

    print("[Producer] Preparing events...")
    events = []
    for idx, row in df.iterrows():
        try:
            events.append(prepare_event(row))
        except Exception:
            continue

    total = len(events)
    treated = sum(1 for e in events if e['T'] == 1.0)
    control = total - treated
    print(f"[Producer] {total:,} events ready")
    print(f"[Producer] Treatment: {treated:,} ({100*treated/total:.1f}%)  "
          f"Control: {control:,} ({100*control/total:.1f}%)\n")

    print(f"[Producer] Connecting to worker at {worker_ip}:{worker_port} ...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((worker_ip, worker_port))
        print(f"[Producer] Connected. Streaming in batches of {batch_size} events...\n")

        sent          = 0
        total_latency = 0.0
        latencies     = []
        start_time    = time.time()

        for batch_start in range(0, total, batch_size):
            batch   = events[batch_start: batch_start + batch_size]
            payload = json.dumps(batch) + "\n"

            t0 = time.perf_counter()
            s.sendall(payload.encode())

            # Wait for acknowledgement from worker
            ack = b""
            while not ack.endswith(b"\n"):
                chunk = s.recv(64)
                if not chunk:
                    break
                ack += chunk
            t1 = time.perf_counter()

            rtt_ms = (t1 - t0) * 1000
            total_latency += rtt_ms
            latencies.append(rtt_ms)
            sent += len(batch)

            if sent % 10000 == 0:
                elapsed    = time.time() - start_time
                avg_lat    = total_latency / len(latencies)
                throughput = sent / elapsed
                print(f"[Producer] Sent {sent:,} events | "
                      f"Avg RTT: {avg_lat:.1f} ms | "
                      f"Throughput: {throughput:,.0f} events/sec")

        # Signal end of stream
        s.sendall(b"DONE\n")

        elapsed    = time.time() - start_time
        avg_lat    = total_latency / len(latencies) if latencies else 0
        throughput = sent / elapsed if elapsed > 0 else 0

        print("\n" + "=" * 60)
        print("PRODUCER FINAL METRICS")
        print("=" * 60)
        print(f"  Total events sent      : {sent:,}")
        print(f"  Batch size             : {batch_size}")
        print(f"  Total time             : {elapsed:.2f} seconds")
        print(f"  Throughput             : {throughput:,.1f} events/sec")
        print(f"  Avg batch RTT          : {avg_lat:.2f} ms")
        print(f"  Min batch RTT          : {min(latencies):.2f} ms")
        print(f"  Max batch RTT          : {max(latencies):.2f} ms")
        print("=" * 60)

        metrics = {
            "total_events":    sent,
            "batch_size":      batch_size,
            "total_time_sec":  round(elapsed, 2),
            "throughput_eps":  round(throughput, 1),
            "avg_rtt_ms":      round(avg_lat, 2),
            "min_rtt_ms":      round(min(latencies), 2),
            "max_rtt_ms":      round(max(latencies), 2),
        }
        with open("producer_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("[Producer] Metrics saved to producer_metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CausalStream distributed experiment - producer node")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to FreshRetailNet-50K dataset (.xlsx or .csv)")
    parser.add_argument("--host", type=str, required=True,
                        help="Worker node IP address")
    parser.add_argument("--port", type=int, default=5000,
                        help="Worker node port (default: 5000)")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Events per micro-batch (default: 100)")
    args = parser.parse_args()

    run_producer(args.data, args.host, args.port, args.batch_size)
