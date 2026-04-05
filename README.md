# CausalStream

**A General Architecture for Real-Time Causal Inference on Data Streams**

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

CausalStream is a streaming architecture that embeds native causal operators to enable real-time treatment effect estimation on high-volume data streams. The core contribution is an incremental doubly robust estimator with O(1) per-event complexity, achieved through coordinated state management and sufficient statistics that eliminate full-history retention.

This repository contains the code for the paper:

> **CausalStream: A General Architecture for Real-Time Causal Inference on Data Streams**  
> Omar El quammah, Cui Weijun, Ouahiba Ouchkhi, Kristina Darbinian  
> *Concurrency and Computation: Practice and Experience* (under review)

## Key Results

| Metric | Value |
|--------|-------|
| Single-node throughput | 2,441 events/sec |
| 30-day monitoring time | 143 seconds (vs 4,500 seconds batch) |
| Speedup over batch reprocessing | 31× architectural / 3.8× windowed |
| Memory (steady-state) | 312 MB |
| Distributed throughput (2 nodes) | 1,933 events/sec |
| ATE deviation across nodes | 0.0003 |
| Bias on IHDP benchmark | 0.021 |
| Bias on 1M synthetic uplift | 0.0019 |

## Repository Structure
```
CausalStream/
├── causalstream_paper.py        # Main single-node implementation
├── producer.py                  # Distributed experiment: event producer
├── worker.py                    # Distributed experiment: causal worker
├── experiments/
│   ├── coverage_probability.py  # Section 5.4.2: Coverage analysis (500 simulations)
│   ├── autocorrelation.py       # Section 5.4.3: ACF analysis across confounding levels
│   ├── robustness.py            # Section 5.4.4: High-dim sparse, non-stationary, confounding
│   ├── baseline_comparison.py   # Section 5.4.5: Comparison vs AIPW, IPW, Causal Forest, BSTS, DML
│   └── multi_dataset.py         # Section 5.4.6: IHDP, Online Shoppers, 1M synthetic uplift
├── requirements.txt
└── README.md
```

## Requirements
```
Python 3.9+
numpy
pandas
scikit-learn
matplotlib
scipy
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Single-Node Experiment
```bash
python causalstream_paper.py
```

### Distributed Two-Node Experiment

On Machine 1 (producer):
```bash
python producer.py --host <worker_ip> --port 5000
```

On Machine 2 (worker):
```bash
python worker.py --port 5000
```

### Reproduce Statistical Validation Experiments
```bash
python experiments/coverage_probability.py    # Coverage probability analysis
python experiments/autocorrelation.py         # Autocorrelation analysis
python experiments/robustness.py              # Robustness experiments
python experiments/baseline_comparison.py     # Baseline comparison
python experiments/multi_dataset.py           # Multi-dataset validation
```

## Dataset

The main experiments use **FreshRetailNet-50K**, a real-world retail dataset containing 350,000 transactions over 30 days with 19 covariates. 

Statistical validation experiments (Sections 5.4.2–5.4.4) use synthetically generated data and do not require external datasets.

Multi-dataset validation uses:
- **IHDP**: Available from [here](https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/IHDP)
- **UCI Online Shoppers**: Available from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)
- **Synthetic Uplift**: Available from [Zenodo](https://zenodo.org/record/6342552)

## Citation

If you use this code in your research, please cite:
```bibtex
@article{elquammah2025causalstream,
  title={CausalStream: A General Architecture for Real-Time Causal Inference on Data Streams},
  author={El quammah, Omar and Weijun, Cui and Ouchkhi, Ouahiba and Darbinian, Kristina},
  journal={Concurrency and Computation: Practice and Experience},
  year={2025}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contact

Omar El quammah — o.elquammah@gmail.com  
School of Business, Nanjing University of Information Science and Technology
