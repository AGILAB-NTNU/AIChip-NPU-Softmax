# AICoreCompiler: A Hardware-Aware AI Compiler Framework

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)

**AICoreCompiler** is a research-oriented AI compiler framework designed to bridge the gap between high-level neural network models (ONNX) and diverse hardware backends (NVIDIA Orin/Thor, Qualcomm NPU, Google Coral, and custom FPGA accelerators).

This framework focuses on **structural optimization, hardware-aware partitioning, and automated graph rewriting**, enabling researchers to experiment with advanced inference strategies such as structural re-parameterization and polynomial approximation.

## 🚀 Key Features

* **Multi-Backend Support**: Seamlessly deploy models to various edge hardware via a unified intermediate representation (Lab-IR).
* **Graph Surgery Engine**: Powered by `onnx-graph-surgeon`, enabling precision structural modifications (fusion, splitting, and approximation).
* **Experiment-Driven Development**: A structured `experiments/` directory to manage configurations, optimization strategies, and benchmarking logs.
* **Hardware-Aware DSE**: Facilitates Design Space Exploration (DSE) by decoupling hardware-agnostic logic from hardware-specific constraints.

## 📁 Project Structure

```text
AICoreCompiler/
├── core/                # Core compilation engine & passes
│   ├── graph/           # IR management & topological processing
│   ├── passes/          # Optimization logic (e.g., Taylor Approximation, Fusion)
│   └── backend/         # Hardware mapping & target generation
├── experiments/         # Experiment execution & data analysis
│   ├── configs/         # Strategy configurations (e.g., Taylor degrees)
│   └── results/         # Performance metrics & logs
├── models/              # Model zoo for benchmarking
├── tests/               # Unit tests for verification
└── main.py              # Main CLI entry point
```
## 🛠 Getting Started

### Prerequisites
* **Python 3.10+**
* **`onnx`**, **`onnx-graph-surgeon`**
* **Target SDKs** (e.g., TensorRT, QNN, Edge TPU Compiler)

## 🔬 Research Focus
This project serves as a testbed for:

* **Operator Approximation**: Replacing non-linear functions with polynomial expansions (Taylor Series).
* **Structural Re-parameterization**: Implementing RepVGG-style structural collapse for inference acceleration.
* **Hardware Mapping**: Heterogeneous resource allocation for CPU-RTL co-design.

## 🤝 Contribution
This framework is currently maintained by the **AGILAB**. We welcome contributions that focus on new optimization passes or backend support. Please refer to `CONTRIBUTING.md` for guidelines.

## 📜 Citation
If you find this framework useful for your research, please cite:

> [Inser our relevant paper or project citation here]

