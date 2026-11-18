# AriaQuanta

**A Python-based quantum computing and quantum machine learning framework.**

AriaQuanta is an open-source Python library designed to explore quantum computation and quantum machine learning through a modular, NumPy-based architecture. It provides tools for constructing and simulating quantum circuits, implementing quantum algorithms, modeling quantum noise, and developing variational quantum machine learning models.

The project aims to provide a flexible foundation for quantum computing research and experimentation, with an emphasis on modularity, extensibility, and clarity.

## ✨ Features

### Quantum Computing

- **Qubits and Quantum States:** Create and manipulate individual and multi-qubit quantum states.
- **Quantum Gates:** A collection of single-qubit, multi-qubit, controlled, and custom quantum gates.
- **Quantum Circuits:** Build and execute quantum circuits using an intuitive circuit interface.
- **Statevector Simulation:** Simulate quantum circuits and obtain measurement statistics.
- **Density Matrix Simulation:** Support for density-matrix representations and quantum noise channels.
- **Noise Models:** Built-in bit-flip, phase-flip, and depolarizing noise models.
- **Visualization:** Visualize quantum circuits and measurement results.

### Quantum Algorithms

AriaQuanta includes implementations of several fundamental and variational quantum algorithms:

- Deutsch–Jozsa Algorithm
- Grover's Search Algorithm
- Quantum Fourier Transform (QFT) and Inverse QFT
- Quantum Phase Estimation (QPE)
- Variational Quantum Eigensolver (VQE)
- Quantum Approximate Optimization Algorithm (QAOA)
- Hamiltonian-based operations and expectation-value calculations

### Quantum Machine Learning

The QML module provides building blocks for designing and training parameterized quantum circuits (PQCs) and hybrid quantum-classical models.

- **Data Encoding:** Basis, amplitude, angle, Hamiltonian, uniform superposition, and QRAM-based encoding.
- **Quantum Feature Maps:** Pauli, Z, ZZ, and IQP feature maps.
- **Parameterized Circuits:** Hardware-efficient and data re-uploading ansätze.
- **Gradient Estimation:** Parameter-shift rule and Simultaneous Perturbation Stochastic Approximation (SPSA).
- **Optimization:** Gradient Descent, Adagrad, RMSProp, Adam, and SPSA optimizers.
- **Loss Functions:** MSE, MAE, Hinge, Binary Cross-Entropy, and Categorical Cross-Entropy.
- **Quantum Models:** Variational Quantum Classifier (VQC) with configurable output mappings.
- **Diagnostics:** Gradient statistics and barren plateau analysis.

## 🏗️ Project Structure

The codebase is organized into independent modules:

```text
AriaQuanta/
│
├── algorithms/             # Quantum algorithms
│
├── aqc/                    # Core quantum computing framework
│   ├── gatelibrary/        # Quantum gate implementations
│   ├── circuit.py          # Quantum circuit construction and execution
│   ├── qubit.py            # Qubit and multi-qubit states
│   ├── noise.py            # Quantum noise models
│   ├── measure.py          # Measurement operations
│   ├── hamiltonian.py
│   └── visualization.py
│
├── backend/                # Simulation backend
│   ├── simulator.py
│   ├── job.py
│   └── result.py
│
├── qml/                    # Quantum machine learning
│   ├── encoding/
│   ├── feature_map/
│   ├── ansatz/
│   ├── gradients/
│   ├── training/
│   ├── models/
│   └── diagnostics/
│
├── config.py               # Runtime and computation settings
└── _utils.py               # Shared utilities
```

## 🚀 Getting Started

### Requirements

- Python 3.9+
- NumPy
- Matplotlib

Optional dependencies:

- CuPy for GPU acceleration.
- tqdm for simulation progress bars.

### Installation

Clone the repository:

```bash
git clone https://github.com/mehdibp/AriaQuanta.git
cd AriaQuanta
```

Install the required dependencies:

```bash
pip install numpy matplotlib
```

For GPU acceleration, install a CuPy version compatible with your CUDA environment.

> Note: The repository currently provides the source code rather than a published pip package. Import AriaQuanta from its parent directory after making the source package available on your Python path.

### Example: Building and Simulating a Quantum Circuit

The following example creates a two-qubit Bell state using a Hadamard gate and a controlled-X gate, then simulates its measurement statistics.

```python
from AriaQuanta.aqc import Circuit
from AriaQuanta.aqc.gatelibrary import H, CX
from AriaQuanta.backend import Simulator

# Create a two-qubit circuit
qc = Circuit(2)

# Prepare a Bell state
qc | H(0) | CX(0, 1)

# Run the simulation
simulator = Simulator()
result = simulator.simulate(
    circuit=qc,
    iterations=1000
)

# Obtain measurement statistics
counts, probabilities = result.count()

print("Counts:", counts)
print("Probabilities:", probabilities)
```

The expected measurement outcomes are approximately:

```text
|00> : 50%
|11> : 50%
|01> :  0%
|10> :  0%
```

## ⚙️ Computational Configuration

AriaQuanta supports configurable CPU/GPU execution through its configuration module.

```python
from AriaQuanta.config import Config

# Enable GPU acceleration when CuPy is available
Config.set_use_gpu(True)

# Check GPU availability and status
print(Config.gpu_info())
```

CPU execution is available by default. GPU acceleration requires a compatible CuPy installation.

The simulator currently supports local execution. HPC, Cloud, and QPU hardware configurations are declared but are not yet implemented.

## 🔬 Project Status

AriaQuanta is an actively evolving research and development project. Its core quantum computing framework and several quantum algorithms are implemented, while the quantum machine learning module continues to expand.

Some advanced capabilities, particularly large-scale simulation and hardware integration, remain areas for further development.

Contributions, suggestions, and discussions are welcome.

## 📄 License



## 👨‍💻 Authors

**Mehdi Bakhshipoor** ----- [@mehdibp](https://github.com/mehdibp)
