# CosmoDyn

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

**CosmoDyn** is an open-source Python framework designed to reconstruct **time-dependent galactic potentials** from cosmological simulations and model the dynamical evolution of compact stellar systems such as **globular clusters (GCs)**, **nuclear star clusters (NSCs)** and **massive black holes (MBHs)**.

It combines analytic gravitational potentials with orbital integration techniques to study the evolution of stellar systems in realistic galaxy assembly histories.

---

## Features

Current version includes

- ✅ Generation of **in-situ globular cluster initial conditions**
- ✅ Static galactic potentials
- ✅ Time-evolving galactic potentials
- ✅ Chandrasekhar dynamical friction
- ✅ Fuzzy Dark Matter (FDM) dynamical friction
- ✅ Tidal mass loss
- ✅ Central capture of compact stellar systems
- ✅ Batch processing of multiple galaxies

---

## Installation

Clone the repository

```bash
git clone https://github.com/Blackholan/CosmoDyn.git
cd CosmoDyn
```

Install CosmoDyn

```bash
pip install .
```

or, for development,

```bash
pip install -e .
```

---

## Python dependencies

The following packages are installed automatically

- numpy
- scipy
- matplotlib
- astropy
- h5py
- galpy
- tqdm

---

## AGAMA dependency

The generation of initial conditions relies on the **AGAMA** package.

**AGAMA is NOT installed automatically.**

This is intentional.

CosmoDyn executes AGAMA in a **separate Python process** in order to avoid runtime conflicts between AGAMA and **galpy** (OpenMP libraries).

### AGAMA is required only for

- generating in-situ initial conditions;
- creating AGAMA particle files.

### AGAMA is NOT required for

- orbit integration;
- dynamical friction;
- tidal mass loss;
- FDM calculations;
- analysing already-generated initial conditions.

After installing AGAMA separately, 

```bash
pip install agama
```
CosmoDyn automatically searches for an existing Python installation containing AGAMA.

Only if no installation is found does the user need to specify the interpreter manually.

---

## Basic example

```python
from cosmodyn import (
    generate_in_situ_gcs,
    run_in_situ_dynamics,
)

generate_in_situ_gcs(...)

run_in_situ_dynamics(...)
```

Try 
```bash
python examples/run_in_situ.py 
```

---

## Input data

CosmoDyn does **not** distribute cosmological simulations.

Users must provide their own

- galaxy catalogues;
- galactic potential files;
- timestep files;
- particle catalogues.

---

## Project roadmap

### Current release

- ✅ In-situ globular clusters
- ✅ Static potentials
- ✅ Time-evolving potentials
- ✅ Chandrasekhar dynamical friction
- ✅ FDM dynamical friction
- ✅ Mass loss

### Planned features

- ⏳ Ex-situ globular clusters
- ⏳ Massive black holes
- ⏳ Nuclear star clusters
- ⏳ Triaxial potentials
- ⏳ Additional dark matter models
- ⏳ Automatic documentation
- ⏳ Jupyter tutorials

---

## Contributing

Contributions are welcome.

If you find a bug or would like to request a feature, please open an Issue or submit a Pull Request.

---

## Citation

If you use **CosmoDyn** in your research, please cite

> Boldrini et al. (in preparation)


---

## License

This project is distributed under the **MIT License**.