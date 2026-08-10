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
- ✅ Formation and evolution of tidal stream
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
- pandas
- joblib

---

## AGAMA dependency

The generation of initial conditions relies on the **AGAMA** package.

**AGAMA is NOT installed automatically and is required only for**

- generating in-situ initial conditions;
- creating AGAMA particle files.

After installing AGAMA separately, 

```bash
pip install agama
```
CosmoDyn automatically searches for an existing Python installation containing AGAMA.

---

## Basic example

```python
from cosmodyn import (generate_in_situ_gcs,generate_plummer_gc,run_in_situ_dynamics,run_in_situ_streams)

generate_in_situ_gcs(...)

run_in_situ_dynamics(...)

run_in_situ_streams(...)
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

---

## Planned features

- ⏳ Ex-situ globular clusters
- ⏳ Massive black holes
- ⏳ Nuclear star clusters
- ⏳ Triaxial potentials
- ⏳ Additional dark matter models

---

## Contributing

Contributions are welcome.


## Citation

If you use **CosmoDyn** in your research, please cite

> Boldrini et al. (in preparation)


---

## License

This project is distributed under the **MIT License**.