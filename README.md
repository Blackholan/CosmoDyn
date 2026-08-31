# CosmoDyn

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

**CosmoDyn** is an open-source Python framework designed to reconstruct **time-dependent galactic potentials** from cosmological simulations and model the dynamical evolution of compact stellar systems such as **globular clusters (GCs)**, **nuclear star clusters (NSCs)** and **massive black holes (MBHs)**.

---

## Features

The current version includes

- ✅ Generation of **in-situ and ex-situ globular cluster initial conditions**
- ✅ Static and time-evolving galactic potentials
- ✅ Moving satellite-galaxy potentials
- ✅ Chandrasekhar dynamical friction in Cold Dark Matter (CDM)
- ✅ Dynamical friction in Fuzzy Dark Matter (FDM)
- ✅ Globular-cluster tidal mass loss, computed either during the orbital integration or in post-processing
- ✅ Orbital integration of globular clusters formed in the host galaxy or accreted from satellite galaxies
- ✅ Automatic release of ex-situ globular clusters from their parent satellites
- ✅ Generation and integration of globular-cluster tidal streams
- ✅ Generation of globular-cluster moving potentials for stream calculations
- ✅ Static, evolving, in-situ, ex-situ and combined execution modes
- ✅ Batch processing of multiple galaxies
- ✅ Computation-time reports

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
- colossus

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

## Getting-started

A complete getting-started guide is provided in Appendix~B of Boldrini et al. 2026

---

## Contributing

Contributions are welcome.


## Citation

If you use **CosmoDyn** in your research, please cite

> Boldrini et al. 2026


---

## License

This project is distributed under the **MIT License**.