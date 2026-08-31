# CosmoDyn

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

**CosmoDyn** is an open-source Python framework designed to reconstruct **time-dependent galactic potentials** from cosmological simulations and model the dynamical evolution of compact stellar systems such as **globular clusters (GCs)**, **nuclear star clusters (NSCs)** and **massive black holes (MBHs)**.

---

## Features

The current version includes


- ✅ Static and time-evolving analytical potentials for host galaxies
- ✅ Reconstruction of satellite trajectories and their moving gravitational potentials
- ✅ Optional inclusion of moving satellites in the total gravitational potential
- ✅ Generation of in-situ and ex-situ initial conditions for globular clusters (GCs), nuclear star clusters (NSCs), and black holes (BHs)
- ✅ User-defined or scaling-relation-based object populations and masses
- ✅ Orbital integration of GCs, NSCs, and BHs within the reconstructed cosmological environment
- ✅ Automatic release of ex-situ objects from their parent satellites
- ✅ Central-capture and escape criteria
- ✅ Chandrasekhar, anisotropic Binney, and Fuzzy Dark Matter dynamical-friction prescriptions
- ✅ Tidal mass loss during orbital integration or in post-processing
- ✅ Generation and integration of GC and NSC tidal streams using moving object potentials
- ✅ Predefined in-situ, ex-situ, and full execution modes, together with a fully configurable custom mode
- ✅ Batch processing of multiple galaxies, parallel stream calculations, diagnostic figures, and computation-time reports

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