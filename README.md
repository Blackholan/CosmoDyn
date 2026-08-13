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

The easiest way to run CosmoDyn is to configure the launcher

```text
examples/run_all.py
```

Select one or several galaxies and an execution mode

```python
GALAXY_IDS = [462710]

# "in_situ": in-situ globular clusters only
# "ex_situ": ex-situ globular clusters only
# "full":    in-situ and ex-situ globular clusters
# "custom":  manually select individual stages
RUN_MODE = "in_situ"

# Generate and integrate tidal streams for the selected populations.
ENABLE_STREAMS = False
```

The main physical and numerical parameters can then be specified in the same
launcher

```python
SNAPSHOT_INDEX = 0
END_SNAPSHOT_INDEX = None

POTENTIAL_MODE = "evolving"  # "static" or "evolving"
DF_MODEL = "cdm"             # "none", "cdm" or "fdm"
MASS_LOSS_MODE = "none"      # "none", "postprocess" or "coupled"

GC_MASS = 1e6                 # Msun
GC_HALF_MASS_RADIUS = 0.01    # kpc
```

Run the pipeline from the root of the repository

```bash
python3 examples/run_all.py
```

For advanced applications, `RUN_MODE = "custom"` allows each initial-condition,
satellite, dynamics and stream stage to be enabled independently.

---

## Input data

CosmoDyn does **not** distribute cosmological simulations. Users must provide
the galaxy histories and galactic potentials extracted from their own
cosmological simulations.

For a galaxy with identifier `<ID>`, the standard input structure is

```text
DataG<ID>/
├── DataG<ID>.txt
├── PotsG<ID>.pkl
├── G<ID>TimeSat.txt
└── GSat<ID>/
```

The required inputs are

- `DataG<ID>.txt`: the evolution of the main host-galaxy properties;
- `PotsG<ID>.pkl`: the host-galaxy potentials in a format compatible with `galpy`;
- a timestep file containing the start time, end time and number of samples for each integration interval;
- `G<ID>TimeSat.txt`: the list and temporal information of the satellite galaxies, required only for ex-situ calculations or moving satellite potentials;
- `GSat<ID>/`: the individual satellite-galaxy histories, required only for ex-situ calculations or moving satellite potentials.

The example script

```text
examples/create_potential.py
```

can be used to generate example `DataG<ID>.txt` and `PotsG<ID>.pkl` files for a
user-defined static or time-evolving galactic potential.

A timestep file may contain one or several intervals. For example, a single
static integration from 0 to 13.803 Gyr with 2000 samples is specified as

```text
0 13.803 2000
```

Input paths and filenames are constructed automatically from `GALAXY_IDS` by
the standard launcher.

---

## Planned features

- ⏳ Initial conditions and dynamical evolution of massive black holes
- ⏳ Initial conditions and dynamical evolution of nuclear star clusters
- ⏳ Axisymmetric and triaxial host-galaxy potentials
- ⏳ Anisotropic dynamical-friction prescriptions
- ⏳ Additional dark-matter models
- ⏳ More flexible input interfaces for other cosmological simulations
- ⏳ Expanded validation tests and example datasets

---

## Contributing

Contributions are welcome.


## Citation

If you use **CosmoDyn** in your research, please cite

> Boldrini et al. (in preparation)


---

## License

This project is distributed under the **MIT License**.