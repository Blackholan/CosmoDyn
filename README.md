# CosmoDyn

CosmoDyn is a semi-analytic framework for reconstructing time-dependent
galactic potentials from cosmological simulations and following the
dynamical evolution of compact stellar systems.

## Current features

- generation of in-situ globular-cluster initial conditions;
- static and time-evolving galactic potentials;
- Chandrasekhar dynamical friction;
- fuzzy-dark-matter dynamical friction;
- tidal mass loss;
- central capture of globular clusters;
- sequential processing of multiple galaxies.

## Installation from GitHub

```bash
python3 -m pip install git+https://github.com/Blackholan/CosmoDyn.git
```

## Development installation

```bash
git clone https://github.com/Blackholan/CosmoDyn.git
cd CosmoDyn
python3 -m pip install -e .
```

## Basic usage

```python
from cosmodyn import (
    generate_in_situ_gcs,
    run_in_situ_dynamics,
)
```

## External dependency

The generation of initial conditions currently requires AGAMA.

## Data

Simulation inputs and production outputs are not distributed with the
package. Users must provide their own potential, timestep, and galaxy
data files.