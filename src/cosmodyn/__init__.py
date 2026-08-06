"""CosmoDyn: dynamics of compact stellar systems in cosmological potentials."""

from .initial_conditions import generate_in_situ_gcs
from .dynamics import run_in_situ_dynamics

__all__ = [
    "generate_in_situ_gcs",
    "run_in_situ_dynamics",
]

__version__ = "0.1.0"