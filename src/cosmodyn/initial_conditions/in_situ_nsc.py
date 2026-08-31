#!/usr/bin/env python3
# coding: utf-8
"""Initial conditions for one in-situ NSC or black hole."""

from pathlib import Path
import pickle

import numpy as np
from astropy import units
from galpy.potential import vcirc

from .nsc_mass import resolve_nsc_mass


HOST_STELLAR_MASS_COLUMN = 5


def vcirc_fixed(radius, potential):
    """Return the physical circular velocity at ``radius`` in kpc."""
    return vcirc(
        potential,
        np.asarray(radius) * units.kpc,
    )


def generate_in_situ_nsc(
    mwpots_path,
    output_file,
    snapshot_index=0,
    initial_radius=0.001,
    object_type="NSC",
    object_mass=None,
    host_data_file=None,
):
    """
    Generate one NSC or BH on a circular orbit around the galaxy centre.

    The output follows galpy's cylindrical Orbit convention:
    ``R, vR, vT, z, vz, phi``.
    """
    mwpots_path = Path(mwpots_path)
    output_file = Path(output_file)
    object_type = str(object_type).strip().upper()

    if object_type not in {"NSC", "BH"}:
        raise ValueError("object_type must be 'NSC' or 'BH'.")

    if not mwpots_path.exists():
        raise FileNotFoundError(f"File not found: {mwpots_path}")
    if initial_radius <= 0.0:
        raise ValueError(
            f"{object_type} initial_radius must be strictly positive."
        )

    resolved_mass = None
    if object_mass is None:
        raise ValueError(f"object_mass is required for a {object_type}.")
    if not np.isfinite(object_mass):
        raise ValueError(
            f"{object_type}_MASS must be finite."
        )
    if object_type == "NSC" and float(object_mass) < 0.0:
        raise ValueError("NSC_MASS must be non-negative.")

    if object_type == "BH":
        if object_mass > 0.0:
            resolved_mass = float(object_mass)
        else:
            if host_data_file is None:
                raise ValueError(
                    "host_data_file is required when BH_MASS <= 0."
                )
            host_data_file = Path(host_data_file)
            if not host_data_file.exists():
                raise FileNotFoundError(f"File not found: {host_data_file}")
            host_data = np.atleast_2d(np.loadtxt(host_data_file))
            if not 0 <= snapshot_index < len(host_data):
                raise IndexError(
                    f"Snapshot {snapshot_index} is absent from "
                    f"{host_data_file}."
                )
            stellar_mass = float(
                host_data[snapshot_index, HOST_STELLAR_MASS_COLUMN]
            )
            if not np.isfinite(stellar_mass) or stellar_mass <= 0.0:
                raise ValueError(
                    "The host stellar mass must be finite and positive."
                )
            resolved_mass = 0.006 * stellar_mass
    else:
        if object_mass > 0.0:
            resolved_mass = float(object_mass)
        else:
            if host_data_file is None:
                raise ValueError(
                    "host_data_file is required when NSC_MASS == 0."
                )
            host_data_file = Path(host_data_file)
            if not host_data_file.exists():
                raise FileNotFoundError(f"File not found: {host_data_file}")
            host_data = np.atleast_2d(np.loadtxt(host_data_file))
            if not 0 <= snapshot_index < len(host_data):
                raise IndexError(
                    f"Snapshot {snapshot_index} is absent from "
                    f"{host_data_file}."
                )
            stellar_mass = float(
                host_data[snapshot_index, HOST_STELLAR_MASS_COLUMN]
            )
            resolved_mass = resolve_nsc_mass(object_mass, stellar_mass)

    with mwpots_path.open("rb") as stream:
        potentials = pickle.load(stream)

    if isinstance(potentials, dict):
        if snapshot_index not in potentials:
            raise KeyError(
                f"Snapshot {snapshot_index} is absent from {mwpots_path}."
            )
        potential = potentials[snapshot_index]
    else:
        potential = potentials

    if not isinstance(potential, (list, tuple)) or len(potential) < 2:
        raise ValueError(
            f"The {object_type} circular velocity requires potential[0] and "
            "potential[1] (stellar and dark-matter components)."
        )

    # Potentials loaded from pickle may retain an incorrect dissipative flag.
    # Both components used for vcirc are gravitational and non-dissipative.
    potential[0].isDissipative = False
    potential[1].isDissipative = False

    circular_velocity = float(
        vcirc_fixed(
            initial_radius,
            potential[0] + potential[1],
        )
    )

    if not np.isfinite(circular_velocity) or circular_velocity <= 0.0:
        raise ValueError(
            f"The circular velocity at the {object_type} initial radius "
            "is invalid: "
            f"{circular_velocity}."
        )

    initial_conditions = np.array(
        [[initial_radius, 0.0, circular_velocity, 0.0, 0.0, 0.0]],
        dtype=float,
    )
    initial_conditions = np.column_stack(
        (initial_conditions, [resolved_mass])
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_file, initial_conditions, fmt="%.16g")

    print(
        f"In-situ {object_type} initial conditions created: "
        f"R={initial_radius:.6g} kpc, "
        f"vcirc={circular_velocity:.6g} km/s."
    )
    print(f"{object_type} initial conditions saved to: {output_file}")
    print(f"{object_type} mass: {resolved_mass:.6e} Msun.")

    return initial_conditions


def generate_in_situ_bh(
    mwpots_path,
    output_file,
    snapshot_index=0,
    initial_radius=0.001,
    bh_mass=0.0,
    host_data_file=None,
):
    """Compatibility wrapper generating one in-situ black hole."""
    return generate_in_situ_nsc(
        mwpots_path=mwpots_path,
        output_file=output_file,
        snapshot_index=snapshot_index,
        initial_radius=initial_radius,
        object_type="BH",
        object_mass=bh_mass,
        host_data_file=host_data_file,
    )
