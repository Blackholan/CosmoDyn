#!/usr/bin/env python3
# coding: utf-8
"""Generate one ex-situ NSC or black hole per satellite galaxy."""

from pathlib import Path
import pickle

import numpy as np
from astropy import units
from galpy.potential import vcirc

from .nsc_mass import resolve_nsc_mass


SATELLITE_STELLAR_MASS_COLUMN = 6


def vcirc_fixed(radius, potential):
    """Return the physical circular velocity at ``radius`` in kpc."""
    return vcirc(potential, np.asarray(radius) * units.kpc)


def _read_satellite_list(satellites_file):
    satellites = []
    with Path(satellites_file).open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped:
                continue
            satellite_id, values = stripped.split(":", maxsplit=1)
            snapshots = [int(value) for value in values.split(",")]
            satellites.append((int(satellite_id), snapshots))
    return satellites


def _find_tagging_snapshot(snapshot_numbers, snapshot_index):
    available = [
        snapshot for snapshot in snapshot_numbers
        if snapshot >= snapshot_index
    ]
    return min(available) if available else None


def _keep_sequential_tail(snapshot_numbers):
    if not snapshot_numbers:
        return []
    result = [snapshot_numbers[0]]
    for snapshot in snapshot_numbers[1:]:
        if snapshot != result[-1] + 1:
            break
        result.append(snapshot)
    return result


def _write_satellite_list(satellites, output_file):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as stream:
        for satellite_id, snapshots in satellites:
            values = ",".join(str(snapshot) for snapshot in snapshots)
            stream.write(f"{satellite_id}: {values}\n")


def generate_ex_situ_nscs(
    galaxy_id,
    satellites_file,
    satellite_data_directory,
    output_directory,
    snapshot_index=0,
    initial_radius=0.001,
    object_type="NSC",
    object_mass=None,
):
    """
    Generate one satellite-centric NSC or BH initial condition per satellite.

    Each output row follows galpy's cylindrical convention:
    ``R, vR, vT, z, vz, phi``.
    """
    satellites_file = Path(satellites_file)
    satellite_data_directory = Path(satellite_data_directory)
    output_directory = Path(output_directory)
    object_type = str(object_type).strip().upper()

    if object_type not in {"NSC", "BH"}:
        raise ValueError("object_type must be 'NSC' or 'BH'.")

    if not satellites_file.exists():
        raise FileNotFoundError(f"File not found: {satellites_file}")
    if initial_radius <= 0.0:
        raise ValueError(
            f"{object_type} initial_radius must be strictly positive."
        )
    if object_mass is None:
        raise ValueError(f"object_mass is required for a {object_type}.")
    if not np.isfinite(object_mass):
        raise ValueError(f"{object_type}_MASS must be finite.")
    if object_type == "NSC" and object_mass < 0.0:
        raise ValueError("NSC_MASS must be non-negative.")

    output_directory.mkdir(parents=True, exist_ok=True)
    satellites = _read_satellite_list(satellites_file)
    generated_files = []
    retained_satellites = []
    tagging_snapshots = {}

    for satellite_id, snapshot_numbers in satellites:
        tagging_snapshot = _find_tagging_snapshot(
            snapshot_numbers,
            snapshot_index,
        )
        if tagging_snapshot is None:
            print(
                f"Satellite {satellite_id} disappeared before snapshot "
                f"{snapshot_index}; ex-situ {object_type} skipped."
            )
            continue

        potential_file = (
            satellite_data_directory
            / f"PotsGSat{galaxy_id}N{satellite_id}.pkl"
        )
        if not potential_file.exists():
            print(
                f"Satellite potential file not found: {potential_file}; "
                f"ex-situ {object_type} skipped."
            )
            continue

        with potential_file.open("rb") as stream:
            potentials = pickle.load(stream)

        if not isinstance(potentials, dict) or tagging_snapshot not in potentials:
            print(
                f"Satellite {satellite_id}: snapshot {tagging_snapshot} "
                f"is absent from {potential_file}; "
                f"ex-situ {object_type} skipped."
            )
            continue

        potential = potentials[tagging_snapshot]
        if not isinstance(potential, (list, tuple)) or len(potential) < 2:
            print(
                f"Satellite {satellite_id}: stellar and dark-matter "
                "potential components are unavailable; "
                f"ex-situ {object_type} skipped."
            )
            continue

        potential[0].isDissipative = False
        potential[1].isDissipative = False
        circular_velocity = float(
            vcirc_fixed(
                initial_radius,
                potential[0] + potential[1],
            )
        )

        if not np.isfinite(circular_velocity) or circular_velocity <= 0.0:
            print(
                f"Satellite {satellite_id}: invalid circular velocity "
                f"{circular_velocity}; ex-situ {object_type} skipped."
            )
            continue

        initial_conditions = np.array(
            [[initial_radius, 0.0, circular_velocity, 0.0, 0.0, 0.0]],
            dtype=float,
        )
        if object_type in {"NSC", "BH"}:
            if object_mass > 0.0:
                resolved_mass = float(object_mass)
            else:
                stellar_mass = float(
                    np.atleast_2d(
                        np.loadtxt(
                            satellite_data_directory
                            / f"G{galaxy_id}Sat{satellite_id}.txt"
                        )
                    )[
                        tagging_snapshot,
                        SATELLITE_STELLAR_MASS_COLUMN,
                    ]
                )
                if not np.isfinite(stellar_mass) or stellar_mass <= 0.0:
                    raise ValueError(
                        f"Satellite {satellite_id}: stellar mass must be "
                        "finite and positive."
                    )
                resolved_mass = (
                    0.006 * stellar_mass
                    if object_type == "BH"
                    else resolve_nsc_mass(object_mass, stellar_mass)
                )
            initial_conditions = np.column_stack(
                (initial_conditions, [resolved_mass])
            )
        output_file = (
            output_directory
            / f"IniG{galaxy_id}Sat{satellite_id}{object_type}.txt"
        )
        np.savetxt(output_file, initial_conditions, fmt="%.16g")
        generated_files.append(output_file)

        tagging_position = snapshot_numbers.index(tagging_snapshot)
        retained_history = _keep_sequential_tail(
            snapshot_numbers[tagging_position:]
        )
        retained_satellites.append((satellite_id, retained_history))
        tagging_snapshots[satellite_id] = tagging_snapshot

        print(
            f"Ex-situ {object_type} created for satellite {satellite_id}: "
            f"snapshot={tagging_snapshot}, R={initial_radius:.6g} kpc, "
            f"vcirc={circular_velocity:.6g} km/s."
        )
        print(f"{object_type} initial conditions saved to: {output_file}")
        print(f"{object_type} mass: {resolved_mass:.6e} Msun.")

    retained_satellites_file = (
        output_directory / f"ExSituNSCSatellitesG{galaxy_id}.txt"
    )
    if object_type == "BH":
        retained_satellites_file = (
            output_directory / f"ExSituBHSatellitesG{galaxy_id}.txt"
        )
    _write_satellite_list(retained_satellites, retained_satellites_file)

    print(
        f"Generated ex-situ {object_type}s: {len(generated_files)}"
    )
    print(
        f"Retained {object_type} satellite list saved to: "
        f"{retained_satellites_file}"
    )

    return {
        "generated_files": generated_files,
        "satellite_ids": [item[0] for item in retained_satellites],
        "satellite_snapshot_lists": [item[1] for item in retained_satellites],
        "tagging_snapshots": tagging_snapshots,
        "satellites_file": retained_satellites_file,
        "object_type": object_type,
    }


def generate_ex_situ_bhs(
    galaxy_id,
    satellites_file,
    satellite_data_directory,
    output_directory,
    snapshot_index=0,
    initial_radius=0.001,
    bh_mass=0.0,
):
    """Compatibility wrapper generating one ex-situ BH per satellite."""
    return generate_ex_situ_nscs(
        galaxy_id=galaxy_id,
        satellites_file=satellites_file,
        satellite_data_directory=satellite_data_directory,
        output_directory=output_directory,
        snapshot_index=snapshot_index,
        initial_radius=initial_radius,
        object_type="BH",
        object_mass=bh_mass,
    )
