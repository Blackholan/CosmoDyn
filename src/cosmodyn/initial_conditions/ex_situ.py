#!/usr/bin/env python3
# coding: utf-8

"""Generate ex-situ GC initial conditions satellite by satellite."""

import os
from pathlib import Path
import subprocess
import pickle

import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy import units
from galpy.potential import (
    evaluatePotentials,
    vcirc,
)
from scipy.interpolate import interp1d

from .in_situ import (
    find_agama_python,
    number_of_globular_clusters,
)


def _vcirc_fixed(radius, potential):
    """Circular velocity wrapper using physical kpc units."""
    return vcirc(
        potential,
        np.asarray(radius) * units.kpc,
    )


def _read_satellite_list(satellites_file):
    """
    Read the legacy ``satellites<ID>.txt`` format.

    Example
    -------
    1: 12,13,14,15
    2: 9,10,11
    """
    satellites = []

    with open(satellites_file, "r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()

            if not stripped:
                continue

            satellite_id, values = stripped.split(": ")
            snapshot_numbers = list(
                map(int, values.split(","))
            )

            satellites.append(
                (
                    int(satellite_id),
                    snapshot_numbers,
                )
            )

    return satellites


def _find_tagging_snapshot(
    snapshot_numbers,
    snapshot_index,
):
    """
    Return the first satellite snapshot >= the global target snapshot.

    If the requested global snapshot exists in the satellite history,
    it is used directly. If the satellite appears later, its first
    available snapshot after the target is used. If the satellite has
    already disappeared before the target, return None.
    """
    available = [
        snapshot_number
        for snapshot_number in snapshot_numbers
        if snapshot_number >= snapshot_index
    ]

    if not available:
        return None

    return min(available)


def _keep_sequential_tail(values):
    """Keep the sequential tail exactly as in the legacy ex-situ script."""
    for index in range(1, len(values)):
        if values[index] != values[index - 1] + 1:
            return values[index:]

    return values


def _write_satellite_list(
    satellite_ids,
    snapshot_lists,
    output_file,
):
    """Write the retained satellite list."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as stream:
        for satellite_id, snapshots in zip(
            satellite_ids,
            snapshot_lists,
        ):
            stream.write(
                f"{satellite_id}: "
                + ",".join(map(str, snapshots))
                + "\n"
            )


def _ensure_ex_situ_agama_file(
    agama_file,
    mass_dm,
    mass_star,
    dm_scale_radius,
    stellar_half_mass_radius,
    tagging_radius_factor,
    n_iter,
    n_particles_per_component,
    agama_builder_path=None,
    agama_python_executable=None,
):
    """Create one satellite AGAMA particle file if needed."""

    agama_file = Path(agama_file)

    if agama_file.exists():
        print(
            "AGAMA satellite particle file already exists: "
            f"{agama_file}"
        )
        print("AGAMA will not be run again.")
        return

    if agama_builder_path is None:
        agama_builder_path = Path(__file__).with_name(
            "agama_ex_situ_builder.py"
        )
    else:
        agama_builder_path = Path(agama_builder_path)

    if not agama_builder_path.exists():
        raise FileNotFoundError(
            "AGAMA ex-situ builder script not found: "
            f"{agama_builder_path}"
        )

    python_executable = find_agama_python(
        agama_python_executable=agama_python_executable,
    )

    command = [
        str(python_executable),
        str(agama_builder_path),
        "--output",
        str(agama_file),
        "--mass-dm",
        str(float(mass_dm)),
        "--mass-star",
        str(float(mass_star)),
        "--dm-scale-radius",
        str(float(dm_scale_radius)),
        "--stellar-half-mass-radius",
        str(float(stellar_half_mass_radius)),
        "--tagging-radius-factor",
        str(float(tagging_radius_factor)),
        "--n-iter",
        str(int(n_iter)),
        "--n-particles",
        str(int(n_particles_per_component)),
    ]

    print("AGAMA satellite particle file not found.")
    print("Running AGAMA in a separate Python process.")
    print(f"AGAMA Python executable: {python_executable}")

    subprocess.run(command, check=True)

    if not agama_file.exists():
        raise RuntimeError(
            "The AGAMA process finished without creating "
            f"the expected file: {agama_file}"
        )


def _select_circular_candidates(
    x,
    y,
    z,
    vx,
    vy,
    vz,
    potential,
    circularity_threshold,
    minimum_tagging_radius,
):
    """Apply the same E-Lz circularity selection as the in-situ code."""

    position = np.sqrt(
        x**2 + y**2 + z**2
    )

    velocity = np.sqrt(
        vx**2 + vy**2 + vz**2
    )

    R = np.sqrt(
        x**2 + y**2
    )

    safe_R = np.where(
        R == 0.0,
        np.nan,
        R,
    )

    vR = (
        x * vx
        + y * vy
    ) / safe_R

    vT = (
        x * vy
        - y * vx
    ) / safe_R

    phi = np.arctan2(
        y,
        x,
    )

    valid = (
        np.isfinite(vR)
        & np.isfinite(vT)
    )

    position = position[valid]
    velocity = velocity[valid]
    R = R[valid]
    z = z[valid]
    vz = vz[valid]
    vR = vR[valid]
    vT = vT[valid]
    phi = phi[valid]

    potential_energy_1 = evaluatePotentials(
        potential[0],
        position * units.kpc,
        0.0 * units.kpc,
    )

    potential_energy_2 = evaluatePotentials(
        potential[1],
        position * units.kpc,
        0.0 * units.kpc,
    )

    energy = (
        0.5 * velocity**2
        + potential_energy_1
        + potential_energy_2
    )

    angular_momentum_z = (
        R * vT
    )

    radius_grid = np.linspace(
        0.01,
        200.0,
        10_000,
    )

    circular_velocity = _vcirc_fixed(
        radius_grid,
        potential[0] + potential[1],
    )

    circular_angular_momentum = (
        radius_grid
        * circular_velocity
    )

    circular_potential_energy_1 = evaluatePotentials(
        potential[0],
        radius_grid * units.kpc,
        0.0 * units.kpc,
    )

    circular_potential_energy_2 = evaluatePotentials(
        potential[1],
        radius_grid * units.kpc,
        0.0 * units.kpc,
    )

    circular_energy = (
        0.5 * circular_velocity**2
        + circular_potential_energy_1
        + circular_potential_energy_2
    )

    order = np.argsort(
        circular_energy
    )

    circular_energy_sorted = (
        circular_energy[order]
    )

    circular_angular_momentum_sorted = (
        circular_angular_momentum[order]
    )

    circular_energy_unique, indices = np.unique(
        circular_energy_sorted,
        return_index=True,
    )

    circular_angular_momentum_unique = (
        circular_angular_momentum_sorted[
            indices
        ]
    )

    lcirc_of_energy = interp1d(
        circular_energy_unique,
        circular_angular_momentum_unique,
        bounds_error=False,
        fill_value=np.nan,
    )

    lcirc_particles = lcirc_of_energy(
        energy
    )

    if minimum_tagging_radius is None:
        radial_selection = np.ones(
            len(position),
            dtype=bool,
        )
    else:
        if minimum_tagging_radius < 0.0:
            raise ValueError(
                "minimum_tagging_radius must be non-negative "
                "or None."
            )

        radial_selection = (
            position >= minimum_tagging_radius
        )

    if circularity_threshold is None:
        # No circularity selection.
        selected = np.where(
            np.isfinite(energy)
            & np.isfinite(angular_momentum_z)
            & radial_selection
        )[0]
    else:
        if not 0.0 <= circularity_threshold <= 1.0:
            raise ValueError(
                "circularity_threshold must be between 0 and 1, "
                "or None."
            )

        selected = np.where(
            np.isfinite(lcirc_particles)
            & (
                angular_momentum_z
                >= circularity_threshold
                * lcirc_particles
            )
            & (
                angular_momentum_z
                <= lcirc_particles
            )
            & radial_selection
        )[0]

    phase_space = {
        "position": position,
        "R": R,
        "vR": vR,
        "vT": vT,
        "z": z,
        "vz": vz,
        "phi": phi,
        "energy": energy,
        "Lz": angular_momentum_z,
    }

    return selected, phase_space


def generate_ex_situ_gcs(
    galaxy_id,
    satellites_file,
    satellite_data_directory,
    output_directory,
    snapshot_index=8,
    ngc=0,
    alpha=5,
    tagging_radius_factor=1.0,
    minimum_tagging_radius=None,
    circularity_threshold=0.6,
    n_iter=20,
    n_particles_per_component=1_000_000,
    random_seed=None,
    keep_agama_files=True,
    agama_builder_path=None,
    agama_python_executable=None,
):
    """
    Generate ex-situ GC initial conditions for all satellites.

    ``snapshot_index`` is the global target snapshot number.

    For each satellite:
      1. If the target snapshot exists, tag there.
      2. If the satellite appears later, use its first snapshot >= target.
      3. If the satellite disappeared before the target, skip it.

    Satellite data columns use the ex-situ convention:
      column 3 : dark-matter mass
      column 5 : stellar mass
      column 8 : NFW scale radius
      column 9 : stellar half-mass radius

    If ``ngc == 0``, each satellite gets a number of GCs from the
    halo-mass relation multiplied by ``alpha``. If ``ngc > 0``,
    every retained satellite gets exactly ``ngc`` GCs.
    """

    galaxy_id = int(galaxy_id)

    satellites_file = Path(
        satellites_file
    )
    satellite_data_directory = Path(
        satellite_data_directory
    )
    output_directory = Path(
        output_directory
    )

    if not satellites_file.exists():
        raise FileNotFoundError(
            f"File not found: {satellites_file}"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    agama_directory = (
        output_directory / "AGAMA"
    )
    agama_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    plots_directory = (
        output_directory / "Plots"
    )
    plots_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    satellites = _read_satellite_list(
        satellites_file
    )

    rng = np.random.default_rng(
        random_seed
    )

    retained_satellite_ids = []
    retained_snapshot_lists = []
    generated_files = []
    tagging_snapshots = {}

    for satellite_id, snapshot_numbers in satellites:

        print()
        print("=" * 70)
        print(
            f"Generating ex-situ GC ICs for "
            f"G{galaxy_id}, satellite {satellite_id}"
        )
        print("=" * 70)

        tagging_snapshot = _find_tagging_snapshot(
            snapshot_numbers,
            snapshot_index,
        )

        if tagging_snapshot is None:
            print(
                f"Satellite {satellite_id} has no snapshot "
                f">= {snapshot_index}. "
                "It disappeared before the tagging epoch. Skipping."
            )
            continue

        local_index = snapshot_numbers.index(
            tagging_snapshot
        )

        print(
            f"Requested global tagging snapshot: {snapshot_index}"
        )
        print(
            f"Satellite tagging snapshot: {tagging_snapshot}"
        )
        print(
            f"Local satellite-data index: {local_index}"
        )

        satellite_data_file = (
            satellite_data_directory
            / (
                f"G{galaxy_id}"
                f"Sat{satellite_id}.txt"
            )
        )

        if not satellite_data_file.exists():
            print(
                f"Satellite data file not found: "
                f"{satellite_data_file}. Skipping."
            )
            continue

        satellite_data = np.atleast_2d(
            np.loadtxt(
                satellite_data_file
            )
        )

        if local_index >= len(
            satellite_data
        ):
            print(
                f"Satellite {satellite_id}: local index "
                f"{local_index} is outside its data file "
                f"({len(satellite_data)} rows). Skipping."
            )
            continue

        mass_dm = float(
            satellite_data[
                local_index,
                3,
            ]
        )

        mass_star = float(
            satellite_data[
                local_index,
                5,
            ]
        )

        dm_scale_radius = float(
            satellite_data[
                local_index,
                8,
            ]
        )

        stellar_half_mass_radius = float(
            satellite_data[
                local_index,
                9,
            ]
        )

        parameters = np.array(
            [
                mass_dm,
                mass_star,
                dm_scale_radius,
                stellar_half_mass_radius,
            ],
            dtype=float,
        )

        if not np.all(
            np.isfinite(parameters)
        ):
            print(
                "Non-finite satellite parameters at the "
                "tagging snapshot. Skipping."
            )
            continue

        if np.any(
            parameters <= 0.0
        ):
            print(
                "Non-positive satellite mass or radius at the "
                "tagging snapshot. Skipping."
            )
            continue

        if ngc == 0:
            number_of_gcs = int(
                round(
                    number_of_globular_clusters(
                        mass_dm
                    )
                    * alpha,
                    0,
                )
            )

            print(
                "Number of GCs computed from the halo-mass "
                f"relation and alpha={alpha}: "
                f"{number_of_gcs}"
            )

        elif ngc > 0:
            number_of_gcs = int(
                ngc
            )

            print(
                "Number of GCs set by NGC_EX_SITU: "
                f"{number_of_gcs}"
            )

        else:
            raise ValueError(
                "NGC_EX_SITU must be greater than "
                "or equal to zero."
            )

        if number_of_gcs == 0:
            print(
                "No GC assigned to this satellite. Skipping."
            )
            continue

        agama_file = (
            agama_directory
            / (
                f"ExSituAGAMA_G{galaxy_id}"
                f"_Sat{satellite_id}"
                f"_Snap{tagging_snapshot}"
                f"_N{n_particles_per_component:.0e}.h5"
            )
        )

        _ensure_ex_situ_agama_file(
            agama_file=agama_file,
            mass_dm=mass_dm,
            mass_star=mass_star,
            dm_scale_radius=dm_scale_radius,
            stellar_half_mass_radius=(
                stellar_half_mass_radius
            ),
            tagging_radius_factor=(
                tagging_radius_factor
            ),
            n_iter=n_iter,
            n_particles_per_component=(
                n_particles_per_component
            ),
            agama_builder_path=(
                agama_builder_path
            ),
            agama_python_executable=(
                agama_python_executable
            ),
        )

        with h5py.File(
            agama_file,
            "r",
        ) as h5:
            x = h5["GCdata/PosX"][:]
            y = h5["GCdata/PosY"][:]
            z = h5["GCdata/PosZ"][:]
            vx = h5["GCdata/vX"][:]
            vy = h5["GCdata/vY"][:]
            vz = h5["GCdata/vZ"][:]

        satellite_potential_file = (
            satellite_data_directory
            / f"PotsGSat{galaxy_id}N{satellite_id}.pkl"
        )

        if not satellite_potential_file.exists():
            print(
                f"Satellite potential file not found: "
                f"{satellite_potential_file}. Skipping."
            )
            continue

        with open(satellite_potential_file, "rb") as stream:
            satellite_potentials = pickle.load(stream)

        if tagging_snapshot not in satellite_potentials:
            print(
                f"Satellite {satellite_id}: snapshot "
                f"{tagging_snapshot} is not available in "
                f"{satellite_potential_file}. Skipping."
            )
            continue

        satellite_potential = satellite_potentials[
            tagging_snapshot
        ]

        if isinstance(satellite_potential, (list, tuple)):
            for component in satellite_potential:
                component.isDissipative = False
        else:
            satellite_potential.isDissipative = False

        candidates, phase_space = (
            _select_circular_candidates(
                x=x,
                y=y,
                z=z,
                vx=vx,
                vy=vy,
                vz=vz,
                potential=satellite_potential,
                circularity_threshold=(
                    circularity_threshold
                ),
                minimum_tagging_radius=(
                    minimum_tagging_radius
                ),
            )
        )

        print(
            "Number of candidate particles available "
            "for GC tagging: "
            f"{len(candidates)} "
            f"(requested: {number_of_gcs})"
        )

        if len(candidates) < number_of_gcs:
            print(
                "Not enough circular candidate particles "
                "for this satellite. Skipping."
            )
            continue

        drawn = rng.choice(
            candidates,
            size=number_of_gcs,
            replace=False,
        )

        output_data = np.column_stack(
            [
                phase_space["R"][drawn],
                phase_space["vR"][drawn],
                phase_space["vT"][drawn],
                phase_space["z"][drawn],
                phase_space["vz"][drawn],
                phase_space["phi"][drawn],
            ]
        )

        output_file = (
            output_directory
            / (
                f"IniG{galaxy_id}"
                f"Sat{satellite_id}GCs.txt"
            )
        )

        np.savetxt(
            output_file,
            output_data,
            fmt="%s",
        )

        generated_files.append(
            output_file
        )

        retained_satellite_ids.append(
            satellite_id
        )

        # The retained history begins at the actual tagging snapshot.
        tagging_position = snapshot_numbers.index(
            tagging_snapshot
        )

        retained_history = snapshot_numbers[
            tagging_position:
        ]

        retained_snapshot_lists.append(
            _keep_sequential_tail(
                retained_history
            )
        )

        tagging_snapshots[
            satellite_id
        ] = tagging_snapshot

        print(
            f"Initial conditions created: "
            f"{output_file}"
        )

        print(
            "Maximum satellite-centric radius of tagged GCs: "
            f"{np.max(phase_space['position'][drawn]):.2f} kpc "
            f"(stellar half-mass radius = "
            f"{stellar_half_mass_radius:.2f} kpc)"
        )

        plt.figure()

        plt.scatter(
            phase_space["Lz"] / 1.0e3,
            phase_space["energy"] / 1.0e5,
            s=1,
        )

        plt.scatter(
            phase_space["Lz"][
                candidates
            ] / 1.0e3,
            phase_space["energy"][
                candidates
            ] / 1.0e5,
            s=1,
        )

        plt.scatter(
            phase_space["Lz"][
                drawn
            ] / 1.0e3,
            phase_space["energy"][
                drawn
            ] / 1.0e5,
            s=8,
            label=(
                "Tagged globular clusters"
            ),
        )

        plt.xlabel(
            r"Angular momentum $L_z$ "
            r"$[10^3\;\mathrm{kpc\,km\,s^{-1}}]$"
        )

        plt.ylabel(
            r"Energy $E$ "
            r"$[10^5\;\mathrm{km^2\,s^{-2}}]$"
        )

        plt.legend()
        plt.tight_layout()

        plot_file = (
            plots_directory
            / (
                f"ExSituELz_G{galaxy_id}"
                f"_Sat{satellite_id}"
                f"_Snap{tagging_snapshot}.png"
            )
        )

        plt.savefig(
            plot_file,
            dpi=200,
        )
        plt.close()

        if not keep_agama_files:
            if agama_file.exists():
                agama_file.unlink()

    retained_satellites_file = (
        output_directory
        / f"ExSituSatellitesG{galaxy_id}.txt"
    )

    _write_satellite_list(
        retained_satellite_ids,
        retained_snapshot_lists,
        retained_satellites_file,
    )

    print()
    print(
        f"Retained ex-situ satellites: "
        f"{len(retained_satellite_ids)}"
    )

    print(
        "Retained satellite list saved to: "
        f"{retained_satellites_file}"
    )

    return {
        "generated_files": (
            generated_files
        ),
        "satellite_ids": (
            retained_satellite_ids
        ),
        "satellite_snapshot_lists": (
            retained_snapshot_lists
        ),
        "tagging_snapshots": (
            tagging_snapshots
        ),
        "satellites_file": (
            retained_satellites_file
        ),
    }
