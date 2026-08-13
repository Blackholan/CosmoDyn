"""Prepare satellite trajectories and potentials for ex-situ GC dynamics.

This module replaces the legacy ``6SatPot.py`` script.  It integrates each
satellite in the host-galaxy potential, stores its galactocentric trajectory,
and constructs the potentials required to evolve GCs while they are still
bound to their satellite.
"""

from pathlib import Path
import pickle

import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy import units
from galpy.orbit import Orbit
from galpy.potential import (
    ChandrasekharDynamicalFrictionForce,
    FDMDynamicalFrictionForce,
    HernquistPotential,
    MovingObjectPotential,
)


def _read_satellite_list(satellites_file):
    """Read lines formatted as ``satellite_id: snapshot,snapshot,...``."""
    satellites = []

    with Path(satellites_file).open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped:
                continue

            satellite_id, values = stripped.split(":", maxsplit=1)
            snapshots = [
                int(value)
                for value in values.strip().split(",")
                if value.strip()
            ]
            satellites.append((int(satellite_id), snapshots))

    return satellites


def _write_satellite_list(satellites, output_file):
    """Write the snapshots successfully prepared for every satellite."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as stream:
        for satellite_id, snapshots in satellites:
            stream.write(
                f"{satellite_id}: "
                + ",".join(map(str, snapshots))
                + "\n"
            )


def _set_non_dissipative(potential):
    """Set ``isDissipative=False`` on all galpy potential components."""
    components = potential if isinstance(potential, (list, tuple)) else [potential]
    for component in components:
        component.isDissipative = False


def _combine_potential_and_force(potential, force):
    """Return a galpy-compatible list containing a potential and a force."""
    if force is None:
        return potential
    if isinstance(potential, (list, tuple)):
        return list(potential) + [force]
    return [potential, force]


def _potential_for_snapshot(
    potentials,
    snapshot_index,
    potential_mode,
    static_potential_index,
):
    """Resolve the requested evolving or static host potential."""
    if potential_mode == "static":
        if isinstance(potentials, dict):
            if static_potential_index not in potentials:
                raise KeyError(
                    f"Static potential index {static_potential_index} is absent. "
                    f"Available indices: {list(potentials.keys())}"
                )
            return potentials[static_potential_index]

        # A directly stored component or list of components is already static.
        return potentials

    if isinstance(potentials, dict):
        if snapshot_index not in potentials:
            raise KeyError(
                f"Snapshot {snapshot_index} is absent from the host potentials."
            )
        return potentials[snapshot_index]

    raise TypeError(
        "potential_mode='evolving' requires host potentials indexed by "
        "snapshot in a dictionary."
    )


def _initial_satellite_phase_space(satellite_row, host_row):
    """Return the satellite phase space relative to the host centre."""
    relative_position = satellite_row[-6:-3] - host_row[-6:-3]
    relative_velocity = satellite_row[-3:] - host_row[-3:]

    x, y, z = relative_position
    vx, vy, vz = relative_velocity
    cylindrical_radius = np.hypot(x, y)

    if cylindrical_radius == 0.0:
        raise ValueError(
            "A satellite cannot be initialized at cylindrical radius R=0."
        )

    phi = np.arctan2(y, x)
    vR = (x * vx + y * vy) / cylindrical_radius
    vT = (x * vy - y * vx) / cylindrical_radius

    return [
        cylindrical_radius * units.kpc,
        vR * units.km / units.s,
        vT * units.km / units.s,
        z * units.kpc,
        vz * units.km / units.s,
        phi * units.rad,
    ]


def _internal_satellite_potential(satellite_row):
    """Construct the legacy two-component Hernquist satellite potential."""
    dark_matter_mass = float(satellite_row[3])
    stellar_mass = float(satellite_row[5])
    dark_matter_half_mass_radius = float(satellite_row[7])
    stellar_half_mass_radius = float(satellite_row[9])

    values = np.array(
        [
            dark_matter_mass,
            stellar_mass,
            dark_matter_half_mass_radius,
            stellar_half_mass_radius,
        ]
    )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(
            "Satellite masses and half-mass radii must be finite and positive."
        )

    conversion_factor = np.sqrt(2.0) + 1.0
    dark_matter_scale_radius = dark_matter_half_mass_radius / conversion_factor
    stellar_scale_radius = stellar_half_mass_radius / conversion_factor

    return [
        HernquistPotential(
            amp=2.0 * dark_matter_mass * units.Msun,
            a=dark_matter_scale_radius * units.kpc,
        ),
        HernquistPotential(
            amp=2.0 * stellar_mass * units.Msun,
            a=stellar_scale_radius * units.kpc,
        ),
    ]


def _satellite_friction_force(
    satellite_row,
    host_potential,
    df_model,
    m22,
):
    """Construct the selected force acting on the satellite itself."""
    dark_matter_mass = float(satellite_row[3])
    stellar_mass = float(satellite_row[5])
    dark_matter_half_mass_radius = float(satellite_row[7])
    stellar_half_mass_radius = float(satellite_row[9])
    total_mass = dark_matter_mass + stellar_mass

    effective_radius = (
        dark_matter_half_mass_radius * dark_matter_mass
        + stellar_half_mass_radius * stellar_mass
    ) / total_mass

    if df_model == "cdm":
        return ChandrasekharDynamicalFrictionForce(
            GMs=total_mass * units.Msun,
            rhm=effective_radius * units.kpc,
            dens=host_potential,
        )

    if df_model == "fdm":
        return FDMDynamicalFrictionForce(
            GMs=total_mass * units.Msun,
            rhm=effective_radius * units.kpc,
            dens=host_potential,
            m=m22 * 1.0e-22 * units.eV,
        )

    return None


def _phase_space_from_orbit_endpoint(orbit, times):
    """Create initial conditions for the next snapshot interval."""
    return [
        orbit.R(times)[-1] * units.kpc,
        orbit.vR(times)[-1] * units.km / units.s,
        orbit.vT(times)[-1] * units.km / units.s,
        orbit.z(times)[-1] * units.kpc,
        orbit.vz(times)[-1] * units.km / units.s,
        orbit.phi(times)[-1] * units.rad,
    ]


def prepare_ex_situ_satellites(
    galaxy_id,
    satellites_file,
    satellite_data_directory,
    host_data_file,
    timestep_file,
    host_potential_file,
    output_directory,
    satellite_data_filename="G{galaxy_id}Sat{satellite_id}.txt",
    integration_method="dop853_c",
    potential_mode="evolving",
    static_potential_index=73,
    df_model="none",
    m22=1.0,
    start_snapshot_index=0,
    end_snapshot_index=None,
    maximum_satellite_radius=1000.0,
    write_legacy_velocity_aliases=True,
    plot_file=None,
):
    """Prepare all satellite trajectories and potentials for one host galaxy.

    Parameters
    ----------
    galaxy_id : int
        Identifier of the host galaxy.
    satellites_file : path-like
        Text file mapping satellite IDs to their available snapshot indices.
    satellite_data_directory : path-like
        Directory containing one satellite history file per satellite.
    host_data_file : path-like
        Host history. Column 0 is the snapshot index; the final six columns are
        Cartesian position and velocity ``x,y,z,vx,vy,vz``.
    timestep_file : path-like
        Rows ``time_start, time_end, number_of_steps`` indexed by snapshot.
    host_potential_file : path-like
        Pickle containing host potentials indexed by snapshot, or one directly
        stored static potential (one component or a list of components).
    output_directory : path-like
        Destination for HDF5 trajectories and potential pickle files.
    potential_mode : {"evolving", "static"}
        Use the host potential corresponding to every interval or hold one
        host potential fixed throughout the satellite integration.
    static_potential_index : int
        Snapshot potential used when ``potential_mode='static'`` and the
        pickle contains a dictionary indexed by snapshot.
    df_model : {"none", "cdm", "fdm"}
        Friction applied to the satellite orbit in the host potential.
    m22 : float
        FDM particle mass in units of 1e-22 eV.
    end_snapshot_index : int or None
        Exclusive upper snapshot bound. None uses every available interval.
    start_snapshot_index : int
        First integration interval. Satellite histories are truncated at this
        index, allowing all satellites present after the GC-tagging epoch to
        contribute to the full host potential.
    plot_file : path-like or None
        Radius-evolution plot. If None, save it as
        ``SatelliteRadiusEvolutionG<ID>.png`` in ``output_directory``.

    Notes
    -----
    The saved HDF5 velocities ``vR``, ``vT`` and ``vz`` are cylindrical about
    the host centre. Optional datasets ``vx`` and ``vy`` are compatibility
    aliases for the historical files; they contain ``vR`` and ``vT``, not
    Cartesian velocities.
    """
    galaxy_id = int(galaxy_id)
    satellites_file = Path(satellites_file)
    satellite_data_directory = Path(satellite_data_directory)
    host_data_file = Path(host_data_file)
    timestep_file = Path(timestep_file)
    host_potential_file = Path(host_potential_file)
    output_directory = Path(output_directory)

    if plot_file is None:
        plot_file = (
            output_directory
            / f"SatelliteRadiusEvolutionG{galaxy_id}.png"
        )
    else:
        plot_file = Path(plot_file)

    required_files = (
        satellites_file,
        host_data_file,
        timestep_file,
        host_potential_file,
    )
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    if potential_mode not in ("evolving", "static"):
        raise ValueError("potential_mode must be 'evolving' or 'static'.")
    if df_model not in ("none", "cdm", "fdm"):
        raise ValueError("df_model must be 'none', 'cdm', or 'fdm'.")
    if df_model == "fdm" and m22 <= 0.0:
        raise ValueError("m22 must be strictly positive for FDM friction.")
    if start_snapshot_index < 0:
        raise ValueError("start_snapshot_index must be non-negative.")
    if end_snapshot_index is not None and end_snapshot_index <= 0:
        raise ValueError("end_snapshot_index must be positive or None.")
    if (
        end_snapshot_index is not None
        and start_snapshot_index >= end_snapshot_index
    ):
        raise ValueError(
            "start_snapshot_index must be smaller than end_snapshot_index."
        )
    if maximum_satellite_radius <= 0.0:
        raise ValueError("maximum_satellite_radius must be strictly positive.")

    output_directory.mkdir(parents=True, exist_ok=True)
    potential_directory = output_directory / "Potentials"
    potential_directory.mkdir(parents=True, exist_ok=True)

    host_data = np.atleast_2d(np.loadtxt(host_data_file))
    timestep_data = np.atleast_2d(np.loadtxt(timestep_file))

    with host_potential_file.open("rb") as stream:
        host_potentials = pickle.load(stream)

    satellites = _read_satellite_list(satellites_file)
    retained_satellites = []
    moving_satellites_by_snapshot = {}
    trajectory_file = output_directory / f"ExSituSatelliteDataG{galaxy_id}.h5"

    figure, axis = plt.subplots(figsize=(10, 8))

    with h5py.File(trajectory_file, "w") as h5:
        h5.attrs["galaxy_id"] = galaxy_id
        h5.attrs["velocity_coordinates"] = "cylindrical_vR_vT_vz"
        h5.attrs["position_coordinates"] = "host_centered_cartesian_xyz"
        h5.attrs["potential_mode"] = potential_mode
        h5.attrs["static_potential_index"] = static_potential_index
        h5.attrs["df_model"] = df_model
        if df_model == "fdm":
            h5.attrs["m22"] = m22

        for satellite_id, requested_snapshots in satellites:
            print()
            print("=" * 70)
            print(f"Preparing G{galaxy_id}, satellite {satellite_id}")
            print("=" * 70)

            satellite_file = satellite_data_directory / satellite_data_filename.format(
                galaxy_id=galaxy_id,
                satellite_id=satellite_id,
            )
            if not satellite_file.exists():
                print(f"Satellite data file not found: {satellite_file}. Skipping.")
                continue

            satellite_data = np.atleast_2d(np.loadtxt(satellite_file))

            # Snapshot values in the retained satellite list are integration
            # indices shared by the satellite history, host history, timestep
            # table, and evolving-potential dictionary.  For example, a list
            # 8,...,19 is integrated strictly over intervals 8 through 19.
            available_snapshots = [
                snapshot
                for snapshot in requested_snapshots
                if 0 <= snapshot < len(satellite_data)
                and 0 <= snapshot < len(host_data)
                and 0 <= snapshot < len(timestep_data)
                and snapshot >= start_snapshot_index
                and (
                    end_snapshot_index is None
                    or snapshot < end_snapshot_index
                )
            ]

            if potential_mode == "evolving" and isinstance(host_potentials, dict):
                available_snapshots = [
                    snapshot
                    for snapshot in available_snapshots
                    if snapshot in host_potentials
                ]

            if not available_snapshots:
                print("No common satellite/host/timestep snapshot. Skipping.")
                continue

            # Only a continuous increasing tail is dynamically meaningful.
            available_snapshots = sorted(dict.fromkeys(available_snapshots))
            sequential_snapshots = [available_snapshots[0]]
            for snapshot in available_snapshots[1:]:
                if snapshot != sequential_snapshots[-1] + 1:
                    print(
                        f"Stopping before non-sequential snapshot {snapshot}; "
                        f"the last retained snapshot is {sequential_snapshots[-1]}."
                    )
                    break
                sequential_snapshots.append(snapshot)

            first_snapshot = sequential_snapshots[0]
            current_phase_space = _initial_satellite_phase_space(
                satellite_data[first_snapshot],
                host_data[first_snapshot],
            )

            internal_potentials = {}
            combined_potentials = {}
            stored_snapshots = []
            time_history = []
            snapshot_history = []
            x_history = []
            y_history = []
            z_history = []
            vR_history = []
            vT_history = []
            vz_history = []
            phi_history = []
            dark_matter_mass_history = []
            stellar_mass_history = []

            for snapshot in sequential_snapshots:
                satellite_row = satellite_data[snapshot]
                host_potential = _potential_for_snapshot(
                    host_potentials,
                    snapshot,
                    potential_mode,
                    static_potential_index,
                )
                _set_non_dissipative(host_potential)

                times_numeric = np.linspace(
                    timestep_data[snapshot, 0],
                    timestep_data[snapshot, 1],
                    int(timestep_data[snapshot, 2]),
                )
                times = times_numeric * units.Gyr

                orbit = Orbit(current_phase_space)
                initial_radius = float(orbit.r())
                if initial_radius >= maximum_satellite_radius:
                    print(
                        f"Satellite radius is {initial_radius:.3f} kpc, which "
                        f"exceeds {maximum_satellite_radius:g} kpc at snapshot "
                        f"{snapshot}. Stopping."
                    )
                    break

                friction_force = _satellite_friction_force(
                    satellite_row,
                    host_potential,
                    df_model,
                    m22,
                )
                integration_potential = _combine_potential_and_force(
                    host_potential,
                    friction_force,
                )
                orbit.integrate(
                    times,
                    integration_potential,
                    method=integration_method,
                )

                x = np.asarray(orbit.x(times))
                if np.any(~np.isfinite(x)):
                    print(f"Non-finite orbit at snapshot {snapshot}. Stopping.")
                    break

                internal_potential = _internal_satellite_potential(satellite_row)
                _set_non_dissipative(internal_potential)

                # In the satellite-centred frame, the host moves along the
                # opposite coordinate transformation encoded by galpy's
                # MovingObjectPotential, matching the legacy implementation.
                moving_host_potential = MovingObjectPotential(
                    orbit,
                    pot=host_potential,
                )
                combined_potentials[snapshot] = (
                    internal_potential + [moving_host_potential]
                )
                internal_potentials[snapshot] = internal_potential

                # Galactocentric representation used by in-situ GCs: the
                # satellite potential follows the satellite orbit through the
                # host frame.
                moving_satellite_potential = MovingObjectPotential(
                    orbit,
                    pot=internal_potential,
                )
                moving_satellites_by_snapshot.setdefault(
                    snapshot,
                    [],
                ).append(moving_satellite_potential)

                sample_count = len(times_numeric)
                time_history.append(times_numeric)
                snapshot_history.append(
                    np.full(sample_count, snapshot, dtype=int)
                )
                x_history.append(x)
                y_history.append(np.asarray(orbit.y(times)))
                z_history.append(np.asarray(orbit.z(times)))
                vR_history.append(np.asarray(orbit.vR(times)))
                vT_history.append(np.asarray(orbit.vT(times)))
                vz_history.append(np.asarray(orbit.vz(times)))
                phi_history.append(np.asarray(orbit.phi(times)))
                dark_matter_mass_history.append(
                    np.full(sample_count, float(satellite_row[3]))
                )
                stellar_mass_history.append(
                    np.full(sample_count, float(satellite_row[5]))
                )
                stored_snapshots.append(snapshot)
                current_phase_space = _phase_space_from_orbit_endpoint(
                    orbit,
                    times,
                )

            if not stored_snapshots:
                print("No satellite interval was successfully integrated. Skipping.")
                continue

            group = h5.create_group(f"Sat_{satellite_id}")
            time_output = np.concatenate(time_history)
            snapshot_output = np.concatenate(snapshot_history)
            x_output = np.concatenate(x_history)
            y_output = np.concatenate(y_history)
            z_output = np.concatenate(z_history)
            vR_output = np.concatenate(vR_history)
            vT_output = np.concatenate(vT_history)
            vz_output = np.concatenate(vz_history)
            phi_output = np.concatenate(phi_history)

            radius_output = np.sqrt(
                x_output**2
                + y_output**2
                + z_output**2
            )

            axis.plot(
                time_output,
                radius_output,
                linewidth=1.2,
                label=f"Satellite {satellite_id}",
            )

            group.create_dataset("time", data=time_output)
            group.create_dataset("snapshot_index", data=snapshot_output)
            group.create_dataset("x", data=x_output)
            group.create_dataset("y", data=y_output)
            group.create_dataset("z", data=z_output)
            group.create_dataset("vR", data=vR_output)
            group.create_dataset("vT", data=vT_output)
            group.create_dataset("vz", data=vz_output)
            group.create_dataset("phi", data=phi_output)
            group.create_dataset(
                "dark_matter_mass",
                data=np.concatenate(dark_matter_mass_history),
            )
            group.create_dataset(
                "stellar_mass",
                data=np.concatenate(stellar_mass_history),
            )

            if write_legacy_velocity_aliases:
                # Historical compatibility only: vx=vR and vy=vT.
                group.create_dataset("vx", data=vR_output)
                group.create_dataset("vy", data=vT_output)
                group.create_dataset("tSat", data=time_output)

            group.attrs["first_snapshot"] = stored_snapshots[0]
            group.attrs["last_snapshot"] = stored_snapshots[-1]
            group.attrs["velocity_coordinates"] = "cylindrical_vR_vT_vz"

            internal_file = (
                potential_directory
                / f"SatellitePotentialG{galaxy_id}Sat{satellite_id}.pkl"
            )
            combined_file = (
                potential_directory
                / f"CombinedPotentialG{galaxy_id}Sat{satellite_id}.pkl"
            )
            with internal_file.open("wb") as stream:
                pickle.dump(
                    internal_potentials,
                    stream,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            with combined_file.open("wb") as stream:
                pickle.dump(
                    combined_potentials,
                    stream,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            retained_satellites.append((satellite_id, stored_snapshots))
            print(
                f"Prepared {len(stored_snapshots)} snapshot interval(s); "
                f"trajectory samples: {len(time_output)}."
            )

    axis.set_xlabel(
        r"Time $t \; [\mathrm{Gyr}]$",
        fontsize=20,
        fontweight="bold",
    )
    axis.set_ylabel(
        r"Galactocentric distance $r \;[\mathrm{kpc}]$",
        fontsize=20,
        fontweight="bold",
    )
    axis.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.7,
        alpha=0.7,
    )
    axis.set_facecolor("whitesmoke")

    if retained_satellites:
        axis.legend()

    figure.tight_layout()
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_file, dpi=200)
    plt.close(figure)

    retained_file = output_directory / f"PreparedSatellitesG{galaxy_id}.txt"
    _write_satellite_list(retained_satellites, retained_file)

    # Construct the evolving galactocentric potential felt by in-situ objects:
    # host galaxy plus every satellite that exists during each interval.
    if end_snapshot_index is None:
        full_end_snapshot_index = min(
            len(timestep_data),
            len(host_data),
        )
    else:
        full_end_snapshot_index = min(
            end_snapshot_index,
            len(timestep_data),
            len(host_data),
        )

    full_host_potentials = {}
    for snapshot in range(
        start_snapshot_index,
        full_end_snapshot_index,
    ):
        if (
            potential_mode == "evolving"
            and isinstance(host_potentials, dict)
            and snapshot not in host_potentials
        ):
            continue

        host_potential = _potential_for_snapshot(
            host_potentials,
            snapshot,
            potential_mode,
            static_potential_index,
        )
        _set_non_dissipative(host_potential)
        host_components = (
            list(host_potential)
            if isinstance(host_potential, (list, tuple))
            else [host_potential]
        )
        full_host_potentials[snapshot] = (
            host_components
            + moving_satellites_by_snapshot.get(snapshot, [])
        )

    full_potential_file = (
        potential_directory
        / f"FullHostPotentialG{galaxy_id}.pkl"
    )
    with full_potential_file.open("wb") as stream:
        pickle.dump(
            full_host_potentials,
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print()
    print(f"Satellite trajectories saved to: {trajectory_file}")
    print(f"Satellite potentials saved to: {potential_directory}")
    print(f"Full host + moving-satellite potential saved to: {full_potential_file}")
    print(f"Prepared satellite list saved to: {retained_file}")
    print(f"Satellite radius-evolution plot saved to: {plot_file}")

    return {
        "trajectory_file": trajectory_file,
        "potential_directory": potential_directory,
        "full_potential_file": full_potential_file,
        "satellites_file": retained_file,
        "plot_file": plot_file,
        "satellites": retained_satellites,
    }
