#!/usr/bin/env python3
# coding: utf-8

"""In-situ tidal-stream evolution for CosmoDyn."""

from pathlib import Path
import pickle
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy import units
from galpy.orbit import Orbit
from galpy.potential import evaluatePotentials
from joblib import Parallel, delayed
from tqdm import tqdm


def _kinetic_energy(speed):
    """Return the specific kinetic energy."""
    return 0.5 * speed**2


def _set_non_dissipative(potential):
    """Explicitly set isDissipative=False on all host-potential components."""
    components = potential if isinstance(potential, (list, tuple)) else [potential]
    for component in components:
        component.isDissipative = False


def _load_plummer_particles(plummer_file):
    """Load the GC-centred Plummer particle distribution."""
    with h5py.File(plummer_file, "r") as h5:
        x = h5["GCdata/PosX"][:]
        y = h5["GCdata/PosY"][:]
        z = h5["GCdata/PosZ"][:]
        vx = h5["GCdata/vX"][:]
        vy = h5["GCdata/vY"][:]
        vz = h5["GCdata/vZ"][:]

    cylindrical_radius = np.sqrt(x**2 + y**2)
    safe_radius = np.where(cylindrical_radius == 0.0, 1.0e-12, cylindrical_radius)
    vR = (x * vx + y * vy) / safe_radius
    vT = (x * vy - y * vx) / safe_radius
    phi = np.arctan2(y, x)

    return np.column_stack([cylindrical_radius, vR, vT, z, vz, phi])


def _moving_potential_filename(
    directory,
    potential_mode,
    df_model,
    mass_loss_mode,
    cluster_index,
):
    """Return the moving-potential file produced by in-situ dynamics."""
    mode_tag = f"{potential_mode}_{df_model}_{mass_loss_mode}"
    return Path(directory) / f"GCpotInSitu_{mode_tag}_GC{cluster_index}.pkl"


def _normalise_moving_potential_entry(entry):
    """Return a common representation for new and legacy potential files."""
    if entry is None:
        return {
            "active": False,
            "gc_potential": None,
            "moving_potential": None,
            "inactive_reason": "missing",
        }

    if isinstance(entry, dict):
        return {
            "active": bool(entry.get("active", False)),
            "gc_potential": entry.get("gc_potential"),
            "moving_potential": entry.get("moving_potential"),
            "inactive_reason": entry.get("inactive_reason"),
        }

    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        return {
            "active": True,
            "gc_potential": entry[0],
            "moving_potential": entry[1],
            "inactive_reason": None,
        }

    raise TypeError(
        "Unsupported GC moving-potential entry. Expected a CosmoDyn "
        "dictionary, a legacy two-element list/tuple, or None."
    )


def _relative_to_galactocentric(
    init_row,
    gc_x,
    gc_y,
    gc_z,
    gc_vR,
    gc_vT,
    gc_vz,
):
    """Convert one GC-centred phase-space point to the galactocentric frame."""
    relative_R = float(init_row[0])
    relative_phi = float(init_row[5])
    relative_x = relative_R * np.cos(relative_phi)
    relative_y = relative_R * np.sin(relative_phi)

    x = relative_x + float(gc_x)
    y = relative_y + float(gc_y)
    z = float(init_row[3]) + float(gc_z)
    vR = float(init_row[1]) + float(gc_vR)
    vT = float(init_row[2]) + float(gc_vT)
    vz = float(init_row[4]) + float(gc_vz)

    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return np.array([R, vR, vT, z, vz, phi], dtype=float)


def _orbit_initial_conditions(row):
    """Convert a numerical phase-space row into galpy units."""
    return [
        row[0] * units.kpc,
        row[1] * units.km / units.s,
        row[2] * units.km / units.s,
        row[3] * units.kpc,
        row[4] * units.km / units.s,
        row[5] * units.rad,
    ]


def _final_state_from_orbit(orbit, times):
    """Return final x, y, z, vR, vT, vz from an integrated galpy orbit."""
    return (
        float(orbit.x(times)[-1]),
        float(orbit.y(times)[-1]),
        float(orbit.z(times)[-1]),
        float(orbit.vR(times)[-1]),
        float(orbit.vT(times)[-1]),
        float(orbit.vz(times)[-1]),
    )


def _state_to_row(x, y, z, vR, vT, vz):
    """Convert a galactocentric final state back to the stored row format."""
    R = np.sqrt(float(x) ** 2 + float(y) ** 2)
    phi = np.arctan2(float(y), float(x))
    return np.array([R, float(vR), float(vT), float(z), float(vz), phi])


def _process_particle_snapshot(
    particle_index,
    init_row,
    already_unbound,
    times,
    host_potential,
    gc_entry,
    gc_x,
    gc_y,
    gc_z,
    gc_vR,
    gc_vT,
    gc_vz,
    integration_method,
):
    """Integrate one stream particle through one snapshot interval."""
    if already_unbound:
        orbit = Orbit(_orbit_initial_conditions(init_row))
        orbit.integrate(times, host_potential, method=integration_method)
        return particle_index, *_final_state_from_orbit(orbit, times), True, False

    if not gc_entry["active"]:
        # GC destroyed/captured: release all still-bound particles at the
        # beginning of this snapshot and continue in the host potential only.
        galactocentric_row = _relative_to_galactocentric(
            init_row,
            gc_x[0], gc_y[0], gc_z[0],
            gc_vR[0], gc_vT[0], gc_vz[0],
        )
        orbit = Orbit(_orbit_initial_conditions(galactocentric_row))
        orbit.integrate(times, host_potential, method=integration_method)
        return particle_index, *_final_state_from_orbit(orbit, times), True, True

    phase1_potential = [
        gc_entry["gc_potential"],
        gc_entry["moving_potential"],
    ]

    orbit = Orbit(_orbit_initial_conditions(init_row))
    orbit.integrate(times, phase1_potential, method=integration_method)

    vx = np.asarray(orbit.vx(times))
    vy = np.asarray(orbit.vy(times))
    vz = np.asarray(orbit.vz(times))
    speed = np.sqrt(vx**2 + vy**2 + vz**2)

    gc_potential_energy = evaluatePotentials(
        gc_entry["gc_potential"],
        orbit.R(times) * units.kpc,
        orbit.z(times) * units.kpc,
    )
    specific_energy = _kinetic_energy(speed) + gc_potential_energy
    unbound_mask = specific_energy > 0.0

    if np.any(unbound_mask):
        switch_index = int(np.argmax(unbound_mask))

        x_escape = float(orbit.x(times)[switch_index]) + float(gc_x[switch_index])
        y_escape = float(orbit.y(times)[switch_index]) + float(gc_y[switch_index])
        z_escape = float(orbit.z(times)[switch_index]) + float(gc_z[switch_index])
        vR_escape = float(orbit.vR(times)[switch_index]) + float(gc_vR[switch_index])
        vT_escape = float(orbit.vT(times)[switch_index]) + float(gc_vT[switch_index])
        vz_escape = float(orbit.vz(times)[switch_index]) + float(gc_vz[switch_index])

        R_escape = np.sqrt(x_escape**2 + y_escape**2)
        phi_escape = np.arctan2(y_escape, x_escape)
        escape_row = np.array(
            [R_escape, vR_escape, vT_escape, z_escape, vz_escape, phi_escape]
        )

        tail_times = times[switch_index:]
        if len(tail_times) < 2:
            return (
                particle_index,
                x_escape, y_escape, z_escape,
                vR_escape, vT_escape, vz_escape,
                True,
                False,
            )

        free_orbit = Orbit(_orbit_initial_conditions(escape_row))
        free_orbit.integrate(tail_times, host_potential, method=integration_method)
        return particle_index, *_final_state_from_orbit(free_orbit, tail_times), True, False

    return particle_index, *_final_state_from_orbit(orbit, times), False, False


def run_in_situ_streams(
    plummer_file,
    timestep_file,
    potential_file,
    dynamics_file,
    gc_moving_potential_directory,
    output_directory,
    plot_directory=None,
    start_snapshot_index=8,
    end_snapshot_index=None,
    potential_mode="evolving",
    df_model="none",
    mass_loss_mode="none",
    integration_method="dop853_c",
    n_jobs=-1,
    batch_size=64,
):
    """Generate and evolve one tidal stream for every in-situ GC."""
    start_clock = time.time()

    plummer_file = Path(plummer_file)
    timestep_file = Path(timestep_file)
    potential_file = Path(potential_file)
    dynamics_file = Path(dynamics_file)
    gc_moving_potential_directory = Path(gc_moving_potential_directory)
    output_directory = Path(output_directory)

    for path in (plummer_file, timestep_file, potential_file, dynamics_file):
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    output_directory.mkdir(parents=True, exist_ok=True)
    if plot_directory is not None:
        plot_directory = Path(plot_directory)
        plot_directory.mkdir(parents=True, exist_ok=True)

    timestep_data = np.loadtxt(timestep_file)
    time_start = timestep_data[:, 0]
    time_end = timestep_data[:, 1]
    number_of_steps = timestep_data[:, 2].astype(int)

    with potential_file.open("rb") as stream:
        host_potentials = pickle.load(stream)

    if end_snapshot_index is None:
        end_snapshot_index = min(len(host_potentials), len(timestep_data))

    if not 0 <= start_snapshot_index < end_snapshot_index:
        raise ValueError("start_snapshot_index must be smaller than end_snapshot_index.")

    with h5py.File(dynamics_file, "r") as h5:
        gc_x_all = h5["GCdata/PosX"][:]
        gc_y_all = h5["GCdata/PosY"][:]
        gc_z_all = h5["GCdata/PosZ"][:]
        gc_vR_all = h5["GCdata/vR"][:]
        gc_vT_all = h5["GCdata/vT"][:]
        gc_vz_all = h5["GCdata/vz"][:]
        gc_mass_all = h5["GCdata/Mass"][:]
        captured = h5["GCdata/Captured"][:].astype(bool)
        dynamics_time = h5["Time"][:]

    number_of_clusters = gc_x_all.shape[0]
    expected_number_of_times = int(
        np.sum(number_of_steps[start_snapshot_index:end_snapshot_index])
    )

    if gc_x_all.shape[1] != expected_number_of_times:
        raise ValueError(
            "The dynamics file time dimension does not match the requested "
            f"timestep intervals: expected {expected_number_of_times}, "
            f"found {gc_x_all.shape[1]}."
        )
    if len(dynamics_time) != expected_number_of_times:
        raise ValueError("The Time dataset does not match the timestep intervals.")

    base_particle_initial_conditions = _load_plummer_particles(plummer_file)
    number_of_particles = len(base_particle_initial_conditions)

    interval_slices = {}
    cursor = 0
    for snapshot_index in range(start_snapshot_index, end_snapshot_index):
        interval_length = int(number_of_steps[snapshot_index])
        interval_slices[snapshot_index] = slice(cursor, cursor + interval_length)
        cursor += interval_length

    output_files = []

    for cluster_index in range(number_of_clusters):
        print("\n" + "=" * 70)
        print(f"Generating stream for GC {cluster_index}/{number_of_clusters - 1}")
        print("=" * 70)

        moving_potential_file = _moving_potential_filename(
            gc_moving_potential_directory,
            potential_mode,
            df_model,
            mass_loss_mode,
            cluster_index,
        )
        if not moving_potential_file.exists():
            raise FileNotFoundError(
                f"GC moving-potential file not found: {moving_potential_file}"
            )

        with moving_potential_file.open("rb") as stream:
            gc_potential_history = pickle.load(stream)

        particle_state = base_particle_initial_conditions.copy()
        particle_unbound = np.zeros(number_of_particles, dtype=bool)
        release_snapshot = np.full(number_of_particles, -1, dtype=int)
        released_by_gc_disappearance = np.zeros(number_of_particles, dtype=bool)

        final_x = final_y = final_z = None
        final_vR = final_vT = final_vz = None

        for snapshot_index in range(start_snapshot_index, end_snapshot_index):
            print(f"--- Stream snapshot {snapshot_index}/{end_snapshot_index - 1} ---")

            times_numeric = np.linspace(
                time_start[snapshot_index],
                time_end[snapshot_index],
                int(number_of_steps[snapshot_index]),
            )
            times = times_numeric * units.Gyr

            # Snapshot-indexed potentials, as in the evolving TNG50 model.
            if isinstance(host_potentials, dict):
                host_potential = host_potentials[snapshot_index]
            else:
                host_potential = host_potentials

            _set_non_dissipative(host_potential)

            entry = _normalise_moving_potential_entry(
                gc_potential_history.get(snapshot_index)
            )
            dynamics_slice = interval_slices[snapshot_index]

            gc_x = gc_x_all[cluster_index, dynamics_slice]
            gc_y = gc_y_all[cluster_index, dynamics_slice]
            gc_z = gc_z_all[cluster_index, dynamics_slice]
            gc_vR = gc_vR_all[cluster_index, dynamics_slice]
            gc_vT = gc_vT_all[cluster_index, dynamics_slice]
            gc_vz = gc_vz_all[cluster_index, dynamics_slice]

            if not entry["active"]:
                print(
                    "GC moving potential inactive "
                    f"({entry['inactive_reason']}). Releasing all remaining "
                    "bound particles into the host potential."
                )

            def process_particle(index):
                return _process_particle_snapshot(
                    particle_index=index,
                    init_row=particle_state[index],
                    already_unbound=particle_unbound[index],
                    times=times,
                    host_potential=host_potential,
                    gc_entry=entry,
                    gc_x=gc_x,
                    gc_y=gc_y,
                    gc_z=gc_z,
                    gc_vR=gc_vR,
                    gc_vT=gc_vT,
                    gc_vz=gc_vz,
                    integration_method=integration_method,
                )

            results = Parallel(
                n_jobs=n_jobs,
                backend="loky",
                batch_size=batch_size,
            )(
                delayed(process_particle)(index)
                for index in tqdm(
                    range(number_of_particles),
                    desc=f"Snapshot {snapshot_index}",
                    unit="part",
                )
            )

            results = np.asarray(results, dtype=object)
            indices = results[:, 0].astype(int)
            final_x = results[:, 1].astype(float)
            final_y = results[:, 2].astype(float)
            final_z = results[:, 3].astype(float)
            final_vR = results[:, 4].astype(float)
            final_vT = results[:, 5].astype(float)
            final_vz = results[:, 6].astype(float)
            unbound_flags = results[:, 7].astype(bool)
            disappearance_flags = results[:, 8].astype(bool)

            newly_released = (~particle_unbound[indices]) & unbound_flags
            release_snapshot[indices[newly_released]] = snapshot_index
            released_by_gc_disappearance[indices] |= disappearance_flags
            particle_unbound[indices] |= unbound_flags

            for local_index, particle_index in enumerate(indices):
                particle_state[particle_index] = _state_to_row(
                    final_x[local_index],
                    final_y[local_index],
                    final_z[local_index],
                    final_vR[local_index],
                    final_vT[local_index],
                    final_vz[local_index],
                )

            print(
                f"Snapshot {snapshot_index} completed. "
                f"{np.sum(particle_unbound)}/{number_of_particles} particles are free."
            )

        # Particles that never escaped remain in the GC-centred frame.
        # Translate them once at the final time for the saved galactocentric output.
        bound_mask = ~particle_unbound
        if np.any(bound_mask):
            final_gc_slice = interval_slices[end_snapshot_index - 1]
            gc_x_final = gc_x_all[cluster_index, final_gc_slice][-1]
            gc_y_final = gc_y_all[cluster_index, final_gc_slice][-1]
            gc_z_final = gc_z_all[cluster_index, final_gc_slice][-1]
            gc_vR_final = gc_vR_all[cluster_index, final_gc_slice][-1]
            gc_vT_final = gc_vT_all[cluster_index, final_gc_slice][-1]
            gc_vz_final = gc_vz_all[cluster_index, final_gc_slice][-1]

            for particle_index in np.where(bound_mask)[0]:
                gal_row = _relative_to_galactocentric(
                    particle_state[particle_index],
                    gc_x_final, gc_y_final, gc_z_final,
                    gc_vR_final, gc_vT_final, gc_vz_final,
                )
                R, vR, vT, z, vz, phi = gal_row
                final_x[particle_index] = R * np.cos(phi)
                final_y[particle_index] = R * np.sin(phi)
                final_z[particle_index] = z
                final_vR[particle_index] = vR
                final_vT[particle_index] = vT
                final_vz[particle_index] = vz

            print(
                f"{np.sum(bound_mask)} particles remain bound at the final time "
                "and were translated to the galactocentric frame for output."
            )

        output_file = output_directory / f"InSituStream_GC{cluster_index}.h5"
        with h5py.File(output_file, "w") as h5:
            group = h5.create_group("StreamData")
            group.create_dataset("PosX", data=final_x)
            group.create_dataset("PosY", data=final_y)
            group.create_dataset("PosZ", data=final_z)
            group.create_dataset("vR", data=final_vR)
            group.create_dataset("vT", data=final_vT)
            group.create_dataset("vz", data=final_vz)

            h5.attrs["gc_index"] = cluster_index
            h5.attrs["potential_mode"] = potential_mode
            h5.attrs["df_model"] = df_model
            h5.attrs["mass_loss_mode"] = mass_loss_mode
            h5.attrs["start_snapshot_index"] = start_snapshot_index
            h5.attrs["end_snapshot_index"] = end_snapshot_index
            h5.attrs["final_gc_mass_msun"] = float(gc_mass_all[cluster_index, -1])
            h5.attrs["gc_captured"] = bool(captured[cluster_index])

        output_files.append(output_file)
        print(f"Final stream saved to: {output_file}")

        if plot_directory is not None:
            final_gc_slice = interval_slices[end_snapshot_index - 1]
            gc_x_final = gc_x_all[cluster_index, final_gc_slice][-1]
            gc_y_final = gc_y_all[cluster_index, final_gc_slice][-1]

            plt.figure(figsize=(6, 6))
            plt.scatter(
                final_x,
                final_y,
                s=0.1,
                alpha=0.7,
                label=f"{np.sum(particle_unbound)} unbound stars",
            )
            plt.scatter(
                gc_x_final,
                gc_y_final,
                s=15,
                marker="*",
                label="GC final position",
            )
            plt.xlabel("X [kpc]")
            plt.ylabel("Y [kpc]")
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig(plot_directory / f"InSituStream_GC{cluster_index}.png", dpi=200)
            plt.close()

    print(f"All in-situ streams completed in {time.time() - start_clock:.2f} s.")
    return output_files
