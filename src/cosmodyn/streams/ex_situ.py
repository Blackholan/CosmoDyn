#!/usr/bin/env python3
# coding: utf-8

"""Ex-situ tidal-stream evolution for CosmoDyn."""

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


def _set_non_dissipative(potential):
    components = potential if isinstance(potential, (list, tuple)) else [potential]
    for component in components:
        component.isDissipative = False


def _load_plummer_particles(plummer_file):
    with h5py.File(plummer_file, "r") as h5:
        x = h5["GCdata/PosX"][:]
        y = h5["GCdata/PosY"][:]
        z = h5["GCdata/PosZ"][:]
        vx = h5["GCdata/vX"][:]
        vy = h5["GCdata/vY"][:]
        vz = h5["GCdata/vZ"][:]

    R = np.hypot(x, y)
    safe_R = np.where(R == 0.0, np.finfo(float).tiny, R)
    vR = (x * vx + y * vy) / safe_R
    vT = (x * vy - y * vx) / safe_R
    phi = np.arctan2(y, x)
    return np.column_stack((R, vR, vT, z, vz, phi))


def _moving_potential_filename(
    directory,
    galaxy_id,
    satellite_id,
    cluster_index,
    potential_mode,
    df_model,
    mass_loss_mode,
):
    mode_tag = f"{potential_mode}_{df_model}_{mass_loss_mode}"
    return Path(directory) / (
        f"GCpotExSitu_{mode_tag}_G{galaxy_id}_"
        f"Sat{satellite_id}_GC{cluster_index}.pkl"
    )


def _orbit_initial_conditions(row):
    return [
        row[0] * units.kpc,
        row[1] * units.km / units.s,
        row[2] * units.km / units.s,
        row[3] * units.kpc,
        row[4] * units.km / units.s,
        row[5] * units.rad,
    ]


def _state_from_orbit(orbit, times):
    x = float(orbit.x(times)[-1])
    y = float(orbit.y(times)[-1])
    return np.array(
        [
            np.hypot(x, y),
            float(orbit.vR(times)[-1]),
            float(orbit.vT(times)[-1]),
            float(orbit.z(times)[-1]),
            float(orbit.vz(times)[-1]),
            np.arctan2(y, x),
        ]
    )


def _cartesian_velocity(vR, vT, phi):
    return (
        vR * np.cos(phi) - vT * np.sin(phi),
        vR * np.sin(phi) + vT * np.cos(phi),
    )


def _relative_to_galactocentric(relative_row, gc_state):
    """Add relative and GC phase space in a common Cartesian basis."""
    rel_R, rel_vR, rel_vT, rel_z, rel_vz, rel_phi = relative_row
    gc_x, gc_y, gc_z, gc_vR, gc_vT, gc_vz = gc_state

    rel_x = rel_R * np.cos(rel_phi)
    rel_y = rel_R * np.sin(rel_phi)
    gc_phi = np.arctan2(gc_y, gc_x)
    rel_vx, rel_vy = _cartesian_velocity(rel_vR, rel_vT, rel_phi)
    gc_vx, gc_vy = _cartesian_velocity(gc_vR, gc_vT, gc_phi)

    x = rel_x + gc_x
    y = rel_y + gc_y
    z = rel_z + gc_z
    vx = rel_vx + gc_vx
    vy = rel_vy + gc_vy
    vz = rel_vz + gc_vz
    R = np.hypot(x, y)
    phi = np.arctan2(y, x)
    safe_R = max(R, np.finfo(float).tiny)
    vR = (x * vx + y * vy) / safe_R
    vT = (x * vy - y * vx) / safe_R
    return np.array([R, vR, vT, z, vz, phi])


def _galactocentric_xyz(row):
    return row[0] * np.cos(row[5]), row[0] * np.sin(row[5]), row[3]


def _normalise_segments(entry, interval_length):
    if entry is None or not isinstance(entry, dict):
        return []
    segments = entry.get("segments")
    if segments is None:
        segments = [entry]
    normalised = []
    for segment in segments:
        if not segment or not segment.get("active", False):
            continue
        start = max(int(segment.get("start_index", 0)), 0)
        end = min(int(segment.get("end_index", interval_length)), interval_length)
        if end <= start:
            continue
        normalised.append(
            {
                "active": True,
                "start_index": start,
                "end_index": end,
                "frame": segment.get("frame"),
                "gc_potential": segment.get("gc_potential"),
                "moving_potential": segment.get("moving_potential"),
            }
        )
    return normalised


def _integrate_free(row, times, host_potential, integration_method):
    if len(times) < 2:
        return row
    orbit = Orbit(_orbit_initial_conditions(row))
    orbit.integrate(times, host_potential, method=integration_method)
    return _state_from_orbit(orbit, times)


def _process_particle_snapshot(
    particle_index,
    initial_row,
    already_unbound,
    times,
    host_potential,
    segments,
    gc_states,
    integration_method,
):
    """Integrate one particle, honoring a mid-snapshot GC release."""
    row = np.asarray(initial_row, dtype=float)
    if already_unbound:
        row = _integrate_free(row, times, host_potential, integration_method)
        return particle_index, row, True, False, -1

    if not segments:
        galactic_row = _relative_to_galactocentric(row, gc_states[0])
        galactic_row = _integrate_free(
            galactic_row, times, host_potential, integration_method
        )
        return particle_index, galactic_row, True, True, 0

    for segment in segments:
        start = segment["start_index"]
        end = segment["end_index"]
        segment_times = times[start:end]
        if len(segment_times) == 0:
            continue

        potential = [
            segment["gc_potential"],
            segment["moving_potential"],
        ]
        _set_non_dissipative(potential)
        orbit = Orbit(_orbit_initial_conditions(row))
        if len(segment_times) >= 2:
            orbit.integrate(
                segment_times,
                potential,
                method=integration_method,
            )

        speed_squared = (
            np.asarray(orbit.vR(segment_times)) ** 2
            + np.asarray(orbit.vT(segment_times)) ** 2
            + np.asarray(orbit.vz(segment_times)) ** 2
        )
        potential_energy = evaluatePotentials(
            segment["gc_potential"],
            np.asarray(orbit.R(segment_times)) * units.kpc,
            np.asarray(orbit.z(segment_times)) * units.kpc,
            phi=np.asarray(orbit.phi(segment_times)),
            t=segment_times,
        )
        unbound = np.where(0.5 * speed_squared + potential_energy > 0.0)[0]

        if unbound.size:
            local_index = int(unbound[0])
            absolute_index = start + local_index
            relative_escape = np.array(
                [
                    float(orbit.R(segment_times)[local_index]),
                    float(orbit.vR(segment_times)[local_index]),
                    float(orbit.vT(segment_times)[local_index]),
                    float(orbit.z(segment_times)[local_index]),
                    float(orbit.vz(segment_times)[local_index]),
                    float(orbit.phi(segment_times)[local_index]),
                ]
            )
            galactic_row = _relative_to_galactocentric(
                relative_escape,
                gc_states[absolute_index],
            )
            galactic_row = _integrate_free(
                galactic_row,
                times[absolute_index:],
                host_potential,
                integration_method,
            )
            return particle_index, galactic_row, True, False, absolute_index

        row = _state_from_orbit(orbit, segment_times)

    return particle_index, row, False, False, -1


def run_ex_situ_streams(
    galaxy_id,
    plummer_file,
    timestep_file,
    host_potential_file,
    dynamics_file,
    gc_moving_potential_directory,
    output_directory,
    plot_directory=None,
    potential_mode="evolving",
    static_potential_index=73,
    df_model="none",
    mass_loss_mode="none",
    integration_method="dop853_c",
    n_jobs=-1,
    batch_size=64,
    object_type="GC",
):
    """Generate and evolve one tidal stream for every ex-situ object."""
    start_clock = time.time()
    galaxy_id = int(galaxy_id)
    object_type = str(object_type).strip().upper()
    if object_type not in {"GC", "NSC"}:
        raise ValueError("object_type must be 'GC' or 'NSC'.")
    plummer_file = Path(plummer_file)
    timestep_file = Path(timestep_file)
    host_potential_file = Path(host_potential_file)
    dynamics_file = Path(dynamics_file)
    gc_moving_potential_directory = Path(gc_moving_potential_directory)
    output_directory = Path(output_directory)

    for path in (
        plummer_file,
        timestep_file,
        host_potential_file,
        dynamics_file,
    ):
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    output_directory.mkdir(parents=True, exist_ok=True)
    if plot_directory is not None:
        plot_directory = Path(plot_directory)
        plot_directory.mkdir(parents=True, exist_ok=True)

    timestep = np.atleast_2d(np.loadtxt(timestep_file))
    with host_potential_file.open("rb") as stream:
        host_potentials = pickle.load(stream)
    base_particles = _load_plummer_particles(plummer_file)
    number_of_particles = len(base_particles)
    output_files = []

    with h5py.File(dynamics_file, "r") as dynamics_h5:
        satellite_names = sorted(
            name for name in dynamics_h5 if name.startswith("Sat_")
        )

        for satellite_name in satellite_names:
            satellite_id = int(satellite_name.split("_")[1])
            satellite_group = dynamics_h5[satellite_name]
            object_names = sorted(
                (
                    name
                    for name in satellite_group
                    if name.startswith(f"{object_type}_")
                ),
                key=lambda name: int(name.split("_")[1]),
            )

            for object_name in object_names:
                cluster_index = int(object_name.split("_")[1])
                gc_group = satellite_group[object_name]
                gc_time = np.asarray(gc_group["time"][:])
                gc_snapshot = np.asarray(gc_group["snapshot"][:], dtype=int)
                gc_x = np.asarray(gc_group["x"][:])
                gc_y = np.asarray(gc_group["y"][:])
                gc_z = np.asarray(gc_group["z"][:])
                gc_vR = np.asarray(gc_group["vR"][:])
                gc_vT = np.asarray(gc_group["vT"][:])
                gc_vz = np.asarray(gc_group["vz"][:])

                potential_file = _moving_potential_filename(
                    gc_moving_potential_directory,
                    galaxy_id,
                    satellite_id,
                    cluster_index,
                    potential_mode,
                    df_model,
                    mass_loss_mode,
                )
                if not potential_file.exists():
                    raise FileNotFoundError(
                        f"{object_type} moving-potential file not found: "
                        f"{potential_file}"
                    )
                with potential_file.open("rb") as stream:
                    potential_history = pickle.load(stream)

                particle_state = base_particles.copy()
                particle_unbound = np.zeros(number_of_particles, dtype=bool)
                release_snapshot = np.full(number_of_particles, -1, dtype=int)
                release_sample = np.full(number_of_particles, -1, dtype=int)
                released_by_gc_disappearance = np.zeros(
                    number_of_particles, dtype=bool
                )

                snapshots = sorted(np.unique(gc_snapshot))
                print("\n" + "=" * 70)
                print(
                    f"Ex-situ stream: G{galaxy_id}, satellite {satellite_id}, "
                    f"{object_type} {cluster_index}"
                )
                print("=" * 70)

                for snapshot in snapshots:
                    mask = gc_snapshot == snapshot
                    order = np.argsort(gc_time[mask])
                    interval_time = gc_time[mask][order]
                    gc_states = np.column_stack(
                        (
                            gc_x[mask][order],
                            gc_y[mask][order],
                            gc_z[mask][order],
                            gc_vR[mask][order],
                            gc_vT[mask][order],
                            gc_vz[mask][order],
                        )
                    )
                    times = interval_time * units.Gyr
                    if potential_mode == "static":
                        host_potential = (
                            host_potentials[static_potential_index]
                            if isinstance(host_potentials, dict)
                            else host_potentials
                        )
                    else:
                        host_potential = host_potentials[snapshot]
                    _set_non_dissipative(host_potential)
                    segments = _normalise_segments(
                        potential_history.get(snapshot),
                        len(interval_time),
                    )

                    def process_particle(index):
                        return _process_particle_snapshot(
                            index,
                            particle_state[index],
                            particle_unbound[index],
                            times,
                            host_potential,
                            segments,
                            gc_states,
                            integration_method,
                        )

                    results = Parallel(
                        n_jobs=n_jobs,
                        backend="loky",
                        batch_size=batch_size,
                    )(
                        delayed(process_particle)(index)
                        for index in tqdm(
                            range(number_of_particles),
                            desc=(
                                f"Sat {satellite_id} {object_type} "
                                f"{cluster_index} S{snapshot}"
                            ),
                            unit="part",
                        )
                    )

                    for index, row, is_unbound, disappeared, sample in results:
                        if not particle_unbound[index] and is_unbound:
                            release_snapshot[index] = snapshot
                            release_sample[index] = sample
                        particle_unbound[index] |= is_unbound
                        released_by_gc_disappearance[index] |= disappeared
                        particle_state[index] = row

                # Translate particles still bound to the GC at the final time.
                bound = ~particle_unbound
                final_gc_state = np.array(
                    [gc_x[-1], gc_y[-1], gc_z[-1], gc_vR[-1], gc_vT[-1], gc_vz[-1]]
                )
                final_rows = particle_state.copy()
                for index in np.where(bound)[0]:
                    final_rows[index] = _relative_to_galactocentric(
                        particle_state[index], final_gc_state
                    )

                final_x = final_rows[:, 0] * np.cos(final_rows[:, 5])
                final_y = final_rows[:, 0] * np.sin(final_rows[:, 5])
                output_file = output_directory / (
                    f"ExSituStream_G{galaxy_id}_Sat{satellite_id}_"
                    f"{object_type}{cluster_index}.h5"
                )
                with h5py.File(output_file, "w") as h5:
                    group = h5.create_group("StreamData")
                    group.create_dataset("PosX", data=final_x)
                    group.create_dataset("PosY", data=final_y)
                    group.create_dataset("PosZ", data=final_rows[:, 3])
                    group.create_dataset("vR", data=final_rows[:, 1])
                    group.create_dataset("vT", data=final_rows[:, 2])
                    group.create_dataset("vz", data=final_rows[:, 4])
                    group.create_dataset(
                        "UnboundFromObject",
                        data=particle_unbound.astype(np.uint8),
                    )
                    group.create_dataset(
                        f"UnboundFrom{object_type}",
                        data=particle_unbound.astype(np.uint8),
                    )
                    group.create_dataset("ReleaseSnapshot", data=release_snapshot)
                    group.create_dataset("ReleaseSample", data=release_sample)
                    group.create_dataset(
                        "ReleasedByObjectDisappearance",
                        data=released_by_gc_disappearance.astype(np.uint8),
                    )
                    group.create_dataset(
                        f"ReleasedBy{object_type}Disappearance",
                        data=released_by_gc_disappearance.astype(np.uint8),
                    )
                    h5.attrs["galaxy_id"] = galaxy_id
                    h5.attrs["satellite_id"] = satellite_id
                    h5.attrs["object_index"] = cluster_index
                    h5.attrs["object_type"] = object_type
                    h5.attrs[f"{object_type.lower()}_index"] = cluster_index
                    h5.attrs["potential_mode"] = potential_mode
                    h5.attrs["df_model"] = df_model
                    h5.attrs["mass_loss_mode"] = mass_loss_mode

                output_files.append(output_file)
                print(f"Final ex-situ stream saved to: {output_file}")

                if plot_directory is not None:
                    plt.figure(figsize=(6, 6))
                    plt.scatter(final_x, final_y, s=0.1, alpha=0.7)
                    plt.scatter(
                        gc_x[-1],
                        gc_y[-1],
                        s=18,
                        marker="*",
                        label=object_type,
                    )
                    plt.xlabel("X [kpc]")
                    plt.ylabel("Y [kpc]")
                    plt.legend(loc="best")
                    plt.tight_layout()
                    plt.savefig(
                        plot_directory
                        / (
                            f"ExSituStream_G{galaxy_id}_Sat{satellite_id}_"
                            f"{object_type}{cluster_index}.png"
                        ),
                        dpi=200,
                    )
                    plt.close()

    print(
        f"All ex-situ {object_type} streams completed in "
        f"{time.time() - start_clock:.2f} s."
    )
    return output_files
