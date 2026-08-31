#!/usr/bin/env python3
# coding: utf-8

"""Ex-situ GC dynamics before and after release from merging satellites."""

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
    MovingObjectPotential,
    PlummerPotential,
    evaluatePotentials,
    ttensor,
)
from scipy.integrate import solve_ivp


def _read_satellite_list(path):
    satellites = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            satellite_id, values = line.split(":", maxsplit=1)
            snapshots = [
                int(value)
                for value in values.strip().split(",")
                if value.strip()
            ]
            satellites.append((int(satellite_id), snapshots))
    return satellites


def _set_non_dissipative(potential):
    components = potential if isinstance(potential, (list, tuple)) else [potential]
    for component in components:
        component.isDissipative = False


def _combine(potential, force):
    if force is None:
        return potential
    if isinstance(potential, (list, tuple)):
        return list(potential) + [force]
    return [potential, force]


def _potential_at(potentials, snapshot, mode, static_index):
    if mode == "static":
        if isinstance(potentials, dict):
            return potentials[static_index]
        return potentials
    if not isinstance(potentials, dict):
        raise TypeError("An evolving potential must be a snapshot dictionary.")
    return potentials[snapshot]


def _phase_space(R, vR, vT, z, vz, phi):
    return [
        float(R) * units.kpc,
        float(vR) * units.km / units.s,
        float(vT) * units.km / units.s,
        float(z) * units.kpc,
        float(vz) * units.km / units.s,
        float(phi) * units.rad,
    ]


def _orbit_endpoint(orbit, times):
    return _phase_space(
        orbit.R(times)[-1],
        orbit.vR(times)[-1],
        orbit.vT(times)[-1],
        orbit.z(times)[-1],
        orbit.vz(times)[-1],
        orbit.phi(times)[-1],
    )


def _cylindrical_to_cartesian_velocity(vR, vT, phi):
    vx = vR * np.cos(phi) - vT * np.sin(phi)
    vy = vR * np.sin(phi) + vT * np.cos(phi)
    return vx, vy


def _cartesian_to_cylindrical(x, y, vx, vy):
    R = np.hypot(x, y)
    phi = np.arctan2(y, x)
    safe_R = np.where(np.asarray(R) == 0.0, np.finfo(float).tiny, R)
    vR = (x * vx + y * vy) / safe_R
    vT = (x * vy - y * vx) / safe_R
    return R, vR, vT, phi


def _relative_plus_satellite(relative_orbit, times, satellite):
    """Convert a satellite-centric GC orbit to galactocentric coordinates."""
    xr = np.asarray(relative_orbit.x(times))
    yr = np.asarray(relative_orbit.y(times))
    zr = np.asarray(relative_orbit.z(times))
    phir = np.asarray(relative_orbit.phi(times))
    vxr, vyr = _cylindrical_to_cartesian_velocity(
        np.asarray(relative_orbit.vR(times)),
        np.asarray(relative_orbit.vT(times)),
        phir,
    )

    x = xr + satellite["x"]
    y = yr + satellite["y"]
    z = zr + satellite["z"]
    vxs, vys = _cylindrical_to_cartesian_velocity(
        satellite["vR"], satellite["vT"], satellite["phi"]
    )
    vx = vxr + vxs
    vy = vyr + vys
    vz = np.asarray(relative_orbit.vz(times)) + satellite["vz"]
    R, vR, vT, phi = _cartesian_to_cylindrical(x, y, vx, vy)
    return {
        "x": x, "y": y, "z": z, "R": R, "phi": phi,
        "vR": vR, "vT": vT, "vz": vz,
    }


def _satellite_segment(group, snapshot, target_times):
    snapshots = np.asarray(group["snapshot_index"][:], dtype=int)
    mask = snapshots == snapshot
    if not np.any(mask):
        raise KeyError(f"No trajectory samples for satellite snapshot {snapshot}.")

    source_time = np.asarray(group["time"][:])[mask]
    order = np.argsort(source_time)
    source_time = source_time[order]
    result = {}
    for name in ("x", "y", "z", "vR", "vT", "vz", "phi"):
        values = np.asarray(group[name][:])[mask][order]
        result[name] = np.interp(target_times, source_time, values)
    return result


def _build_df_force(model, density, mass, half_mass_radius, m22):
    if model == "cdm":
        return ChandrasekharDynamicalFrictionForce(
            GMs=mass * units.Msun,
            rhm=half_mass_radius * units.kpc,
            dens=density,
        )
    if model == "fdm":
        return FDMDynamicalFrictionForce(
            GMs=mass * units.Msun,
            rhm=half_mass_radius * units.kpc,
            dens=density,
            m=m22 * 1.0e-22 * units.eV,
        )
    return None


def _satellite_energy(relative_orbit, times, internal_potential):
    """Return the satellite-centric energy, summing components explicitly."""
    speed_squared = (
        np.asarray(relative_orbit.vR(times)) ** 2
        + np.asarray(relative_orbit.vT(times)) ** 2
        + np.asarray(relative_orbit.vz(times)) ** 2
    )

    components = (
        list(internal_potential)
        if isinstance(internal_potential, (list, tuple))
        else [internal_potential]
    )
    potential_energy = np.zeros_like(speed_squared, dtype=float)

    for component in components:
        # galpy must receive each conservative component separately here.
        # Passing the complete list can make its physical-unit wrapper look
        # for a shared ro scale and obtain None.
        component.isDissipative = False
        component_energy = evaluatePotentials(
            component,
            np.asarray(relative_orbit.R(times)) * units.kpc,
            np.asarray(relative_orbit.z(times)) * units.kpc,
            phi=np.asarray(relative_orbit.phi(times)),
            t=times,
        )
        potential_energy += np.asarray(component_energy, dtype=float)

    return 0.5 * speed_squared + potential_energy


def _tidal_strength(potential, x, y, z):
    result = np.zeros(len(x), dtype=float)
    _set_non_dissipative(potential)
    for index in range(len(x)):
        eigenvalues = ttensor(
            potential,
            np.hypot(x[index], y[index]) * units.kpc,
            z[index] * units.kpc,
            phi=np.arctan2(y[index], x[index]),
            t=0.0,
            eigenval=True,
        )
        result[index] = (
            np.max(eigenvalues)
            - 0.5 * (eigenvalues[1] + eigenvalues[2])
        )
    return result


def _mass_derivative(
    time, mass, gamma, tidal_values, time_values,
    tidal_reference, normalization,
):
    tidal = max(
        float(np.interp(time, time_values, tidal_values)),
        np.finfo(float).tiny,
    )
    current_mass = max(float(mass[0]), 0.0)
    if current_mass == 0.0:
        return [0.0]
    tidal_factor = (tidal / tidal_reference) ** -0.5
    return [-(current_mass ** (1.0 - gamma)) / (normalization * tidal_factor)]


def _integrate_mass(initial_mass, times, tidal, gamma, reference, normalization):
    initial_mass = max(float(initial_mass), 0.0)
    if initial_mass == 0.0 or len(times) < 2 or times[-1] <= times[0]:
        return np.full(len(times), initial_mass)
    solution = solve_ivp(
        _mass_derivative,
        (times[0], times[-1]),
        [initial_mass],
        args=(gamma, tidal, times, reference, normalization),
        t_eval=times,
        atol=1.0e-8,
        rtol=1.0e-6,
    )
    if not solution.success or solution.y.shape[1] != len(times):
        raise RuntimeError(f"Mass-loss integration failed: {solution.message}")
    return np.maximum(solution.y[0], 0.0)


def _gc_moving_potential_entry(
    gc_orbit,
    external_potential,
    gc_mass,
    gc_scale_radius,
    start_index,
    end_index,
    frame,
    active=True,
    inactive_reason=None,
):
    """Build one GC-frame potential segment for a future stream run."""
    if not active or gc_orbit is None or gc_mass <= 0.0:
        return {
            "active": False,
            "gc_potential": None,
            "moving_potential": None,
            "gc_orbit": None,
            "start_index": int(start_index),
            "end_index": int(end_index),
            "frame": frame,
            "gc_mass": max(float(gc_mass), 0.0),
            "inactive_reason": inactive_reason,
            "fallback": "external_only",
        }

    gc_potential = PlummerPotential(
        amp=float(gc_mass) * units.Msun,
        b=float(gc_scale_radius) * units.kpc,
    )
    moving_potential = MovingObjectPotential(
        gc_orbit,
        pot=external_potential,
    )
    return {
        "active": True,
        "gc_potential": gc_potential,
        "moving_potential": moving_potential,
        "gc_orbit": gc_orbit,
        "start_index": int(start_index),
        "end_index": int(end_index),
        "frame": frame,
        "gc_mass": float(gc_mass),
        "inactive_reason": None,
        "fallback": None,
    }


def _append_segment(storage, time, phase, mass, snapshot, bound):
    storage["time"].append(np.asarray(time))
    for key in ("x", "y", "z", "vR", "vT", "vz"):
        storage[key].append(np.asarray(phase[key]))
    storage["mass"].append(np.asarray(mass))
    storage["snapshot"].append(np.full(len(time), snapshot, dtype=int))
    if np.ndim(bound) == 0:
        bound_values = np.full(len(time), bound, dtype=np.uint8)
    else:
        bound_values = np.asarray(bound, dtype=np.uint8)
    storage["bound"].append(bound_values)


def _empty_storage():
    return {
        key: []
        for key in (
            "time", "x", "y", "z", "vR", "vT", "vz",
            "mass", "snapshot", "bound",
        )
    }


def run_ex_situ_dynamics(
    galaxy_id,
    satellites_file,
    initial_conditions_directory,
    satellite_trajectory_file,
    satellite_potential_directory,
    timestep_file,
    host_orbit_potential_file,
    host_density_potential_file,
    output_file,
    plot_file=None,
    start_snapshot_index=8,
    end_snapshot_index=None,
    integration_method="dop853_c",
    potential_mode="evolving",
    static_potential_index=73,
    df_model="none",
    gc_mass=1.0e6,
    gc_half_mass_radius=0.01,
    m22=1.0,
    mass_loss_mode="none",
    mass_loss_gamma=0.7,
    tidal_strength_reference=7.01e2,
    dissolution_time_normalization=0.0107,
    central_capture_radius=0.01,
    release_energy_tolerance=0.0,
    generate_gc_moving_potentials=False,
    gc_moving_potential_directory=None,
    gc_potential_scale_radius=None,
    object_type="GC",
    initial_conditions_filename_template=None,
):
    """Integrate ex-situ GCs in the satellite frame, then in the host frame.

    Bound GCs use the prepared combined satellite-frame potential. Their DF
    and tidal mass loss use only the smooth internal satellite potential.
    Released GCs use ``host_orbit_potential_file`` for their orbits and the
    smooth ``host_density_potential_file`` for DF and tidal mass loss.

    A GC is released at the first sample with satellite-centric energy above
    ``release_energy_tolerance``. If it is still bound when the satellite
    history ends, it is automatically released at the final satellite state.
    """
    galaxy_id = int(galaxy_id)
    satellites_file = Path(satellites_file)
    initial_conditions_directory = Path(initial_conditions_directory)
    satellite_trajectory_file = Path(satellite_trajectory_file)
    satellite_potential_directory = Path(satellite_potential_directory)
    timestep_file = Path(timestep_file)
    host_orbit_potential_file = Path(host_orbit_potential_file)
    host_density_potential_file = Path(host_density_potential_file)
    output_file = Path(output_file)
    plot_file = Path(plot_file) if plot_file is not None else None

    object_type = str(object_type).strip().upper()
    if object_type not in {"GC", "NSC", "BH"}:
        raise ValueError("object_type must be 'GC', 'NSC', or 'BH'.")

    # A BH is a point mass. It never loses mass and never needs a moving
    # potential because no BH stream is generated.
    if object_type == "BH":
        gc_half_mass_radius = 0.0
        mass_loss_mode = "none"
        generate_gc_moving_potentials = False
        gc_potential_scale_radius = 0.0
    if initial_conditions_filename_template is None:
        suffix = "GCs" if object_type == "GC" else object_type
        initial_conditions_filename_template = (
            f"IniG{{galaxy_id}}Sat{{satellite_id}}{suffix}.txt"
        )

    if gc_potential_scale_radius is None:
        gc_potential_scale_radius = gc_half_mass_radius
    if gc_moving_potential_directory is None:
        gc_moving_potential_directory = output_file.parent / "GCPotential"
    else:
        gc_moving_potential_directory = Path(gc_moving_potential_directory)
    if generate_gc_moving_potentials and gc_potential_scale_radius <= 0.0:
        raise ValueError("gc_potential_scale_radius must be positive.")
    if generate_gc_moving_potentials:
        gc_moving_potential_directory.mkdir(parents=True, exist_ok=True)

    for path in (
        satellites_file, satellite_trajectory_file, timestep_file,
        host_orbit_potential_file, host_density_potential_file,
    ):
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
    if potential_mode not in ("evolving", "static"):
        raise ValueError("potential_mode must be 'evolving' or 'static'.")
    if df_model not in ("none", "cdm", "fdm"):
        raise ValueError("df_model must be 'none', 'cdm', or 'fdm'.")
    if mass_loss_mode not in ("none", "postprocess", "coupled"):
        raise ValueError("mass_loss_mode must be 'none', 'postprocess', or 'coupled'.")
    if central_capture_radius <= 0.0:
        raise ValueError("central_capture_radius must be positive.")
    if object_type == "BH":
        if gc_half_mass_radius != 0.0:
            raise ValueError("A BH must have gc_half_mass_radius=0.")
    elif gc_half_mass_radius <= 0.0:
        raise ValueError("Object size must be positive.")
    elif object_type == "GC" and gc_mass <= 0.0:
        raise ValueError("GC mass must be positive.")
    if df_model == "fdm" and m22 <= 0.0:
        raise ValueError("m22 must be positive for FDM friction.")

    timestep = np.atleast_2d(np.loadtxt(timestep_file))
    if end_snapshot_index is None:
        end_snapshot_index = len(timestep)

    with host_orbit_potential_file.open("rb") as stream:
        host_orbit_potentials = pickle.load(stream)
    with host_density_potential_file.open("rb") as stream:
        host_density_potentials = pickle.load(stream)

    satellites = _read_satellite_list(satellites_file)
    output = {}
    satellite_plot_tracks = {}
    total_clusters = 0
    gc_moving_potential_files = []
    all_initial_object_masses = []

    # Building a galpy dynamical-friction force is expensive because its
    # velocity-dispersion information is initialized from the density
    # potential.  Reuse one host-force template per snapshot for every GC
    # and every satellite.  In coupled mode only GMs is changed before an
    # individual orbit integration.
    host_df_force_cache = {}

    with h5py.File(satellite_trajectory_file, "r") as trajectory_h5:
        for satellite_id, listed_snapshots in satellites:
            ic_file = (
                initial_conditions_directory
                / initial_conditions_filename_template.format(
                    galaxy_id=galaxy_id,
                    satellite_id=satellite_id,
                )
            )
            group_name = f"Sat_{satellite_id}"
            if not ic_file.exists():
                print(
                    f"No {object_type} initial conditions for satellite "
                    f"{satellite_id}; skipping."
                )
                continue
            if group_name not in trajectory_h5:
                print(f"No prepared trajectory for satellite {satellite_id}; skipping.")
                continue

            combined_file = (
                satellite_potential_directory
                / f"CombinedPotentialG{galaxy_id}Sat{satellite_id}.pkl"
            )
            internal_file = (
                satellite_potential_directory
                / f"SatellitePotentialG{galaxy_id}Sat{satellite_id}.pkl"
            )
            if not combined_file.exists() or not internal_file.exists():
                print(f"Missing prepared potential(s) for satellite {satellite_id}; skipping.")
                continue

            with combined_file.open("rb") as stream:
                combined_potentials = pickle.load(stream)
            with internal_file.open("rb") as stream:
                internal_potentials = pickle.load(stream)

            initial_conditions = np.atleast_2d(np.loadtxt(ic_file))
            number_of_clusters = len(initial_conditions)
            if object_type in {"NSC", "BH"}:
                if initial_conditions.shape[1] < 7:
                    raise ValueError(
                        f"{object_type} initial conditions must contain a seventh "
                        f"mass column: {ic_file}"
                    )
                initial_masses = np.asarray(
                    initial_conditions[:, 6], dtype=float
                )
            else:
                initial_masses = np.full(
                    number_of_clusters, gc_mass, dtype=float
                )
            if np.any(~np.isfinite(initial_masses)) or np.any(
                initial_masses <= 0.0
            ):
                raise ValueError(
                    f"{object_type} masses must be finite and positive."
                )
            all_initial_object_masses.extend(initial_masses.tolist())
            total_clusters += number_of_clusters
            satellite_group = trajectory_h5[group_name]
            satellite_plot_tracks[satellite_id] = {
                "time": np.asarray(satellite_group["time"][:]),
                "radius": np.sqrt(
                    np.asarray(satellite_group["x"][:]) ** 2
                    + np.asarray(satellite_group["y"][:]) ** 2
                    + np.asarray(satellite_group["z"][:]) ** 2
                ),
            }
            prepared_snapshots = sorted(
                set(combined_potentials) & set(internal_potentials)
            )
            requested = [
                snapshot for snapshot in listed_snapshots
                if start_snapshot_index <= snapshot < end_snapshot_index
                and snapshot in prepared_snapshots
            ]
            if not requested:
                print(f"No usable snapshot for satellite {satellite_id}; skipping.")
                continue

            first_snapshot = requested[0]
            current = [
                _phase_space(*row[:6])
                for row in initial_conditions
            ]
            released = np.zeros(number_of_clusters, dtype=bool)
            captured = np.zeros(number_of_clusters, dtype=bool)
            release_time = np.full(number_of_clusters, np.nan)
            release_snapshot = np.full(number_of_clusters, -1, dtype=int)
            current_mass = initial_masses.copy()
            storage = [_empty_storage() for _ in range(number_of_clusters)]
            moving_potential_histories = [
                {} for _ in range(number_of_clusters)
            ]
            satellite_df_force_cache = {}

            print(
                f"Integrating G{galaxy_id} satellite {satellite_id}: "
                f"{number_of_clusters} {object_type}(s), snapshots "
                f"{first_snapshot}-{end_snapshot_index - 1}."
            )
            if object_type in {"NSC", "BH"}:
                print(
                    f"{object_type} mass(es) from initial conditions: "
                    + ", ".join(
                        f"{mass:.6e} Msun" for mass in initial_masses
                    )
                )

            for snapshot in range(first_snapshot, end_snapshot_index):
                times_numeric = np.linspace(
                    timestep[snapshot, 0],
                    timestep[snapshot, 1],
                    int(timestep[snapshot, 2]),
                )
                times = times_numeric * units.Gyr
                host_orbit = _potential_at(
                    host_orbit_potentials, snapshot,
                    potential_mode, static_potential_index,
                )
                host_density = _potential_at(
                    host_density_potentials, snapshot,
                    potential_mode, static_potential_index,
                )
                _set_non_dissipative(host_orbit)
                _set_non_dissipative(host_density)
                satellite_exists = snapshot in requested
                satellite_segment = (
                    _satellite_segment(satellite_group, snapshot, times_numeric)
                    if satellite_exists else None
                )

                if (
                    df_model != "none"
                    and np.any(released & ~captured)
                    and snapshot not in host_df_force_cache
                ):
                    host_df_force_cache[snapshot] = _build_df_force(
                        df_model,
                        host_density,
                        float(np.max(initial_masses)),
                        gc_half_mass_radius,
                        m22,
                    )

                if (
                    df_model != "none"
                    and satellite_exists
                    and np.any(~released & ~captured)
                    and snapshot not in satellite_df_force_cache
                ):
                    internal_density = internal_potentials[snapshot]
                    _set_non_dissipative(internal_density)
                    satellite_df_force_cache[snapshot] = _build_df_force(
                        df_model,
                        internal_density,
                        float(np.max(initial_masses)),
                        gc_half_mass_radius,
                        m22,
                    )

                for cluster_index in range(number_of_clusters):
                    if captured[cluster_index]:
                        if generate_gc_moving_potentials:
                            moving_potential_histories[cluster_index][snapshot] = {
                                "active": False,
                                "segments": [],
                                "inactive_reason": "captured",
                                "fallback": "external_only",
                                "gc_mass": float(current_mass[cluster_index]),
                            }
                        continue

                    cluster_times_numeric = times_numeric.copy()
                    mass_at_start = current_mass[cluster_index]
                    mass_for_df = (
                        mass_at_start
                        if mass_loss_mode == "coupled"
                        else initial_masses[cluster_index]
                    )
                    interval_phase = None
                    tidal_parts = []
                    potential_segments = []

                    if not released[cluster_index] and satellite_exists:
                        combined = combined_potentials[snapshot]
                        internal = internal_potentials[snapshot]
                        _set_non_dissipative(combined)
                        _set_non_dissipative(internal)
                        force = (
                            satellite_df_force_cache.get(snapshot)
                            if mass_for_df > 0.0
                            else None
                        )
                        if force is not None:
                            force.GMs = mass_for_df * units.Msun
                        relative_orbit = Orbit(current[cluster_index])
                        relative_orbit.integrate(
                            times, _combine(combined, force),
                            method=integration_method,
                        )
                        energy = _satellite_energy(relative_orbit, times, internal)
                        positive = np.where(energy > release_energy_tolerance)[0]
                        release_index = int(positive[0]) if positive.size else None

                        if release_index is None:
                            galactic = _relative_plus_satellite(
                                relative_orbit, times, satellite_segment
                            )
                            interval_phase = galactic
                            tidal_parts.append(
                                (
                                    0,
                                    len(times_numeric),
                                    internal,
                                    np.asarray(relative_orbit.x(times)),
                                    np.asarray(relative_orbit.y(times)),
                                    np.asarray(relative_orbit.z(times)),
                                )
                            )
                            current[cluster_index] = _orbit_endpoint(relative_orbit, times)
                            potential_segments.append(
                                _gc_moving_potential_entry(
                                    relative_orbit,
                                    internal,
                                    mass_at_start,
                                    gc_potential_scale_radius,
                                    0,
                                    len(times_numeric),
                                    "satellite_centered",
                                )
                            )

                            if snapshot == requested[-1]:
                                released[cluster_index] = True
                                release_time[cluster_index] = times_numeric[-1]
                                release_snapshot[cluster_index] = snapshot
                                current[cluster_index] = _phase_space(
                                    galactic["R"][-1], galactic["vR"][-1],
                                    galactic["vT"][-1], galactic["z"][-1],
                                    galactic["vz"][-1], galactic["phi"][-1],
                                )
                        else:
                            bound_count = release_index
                            galactic_bound = _relative_plus_satellite(
                                relative_orbit, times, satellite_segment
                            )
                            release_phase = _phase_space(
                                galactic_bound["R"][release_index],
                                galactic_bound["vR"][release_index],
                                galactic_bound["vT"][release_index],
                                galactic_bound["z"][release_index],
                                galactic_bound["vz"][release_index],
                                galactic_bound["phi"][release_index],
                            )
                            released[cluster_index] = True
                            release_time[cluster_index] = times_numeric[release_index]
                            release_snapshot[cluster_index] = snapshot
                            host_times = times[release_index:]
                            host_orbit_gc = Orbit(release_phase)
                            if (
                                df_model != "none"
                                and snapshot not in host_df_force_cache
                            ):
                                host_df_force_cache[snapshot] = _build_df_force(
                                    df_model,
                                    host_density,
                                    initial_masses[cluster_index],
                                    gc_half_mass_radius,
                                    m22,
                                )
                            host_force = (
                                host_df_force_cache.get(snapshot)
                                if mass_for_df > 0.0
                                else None
                            )
                            if host_force is not None:
                                host_force.GMs = mass_for_df * units.Msun
                            host_orbit_gc.integrate(
                                host_times, _combine(host_orbit, host_force),
                                method=integration_method,
                            )
                            host_phase = {
                                "x": np.asarray(host_orbit_gc.x(host_times)),
                                "y": np.asarray(host_orbit_gc.y(host_times)),
                                "z": np.asarray(host_orbit_gc.z(host_times)),
                                "vR": np.asarray(host_orbit_gc.vR(host_times)),
                                "vT": np.asarray(host_orbit_gc.vT(host_times)),
                                "vz": np.asarray(host_orbit_gc.vz(host_times)),
                            }
                            interval_phase = {
                                key: np.concatenate(
                                    (galactic_bound[key][:bound_count], host_phase[key])
                                )
                                for key in ("x", "y", "z", "vR", "vT", "vz")
                            }
                            tidal_parts.extend(
                                [
                                    (
                                        0,
                                        bound_count,
                                        internal,
                                        np.asarray(relative_orbit.x(times))[
                                            :bound_count
                                        ],
                                        np.asarray(relative_orbit.y(times))[
                                            :bound_count
                                        ],
                                        np.asarray(relative_orbit.z(times))[
                                            :bound_count
                                        ],
                                    ),
                                    (
                                        bound_count,
                                        len(times_numeric),
                                        host_density,
                                        host_phase["x"],
                                        host_phase["y"],
                                        host_phase["z"],
                                    ),
                                ]
                            )
                            current[cluster_index] = _orbit_endpoint(host_orbit_gc, host_times)
                            if bound_count > 0:
                                potential_segments.append(
                                    _gc_moving_potential_entry(
                                        relative_orbit,
                                        internal,
                                        mass_at_start,
                                        gc_potential_scale_radius,
                                        0,
                                        bound_count,
                                        "satellite_centered",
                                    )
                                )
                            potential_segments.append(
                                _gc_moving_potential_entry(
                                    host_orbit_gc,
                                    host_density,
                                    mass_at_start,
                                    gc_potential_scale_radius,
                                    bound_count,
                                    len(times_numeric),
                                    "galactocentric",
                                )
                            )
                    else:
                        # A satellite that ended without an earlier positive
                        # energy releases its GCs from the saved endpoint.
                        if not released[cluster_index]:
                            released[cluster_index] = True
                            release_time[cluster_index] = times_numeric[0]
                            release_snapshot[cluster_index] = snapshot
                        host_force = (
                            host_df_force_cache.get(snapshot)
                            if mass_for_df > 0.0
                            else None
                        )
                        if host_force is not None:
                            host_force.GMs = mass_for_df * units.Msun
                        host_orbit_gc = Orbit(current[cluster_index])
                        host_orbit_gc.integrate(
                            times, _combine(host_orbit, host_force),
                            method=integration_method,
                        )
                        interval_phase = {
                            "x": np.asarray(host_orbit_gc.x(times)),
                            "y": np.asarray(host_orbit_gc.y(times)),
                            "z": np.asarray(host_orbit_gc.z(times)),
                            "vR": np.asarray(host_orbit_gc.vR(times)),
                            "vT": np.asarray(host_orbit_gc.vT(times)),
                            "vz": np.asarray(host_orbit_gc.vz(times)),
                        }
                        tidal_parts.append(
                            (
                                0,
                                len(times_numeric),
                                host_density,
                                interval_phase["x"],
                                interval_phase["y"],
                                interval_phase["z"],
                            )
                        )
                        current[cluster_index] = _orbit_endpoint(host_orbit_gc, times)
                        potential_segments.append(
                            _gc_moving_potential_entry(
                                host_orbit_gc,
                                host_density,
                                mass_at_start,
                                gc_potential_scale_radius,
                                0,
                                len(times_numeric),
                                "galactocentric",
                            )
                        )

                    radius = np.sqrt(
                        interval_phase["x"] ** 2
                        + interval_phase["y"] ** 2
                        + interval_phase["z"] ** 2
                    )
                    capture_indices = np.where(radius <= central_capture_radius)[0]
                    if capture_indices.size:
                        stop = int(capture_indices[0]) + 1
                        captured[cluster_index] = True
                        for key in interval_phase:
                            interval_phase[key] = interval_phase[key][:stop]
                        cluster_times_numeric = cluster_times_numeric[:stop]

                    if mass_loss_mode == "none":
                        mass_history = np.full(
                            len(cluster_times_numeric), mass_at_start
                        )
                    else:
                        tidal = np.zeros(len(cluster_times_numeric))
                        for begin, end, density, tx, ty, tz in tidal_parts:
                            clipped_end = min(end, len(cluster_times_numeric))
                            sample_count = clipped_end - begin
                            if sample_count > 0:
                                tidal[begin:clipped_end] = _tidal_strength(
                                    density,
                                    np.asarray(tx)[:sample_count],
                                    np.asarray(ty)[:sample_count],
                                    np.asarray(tz)[:sample_count],
                                )
                        mass_history = _integrate_mass(
                            mass_at_start, cluster_times_numeric, tidal,
                            mass_loss_gamma, tidal_strength_reference,
                            dissolution_time_normalization,
                        )
                        current_mass[cluster_index] = mass_history[-1]

                    is_bound = (
                        np.asarray(cluster_times_numeric)
                        < release_time[cluster_index]
                        if released[cluster_index]
                        else np.ones(len(cluster_times_numeric), dtype=bool)
                    )
                    _append_segment(
                        storage[cluster_index], cluster_times_numeric,
                        interval_phase, mass_history, snapshot, is_bound,
                    )

                    if generate_gc_moving_potentials:
                        if captured[cluster_index]:
                            for segment in potential_segments:
                                segment["end_index"] = min(
                                    segment["end_index"],
                                    len(cluster_times_numeric),
                                )
                            potential_segments = [
                                segment
                                for segment in potential_segments
                                if segment["end_index"] > segment["start_index"]
                            ]

                        if len(potential_segments) == 1:
                            entry = dict(potential_segments[0])
                            entry["segments"] = potential_segments
                        else:
                            entry = {
                                "active": any(
                                    segment["active"]
                                    for segment in potential_segments
                                ),
                                "gc_potential": None,
                                "moving_potential": None,
                                "segments": potential_segments,
                                "gc_mass": float(mass_at_start),
                                "fallback": None,
                                "inactive_reason": None,
                            }
                        moving_potential_histories[cluster_index][snapshot] = entry

            output[satellite_id] = {
                "storage": storage,
                "initial_masses": initial_masses,
                "released": released,
                "captured": captured,
                "release_time": release_time,
                "release_snapshot": release_snapshot,
            }

            if generate_gc_moving_potentials:
                mode_tag = f"{potential_mode}_{df_model}_{mass_loss_mode}"
                for cluster_index, history in enumerate(
                    moving_potential_histories
                ):
                    potential_path = gc_moving_potential_directory / (
                        f"GCpotExSitu_{mode_tag}_G{galaxy_id}_"
                        f"Sat{satellite_id}_GC{cluster_index}.pkl"
                    )
                    with potential_path.open("wb") as stream:
                        pickle.dump(
                            history,
                            stream,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                    gc_moving_potential_files.append(str(potential_path))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_file, "w") as h5:
        h5.attrs["galaxy_id"] = galaxy_id
        h5.attrs["potential_mode"] = potential_mode
        h5.attrs["df_model"] = df_model
        h5.attrs["mass_loss_mode"] = mass_loss_mode
        h5.attrs["object_type"] = object_type
        h5.attrs["initial_object_mass_msun"] = (
            np.asarray(all_initial_object_masses, dtype=float)
            if object_type in {"NSC", "BH"}
            else gc_mass
        )
        h5.attrs["object_half_mass_radius_kpc"] = gc_half_mass_radius
        h5.attrs["host_orbit_potential_file"] = str(host_orbit_potential_file)
        h5.attrs["host_density_potential_file"] = str(host_density_potential_file)
        h5.attrs["release_energy_tolerance"] = release_energy_tolerance
        h5.attrs["initial_gc_mass_msun"] = (
            np.asarray(all_initial_object_masses, dtype=float)
            if object_type in {"NSC", "BH"}
            else gc_mass
        )
        h5.attrs["gc_half_mass_radius_kpc"] = gc_half_mass_radius
        h5.attrs["generate_gc_moving_potentials"] = (
            generate_gc_moving_potentials
        )
        h5.attrs["gc_moving_potential_directory"] = str(
            gc_moving_potential_directory
        )
        h5.attrs["gc_potential_scale_radius_kpc"] = (
            gc_potential_scale_radius
        )
        h5.attrs["gc_moving_potential_convention"] = (
            "gc_potential_plus_moving_external_potential"
        )
        for satellite_id, result in output.items():
            satellite_group = h5.create_group(f"Sat_{satellite_id}")
            satellite_group.create_dataset("Released", data=result["released"].astype(np.uint8))
            satellite_group.create_dataset("Captured", data=result["captured"].astype(np.uint8))
            satellite_group.create_dataset("ReleaseTime", data=result["release_time"])
            satellite_group.create_dataset("ReleaseSnapshot", data=result["release_snapshot"])
            satellite_group.create_dataset(
                "InitialMass", data=result["initial_masses"]
            )
            for cluster_index, storage in enumerate(result["storage"]):
                gc_group = satellite_group.create_group(
                    f"{object_type}_{cluster_index}"
                )
                for key, values in storage.items():
                    data = np.concatenate(values) if values else np.array([])
                    gc_group.create_dataset(key, data=data)

    if plot_file is not None:
        plot_file.parent.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(10, 8))
        gc_label_added = False

        for satellite_id, result in output.items():
            for cluster_index, storage in enumerate(result["storage"]):
                if not storage["time"]:
                    continue
                time = np.concatenate(storage["time"])
                radius = np.sqrt(
                    np.concatenate(storage["x"]) ** 2
                    + np.concatenate(storage["y"]) ** 2
                    + np.concatenate(storage["z"]) ** 2
                )
                axis.plot(
                    time,
                    radius,
                    color="black",
                    linewidth=0.55,
                    alpha=0.7,
                    label=(
                        f"Ex-situ {object_type}s"
                        if not gc_label_added
                        else None
                    ),
                )
                gc_label_added = True

        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for color_index, (satellite_id, track) in enumerate(
            satellite_plot_tracks.items()
        ):
            axis.plot(
                track["time"],
                track["radius"],
                color=color_cycle[color_index % len(color_cycle)],
                linewidth=2.0,
                alpha=0.95,
                label=f"Satellite {satellite_id}",
                zorder=3,
            )

        axis.set_xlabel(
            r"Time $t \; [\mathrm{Gyr}]$", fontsize=20, fontweight="bold"
        )
        axis.set_ylabel(
            r"Galactocentric distance $r \;[\mathrm{kpc}]$",
            fontsize=20, fontweight="bold",
        )
        axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
        axis.set_facecolor("whitesmoke")
        if output or satellite_plot_tracks:
            axis.legend()
        figure.tight_layout()
        figure.savefig(plot_file, dpi=200)
        plt.close(figure)

    print(f"Ex-situ dynamics saved to: {output_file}")
    if generate_gc_moving_potentials:
        print(
            "Ex-situ GC moving potentials saved to: "
            f"{gc_moving_potential_directory}"
        )
    if plot_file is not None:
        print(f"Ex-situ radius-evolution plot saved to: {plot_file}")
    return {
        "output_file": output_file,
        "plot_file": plot_file,
        "number_of_clusters": total_clusters,
        "number_of_objects": total_clusters,
        "object_type": object_type,
        "initial_object_masses": np.asarray(
            all_initial_object_masses, dtype=float
        ),
        "gc_moving_potential_files": gc_moving_potential_files,
    }
