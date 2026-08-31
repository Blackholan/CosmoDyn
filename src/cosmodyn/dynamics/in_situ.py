#!/usr/bin/env python3
# coding: utf-8

"""In-situ GC orbit integration with CDM/FDM friction and tidal mass loss."""

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
    ttensor,
)
from scipy.integrate import solve_ivp


def _format_float_for_filename(value):
    """Return a compact, file-name-safe representation of a float."""
    return f"{value:.6g}".replace("+", "").replace(".", "p")


def _set_non_dissipative(potential):
    """Explicitly set isDissipative=False on all potential components."""
    components = potential if isinstance(potential, (list, tuple)) else [potential]
    for component in components:
        component.isDissipative = False



def _last_apocenter(radius_history):
    """
    Return the last local maximum of a radial orbit history.

    If no local maximum is found within the snapshot interval, the
    final radius is returned as a fallback.
    """
    radius_history = np.asarray(radius_history, dtype=float)

    if radius_history.size == 0:
        return np.nan

    if radius_history.size < 3:
        return float(radius_history[-1])

    local_maxima = np.where(
        (radius_history[1:-1] > radius_history[:-2])
        & (radius_history[1:-1] >= radius_history[2:])
    )[0] + 1

    if local_maxima.size == 0:
        return float(radius_history[-1])

    return float(radius_history[local_maxima[-1]])

def _combine_potential_and_force(potential, force):
    """Return a galpy-compatible list containing a potential and one force."""
    if force is None:
        return potential
    if isinstance(potential, list):
        return potential + [force]
    if isinstance(potential, tuple):
        return list(potential) + [force]
    return [potential, force]


def _df_cache_filename(
    potential_mode,
    df_model,
    gc_mass,
    gc_half_mass_radius,
    start_snapshot_index,
    end_snapshot_index,
    static_potential_index,
    m22,
):
    """Build a unique file name for cached dynamical-friction forces."""
    mass_tag = _format_float_for_filename(gc_mass)
    radius_tag = _format_float_for_filename(gc_half_mass_radius)

    if potential_mode == "evolving":
        mode_tag = f"evolving_S{start_snapshot_index}-{end_snapshot_index - 1}"
    else:
        mode_tag = f"static_P{static_potential_index}"

    if df_model == "cdm":
        return f"DF_CDM_{mode_tag}_M{mass_tag}Msun_R{radius_tag}kpc.pkl"

    if df_model == "fdm":
        m22_tag = _format_float_for_filename(m22)
        return (
            f"DF_FDM_{mode_tag}_m22_{m22_tag}"
            f"_M{mass_tag}Msun_R{radius_tag}kpc.pkl"
        )

    return None


def _build_df_force(df_model, potential, gc_mass, gc_half_mass_radius, m22):
    """Create one CDM or FDM dynamical-friction force."""
    if df_model == "cdm":
        return ChandrasekharDynamicalFrictionForce(
            GMs=gc_mass * units.Msun,
            rhm=gc_half_mass_radius * units.kpc,
            dens=potential,
        )

    if df_model == "fdm":
        return FDMDynamicalFrictionForce(
            GMs=gc_mass * units.Msun,
            rhm=gc_half_mass_radius * units.kpc,
            dens=potential,
            m=m22 * 1.0e-22 * units.eV,
        )

    return None



def _build_gc_moving_potential(
    orbit,
    host_potential,
    gc_mass,
    gc_scale_radius,
    active=True,
    inactive_reason=None,
):
    """
    Build the GC potential information used by the stream calculation.

    This intentionally follows the construction of the legacy
    ``10MWGCsInStream.py`` script:

    ``[GC Plummer potential, MovingObjectPotential(orbit, pot=host_potential)]``

    An entry is created for every snapshot. When the GC should no longer
    contribute a moving potential (for example after capture or after its
    mass reaches zero), ``moving_potential`` is set to None and ``active``
    is False. This allows the future stream integrator to switch explicitly
    to the host-galaxy potential only.

    Parameters
    ----------
    orbit : galpy.orbit.Orbit or None
        Integrated GC orbit over the current snapshot interval.
    host_potential : galpy potential or list
        Galactic potential of the current snapshot.
    gc_mass : float
        GC mass in Msun for the current snapshot interval.
    gc_scale_radius : float
        Plummer scale radius in kpc.
    active : bool, optional
        Whether stream particles should still feel the GC moving potential.
    inactive_reason : str, optional
        Reason why the moving potential is no longer active.

    Returns
    -------
    dict
        Snapshot-level stream-potential information.
    """
    if (not active) or orbit is None or gc_mass <= 0.0:
        return {
            "active": False,
            "gc_potential": None,
            "moving_potential": None,
            "fallback": "host_only",
            "inactive_reason": inactive_reason,
            "gc_mass": max(float(gc_mass), 0.0),
        }

    gc_potential = PlummerPotential(
        amp=gc_mass * units.Msun,
        b=gc_scale_radius * units.kpc,
    )

    # Keep the exact construction used in 10MWGCsInStream.py.
    moving_potential = MovingObjectPotential(
        orbit,
        pot=host_potential,
    )

    return {
        "active": True,
        "gc_potential": gc_potential,
        "moving_potential": moving_potential,
        "fallback": None,
        "inactive_reason": None,
        "gc_mass": float(gc_mass),
    }


def _mass_loss_derivative(
    time,
    mass,
    gamma,
    tidal_strength_values,
    time_values,
    tidal_strength_reference,
    dissolution_time_normalization,
):
    """Return dM/dt for the adopted tidal-disruption prescription."""
    tidal_strength = np.interp(
        time,
        time_values,
        tidal_strength_values,
    )
    tidal_strength = max(tidal_strength, np.finfo(float).tiny)
    tidal_factor = (tidal_strength / tidal_strength_reference) ** (-0.5)

    current_mass = max(float(mass[0]), 0.0)
    if current_mass == 0.0:
        return [0.0]

    return [
        -(current_mass ** (1.0 - gamma))
        / (dissolution_time_normalization * tidal_factor)
    ]


def _compute_tidal_strength_history(potential, x, y, z):
    """Compute the tidal-strength history along one or several GC orbits."""
    cylindrical_radius = np.sqrt(x**2 + y**2)
    number_of_clusters, number_of_times = cylindrical_radius.shape
    tidal_strength = np.zeros((number_of_clusters, number_of_times))

    _set_non_dissipative(potential)

    for cluster_index in range(number_of_clusters):
        for time_index in range(number_of_times):
            eigenvalues = ttensor(
                potential,
                cylindrical_radius[cluster_index, time_index] * units.kpc,
                z[cluster_index, time_index] * units.kpc,
                phi=0.0,
                t=0.0,
                eigenval=True,
            )
            tidal_strength[cluster_index, time_index] = (
                np.max(eigenvalues)
                - 0.5 * (eigenvalues[1] + eigenvalues[2])
            )

    return tidal_strength


def _integrate_mass_loss_interval(
    initial_masses,
    times,
    tidal_strength,
    gamma,
    tidal_strength_reference,
    dissolution_time_normalization,
):
    """Integrate the GC masses over one snapshot interval."""
    number_of_clusters = len(initial_masses)
    mass_history = np.zeros((number_of_clusters, len(times)))

    for cluster_index in range(number_of_clusters):
        initial_mass = max(float(initial_masses[cluster_index]), 0.0)

        if initial_mass == 0.0 or len(times) < 2 or times[-1] <= times[0]:
            mass_history[cluster_index, :] = initial_mass
            continue

        solution = solve_ivp(
            _mass_loss_derivative,
            (times[0], times[-1]),
            [initial_mass],
            args=(
                gamma,
                tidal_strength[cluster_index],
                times,
                tidal_strength_reference,
                dissolution_time_normalization,
            ),
            t_eval=times,
            atol=1.0e-8,
            rtol=1.0e-6,
        )

        if not solution.success or solution.y.shape[1] != len(times):
            raise RuntimeError(
                "Mass-loss integration failed for GC "
                f"{cluster_index}: {solution.message}"
            )

        mass_history[cluster_index] = np.maximum(solution.y[0], 0.0)

    return mass_history


def run_in_situ_dynamics(
    initial_conditions_file,
    timestep_file,
    potential_file,
    output_file,
    plot_file=None,
    density_potential_file=None,
    start_snapshot_index=8,
    end_snapshot_index=None,
    integration_method="dop853_c",
    potential_mode="evolving",
    df_model="none",
    static_potential_index=73,
    gc_mass=1.0e6,
    gc_half_mass_radius=0.01,
    m22=1.0,
    df_cache_directory=None,
    reuse_df_cache=True,
    mass_loss_mode="none",
    mass_loss_gamma=0.7,
    tidal_strength_reference=7.01e2,
    dissolution_time_normalization=0.0107,
    central_capture_radius=0.01,
    generate_gc_moving_potentials=False,
    gc_moving_potential_directory=None,
    gc_potential_scale_radius=None,
    object_type="GC",
):
    """
    Integrate in-situ GC orbits.

    Parameters
    ----------
    potential_mode : {"evolving", "static"}
        Use the time-dependent potential or one fixed snapshot potential.
    density_potential_file : path-like or None
        Smooth host-galaxy potential used for dynamical friction and tidal
        mass loss. If None, ``potential_file`` is used. Set this to the
        host-only MW potential when ``potential_file`` contains moving
        satellites.
    df_model : {"none", "cdm", "fdm"}
        Dynamical-friction prescription.
    m22 : float
        FDM particle mass in units of 1e-22 eV.
    mass_loss_mode : {"none", "postprocess", "coupled"}
        ``none`` disables mass loss. ``postprocess`` computes the full mass
        history after the orbit integration and does not feed it back into DF.
        ``coupled`` computes mass loss after every snapshot and updates the DF
        force through ``force.GMs`` before the next orbit integration.
    generate_gc_moving_potentials : bool, optional
        If True, construct and save the moving Plummer potential of every GC
        along its integrated orbit for each snapshot interval.
    gc_moving_potential_directory : str or pathlib.Path, optional
        Directory used to save the per-GC moving-potential pickle files.
        The default is ``<output directory>/GCPotential``.
    gc_potential_scale_radius : float, optional
        Plummer scale radius of the GC potential in kpc. If None,
        ``gc_half_mass_radius`` is used, reproducing the scale adopted in
        the previous stream-generation script.
    """
    initial_conditions_file = Path(initial_conditions_file)
    timestep_file = Path(timestep_file)
    orbit_potential_file = Path(potential_file)
    if density_potential_file is None:
        density_potential_file = orbit_potential_file
    else:
        density_potential_file = Path(density_potential_file)
    output_file = Path(output_file)
    object_type = str(object_type).strip().upper()
    if object_type not in {"GC", "NSC", "BH"}:
        raise ValueError("object_type must be 'GC', 'NSC', or 'BH'.")

    # Black holes are point masses: these properties are fixed internally
    # and cannot be overridden by launcher parameters.
    if object_type == "BH":
        gc_half_mass_radius = 0.0
        mass_loss_mode = "none"
        generate_gc_moving_potentials = False
        gc_potential_scale_radius = 0.0

    if df_cache_directory is None:
        df_cache_directory = output_file.parent / "DynamicalFriction"
    else:
        df_cache_directory = Path(df_cache_directory)

    if gc_potential_scale_radius is None:
        gc_potential_scale_radius = gc_half_mass_radius

    if generate_gc_moving_potentials and gc_potential_scale_radius <= 0:
        raise ValueError(
            "gc_potential_scale_radius must be strictly positive."
        )

    if gc_moving_potential_directory is None:
        gc_moving_potential_directory = (
            output_file.parent / "GCPotential"
        )
    else:
        gc_moving_potential_directory = Path(
            gc_moving_potential_directory
        )

    if generate_gc_moving_potentials:
        gc_moving_potential_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    for path in (
        initial_conditions_file,
        timestep_file,
        orbit_potential_file,
        density_potential_file,
    ):
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    if potential_mode not in ("evolving", "static"):
        raise ValueError("potential_mode must be 'evolving' or 'static'.")
    if df_model not in ("none", "cdm", "fdm"):
        raise ValueError("df_model must be 'none', 'cdm', or 'fdm'.")
    if mass_loss_mode not in ("none", "postprocess", "coupled"):
        raise ValueError(
            "mass_loss_mode must be 'none', 'postprocess', or 'coupled'."
        )
    if central_capture_radius <= 0:
        raise ValueError(
            "central_capture_radius must be strictly positive."
        )
    if object_type == "BH":
        if gc_half_mass_radius != 0.0:
            raise ValueError("A BH must have gc_half_mass_radius=0.")
    elif gc_half_mass_radius <= 0:
        raise ValueError("gc_half_mass_radius must be strictly positive.")
    if df_model == "fdm" and m22 <= 0:
        raise ValueError("m22 must be strictly positive.")

    timestep_data = np.atleast_2d(np.loadtxt(timestep_file))
    time_start = timestep_data[:, 0]
    time_end = timestep_data[:, 1]
    number_of_steps = timestep_data[:, 2].astype(int)

    initial_conditions = np.atleast_2d(np.loadtxt(initial_conditions_file))
    number_of_clusters = len(initial_conditions)

    if object_type in {"NSC", "BH"}:
        if initial_conditions.shape[1] < 7:
            raise ValueError(
                f"{object_type} initial conditions must contain a seventh "
                "mass column."
            )
        if number_of_clusters != 1:
            raise ValueError(
                f"Exactly one in-situ {object_type} is expected."
            )
        gc_mass = float(initial_conditions[0, 6])

    if not np.isfinite(gc_mass) or gc_mass <= 0.0:
        raise ValueError(f"{object_type} mass must be finite and positive.")

    if object_type in {"NSC", "BH"}:
        print(
            f"Using in-situ {object_type} mass from initial conditions: "
            f"{gc_mass:.6e} Msun."
        )

    current_phase_space = [
        [
            row[0] * units.kpc,
            row[1] * units.km / units.s,
            row[2] * units.km / units.s,
            row[3] * units.kpc,
            row[4] * units.km / units.s,
            row[5] * 180.0 / np.pi * units.deg,
        ]
        for row in initial_conditions
    ]

    with orbit_potential_file.open("rb") as stream:
        potentials = pickle.load(stream)

    if density_potential_file == orbit_potential_file:
        density_potentials = potentials
    else:
        with density_potential_file.open("rb") as stream:
            density_potentials = pickle.load(stream)

    if end_snapshot_index is None:
        if isinstance(potentials, dict):
            if not potentials:
                raise ValueError("The orbit-potential dictionary is empty.")
            potential_end_index = max(potentials) + 1
        else:
            potential_end_index = len(timestep_data)
        end_snapshot_index = min(
            potential_end_index,
            len(timestep_data),
        )
    if not 0 <= start_snapshot_index < end_snapshot_index:
        raise ValueError(
            "start_snapshot_index must be smaller than end_snapshot_index."
        )

    static_potential = None
    static_density_potential = None

    if potential_mode == "static":

        # Potentials indexed by snapshot, as in the TNG50 pipeline.
        if isinstance(potentials, dict):
            if static_potential_index not in potentials:
                raise KeyError(
                    f"static_potential_index="
                    f"{static_potential_index} is not available. "
                    f"Available indices: {list(potentials.keys())}"
                )

            static_potential = potentials[
                static_potential_index
            ]

        # A list or tuple directly represents several components
        # of the same static potential.
        elif isinstance(potentials, (list, tuple)):
            static_potential = potentials

        # A single static galpy potential.
        else:
            static_potential = potentials

        _set_non_dissipative(static_potential)

        if isinstance(density_potentials, dict):
            if static_potential_index not in density_potentials:
                raise KeyError(
                    f"static_potential_index={static_potential_index} "
                    "is not available in the density potentials."
                )
            static_density_potential = density_potentials[
                static_potential_index
            ]
        else:
            static_density_potential = density_potentials

        _set_non_dissipative(static_density_potential)

    potential_description = (
        f"static potential from index {static_potential_index}"
        if potential_mode == "static"
        else "time-evolving potential"
    )
    print(
        f"Using the {potential_description}; DF model={df_model}; "
        f"mass-loss mode={mass_loss_mode}."
    )

    # Load or build one reusable DF-force template per potential.
    dynamical_friction_forces = {}
    static_dynamical_friction_force = None
    df_cache_file = None

    if df_model != "none":
        df_cache_directory.mkdir(parents=True, exist_ok=True)
        df_cache_file = df_cache_directory / _df_cache_filename(
            potential_mode,
            df_model,
            gc_mass,
            gc_half_mass_radius,
            start_snapshot_index,
            end_snapshot_index,
            static_potential_index,
            m22,
        )

        if reuse_df_cache and df_cache_file.exists():
            print(f"Loading cached dynamical-friction force(s): {df_cache_file}")
            with df_cache_file.open("rb") as stream:
                cached_df = pickle.load(stream)
            if potential_mode == "evolving":
                dynamical_friction_forces = cached_df
            else:
                static_dynamical_friction_force = cached_df

        elif potential_mode == "evolving":
            print(f"Computing one {df_model.upper()} DF force per snapshot.")
            for snapshot_index in range(start_snapshot_index, end_snapshot_index):
                density_potential = density_potentials[snapshot_index]
                _set_non_dissipative(density_potential)
                dynamical_friction_forces[snapshot_index] = _build_df_force(
                    df_model,
                    density_potential,
                    gc_mass,
                    gc_half_mass_radius,
                    m22,
                )
            with df_cache_file.open("wb") as stream:
                pickle.dump(
                    dynamical_friction_forces,
                    stream,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            print(f"Saved DF forces to: {df_cache_file}")

        else:
            print(f"Computing one static {df_model.upper()} DF force.")
            static_dynamical_friction_force = _build_df_force(
                df_model,
                static_density_potential,
                gc_mass,
                gc_half_mass_radius,
                m22,
            )
            with df_cache_file.open("wb") as stream:
                pickle.dump(
                    static_dynamical_friction_force,
                    stream,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            print(f"Saved the static DF force to: {df_cache_file}")

    current_masses = np.full(number_of_clusters, gc_mass, dtype=float)
    captured = np.zeros(number_of_clusters, dtype=bool)
    captured_x = np.zeros(number_of_clusters, dtype=float)
    captured_y = np.zeros(number_of_clusters, dtype=float)
    captured_z = np.zeros(number_of_clusters, dtype=float)
    last_apocenter_values = np.full(
        number_of_clusters,
        np.nan,
        dtype=float,
    )

    times_all = []
    x_all = []
    y_all = []
    z_all = []
    vR_all = []
    vT_all = []
    vz_all = []
    mass_all = []
    density_interval_potentials = []

    gc_moving_potentials = (
        [dict() for _ in range(number_of_clusters)]
        if generate_gc_moving_potentials
        else None
    )

    for snapshot_index in range(start_snapshot_index, end_snapshot_index):
        print(
            f"Integrating snapshot interval "
            f"{snapshot_index}/{end_snapshot_index - 1}"
        )

        times_numeric = np.linspace(
            time_start[snapshot_index],
            time_end[snapshot_index],
            number_of_steps[snapshot_index],
        )
        times = times_numeric * units.Gyr

        potential = (
            static_potential
            if potential_mode == "static"
            else potentials[snapshot_index]
        )
        _set_non_dissipative(potential)

        density_potential = (
            static_density_potential
            if potential_mode == "static"
            else density_potentials[snapshot_index]
        )
        _set_non_dissipative(density_potential)
        density_interval_potentials.append(density_potential)

        if df_model == "none":
            force_template = None
        elif potential_mode == "evolving":
            force_template = dynamical_friction_forces[snapshot_index]
        else:
            force_template = static_dynamical_friction_force

        if mass_loss_mode == "coupled" and force_template is not None:
            # Each GC now has its own mass, so orbits must be integrated
            # individually. The expensive force object is reused; only GMs is
            # updated before each integration.
            x = np.zeros((number_of_clusters, len(times_numeric)))
            y = np.zeros_like(x)
            z = np.zeros_like(x)
            R = np.zeros_like(x)
            phi = np.zeros_like(x)
            vR = np.zeros_like(x)
            vT = np.zeros_like(x)
            vz = np.zeros_like(x)

            for cluster_index in range(number_of_clusters):
                if captured[cluster_index]:
                    # Keep an explicit inactive stream-potential entry.
                    # The future stream integrator can then switch to the
                    # host-galaxy potential only for this snapshot.
                    if generate_gc_moving_potentials:
                        gc_moving_potentials[cluster_index][
                            snapshot_index
                        ] = _build_gc_moving_potential(
                            orbit=None,
                            host_potential=density_potential,
                            gc_mass=current_masses[cluster_index],
                            gc_scale_radius=gc_potential_scale_radius,
                            active=False,
                            inactive_reason="captured",
                        )

                    # Keep the GC fixed at its last integrated position
                    # to avoid an artificial jump in the trajectory plot.
                    x[cluster_index, :] = captured_x[cluster_index]
                    y[cluster_index, :] = captured_y[cluster_index]
                    z[cluster_index, :] = captured_z[cluster_index]

                    R[cluster_index, :] = np.sqrt(
                        captured_x[cluster_index] ** 2
                        + captured_y[cluster_index] ** 2
                    )
                    phi[cluster_index, :] = np.arctan2(
                        captured_y[cluster_index],
                        captured_x[cluster_index],
                    )

                    vR[cluster_index, :] = 0.0
                    vT[cluster_index, :] = 0.0
                    vz[cluster_index, :] = 0.0
                    continue

                current_mass = current_masses[cluster_index]

                # A GC that reached zero mass during the previous snapshot
                # is integrated without dynamical friction from this
                # snapshot onward.
                if current_mass > 0.0:
                    force_template.GMs = current_mass * units.Msun
                    integration_potential = _combine_potential_and_force(
                        potential,
                        force_template,
                    )
                else:
                    integration_potential = potential

                orbit = Orbit(current_phase_space[cluster_index])
                orbit.integrate(
                    times,
                    integration_potential,
                    method=integration_method,
                )

                if generate_gc_moving_potentials:
                    # In coupled mode, use the GC mass available at the
                    # beginning of this snapshot interval. If the mass is
                    # already zero, keep an explicit inactive entry so that
                    # stream particles use the host potential only.
                    gc_moving_potentials[cluster_index][
                        snapshot_index
                    ] = _build_gc_moving_potential(
                        orbit=orbit if current_mass > 0.0 else None,
                        host_potential=density_potential,
                        gc_mass=current_mass,
                        gc_scale_radius=gc_potential_scale_radius,
                        active=(current_mass > 0.0),
                        inactive_reason=(
                            None
                            if current_mass > 0.0
                            else "zero_mass"
                        ),
                    )

                x[cluster_index] = np.asarray(orbit.x(times))
                y[cluster_index] = np.asarray(orbit.y(times))
                z[cluster_index] = np.asarray(orbit.z(times))
                R[cluster_index] = np.asarray(orbit.R(times))
                phi[cluster_index] = np.asarray(orbit.phi(times))
                vR[cluster_index] = np.asarray(orbit.vR(times))
                vT[cluster_index] = np.asarray(orbit.vT(times))
                vz[cluster_index] = np.asarray(orbit.vz(times))
        else:
            integration_potential = _combine_potential_and_force(
                potential,
                force_template,
            )

            if np.any(captured):
                # Captured GCs must not be integrated again. Keep them fixed
                # at their last integrated positions and integrate only the
                # non-captured population.
                x = np.zeros((number_of_clusters, len(times_numeric)))
                y = np.zeros_like(x)
                z = np.zeros_like(x)
                R = np.zeros_like(x)
                phi = np.zeros_like(x)
                vR = np.zeros_like(x)
                vT = np.zeros_like(x)
                vz = np.zeros_like(x)

                captured_indices = np.where(captured)[0]
                active_indices = np.where(~captured)[0]

                for cluster_index in captured_indices:
                    if generate_gc_moving_potentials:
                        gc_moving_potentials[cluster_index][
                            snapshot_index
                        ] = _build_gc_moving_potential(
                            orbit=None,
                            host_potential=density_potential,
                            gc_mass=current_masses[cluster_index],
                            gc_scale_radius=gc_potential_scale_radius,
                            active=False,
                            inactive_reason="captured",
                        )

                    x[cluster_index, :] = captured_x[cluster_index]
                    y[cluster_index, :] = captured_y[cluster_index]
                    z[cluster_index, :] = captured_z[cluster_index]

                    R[cluster_index, :] = np.sqrt(
                        captured_x[cluster_index] ** 2
                        + captured_y[cluster_index] ** 2
                    )
                    phi[cluster_index, :] = np.arctan2(
                        captured_y[cluster_index],
                        captured_x[cluster_index],
                    )

                    vR[cluster_index, :] = 0.0
                    vT[cluster_index, :] = 0.0
                    vz[cluster_index, :] = 0.0

                if active_indices.size > 0:
                    active_phase_space = [
                        current_phase_space[index]
                        for index in active_indices
                    ]

                    orbit = Orbit(active_phase_space)
                    orbit.integrate(
                        times,
                        integration_potential,
                        method=integration_method,
                    )

                    if generate_gc_moving_potentials:
                        for local_index, cluster_index in enumerate(
                            active_indices
                        ):
                            if mass_loss_mode == "coupled":
                                potential_mass = current_masses[
                                    cluster_index
                                ]
                            else:
                                # none and postprocess do not feed mass
                                # evolution back into the GC potential.
                                potential_mass = gc_mass

                            gc_moving_potentials[cluster_index][
                                snapshot_index
                            ] = _build_gc_moving_potential(
                                orbit=orbit[local_index],
                                host_potential=density_potential,
                                gc_mass=potential_mass,
                                gc_scale_radius=(
                                    gc_potential_scale_radius
                                ),
                            )

                    x_active = np.asarray(orbit.x(times))
                    y_active = np.asarray(orbit.y(times))
                    z_active = np.asarray(orbit.z(times))
                    R_active = np.asarray(orbit.R(times))
                    phi_active = np.asarray(orbit.phi(times))
                    vR_active = np.asarray(orbit.vR(times))
                    vT_active = np.asarray(orbit.vT(times))
                    vz_active = np.asarray(orbit.vz(times))

                    if active_indices.size == 1:
                        x_active = np.atleast_2d(x_active)
                        y_active = np.atleast_2d(y_active)
                        z_active = np.atleast_2d(z_active)
                        R_active = np.atleast_2d(R_active)
                        phi_active = np.atleast_2d(phi_active)
                        vR_active = np.atleast_2d(vR_active)
                        vT_active = np.atleast_2d(vT_active)
                        vz_active = np.atleast_2d(vz_active)

                    x[active_indices, :] = x_active
                    y[active_indices, :] = y_active
                    z[active_indices, :] = z_active
                    R[active_indices, :] = R_active
                    phi[active_indices, :] = phi_active
                    vR[active_indices, :] = vR_active
                    vT[active_indices, :] = vT_active
                    vz[active_indices, :] = vz_active

            else:
                orbit = Orbit(current_phase_space)
                orbit.integrate(
                    times,
                    integration_potential,
                    method=integration_method,
                )

                if generate_gc_moving_potentials:
                    for cluster_index in range(number_of_clusters):
                        if mass_loss_mode == "coupled":
                            potential_mass = current_masses[
                                cluster_index
                            ]
                        else:
                            # none and postprocess keep the GC potential
                            # at the initial mass throughout the orbit run.
                            potential_mass = gc_mass

                        gc_moving_potentials[cluster_index][
                            snapshot_index
                        ] = _build_gc_moving_potential(
                            orbit=orbit[cluster_index],
                            host_potential=density_potential,
                            gc_mass=potential_mass,
                            gc_scale_radius=gc_potential_scale_radius,
                        )

                x = np.asarray(orbit.x(times))
                y = np.asarray(orbit.y(times))
                z = np.asarray(orbit.z(times))
                R = np.asarray(orbit.R(times))
                phi = np.asarray(orbit.phi(times))
                vR = np.asarray(orbit.vR(times))
                vT = np.asarray(orbit.vT(times))
                vz = np.asarray(orbit.vz(times))

        snapshot_radius = np.sqrt(x**2 + y**2 + z**2)

        last_apocenter_values = np.array(
            [
                _last_apocenter(snapshot_radius[index])
                for index in range(number_of_clusters)
            ],
            dtype=float,
        )

        newly_captured = (
            (~captured)
            & (last_apocenter_values <= central_capture_radius)
        )
        if np.any(newly_captured):
            captured_indices = np.where(newly_captured)[0]

            for index in captured_indices:
                captured_x[index] = x[index, -1]
                captured_y[index] = y[index, -1]
                captured_z[index] = z[index, -1]

            captured[newly_captured] = True

            print(
                f"{len(captured_indices)} GC(s) captured after snapshot "
                f"{snapshot_index}: last apocenter <= "
                f"{central_capture_radius:.4f} kpc."
            )

        if mass_loss_mode == "coupled":
            tidal_strength = _compute_tidal_strength_history(
                density_potential,
                x,
                y,
                z,
            )
            interval_mass_history = _integrate_mass_loss_interval(
                current_masses,
                times_numeric,
                tidal_strength,
                mass_loss_gamma,
                tidal_strength_reference,
                dissolution_time_normalization,
            )
            previous_masses = current_masses.copy()
            current_masses = interval_mass_history[:, -1]
            mass_all.append(interval_mass_history)

            newly_dissolved = np.where(
                (previous_masses > 0.0) & (current_masses == 0.0)
            )[0]

            if len(newly_dissolved) > 0:
                print(
                    f"{len(newly_dissolved)} GC(s) reached zero mass "
                    f"during snapshot {snapshot_index}. Dynamical friction "
                    "will be disabled for them from the next snapshot."
                )

            print(
                f"GC mass range after snapshot {snapshot_index}: "
                f"{current_masses.min():.3e}-{current_masses.max():.3e} Msun"
            )

        times_all.append(times_numeric)
        x_all.append(x)
        y_all.append(y)
        z_all.append(z)
        vR_all.append(vR)
        vT_all.append(vT)
        vz_all.append(vz)

        current_phase_space = []

        for index in range(number_of_clusters):
            if captured[index]:
                captured_R = np.sqrt(
                    captured_x[index] ** 2
                    + captured_y[index] ** 2
                )
                captured_phi = np.arctan2(
                    captured_y[index],
                    captured_x[index],
                )

                current_phase_space.append(
                    [
                        captured_R * units.kpc,
                        0.0 * units.km / units.s,
                        0.0 * units.km / units.s,
                        captured_z[index] * units.kpc,
                        0.0 * units.km / units.s,
                        captured_phi * units.rad,
                    ]
                )
            else:
                current_phase_space.append(
                    [
                        R[index, -1] * units.kpc,
                        vR[index, -1] * units.km / units.s,
                        vT[index, -1] * units.km / units.s,
                        z[index, -1] * units.kpc,
                        vz[index, -1] * units.km / units.s,
                        phi[index, -1]
                        * 180.0
                        / np.pi
                        * units.deg,
                    ]
                )

    time_output = np.concatenate(times_all)
    x_output = np.hstack(x_all)
    y_output = np.hstack(y_all)
    z_output = np.hstack(z_all)
    vR_output = np.hstack(vR_all)
    vT_output = np.hstack(vT_all)
    vz_output = np.hstack(vz_all)

    if mass_loss_mode == "postprocess":
        print("Computing the mass-loss history after the orbit integration.")
        current_masses = np.full(number_of_clusters, gc_mass, dtype=float)
        mass_all = []
        for interval_index, density_potential in enumerate(
            density_interval_potentials
        ):
            x_interval = x_all[interval_index]
            y_interval = y_all[interval_index]
            z_interval = z_all[interval_index]
            times_interval = times_all[interval_index]

            tidal_strength = _compute_tidal_strength_history(
                density_potential,
                x_interval,
                y_interval,
                z_interval,
            )
            interval_mass_history = _integrate_mass_loss_interval(
                current_masses,
                times_interval,
                tidal_strength,
                mass_loss_gamma,
                tidal_strength_reference,
                dissolution_time_normalization,
            )
            current_masses = interval_mass_history[:, -1]
            mass_all.append(interval_mass_history)

    mass_output = (
        np.hstack(mass_all)
        if mass_loss_mode != "none"
        else np.full((number_of_clusters, len(time_output)), gc_mass)
    )

    gc_moving_potential_files = []

    if generate_gc_moving_potentials:
        potential_mode_tag = (
            f"{potential_mode}_{df_model}_{mass_loss_mode}"
        )

        for cluster_index, potential_history in enumerate(
            gc_moving_potentials
        ):
            gc_potential_file = (
                gc_moving_potential_directory
                / (
                    f"GCpotInSitu_{potential_mode_tag}_"
                    f"GC{cluster_index}.pkl"
                )
            )

            with gc_potential_file.open("wb") as stream:
                pickle.dump(
                    potential_history,
                    stream,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            gc_moving_potential_files.append(
                str(gc_potential_file)
            )

        print(
            "GC moving potentials saved to: "
            f"{gc_moving_potential_directory}"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_file, "w") as h5:
        group = h5.create_group("GCdata")
        group.create_dataset("PosX", data=x_output)
        group.create_dataset("PosY", data=y_output)
        group.create_dataset("PosZ", data=z_output)
        group.create_dataset("vR", data=vR_output)
        group.create_dataset("vT", data=vT_output)
        group.create_dataset("vz", data=vz_output)
        group.create_dataset("Mass", data=mass_output)
        group.create_dataset(
            "Captured",
            data=captured.astype(np.uint8),
        )
        h5.create_dataset("Time", data=time_output)

        h5.attrs["potential_mode"] = potential_mode
        h5.attrs["orbit_potential_file"] = str(
            orbit_potential_file
        )
        h5.attrs["density_potential_file"] = str(
            density_potential_file
        )
        h5.attrs["df_model"] = df_model
        h5.attrs["mass_loss_mode"] = mass_loss_mode
        h5.attrs["object_type"] = object_type
        h5.attrs["initial_object_mass_msun"] = gc_mass
        h5.attrs["object_half_mass_radius_kpc"] = gc_half_mass_radius
        h5.attrs["initial_gc_mass_msun"] = gc_mass
        h5.attrs["central_capture_radius_kpc"] = central_capture_radius
        h5.attrs["gc_half_mass_radius_kpc"] = gc_half_mass_radius
        h5.attrs["generate_gc_moving_potentials"] = (
            generate_gc_moving_potentials
        )
        h5.attrs["gc_potential_scale_radius_kpc"] = (
            gc_potential_scale_radius
        )
        h5.attrs["gc_moving_potential_fallback"] = "host_only"
        h5.attrs["mass_loss_gamma"] = mass_loss_gamma
        h5.attrs["tidal_strength_reference"] = tidal_strength_reference
        h5.attrs["dissolution_time_normalization"] = (
            dissolution_time_normalization
        )
        if df_model == "fdm":
            h5.attrs["m22"] = m22

    print(f"Orbit and mass evolution saved to: {output_file}")

    if plot_file is not None:
        plot_file = Path(plot_file)
        plot_file.parent.mkdir(parents=True, exist_ok=True)
        radius = np.sqrt(x_output**2 + y_output**2 + z_output**2)

        for cluster_index in range(number_of_clusters):
            plt.plot(time_output, radius[cluster_index], linewidth=0.4)

        plt.xlabel(r"Time $t\;[\mathrm{Gyr}]$")
        plt.ylabel(r"Galactocentric distance $r\;[\mathrm{kpc}]$")
        plt.tight_layout()
        plt.savefig(plot_file, dpi=200)
        plt.close()
        print(f"Radius-evolution plot saved to: {plot_file}")

    return {
        "time": time_output,
        "x": x_output,
        "y": y_output,
        "z": z_output,
        "vR": vR_output,
        "vT": vT_output,
        "vz": vz_output,
        "mass": mass_output,
        "captured": captured,
        "central_capture_radius": central_capture_radius,
        "potential_mode": potential_mode,
        "orbit_potential_file": str(orbit_potential_file),
        "density_potential_file": str(density_potential_file),
        "df_model": df_model,
        "mass_loss_mode": mass_loss_mode,
        "m22": m22 if df_model == "fdm" else None,
        "df_cache_file": str(df_cache_file) if df_cache_file else None,
        "gc_moving_potential_files": gc_moving_potential_files,
        "gc_potential_scale_radius": gc_potential_scale_radius,
        "object_type": object_type,
        "number_of_objects": number_of_clusters,
    }
