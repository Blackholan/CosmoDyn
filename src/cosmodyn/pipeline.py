#!/usr/bin/env python3
# coding: utf-8
from pathlib import Path
import os
import traceback

import numpy as np

from cosmodyn import (
    generate_ex_situ_gcs,
    generate_in_situ_gcs,
    generate_plummer_gc,
    prepare_ex_situ_satellites,
    run_ex_situ_dynamics,
    run_ex_situ_streams,
    run_in_situ_dynamics,
    run_in_situ_streams,
)
from cosmodyn.timing import (
    record_stage_timing,
    start_stage_timer,
    write_timing_report,
)


DEFAULT_PARAMETERS = {
    # General
    "CONTINUE_ON_ERROR": True,
    "RUN_MODE": "in_situ",
    "ENABLE_STREAMS": False,

    # Stage switches used only by RUN_MODE = "custom"
    "RUN_ICS": False,
    "RUN_EX_SITU_ICS": False,
    "RUN_EX_SITU_SATELLITES": False,
    "RUN_IN_SITU_DYNAMICS": False,
    "RUN_EX_SITU_DYNAMICS": False,
    "RUN_STREAM_ICS": False,
    "RUN_IN_SITU_STREAMS": False,
    "RUN_EX_SITU_STREAMS": False,

    # In-situ ICs
    "SNAPSHOT_INDEX": 0,
    "NGC": 0,
    "ALPHA": 3,
    "CIRCULARITY_THRESHOLD": None,
    "TAGGING_RADIUS_FACTOR": 3,
    "MINIMUM_TAGGING_RADIUS": 0.5,
    "N_ITER": 20,
    "N_PARTICLES_PER_COMPONENT": 100_000,
    "RANDOM_SEED": None,
    "KEEP_AGAMA_FILE": True,

    # Ex-situ ICs
    "NGC_EX_SITU": 0,
    "ALPHA_EX_SITU": 3,
    "EX_SITU_TAGGING_RADIUS_FACTOR": 3,
    "EX_SITU_MINIMUM_TAGGING_RADIUS": 0.5,
    "EX_SITU_CIRCULARITY_THRESHOLD": None,
    "EX_SITU_N_ITER": 20,
    "EX_SITU_N_PARTICLES_PER_COMPONENT": 100_000,
    "EX_SITU_RANDOM_SEED": None,
    "KEEP_EX_SITU_AGAMA_FILES": True,

    # Dynamics
    "END_SNAPSHOT_INDEX": None,
    "INTEGRATION_METHOD": "dop853_c",
    "TIMESTEP_FILE": "TimeStepGTNG50.txt",
    "POTENTIAL_MODE": "evolving",
    "STATIC_POTENTIAL_INDEX": 73,
    "DF_MODEL": "cdm",
    "M22": 1,
    "INCLUDE_MOVING_SATELLITES": True,
    "MAXIMUM_SATELLITE_RADIUS": 1000.0,
    "RELEASE_ENERGY_TOLERANCE": 0.0,
    "GC_MASS": 1e6,
    "GC_HALF_MASS_RADIUS": 0.01,
    "REUSE_DF_CACHE": True,
    "CENTRAL_CAPTURE_RADIUS": 0.01,
    "GENERATE_GC_MOVING_POTENTIALS": True,
    "GC_POTENTIAL_SCALE_RADIUS": None,

    # Mass loss
    "MASS_LOSS_MODE": "none",
    "MASS_LOSS_GAMMA": 0.7,
    "TIDAL_STRENGTH_REFERENCE": 7.01e2,
    "DISSOLUTION_TIME_NORMALIZATION": 0.0107,

    # Streams
    "N_STREAM_PARTICLES": 100,
    "N_STREAM_ITER": 30,
    "OVERWRITE_STREAM_ICS": False,
    "STREAM_N_JOBS": -1,
    "STREAM_BATCH_SIZE": 64,
}


def run_pipeline(configuration):
    """Execute CosmoDyn using parameters supplied by a small launcher."""

    if "GALAXY_IDS" not in configuration:
        raise ValueError("Missing required parameter: GALAXY_IDS")

    parameters = DEFAULT_PARAMETERS | {
        name: value
        for name, value in configuration.items()
        if name in DEFAULT_PARAMETERS or name == "GALAXY_IDS"
    }

    run_mode = str(parameters["RUN_MODE"]).lower()
    valid_modes = {"in_situ", "ex_situ", "full", "custom"}

    if run_mode not in valid_modes:
        raise ValueError(
            f"Unknown RUN_MODE={run_mode!r}; "
            f"choose one of {sorted(valid_modes)}"
        )

    if run_mode != "custom":
        use_in_situ = run_mode in {"in_situ", "full"}
        use_ex_situ = run_mode in {"ex_situ", "full"}
        use_streams = bool(parameters["ENABLE_STREAMS"])

        use_moving_satellites = bool(
            parameters["INCLUDE_MOVING_SATELLITES"]
        )

        parameters.update(
            RUN_ICS=use_in_situ,
            RUN_EX_SITU_ICS=use_ex_situ,

            # The satellite trajectories and potentials are also needed
            # when in-situ GCs feel the moving satellites.
            RUN_EX_SITU_SATELLITES=(
                use_ex_situ or use_moving_satellites
            ),

            RUN_IN_SITU_DYNAMICS=use_in_situ,
            RUN_EX_SITU_DYNAMICS=use_ex_situ,
            RUN_STREAM_ICS=use_streams,
            RUN_IN_SITU_STREAMS=(
                use_streams and use_in_situ
            ),
            RUN_EX_SITU_STREAMS=(
                use_streams and use_ex_situ
            ),
        )

    # Copy only declared parameters.
    # Imports and unrelated launcher variables are ignored.
    for name, value in parameters.items():
        globals()[name] = value

    global N_CPUS
    N_CPUS = os.cpu_count() or 1

    # ==========================================================
    # Global output files
    # ==========================================================

    GLOBAL_OUTPUT_DIR = Path("Outputs")
    GLOBAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # One Plummer particle distribution shared by all galaxies.
    GC_PLUMMER_FILE = (
        GLOBAL_OUTPUT_DIR
        / (
            f"GCPlummer_M{GC_MASS:.0e}"
            f"_R{GC_HALF_MASS_RADIUS}"
            f"_N{N_STREAM_PARTICLES:.0e}.h5"
        )
    )

    SATELLITE_POTENTIAL_TAG = (
        "with_moving_satellites"
        if INCLUDE_MOVING_SATELLITES
        else "host_only"
    )

    MODE_TAG = (
        f"{POTENTIAL_MODE}_"
        f"{SATELLITE_POTENTIAL_TAG}_"
        f"{DF_MODEL}_"
        f"{MASS_LOSS_MODE}"
    )

    TIMING_FILE = (
        GLOBAL_OUTPUT_DIR
        / f"ComputationTime_{MODE_TAG}.txt"
    )

    # ==========================================================
    # Computation-time monitoring
    # ==========================================================

    TIMING_RECORDS = []

    def _stream_cpu_count():
        """Return the number of CPUs allocated to stream calculations."""

        if STREAM_N_JOBS == -1:
            return os.cpu_count() or 1

        return max(int(STREAM_N_JOBS), 1)

    def _read_number_of_gcs(initial_conditions_file):
        """Return the number of GCs stored in an existing IC file."""

        initial_conditions_file = Path(initial_conditions_file)

        if not initial_conditions_file.exists():
            return None

        data = np.atleast_2d(
            np.loadtxt(initial_conditions_file)
        )

        return len(data)

    # ==========================================================
    # Generate the global stream initial conditions
    # ==========================================================

    # The Plummer distribution depends only on the global GC
    # parameters. It is generated once and shared by all galaxies.

    if RUN_STREAM_ICS:
        stage_start = start_stage_timer()

        generate_plummer_gc(
            output_file=GC_PLUMMER_FILE,
            gc_mass=GC_MASS,
            gc_half_mass_radius=GC_HALF_MASS_RADIUS,
            n_particles=N_STREAM_PARTICLES,
            n_iter=N_STREAM_ITER,
            overwrite=OVERWRITE_STREAM_ICS,
        )

        # galaxy_id=-1 marks a global stage in the timing report.
        record_stage_timing(
            records=TIMING_RECORDS,
            galaxy_id=-1,
            stage="RUN_STREAM_ICS",
            start_time=stage_start,
            n_gcs=None,
            n_particles=N_STREAM_PARTICLES,
            n_cpus=1,
        )

    def run_galaxy(galaxy_id):
        """Run the selected pipeline stages for one galaxy."""

        print()
        print("=" * 70)
        print(f"Starting galaxy G{galaxy_id}")
        print("=" * 70)

        # ==========================================================
        # Automatically generated file names
        # ==========================================================

        INPUT_DIR = Path(f"DataG{galaxy_id}")
        MWDATA_PATH = INPUT_DIR / f"DataG{galaxy_id}.txt"
        MWPOTS_PATH = INPUT_DIR / f"PotsG{galaxy_id}.pkl"

        SATELLITES_FILE = (
            INPUT_DIR
            / f"G{galaxy_id}TimeSat.txt"
        )

        SATELLITE_DATA_DIRECTORY = (
            INPUT_DIR
            / f"GSat{galaxy_id}"
        )

        OUTPUT_DIR = (
            GLOBAL_OUTPUT_DIR
            / f"G{galaxy_id}"
        )
        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        IN_SITU_OUTPUT_DIR = (
            OUTPUT_DIR
            / "InSitu"
        )

        IN_SITU_AGAMA_DIR = (
            IN_SITU_OUTPUT_DIR
            / "AGAMA"
        )

        IN_SITU_PLOT_DIR = (
            IN_SITU_OUTPUT_DIR
            / "Plots"
        )

        IN_SITU_OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )
        IN_SITU_AGAMA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )
        IN_SITU_PLOT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        AGAMA_FILE = (
            IN_SITU_AGAMA_DIR
            / (
                f"ICGCG{galaxy_id}"
                f"N{N_PARTICLES_PER_COMPONENT:.0e}.h5"
            )
        )

        OUTPUT_FILE = (
            IN_SITU_OUTPUT_DIR
            / f"IniGCG{galaxy_id}.txt"
        )

        PLOT_FILE = (
            IN_SITU_PLOT_DIR
            / f"G{galaxy_id}.png"
        )

        MODE_TAG = (
            f"{POTENTIAL_MODE}_"
            f"{SATELLITE_POTENTIAL_TAG}_"
            f"{DF_MODEL}_"
            f"{MASS_LOSS_MODE}"
        )

        GC_MOVING_POTENTIAL_DIRECTORY = (
            IN_SITU_OUTPUT_DIR
            / "GCPotential"
        )

        STREAM_OUTPUT_DIRECTORY = (
            IN_SITU_OUTPUT_DIR
            / f"Streams_{MODE_TAG}"
        )

        STREAM_PLOT_DIRECTORY = (
            IN_SITU_OUTPUT_DIR
            / "Plots"
            / f"Streams_{MODE_TAG}"
        )

        DF_CACHE_DIRECTORY = (
            IN_SITU_OUTPUT_DIR
            / "DynamicalFrictionForce"
        )

        # ==========================================================
        # In-situ dynamics outputs
        # ==========================================================

        DYNAMICS_OUTPUT_FILE = (
            IN_SITU_OUTPUT_DIR
            / (
                f"InSituDynamics_"
                f"{MODE_TAG}_G{galaxy_id}.h5"
            )
        )

        DYNAMICS_PLOT_FILE = (
            IN_SITU_OUTPUT_DIR
            / "Plots"
            / (
                f"InSituRadiusEvolution_"
                f"{MODE_TAG}_G{galaxy_id}.png"
            )
        )

        # ==========================================================
        # Ex-situ paths
        # ==========================================================

        EX_SITU_OUTPUT_DIRECTORY = (
            OUTPUT_DIR
            / "ExSitu"
        )

        EX_SITU_SATELLITE_OUTPUT_DIRECTORY = (
            EX_SITU_OUTPUT_DIRECTORY
            / "Satellites"
        )

        EX_SITU_RETAINED_SATELLITES_FILE = (
            EX_SITU_OUTPUT_DIRECTORY
            / f"ExSituSatellitesG{galaxy_id}.txt"
        )

        EX_SITU_SATELLITE_TRAJECTORY_FILE = (
            EX_SITU_SATELLITE_OUTPUT_DIRECTORY
            / f"ExSituSatelliteDataG{galaxy_id}.h5"
        )

        EX_SITU_SATELLITE_POTENTIAL_DIRECTORY = (
            EX_SITU_SATELLITE_OUTPUT_DIRECTORY
            / "Potentials"
        )

        EX_SITU_DYNAMICS_OUTPUT_FILE = (
            EX_SITU_OUTPUT_DIRECTORY
            / (
                f"ExSituDynamics_"
                f"{MODE_TAG}_G{galaxy_id}.h5"
            )
        )

        EX_SITU_DYNAMICS_PLOT_FILE = (
            EX_SITU_OUTPUT_DIRECTORY
            / "Plots"
            / (
                f"ExSituRadiusEvolution_"
                f"{MODE_TAG}_G{galaxy_id}.png"
            )
        )

        EX_SITU_GC_MOVING_POTENTIAL_DIRECTORY = (
            EX_SITU_OUTPUT_DIRECTORY
            / "GCPotential"
        )

        EX_SITU_STREAM_OUTPUT_DIRECTORY = (
            EX_SITU_OUTPUT_DIRECTORY
            / f"Streams_{MODE_TAG}"
        )

        EX_SITU_STREAM_PLOT_DIRECTORY = (
            EX_SITU_OUTPUT_DIRECTORY
            / "Plots"
            / f"Streams_{MODE_TAG}"
        )

        FULL_POTENTIAL_FILE = (
            EX_SITU_SATELLITE_OUTPUT_DIRECTORY
            / "Potentials"
            / f"FullHostPotentialG{galaxy_id}.pkl"
        )

        IN_SITU_ORBIT_POTENTIAL_FILE = (
            FULL_POTENTIAL_FILE
            if INCLUDE_MOVING_SATELLITES
            else MWPOTS_PATH
        )

        if (
            INCLUDE_MOVING_SATELLITES
            and POTENTIAL_MODE != "evolving"
        ):
            raise ValueError(
                "INCLUDE_MOVING_SATELLITES=True requires "
                "POTENTIAL_MODE='evolving'."
            )

        number_of_gcs = _read_number_of_gcs(
            OUTPUT_FILE
        )

        # ==========================================================
        # Run the in-situ GC initial conditions
        # ==========================================================

        if RUN_ICS:
            stage_start = start_stage_timer()

            gc_initial_conditions = generate_in_situ_gcs(
                galaxy_id=galaxy_id,
                mwdata_path=MWDATA_PATH,
                mwpots_path=MWPOTS_PATH,
                agama_file=AGAMA_FILE,
                output_file=OUTPUT_FILE,
                plot_file=PLOT_FILE,
                snapshot_index=SNAPSHOT_INDEX,
                ngc=NGC,
                alpha=ALPHA,
                tagging_radius_factor=(
                    TAGGING_RADIUS_FACTOR
                ),
                minimum_tagging_radius=(
                    MINIMUM_TAGGING_RADIUS
                ),
                circularity_threshold=(
                    CIRCULARITY_THRESHOLD
                ),
                n_iter=N_ITER,
                n_particles_per_component=(
                    N_PARTICLES_PER_COMPONENT
                ),
                random_seed=RANDOM_SEED,
                keep_agama_file=KEEP_AGAMA_FILE,
            )

            number_of_gcs = len(
                np.atleast_2d(gc_initial_conditions)
            )

            record_stage_timing(
                records=TIMING_RECORDS,
                galaxy_id=galaxy_id,
                stage="RUN_IN_SITU_ICS",
                start_time=stage_start,
                n_gcs=number_of_gcs,
                n_particles=(
                    N_PARTICLES_PER_COMPONENT
                ),
                n_cpus=1,
            )

        if number_of_gcs is None:
            number_of_gcs = _read_number_of_gcs(
                OUTPUT_FILE
            )

        # ==========================================================
        # Run the ex-situ GC initial conditions
        # ==========================================================

        if RUN_EX_SITU_ICS:
            stage_start = start_stage_timer()

            ex_situ_results = generate_ex_situ_gcs(
                galaxy_id=galaxy_id,
                satellites_file=SATELLITES_FILE,
                satellite_data_directory=(
                    SATELLITE_DATA_DIRECTORY
                ),
                output_directory=(
                    EX_SITU_OUTPUT_DIRECTORY
                ),
                snapshot_index=SNAPSHOT_INDEX,
                ngc=NGC_EX_SITU,
                alpha=ALPHA_EX_SITU,
                tagging_radius_factor=(
                    EX_SITU_TAGGING_RADIUS_FACTOR
                ),
                minimum_tagging_radius=(
                    EX_SITU_MINIMUM_TAGGING_RADIUS
                ),
                circularity_threshold=(
                    EX_SITU_CIRCULARITY_THRESHOLD
                ),
                n_iter=EX_SITU_N_ITER,
                n_particles_per_component=(
                    EX_SITU_N_PARTICLES_PER_COMPONENT
                ),
                random_seed=EX_SITU_RANDOM_SEED,
                keep_agama_files=(
                    KEEP_EX_SITU_AGAMA_FILES
                ),
            )

            number_of_ex_situ_gcs = 0

            for gc_file in ex_situ_results[
                "generated_files"
            ]:
                data = np.atleast_2d(
                    np.loadtxt(gc_file)
                )
                number_of_ex_situ_gcs += len(data)

            record_stage_timing(
                records=TIMING_RECORDS,
                galaxy_id=galaxy_id,
                stage="RUN_EX_SITU_ICS",
                start_time=stage_start,
                n_gcs=number_of_ex_situ_gcs,
                n_particles=(
                    EX_SITU_N_PARTICLES_PER_COMPONENT
                ),
                n_cpus=1,
            )

        # ==========================================================
        # Prepare ex-situ satellite trajectories and potentials
        # ==========================================================

        if RUN_EX_SITU_SATELLITES:
            stage_start = start_stage_timer()

            prepare_ex_situ_satellites(
                galaxy_id=galaxy_id,
                satellites_file=SATELLITES_FILE,
                satellite_data_directory=(
                    SATELLITE_DATA_DIRECTORY
                ),
                host_data_file=MWDATA_PATH,
                timestep_file=TIMESTEP_FILE,
                host_potential_file=MWPOTS_PATH,
                output_directory=(
                    EX_SITU_SATELLITE_OUTPUT_DIRECTORY
                ),
                start_snapshot_index=SNAPSHOT_INDEX,
                end_snapshot_index=END_SNAPSHOT_INDEX,
                integration_method=INTEGRATION_METHOD,
                potential_mode=POTENTIAL_MODE,
                static_potential_index=(
                    STATIC_POTENTIAL_INDEX
                ),
                df_model=DF_MODEL,
                m22=M22,
                maximum_satellite_radius=(
                    MAXIMUM_SATELLITE_RADIUS
                ),
                write_legacy_velocity_aliases=False,
            )

            record_stage_timing(
                records=TIMING_RECORDS,
                galaxy_id=galaxy_id,
                stage="RUN_EX_SITU_SATELLITES",
                start_time=stage_start,
                n_gcs=None,
                n_particles=None,
                n_cpus=1,
            )

        # ==========================================================
        # Run the in-situ dynamics
        # ==========================================================

        if RUN_IN_SITU_DYNAMICS:
            stage_start = start_stage_timer()

            run_in_situ_dynamics(
                initial_conditions_file=OUTPUT_FILE,
                timestep_file=TIMESTEP_FILE,
                potential_file=(
                    IN_SITU_ORBIT_POTENTIAL_FILE
                ),
                density_potential_file=MWPOTS_PATH,
                output_file=DYNAMICS_OUTPUT_FILE,
                plot_file=DYNAMICS_PLOT_FILE,
                start_snapshot_index=SNAPSHOT_INDEX,
                end_snapshot_index=END_SNAPSHOT_INDEX,
                integration_method=INTEGRATION_METHOD,
                potential_mode=POTENTIAL_MODE,
                df_model=DF_MODEL,
                static_potential_index=(
                    STATIC_POTENTIAL_INDEX
                ),
                gc_mass=GC_MASS,
                gc_half_mass_radius=(
                    GC_HALF_MASS_RADIUS
                ),
                m22=M22,
                df_cache_directory=DF_CACHE_DIRECTORY,
                reuse_df_cache=REUSE_DF_CACHE,
                mass_loss_mode=MASS_LOSS_MODE,
                mass_loss_gamma=MASS_LOSS_GAMMA,
                tidal_strength_reference=(
                    TIDAL_STRENGTH_REFERENCE
                ),
                dissolution_time_normalization=(
                    DISSOLUTION_TIME_NORMALIZATION
                ),
                central_capture_radius=(
                    CENTRAL_CAPTURE_RADIUS
                ),
                generate_gc_moving_potentials=(
                    GENERATE_GC_MOVING_POTENTIALS
                ),
                gc_moving_potential_directory=(
                    GC_MOVING_POTENTIAL_DIRECTORY
                ),
                gc_potential_scale_radius=(
                    GC_POTENTIAL_SCALE_RADIUS
                ),
            )

            record_stage_timing(
                records=TIMING_RECORDS,
                galaxy_id=galaxy_id,
                stage="RUN_IN_SITU_DYNAMICS",
                start_time=stage_start,
                n_gcs=number_of_gcs,
                n_particles=None,
                n_cpus=N_CPUS,
            )

        # ==========================================================
        # Run the ex-situ dynamics
        # ==========================================================

        if RUN_EX_SITU_DYNAMICS:
            stage_start = start_stage_timer()

            ex_situ_dynamics_results = (
                run_ex_situ_dynamics(
                    galaxy_id=galaxy_id,

                    # Satellites containing tagged GCs.
                    satellites_file=(
                        EX_SITU_RETAINED_SATELLITES_FILE
                    ),

                    initial_conditions_directory=(
                        EX_SITU_OUTPUT_DIRECTORY
                    ),

                    satellite_trajectory_file=(
                        EX_SITU_SATELLITE_TRAJECTORY_FILE
                    ),

                    satellite_potential_directory=(
                        EX_SITU_SATELLITE_POTENTIAL_DIRECTORY
                    ),

                    timestep_file=TIMESTEP_FILE,

                    # MW + moving satellites when enabled.
                    host_orbit_potential_file=(
                        IN_SITU_ORBIT_POTENTIAL_FILE
                    ),

                    # Smooth MW for DF and mass loss.
                    host_density_potential_file=(
                        MWPOTS_PATH
                    ),

                    output_file=(
                        EX_SITU_DYNAMICS_OUTPUT_FILE
                    ),

                    plot_file=(
                        EX_SITU_DYNAMICS_PLOT_FILE
                    ),

                    start_snapshot_index=SNAPSHOT_INDEX,
                    end_snapshot_index=(
                        END_SNAPSHOT_INDEX
                    ),

                    integration_method=(
                        INTEGRATION_METHOD
                    ),
                    potential_mode=POTENTIAL_MODE,
                    static_potential_index=(
                        STATIC_POTENTIAL_INDEX
                    ),

                    df_model=DF_MODEL,
                    gc_mass=GC_MASS,
                    gc_half_mass_radius=(
                        GC_HALF_MASS_RADIUS
                    ),
                    m22=M22,

                    mass_loss_mode=MASS_LOSS_MODE,
                    mass_loss_gamma=(
                        MASS_LOSS_GAMMA
                    ),
                    tidal_strength_reference=(
                        TIDAL_STRENGTH_REFERENCE
                    ),
                    dissolution_time_normalization=(
                        DISSOLUTION_TIME_NORMALIZATION
                    ),

                    central_capture_radius=(
                        CENTRAL_CAPTURE_RADIUS
                    ),

                    release_energy_tolerance=(
                        RELEASE_ENERGY_TOLERANCE
                    ),

                    generate_gc_moving_potentials=(
                        GENERATE_GC_MOVING_POTENTIALS
                    ),

                    gc_moving_potential_directory=(
                        EX_SITU_GC_MOVING_POTENTIAL_DIRECTORY
                    ),

                    gc_potential_scale_radius=(
                        GC_POTENTIAL_SCALE_RADIUS
                    ),
                )
            )

            number_of_ex_situ_gcs = (
                ex_situ_dynamics_results[
                    "number_of_clusters"
                ]
            )

            record_stage_timing(
                records=TIMING_RECORDS,
                galaxy_id=galaxy_id,
                stage="RUN_EX_SITU_DYNAMICS",
                start_time=stage_start,
                n_gcs=number_of_ex_situ_gcs,
                n_particles=None,
                n_cpus=1,
            )

        # ==========================================================
        # Run the in-situ streams
        # ==========================================================

        if RUN_IN_SITU_STREAMS:
            stage_start = start_stage_timer()

            run_in_situ_streams(
                plummer_file=GC_PLUMMER_FILE,
                timestep_file=TIMESTEP_FILE,
                potential_file=MWPOTS_PATH,
                dynamics_file=DYNAMICS_OUTPUT_FILE,
                gc_moving_potential_directory=(
                    GC_MOVING_POTENTIAL_DIRECTORY
                ),
                output_directory=(
                    STREAM_OUTPUT_DIRECTORY
                ),
                plot_directory=(
                    STREAM_PLOT_DIRECTORY
                ),
                start_snapshot_index=SNAPSHOT_INDEX,
                end_snapshot_index=(
                    END_SNAPSHOT_INDEX
                ),
                potential_mode=POTENTIAL_MODE,
                df_model=DF_MODEL,
                mass_loss_mode=MASS_LOSS_MODE,
                integration_method=(
                    INTEGRATION_METHOD
                ),
                n_jobs=STREAM_N_JOBS,
                batch_size=STREAM_BATCH_SIZE,
            )

            record_stage_timing(
                records=TIMING_RECORDS,
                galaxy_id=galaxy_id,
                stage="RUN_IN_SITU_STREAMS",
                start_time=stage_start,
                n_gcs=number_of_gcs,
                n_particles=N_STREAM_PARTICLES,
                n_cpus=N_CPUS,
            )

        # ==========================================================
        # Run the ex-situ streams
        # ==========================================================

        if RUN_EX_SITU_STREAMS:
            stage_start = start_stage_timer()

            ex_situ_stream_files = run_ex_situ_streams(
                galaxy_id=galaxy_id,

                plummer_file=GC_PLUMMER_FILE,
                timestep_file=TIMESTEP_FILE,

                # Smooth MW after escape from the GC.
                host_potential_file=MWPOTS_PATH,

                dynamics_file=(
                    EX_SITU_DYNAMICS_OUTPUT_FILE
                ),

                gc_moving_potential_directory=(
                    EX_SITU_GC_MOVING_POTENTIAL_DIRECTORY
                ),

                output_directory=(
                    EX_SITU_STREAM_OUTPUT_DIRECTORY
                ),

                plot_directory=(
                    EX_SITU_STREAM_PLOT_DIRECTORY
                ),

                potential_mode=POTENTIAL_MODE,

                static_potential_index=(
                    STATIC_POTENTIAL_INDEX
                ),

                df_model=DF_MODEL,
                mass_loss_mode=MASS_LOSS_MODE,

                integration_method=(
                    INTEGRATION_METHOD
                ),

                n_jobs=STREAM_N_JOBS,
                batch_size=STREAM_BATCH_SIZE,
            )

            record_stage_timing(
                records=TIMING_RECORDS,
                galaxy_id=galaxy_id,
                stage="RUN_EX_SITU_STREAMS",
                start_time=stage_start,
                n_gcs=len(ex_situ_stream_files),
                n_particles=N_STREAM_PARTICLES,
                n_cpus=_stream_cpu_count(),
            )

        print(f"Finished galaxy G{galaxy_id}")

        return number_of_gcs

    # ==========================================================
    # Run all galaxies
    # ==========================================================

    failed_galaxies = []
    successful_galaxies = []

    for GALAXY_ID in GALAXY_IDS:
        try:
            run_galaxy(GALAXY_ID)
            successful_galaxies.append(GALAXY_ID)

        except Exception:
            failed_galaxies.append(GALAXY_ID)

            print()
            print(f"Galaxy G{GALAXY_ID} failed.")
            traceback.print_exc()

            if not CONTINUE_ON_ERROR:
                raise

    # ==========================================================
    # Save the computation-time report
    # ==========================================================

    RUN_PARAMETERS = {
        # General execution
        "galaxy_ids": GALAXY_IDS,
        "continue_on_error": CONTINUE_ON_ERROR,
        "run_mode": RUN_MODE,
        "enable_streams": ENABLE_STREAMS,
        "available_cpus": N_CPUS,
        "snapshot_index": SNAPSHOT_INDEX,
        "end_snapshot_index": END_SNAPSHOT_INDEX,
        "integration_method": INTEGRATION_METHOD,
        "timestep_file": TIMESTEP_FILE,

        # In-situ initial conditions
        "ngc_in_situ": NGC,
        "alpha_in_situ": ALPHA,
        "circularity_threshold": (
            CIRCULARITY_THRESHOLD
        ),
        "tagging_radius_factor": (
            TAGGING_RADIUS_FACTOR
        ),
        "minimum_tagging_radius": (
            MINIMUM_TAGGING_RADIUS
        ),
        "n_iter_in_situ": N_ITER,
        "n_particles_per_component": (
            N_PARTICLES_PER_COMPONENT
        ),
        "random_seed_in_situ": RANDOM_SEED,
        "keep_agama_file": KEEP_AGAMA_FILE,

        # Ex-situ initial conditions
        "ngc_ex_situ": NGC_EX_SITU,
        "alpha_ex_situ": ALPHA_EX_SITU,
        "ex_situ_tagging_radius_factor": (
            EX_SITU_TAGGING_RADIUS_FACTOR
        ),
        "ex_situ_minimum_tagging_radius": (
            EX_SITU_MINIMUM_TAGGING_RADIUS
        ),
        "ex_situ_circularity_threshold": (
            EX_SITU_CIRCULARITY_THRESHOLD
        ),
        "ex_situ_n_iter": EX_SITU_N_ITER,
        "ex_situ_n_particles_per_component": (
            EX_SITU_N_PARTICLES_PER_COMPONENT
        ),
        "ex_situ_random_seed": (
            EX_SITU_RANDOM_SEED
        ),
        "keep_ex_situ_agama_files": (
            KEEP_EX_SITU_AGAMA_FILES
        ),

        # Satellite and host potentials
        "include_moving_satellites": (
            INCLUDE_MOVING_SATELLITES
        ),
        "maximum_satellite_radius": (
            MAXIMUM_SATELLITE_RADIUS
        ),

        # Dynamics
        "potential_mode": POTENTIAL_MODE,
        "static_potential_index": (
            STATIC_POTENTIAL_INDEX
        ),
        "df_model": DF_MODEL,
        "m22": M22,
        "gc_mass": GC_MASS,
        "gc_half_mass_radius": (
            GC_HALF_MASS_RADIUS
        ),
        "reuse_df_cache": REUSE_DF_CACHE,
        "central_capture_radius": (
            CENTRAL_CAPTURE_RADIUS
        ),
        "generate_gc_moving_potentials": (
            GENERATE_GC_MOVING_POTENTIALS
        ),
        "gc_potential_scale_radius": (
            GC_POTENTIAL_SCALE_RADIUS
        ),
        "mass_loss_mode": MASS_LOSS_MODE,
        "mass_loss_gamma": MASS_LOSS_GAMMA,
        "tidal_strength_reference": (
            TIDAL_STRENGTH_REFERENCE
        ),
        "dissolution_time_normalization": (
            DISSOLUTION_TIME_NORMALIZATION
        ),
        "release_energy_tolerance": (
            RELEASE_ENERGY_TOLERANCE
        ),

        # Streams
        "n_stream_particles": (
            N_STREAM_PARTICLES
        ),
        "n_stream_iter": N_STREAM_ITER,
        "overwrite_stream_ics": (
            OVERWRITE_STREAM_ICS
        ),
        "stream_n_jobs": STREAM_N_JOBS,
        "stream_cpu_count": (
            _stream_cpu_count()
        ),
        "stream_batch_size": STREAM_BATCH_SIZE,
    }

    ENABLED_STAGES = {
        "RUN_IN_SITU_ICS": RUN_ICS,
        "RUN_EX_SITU_ICS": RUN_EX_SITU_ICS,
        "RUN_EX_SITU_SATELLITES": (
            RUN_EX_SITU_SATELLITES
        ),
        "RUN_STREAM_ICS": RUN_STREAM_ICS,
        "RUN_IN_SITU_DYNAMICS": (
            RUN_IN_SITU_DYNAMICS
        ),
        "RUN_EX_SITU_DYNAMICS": (
            RUN_EX_SITU_DYNAMICS
        ),
        "RUN_IN_SITU_STREAMS": (
            RUN_IN_SITU_STREAMS
        ),
        "RUN_EX_SITU_STREAMS": (
            RUN_EX_SITU_STREAMS
        ),
    }

    write_timing_report(
        output_file=TIMING_FILE,
        records=TIMING_RECORDS,
        galaxy_ids=GALAXY_IDS,
        run_parameters=RUN_PARAMETERS,
        enabled_stages=ENABLED_STAGES,
        successful_galaxies=successful_galaxies,
        failed_galaxies=failed_galaxies,
        output_directory=GLOBAL_OUTPUT_DIR,
    )

    # ==========================================================
    # Final summary
    # ==========================================================

    print()
    print("=" * 70)
    print("Multi-galaxy run completed.")
    print(f"Successful galaxies: {len(successful_galaxies)}")
    print(f"Failed galaxies: {len(failed_galaxies)}")

    if failed_galaxies:
        print(
            "Failed galaxy IDs: "
            + ", ".join(
                str(galaxy_id)
                for galaxy_id in failed_galaxies
            )
        )