#!/usr/bin/env python3
# coding: utf-8

from pathlib import Path
import os
import traceback
import numpy as np
from cosmodyn import (generate_in_situ_gcs,generate_ex_situ_gcs,generate_plummer_gc,run_in_situ_dynamics,run_in_situ_streams)
from cosmodyn.timing import (record_stage_timing,start_stage_timer,write_timing_report,)


# ==========================================================
# Galaxy parameters
# ==========================================================
# Single galaxy:
GALAXY_IDS = [1]
# Several galaxies:
# GALAXY_IDS = [613192,462710]
# Or read all galaxy IDs from a text file containing one ID per line:
# GALAXY_IDS = np.loadtxt("GalaxyList.txt", dtype=int).tolist()
CONTINUE_ON_ERROR = True  # continue with the next galaxy if one run fails
N_CPUS = os.cpu_count() or 1

# ==========================================================
# In-situ ICs
# ==========================================================

RUN_ICS = False

SNAPSHOT_INDEX = 0  # index of the snapshot at which the objects are tagged
NGC = 5  # 0 = use the halo-mass relation; >0 = impose this number of GCs
ALPHA = 1  # used only when NGC = 0
CIRCULARITY_THRESHOLD = None # Select particles with 0.6 <= Lz/Lcirc(E) <= 1 if not put None
TAGGING_RADIUS_FACTOR = 2 # inside x half-mass radius of stars
MINIMUM_TAGGING_RADIUS = 0.5  # kpc; None disables the constraint None otherwise
N_ITER = 20
N_PARTICLES_PER_COMPONENT = 1_000_000  # use this format, not 1e6
RANDOM_SEED = None  # or 42 to obtain the same sample at each run
KEEP_AGAMA_FILE = False  # keep the AGAMA file after generating the ICs

# ==========================================================
# Ex-situ ICs
# ==========================================================

RUN_EX_SITU_ICS = False

NGC_EX_SITU = 3
ALPHA_EX_SITU = 3
EX_SITU_TAGGING_RADIUS_FACTOR = TAGGING_RADIUS_FACTOR
EX_SITU_CIRCULARITY_THRESHOLD = CIRCULARITY_THRESHOLD
EX_SITU_N_ITER = N_ITER
EX_SITU_N_PARTICLES_PER_COMPONENT = N_PARTICLES_PER_COMPONENT
EX_SITU_RANDOM_SEED = RANDOM_SEED
KEEP_EX_SITU_AGAMA_FILES = True

# Dynamics

RUN_DYNAMICS = True  # integrate the GC orbits after generating the ICs

END_SNAPSHOT_INDEX = None  # None = integrate to the end
INTEGRATION_METHOD = "dop853_c"
TIMESTEP_FILE = f"TimeStepG1.txt" # [start time (Gyr), end time (Gyr), number of integration steps]
POTENTIAL_MODE = "static"  # "evolving" or "static"
STATIC_POTENTIAL_INDEX = 0     # used only for "static"

DF_MODEL = "cdm"           # "none", "cdm", or "fdm"
M22 = 1                  # 1.0 corresponds to 1e-22 eV for fdm model

# GC parameters
GC_MASS = 5e5                 # Msun, same value for all GCs
GC_HALF_MASS_RADIUS = 0.01      # kpc, same value for all GCs
REUSE_DF_CACHE = True  # Load existing dynamical-friction forces if available instead of recomputing them
CENTRAL_CAPTURE_RADIUS = 0.01  # kpc; capture if the snapshot apocenter is below this radius

# Mass loss parameters
MASS_LOSS_MODE = "none"  # "none", "postprocess", or "coupled"
MASS_LOSS_GAMMA = 0.7 # Based on Kruijssen+11 mass loss model
TIDAL_STRENGTH_REFERENCE = 7.01e2 # Based on Kruijssen+11 mass loss model
DISSOLUTION_TIME_NORMALIZATION = 0.0107 # Based on Kruijssen+11 mass loss model

#Streams
GENERATE_GC_MOVING_POTENTIALS = True

GC_POTENTIAL_SCALE_RADIUS = None  # kpc same as GC_HALF_MASS_RADIUS

RUN_STREAM_ICS = True  # generate the Plummer particle distribution

N_STREAM_PARTICLES = 100000
N_STREAM_ITER = 30
OVERWRITE_STREAM_ICS = False # no rerun of AGAMA for stream

RUN_STREAMS = True  # integrate the stream particles
STREAM_N_JOBS = -1
STREAM_BATCH_SIZE = 64



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

MODE_TAG = f"{POTENTIAL_MODE}_{DF_MODEL}_{MASS_LOSS_MODE}"

TIMING_FILE = GLOBAL_OUTPUT_DIR / f"ComputationTime_{MODE_TAG}.txt"


# ==========================================================
# Computation-time monitoring
# ==========================================================

TIMING_RECORDS = []


def _stream_cpu_count():
    """Return the number of CPUs allocated to the stream calculation."""
    if STREAM_N_JOBS == -1:
        return os.cpu_count() or 1
    return max(int(STREAM_N_JOBS), 1)


def _read_number_of_gcs(initial_conditions_file):
    """Return the number of GCs stored in an existing IC file."""
    initial_conditions_file = Path(initial_conditions_file)

    if not initial_conditions_file.exists():
        return None

    data = np.atleast_2d(np.loadtxt(initial_conditions_file))
    return len(data)


# ==========================================================
# Generate the global stream initial conditions
# ==========================================================

# The Plummer distribution depends only on the global GC parameters.
# It is therefore generated once and shared by all galaxies.

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
    """Run the complete in-situ pipeline for one galaxy."""

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

    SATELLITES_FILE = INPUT_DIR / f"G{galaxy_id}TimeSat.txt"
    SATELLITE_DATA_DIRECTORY = INPUT_DIR / f"GSat{galaxy_id}"

    OUTPUT_DIR = GLOBAL_OUTPUT_DIR / f"G{galaxy_id}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    IN_SITU_OUTPUT_DIR = OUTPUT_DIR / "InSitu"
    IN_SITU_AGAMA_DIR = IN_SITU_OUTPUT_DIR / "AGAMA"
    IN_SITU_PLOT_DIR = IN_SITU_OUTPUT_DIR / "Plots"

    IN_SITU_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IN_SITU_AGAMA_DIR.mkdir(parents=True, exist_ok=True)
    IN_SITU_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    AGAMA_FILE = (
        IN_SITU_AGAMA_DIR
        / f"ICGCG{galaxy_id}N{N_PARTICLES_PER_COMPONENT:.0e}.h5"
    )

    OUTPUT_FILE = (
        IN_SITU_OUTPUT_DIR
        / f"IniGCG{galaxy_id}.txt"
    )

    PLOT_FILE = (
        IN_SITU_PLOT_DIR
        / f"G{galaxy_id}.png"
    )

    MODE_TAG = f"{POTENTIAL_MODE}_{DF_MODEL}_{MASS_LOSS_MODE}"

    GC_MOVING_POTENTIAL_DIRECTORY = (
        IN_SITU_OUTPUT_DIR / "GCPotential"
    )

    STREAM_OUTPUT_DIRECTORY = (
        IN_SITU_OUTPUT_DIR / f"Streams_{MODE_TAG}"
    )

    STREAM_PLOT_DIRECTORY = (
        IN_SITU_OUTPUT_DIR / "Plots" / f"Streams_{MODE_TAG}"
    )

    DF_CACHE_DIRECTORY = (
        IN_SITU_OUTPUT_DIR / "DynamicalFrictionForce"
    )

    # ==========================================================
    # In-situ dynamics outputs
    # ==========================================================

    DYNAMICS_OUTPUT_FILE = (
        IN_SITU_OUTPUT_DIR
        / f"InSituDynamics_{MODE_TAG}_G{galaxy_id}.h5"
    )

    DYNAMICS_PLOT_FILE = (
        IN_SITU_OUTPUT_DIR
        / "Plots"
        / f"InSituRadiusEvolution_{MODE_TAG}_G{galaxy_id}.png"
    )

    EX_SITU_OUTPUT_DIRECTORY = OUTPUT_DIR / "ExSitu"

    number_of_gcs = _read_number_of_gcs(OUTPUT_FILE)

    # ==========================================================
    # Run the in situ GC initial conditions
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
            tagging_radius_factor=TAGGING_RADIUS_FACTOR,
            minimum_tagging_radius=MINIMUM_TAGGING_RADIUS,
            circularity_threshold=CIRCULARITY_THRESHOLD,
            n_iter=N_ITER,
            n_particles_per_component=N_PARTICLES_PER_COMPONENT,
            random_seed=RANDOM_SEED,
            keep_agama_file=KEEP_AGAMA_FILE,
        )

        number_of_gcs = len(np.atleast_2d(gc_initial_conditions))

        record_stage_timing(
            records=TIMING_RECORDS,
            galaxy_id=galaxy_id,
            stage="RUN_IN_SITU_ICS",
            start_time=stage_start,
            n_gcs=number_of_gcs,
            n_particles=N_PARTICLES_PER_COMPONENT,
            n_cpus=1,
        )

    if number_of_gcs is None:
        number_of_gcs = _read_number_of_gcs(OUTPUT_FILE)

    # ==========================================================
    # Run the ex situ GC initial conditions
    # ==========================================================

    if RUN_EX_SITU_ICS:
        stage_start = start_stage_timer()

        ex_situ_results = generate_ex_situ_gcs(
            galaxy_id=galaxy_id,
            satellites_file=SATELLITES_FILE,
            satellite_data_directory=SATELLITE_DATA_DIRECTORY,
            output_directory=EX_SITU_OUTPUT_DIRECTORY,
            snapshot_index=SNAPSHOT_INDEX,
            ngc=NGC_EX_SITU,
            alpha=ALPHA_EX_SITU,
            tagging_radius_factor=EX_SITU_TAGGING_RADIUS_FACTOR,
            circularity_threshold=EX_SITU_CIRCULARITY_THRESHOLD,
            n_iter=EX_SITU_N_ITER,
            n_particles_per_component=EX_SITU_N_PARTICLES_PER_COMPONENT,
            random_seed=EX_SITU_RANDOM_SEED,
            keep_agama_files=KEEP_EX_SITU_AGAMA_FILES,
        )

        number_of_ex_situ_gcs = 0

        for gc_file in ex_situ_results["generated_files"]:
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
            n_particles=EX_SITU_N_PARTICLES_PER_COMPONENT,
            n_cpus=1,
        )

    # ==========================================================
    # Run the in-situ dynamics
    # ==========================================================

    if RUN_DYNAMICS:
        stage_start = start_stage_timer()

        run_in_situ_dynamics(
            initial_conditions_file=OUTPUT_FILE,
            timestep_file=TIMESTEP_FILE,
            potential_file=MWPOTS_PATH,
            output_file=DYNAMICS_OUTPUT_FILE,
            plot_file=DYNAMICS_PLOT_FILE,
            start_snapshot_index=SNAPSHOT_INDEX,
            end_snapshot_index=END_SNAPSHOT_INDEX,
            integration_method=INTEGRATION_METHOD,
            potential_mode=POTENTIAL_MODE,
            df_model=DF_MODEL,
            static_potential_index=STATIC_POTENTIAL_INDEX,
            gc_mass=GC_MASS,
            gc_half_mass_radius=GC_HALF_MASS_RADIUS,
            m22=M22,
            df_cache_directory=DF_CACHE_DIRECTORY,
            reuse_df_cache=REUSE_DF_CACHE,
            mass_loss_mode=MASS_LOSS_MODE,
            mass_loss_gamma=MASS_LOSS_GAMMA,
            tidal_strength_reference=TIDAL_STRENGTH_REFERENCE,
            dissolution_time_normalization=DISSOLUTION_TIME_NORMALIZATION,
            central_capture_radius=CENTRAL_CAPTURE_RADIUS,
            generate_gc_moving_potentials=GENERATE_GC_MOVING_POTENTIALS,
            gc_moving_potential_directory=GC_MOVING_POTENTIAL_DIRECTORY,
            gc_potential_scale_radius=GC_POTENTIAL_SCALE_RADIUS,
        )

        record_stage_timing(
            records=TIMING_RECORDS,
            galaxy_id=galaxy_id,
            stage="RUN_DYNAMICS",
            start_time=stage_start,
            n_gcs=number_of_gcs,
            n_particles=None,
            n_cpus=N_CPUS,
        )

    # ==========================================================
    # Run the in-situ streams
    # ==========================================================

    if RUN_STREAMS:
        stage_start = start_stage_timer()

        run_in_situ_streams(
            plummer_file=GC_PLUMMER_FILE,
            timestep_file=TIMESTEP_FILE,
            potential_file=MWPOTS_PATH,
            dynamics_file=DYNAMICS_OUTPUT_FILE,
            gc_moving_potential_directory=GC_MOVING_POTENTIAL_DIRECTORY,
            output_directory=STREAM_OUTPUT_DIRECTORY,
            plot_directory=STREAM_PLOT_DIRECTORY,
            start_snapshot_index=SNAPSHOT_INDEX,
            end_snapshot_index=END_SNAPSHOT_INDEX,
            potential_mode=POTENTIAL_MODE,
            df_model=DF_MODEL,
            mass_loss_mode=MASS_LOSS_MODE,
            integration_method=INTEGRATION_METHOD,
            n_jobs=STREAM_N_JOBS,
            batch_size=STREAM_BATCH_SIZE,
        )

        record_stage_timing(
            records=TIMING_RECORDS,
            galaxy_id=galaxy_id,
            stage="RUN_STREAMS",
            start_time=stage_start,
            n_gcs=number_of_gcs,
            n_particles=N_STREAM_PARTICLES,
            n_cpus=N_CPUS,
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
    "snapshot_index": SNAPSHOT_INDEX,
    "circularity_threshold": CIRCULARITY_THRESHOLD,
    "tagging_radius_factor": TAGGING_RADIUS_FACTOR,
    "n_particles_per_component": N_PARTICLES_PER_COMPONENT,

    # Ex-situ ICs
    "ngc_ex_situ": NGC_EX_SITU,
    "alpha_ex_situ": ALPHA_EX_SITU,
    "ex_situ_tagging_radius_factor": EX_SITU_TAGGING_RADIUS_FACTOR,
    "ex_situ_circularity_threshold": EX_SITU_CIRCULARITY_THRESHOLD,
    "ex_situ_n_particles_per_component": EX_SITU_N_PARTICLES_PER_COMPONENT,

    # Dynamics
    "potential_mode": POTENTIAL_MODE,
    "df_model": DF_MODEL,
    "m22": M22,
    "gc_mass": GC_MASS,
    "gc_half_mass_radius": GC_HALF_MASS_RADIUS,
    "central_capture_radius": CENTRAL_CAPTURE_RADIUS,
    "mass_loss_mode": MASS_LOSS_MODE,

    # Streams
    "n_stream_particles": N_STREAM_PARTICLES,
}

ENABLED_STAGES = {
    "RUN_IN_SITU_ICS": RUN_ICS,
    "RUN_EX_SITU_ICS": RUN_EX_SITU_ICS,
    "RUN_STREAM_ICS": RUN_STREAM_ICS,
    "RUN_DYNAMICS": RUN_DYNAMICS,
    "RUN_STREAMS": RUN_STREAMS,
}

write_timing_report(
    output_file=TIMING_FILE,
    records=TIMING_RECORDS,
    galaxy_ids=GALAXY_IDS,
    run_parameters=RUN_PARAMETERS,
    enabled_stages=ENABLED_STAGES,
    successful_galaxies=successful_galaxies,
    failed_galaxies=failed_galaxies,
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
        + ", ".join(str(galaxy_id) for galaxy_id in failed_galaxies)
    )
