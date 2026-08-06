#!/usr/bin/env python3
# coding: utf-8
from pathlib import Path
import traceback
from cosmodyn import (generate_in_situ_gcs,run_in_situ_dynamics)

# ==========================================================
# Galaxy parameters
# ==========================================================
# Single galaxy:
GALAXY_IDS = [462710]
# Several galaxies:
# GALAXY_IDS = [613192,462710]
# Or read all galaxy IDs from a text file containing one ID per line:
# GALAXY_IDS = np.loadtxt("GalaxyList.txt", dtype=int).tolist()
CONTINUE_ON_ERROR = True  # continue with the next galaxy if one run fails

# ICs
RUN_ICS = True
SNAPSHOT_INDEX = 8  # index of the snapshot at which the objects are tagged
NGC = 5  # 0 = use the halo-mass relation; >0 = impose this number of GCs
ALPHA = 3  # used only when NGC = 0
CIRCULARITY_THRESHOLD = 0.8 # Select particles with 0.6 <= Lz/Lcirc(E) <= 1
TAGGING_RADIUS_FACTOR = 2 # inside x half-mass radius of stars
N_ITER = 20
N_PARTICLES_PER_COMPONENT = 1_000_00  # use this format, not 1e6
RANDOM_SEED = None  # or 42 to obtain the same sample at each run
KEEP_AGAMA_FILE = True  # keep the AGAMA file after generating the ICs

# Dynamics
RUN_DYNAMICS = True  # integrate the GC orbits after generating the ICs
END_SNAPSHOT_INDEX = None  # None = integrate to the end
INTEGRATION_METHOD = "dop853_c"
TIMESTEP_FILE = f"TimeStepGTNG50.txt" # [start time (Gyr), end time (Gyr), number of integration steps]
POTENTIAL_MODE = "evolving"  # "evolving" or "static"
STATIC_POTENTIAL_INDEX = 73     # used only for "static"


DF_MODEL = "fdm"           # "none", "cdm", or "fdm"
M22 = 1.0                  # 1.0 corresponds to 1e-22 eV for fdm model

# GC parameters
GC_MASS = 1e6                 # Msun, same value for all GCs
GC_HALF_MASS_RADIUS = 0.01      # kpc, same value for all GCs
REUSE_DF_CACHE = True  # Load existing dynamical-friction forces if available instead of recomputing them
CENTRAL_CAPTURE_RADIUS = 0.01  # kpc; capture if the snapshot apocenter is below this radius

# Mass loss parameters
MASS_LOSS_MODE = "coupled"  # "none", "postprocess", or "coupled"
MASS_LOSS_GAMMA = 0.7 # Based on Kruijssen+11 mass loss model
TIDAL_STRENGTH_REFERENCE = 7.01e2 # Based on Kruijssen+11 mass loss model
DISSOLUTION_TIME_NORMALIZATION = 0.0107 # Based on Kruijssen+11 mass loss model


def run_galaxy(galaxy_id):
    """Run the in-situ pipeline for one galaxy."""

    print()
    print("=" * 70)
    print(f"Starting galaxy G{galaxy_id}")
    print("=" * 70)

    # ==========================================================
    # Automatically generated file names
    # ==========================================================
    INPUT_DIR = Path(f"DataG{galaxy_id}")
    OUTPUT_DIR = Path("Outputs") / f"G{galaxy_id}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    MWDATA_PATH = INPUT_DIR / f"DataG{galaxy_id}.txt"
    MWPOTS_PATH = INPUT_DIR / f"PotsG{galaxy_id}.pkl"
    AGAMA_FILE = OUTPUT_DIR / f"ICGCG{galaxy_id}N{N_PARTICLES_PER_COMPONENT:.0e}.h5"
    OUTPUT_FILE = OUTPUT_DIR / f"IniGCG{galaxy_id}.txt" # format [R, vR, vT, z, vz, phi]
    PLOT_FILE = OUTPUT_DIR / f"G{galaxy_id}.png"


    # ==========================================================
    # Run the ICs
    # ==========================================================

    if RUN_ICS:
        generate_in_situ_gcs(
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
            circularity_threshold=CIRCULARITY_THRESHOLD,
            n_iter=N_ITER,
            n_particles_per_component=N_PARTICLES_PER_COMPONENT,
            random_seed=RANDOM_SEED,
            keep_agama_file=KEEP_AGAMA_FILE,
            agama_python_executable=AGAMA_PYTHON_EXECUTABLE)


    DF_CACHE_DIRECTORY = OUTPUT_DIR / "DynamicalFrictionForce"  # Directory where the precomputed dynamical-friction forces are stored

    MODE_TAG = f"{POTENTIAL_MODE}_{DF_MODEL}_{MASS_LOSS_MODE}"
    DYNAMICS_OUTPUT_FILE = OUTPUT_DIR / f"InSituDynamics_{MODE_TAG}_G{galaxy_id}.h5"
    # format [Captured (0 or 1), Mass in Msol, x in kpc, y, z, vR in km/s, vT, vz] for number_of_gcs × number_of_times + Time
    DYNAMICS_PLOT_FILE = OUTPUT_DIR / f"InSituRadiusEvolution_{MODE_TAG}_G{galaxy_id}.png"

    # ==========================================================
    # Run the in-situ dynamics
    # ==========================================================

    if RUN_DYNAMICS:
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
        )

    print(f"Finished galaxy G{galaxy_id}")


# ==========================================================
# Run all galaxies
# ==========================================================

failed_galaxies = []

for GALAXY_ID in GALAXY_IDS:
    try:
        run_galaxy(GALAXY_ID)

    except Exception:
        failed_galaxies.append(GALAXY_ID)

        print()
        print(f"Galaxy G{GALAXY_ID} failed.")
        traceback.print_exc()

        if not CONTINUE_ON_ERROR:
            raise


print()
print("=" * 70)
print("Multi-galaxy run completed.")
print(f"Successful galaxies: {len(GALAXY_IDS) - len(failed_galaxies)}")
print(f"Failed galaxies: {len(failed_galaxies)}")

if failed_galaxies:
    print(
        "Failed galaxy IDs: "
        + ", ".join(str(galaxy_id) for galaxy_id in failed_galaxies)
    )
