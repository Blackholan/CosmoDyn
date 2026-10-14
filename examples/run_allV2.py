#!/usr/bin/env python3
# coding: utf-8
import numpy as np
from cosmodyn.pipeline import run_pipeline

# ==========================================================
# 1. RUN SELECTION
# ==========================================================

GALAXY_IDS = [613192]
CONTINUE_ON_ERROR = True
RUN_MODE = "full"
ENABLE_STREAMS = True
ENABLE_NSC_STREAMS = False

# ==========================================================
# 2. INITIAL CONDITIONS
# ==========================================================

SNAPSHOT_INDEX = 0
NGC = 0                  # 0 = use the halo-mass relation
ALPHA = 3                # used only when NGC = 0
CIRCULARITY_THRESHOLD = 0.6 # None
TAGGING_RADIUS_FACTOR = 1 
MINIMUM_TAGGING_RADIUS = None  # kpc

N_ITER = 20
N_PARTICLES_PER_COMPONENT = 100_000
RANDOM_SEED = None    
KEEP_AGAMA_FILE = True

# ==========================================================
# NUCLEAR STAR CLUSTERS
# ==========================================================

ENABLE_NSCS = True

NSC_INITIAL_RADIUS = 0.001          # kpc 
NSC_MASS = 0                      # Msun
NSC_HALF_MASS_RADIUS = 0.01        # kpc 
NSC_CENTRAL_CAPTURE_RADIUS = 0.0001 # kpc
GENERATE_NSC_MOVING_POTENTIALS = False

# ==========================================================
# BLACK HOLES
# ==========================================================
ENABLE_BHS = True

BH_MASS = 0                         # Msun
BH_INITIAL_RADIUS = 0.001           # kpc 
BH_CENTRAL_CAPTURE_RADIUS = 0.0001  # kpc

# ==========================================================
# 3. DYNAMICS — SHARED BY IN-SITU AND EX-SITU GCs
# ==========================================================

END_SNAPSHOT_INDEX = None 
INTEGRATION_METHOD = "dop853_c"
TIMESTEP_FILE = "TimeStepGTNG50.txt"

POTENTIAL_MODE = "evolving"  
STATIC_POTENTIAL_INDEX = 73
INCLUDE_MOVING_SATELLITES = False

DF_MODEL = "cdm"             # "none", "cdm" or "fdm"
GC_MASS = 1e6                  # Msun
GC_HALF_MASS_RADIUS = 0.01     # kpc
CENTRAL_CAPTURE_RADIUS = 0.01  # kpc
MASS_LOSS_MODE = "coupled" 

# ==========================================================
# 4. OPTIONAL EX-SITU OVERRIDES
# Ignored in RUN_MODE = "in_situ".
# ==========================================================

NGC_EX_SITU = NGC
ALPHA_EX_SITU = ALPHA
MAXIMUM_SATELLITE_RADIUS = 1000.0  # kpc
RELEASE_ENERGY_TOLERANCE = 0.0


# ==========================================================
# 5. OPTIONAL STREAM PARAMETERS
# Used only when ENABLE_STREAMS = True.
# ==========================================================

N_STREAM_PARTICLES = 100
N_STREAM_ITER = 30
OVERWRITE_STREAM_ICS = False
STREAM_N_JOBS = -1
STREAM_BATCH_SIZE = 64


if __name__ == "__main__":
    run_pipeline(locals())