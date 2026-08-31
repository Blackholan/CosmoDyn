#!/usr/bin/env python3
# coding: utf-8
import numpy as np

from cosmodyn.pipeline import run_pipeline


# ==========================================================
# 1. RUN SELECTION
# ==========================================================

GALAXY_IDS = [613192] # [613192, 462710]
# GALAXY_IDS = np.loadtxt("GalaxyList.txt", dtype=int).tolist()
CONTINUE_ON_ERROR = True # Continue with the remaining galaxies if one galaxy fails
RUN_MODE = "custom" # "in_situ", "ex_situ", "full", "custom"
ENABLE_STREAMS = True
ENABLE_NSC_STREAMS = False

# ==========================================================
# 2. INITIAL CONDITIONS
# ==========================================================

SNAPSHOT_INDEX = 0
NGC = 2                  # 0: the halo-mass relation; >0: the number directly
ALPHA = 3                # used only when NGC = 0
CIRCULARITY_THRESHOLD = 0.6  # Minimum Lz/Lcirc; None disables the circularity selection
TAGGING_RADIUS_FACTOR = 1 # Maximum tagging radius in units of the stellar half-mass radius
MINIMUM_TAGGING_RADIUS = None  # Minimum tagging radius in kpc; None disables this lower limit

# AGAMA parameters
N_ITER = 20
N_PARTICLES_PER_COMPONENT = 100_000
RANDOM_SEED = None    
KEEP_AGAMA_FILE = True # Keep the generated AGAMA particle file for reuse

# ==========================================================
# NUCLEAR STAR CLUSTERS
# ==========================================================

ENABLE_NSCS = True

NSC_INITIAL_RADIUS = 0.001          # kpc 
NSC_MASS = 0                        # Msun
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

END_SNAPSHOT_INDEX = None # Final snapshot index; None integrates to the last available snapshot
INTEGRATION_METHOD = "dop853_c"
TIMESTEP_FILE = "TimeStepGTNG50.txt"

POTENTIAL_MODE = "evolving"  # "evolving" or "static"
STATIC_POTENTIAL_INDEX = 73 # used only with POTENTIAL_MODE = "static"
INCLUDE_MOVING_SATELLITES = True # include the gravitational field of moving satellite galaxies

DF_MODEL = "cdm"             # "none", "cdm", "binney", "fdm"
M22 = 1                       # used only with DF_MODEL = "fdm"
REUSE_DF_CACHE = True

GC_MASS = 1e6                  # Msun
GC_HALF_MASS_RADIUS = 0.01     # kpc
CENTRAL_CAPTURE_RADIUS = 0.001  # kpc
MASS_LOSS_MODE = "coupled" # "none", "postprocess" or "coupled"

# Parameters of the Kruijssen mass-loss prescription
MASS_LOSS_GAMMA = 0.7
TIDAL_STRENGTH_REFERENCE = 7.01e2
DISSOLUTION_TIME_NORMALIZATION = 0.0107

GENERATE_GC_MOVING_POTENTIALS = True

# ==========================================================
# 4. OPTIONAL EX-SITU OVERRIDES
# Ignored in RUN_MODE = "in_situ".
# ==========================================================

NGC_EX_SITU = NGC
ALPHA_EX_SITU = ALPHA
EX_SITU_CIRCULARITY_THRESHOLD = CIRCULARITY_THRESHOLD
EX_SITU_TAGGING_RADIUS_FACTOR = TAGGING_RADIUS_FACTOR
EX_SITU_MINIMUM_TAGGING_RADIUS = MINIMUM_TAGGING_RADIUS
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


# ==========================================================
# 6. CUSTOM STAGES — USED ONLY WITH RUN_MODE = "custom"
# ==========================================================

#Globular clusters
RUN_ICS = True
RUN_EX_SITU_ICS = True
RUN_EX_SITU_SATELLITES = True

RUN_IN_SITU_DYNAMICS = True
RUN_EX_SITU_DYNAMICS = True

RUN_STREAM_ICS = True
RUN_IN_SITU_STREAMS = True
RUN_EX_SITU_STREAMS = True

#Nuclear star clusters
RUN_IN_SITU_NSC_ICS = False
RUN_EX_SITU_NSC_ICS = True

RUN_IN_SITU_NSC_DYNAMICS = False
RUN_EX_SITU_NSC_DYNAMICS = True

RUN_NSC_STREAM_ICS = True
RUN_IN_SITU_NSC_STREAMS = False
RUN_EX_SITU_NSC_STREAMS = True

#Black holes
RUN_IN_SITU_BH_ICS = False
RUN_EX_SITU_BH_ICS = True
RUN_IN_SITU_BH_DYNAMICS = False
RUN_EX_SITU_BH_DYNAMICS = True


if __name__ == "__main__":
    run_pipeline(locals())