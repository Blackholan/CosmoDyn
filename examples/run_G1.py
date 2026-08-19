#!/usr/bin/env python3
# coding: utf-8
import numpy as np

from cosmodyn.pipeline import run_pipeline


# ==========================================================
# 1. RUN SELECTION
# ==========================================================

GALAXY_IDS = [1]
CONTINUE_ON_ERROR = True

# "in_situ": in-situ GCs only
# "ex_situ": ex-situ GCs only
# "full":    in-situ + ex-situ GCs
# "custom":  use the individual stage switches at the end of this file
RUN_MODE = "in_situ"

# Generate and integrate streams for the populations selected by RUN_MODE.
ENABLE_STREAMS = False
ENABLE_NSC_STREAMS = False

# ==========================================================
# 2. INITIAL CONDITIONS
# ==========================================================

SNAPSHOT_INDEX = 0
NGC = 3                  # 0 = use the halo-mass relation
ALPHA = 1                # used only when NGC = 0
CIRCULARITY_THRESHOLD = None # None
TAGGING_RADIUS_FACTOR = 2 
MINIMUM_TAGGING_RADIUS = 0.4  # kpc; None disables the constraint

N_ITER = 20
N_PARTICLES_PER_COMPONENT = 100_000
RANDOM_SEED = None       # for example 42 for reproducible sampling
KEEP_AGAMA_FILE = True

# ==========================================================
# NUCLEAR STAR CLUSTERS
# ==========================================================

ENABLE_NSCS = False

NSC_INITIAL_RADIUS = 0.2          # kpc = 1 pc
NSC_MASS = 5e7                      # Msun
NSC_HALF_MASS_RADIUS = 0.01        # kpc = 5 pc
NSC_CENTRAL_CAPTURE_RADIUS = 0.0001 # kpc = 0.1 pc
GENERATE_NSC_MOVING_POTENTIALS = False

# ==========================================================
# BLACK HOLES
# ==========================================================
ENABLE_BHS = False

BH_MASS = 0
BH_INITIAL_RADIUS = 0.001       # kpc = 1 pc
BH_CENTRAL_CAPTURE_RADIUS = 0.001

# ==========================================================
# 3. DYNAMICS — SHARED BY IN-SITU AND EX-SITU GCs
# ==========================================================

END_SNAPSHOT_INDEX = None  # None = integrate to the final snapshot
INTEGRATION_METHOD = "dop853_c"
TIMESTEP_FILE = "TimeStepG1.txt"

POTENTIAL_MODE = "static"  # "evolving" or "static"
STATIC_POTENTIAL_INDEX = 0   # used only with POTENTIAL_MODE = "static"

# Include the gravitational field of moving satellite galaxies
# in the orbital potential of the in-situ and/or ex-situ GCs.
INCLUDE_MOVING_SATELLITES = False

DF_MODEL = "cdm"             # "none", "cdm" or "fdm"
M22 = 1                       # used only with DF_MODEL = "fdm"

GC_MASS = 1e6                  # Msun
GC_HALF_MASS_RADIUS = 0.01     # kpc
CENTRAL_CAPTURE_RADIUS = 0.01  # kpc

MASS_LOSS_MODE = "coupled"  # "none", "postprocess" or "coupled"


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


# ==========================================================
# 6. CUSTOM STAGES — USED ONLY WITH RUN_MODE = "custom"
# ==========================================================

#Globular clusters
RUN_ICS = False
RUN_EX_SITU_ICS = False
RUN_EX_SITU_SATELLITES = True

RUN_IN_SITU_DYNAMICS = False
RUN_EX_SITU_DYNAMICS = False

RUN_STREAM_ICS = False
RUN_IN_SITU_STREAMS = False
RUN_EX_SITU_STREAMS = False

#Nuclear star clusters
RUN_IN_SITU_NSC_ICS = False
RUN_EX_SITU_NSC_ICS = False

RUN_IN_SITU_NSC_DYNAMICS = False
RUN_EX_SITU_NSC_DYNAMICS = False

RUN_NSC_STREAM_ICS = False
RUN_IN_SITU_NSC_STREAMS = False
RUN_EX_SITU_NSC_STREAMS = False

#Black holes
RUN_IN_SITU_BH_ICS = True
RUN_EX_SITU_BH_ICS = True
RUN_IN_SITU_BH_DYNAMICS = True
RUN_EX_SITU_BH_DYNAMICS = True


if __name__ == "__main__":
    run_pipeline(locals())