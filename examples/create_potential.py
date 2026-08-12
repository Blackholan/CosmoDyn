from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from astropy import units
from colossus.cosmology import cosmology
from colossus.halo import concentration, profile_nfw
from galpy.potential import HernquistPotential, NFWPotential, vcirc


# -----------------------------------------------------------------------------
# User parameters
# -----------------------------------------------------------------------------

GALAXY_ID = 1
DM_HALO_MASS = 1e9 # Msol
STELLAR_MASS = 1e7 # Msol
REDSHIFT = 0.0
STELLAR_SCALE_RADIUS = 0.8 #kpc

# WMAP7 has h=0.704, as assumed by the mass and radius conversions below.
COLOSSUS_COSMOLOGY = "WMAP7"
HUBBLE_PARAMETER = 0.704
CONCENTRATION_MODEL = "prada12"

OUTPUT_DIRECTORY = (Path(__file__).resolve().parent.parent/ f"DataG{GALAXY_ID}")
POTENTIAL_FILE = (OUTPUT_DIRECTORY / f"PotsG{GALAXY_ID}.pkl")
DATA_FILE = OUTPUT_DIRECTORY / f"DataG{GALAXY_ID}.txt"

def build_potential():
    """Return the two-component galpy potential used in this example."""

    halo_mass = DM_HALO_MASS
    stellar_mass = STELLAR_MASS

    cosmology.setCosmology(COLOSSUS_COSMOLOGY)

    # Colossus expects halo masses in Msun/h and returns radii in kpc/h.
    halo_mass_colossus = halo_mass * HUBBLE_PARAMETER
    halo_concentration = concentration.concentration(
        halo_mass_colossus,
        "vir",
        REDSHIFT,
        model=CONCENTRATION_MODEL,
    )

    nfw_profile = profile_nfw.NFWProfile(
        M=halo_mass_colossus,
        c=halo_concentration,
        z=REDSHIFT,
        mdef="vir",
    )

    halo_scale_radius = nfw_profile.par["rs"] / HUBBLE_PARAMETER
    nfw_mass_factor = (
        np.log(1.0 + halo_concentration)
        - halo_concentration / (1.0 + halo_concentration)
    )
    halo_amplitude = halo_mass / nfw_mass_factor

    dark_matter_potential = NFWPotential(
        amp=halo_amplitude * units.Msun,
        a=halo_scale_radius * units.kpc,
    )

    # In galpy's Hernquist convention, amp=2*M gives a total mass M.
    stellar_potential = HernquistPotential(
        amp=2.0 * stellar_mass * units.Msun,
        a=STELLAR_SCALE_RADIUS * units.kpc,
    )


    print(f"Halo mass:            {halo_mass:.3e} Msun")
    print(f"Stellar mass:         {stellar_mass:.3e} Msun")
    print(f"Redshift:             {REDSHIFT:.3f}")
    print(f"Halo concentration:   {halo_concentration:.3f}")
    print(f"NFW scale radius:     {halo_scale_radius:.3f} kpc")

    stellar_amplitude = 2.0 * stellar_mass

    data1 = np.array(
        [0,np.nan,
            REDSHIFT,
            GALAXY_ID,
            halo_mass,
            halo_amplitude,
            np.nan,
            np.nan,
            np.nan,
            halo_scale_radius,
            np.nan,
        ],
        dtype=float,
    )

    data1 = np.vstack(
        [
            data1,
            data1,
        ]
    )

    data2 = np.array(
        [0,np.nan,
            REDSHIFT,
            GALAXY_ID,
            halo_mass,
            halo_amplitude,
            stellar_mass,
            stellar_amplitude,
            np.nan,
            halo_scale_radius,
            STELLAR_SCALE_RADIUS,
        ],
        dtype=float,
    )

    data2 = np.vstack(
        [
            data2,
            data2,
        ]
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savetxt(
        DATA_FILE,
        data2.reshape(1, -1), #data1 for 1 component and data2 for 2 components
        fmt="%.8e",
    )

    #return [dark_matter_potential, stellar_potential]
    return dark_matter_potential #if only one component


def save_potential(potential, output_file):
    """Save a galpy potential to a pickle file."""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as stream:
        pickle.dump(potential, stream)


def load_potential(input_file):
    """Load a galpy potential and restore compatibility attributes."""

    with input_file.open("rb") as stream:
        potential = pickle.load(stream)

    components = potential if isinstance(potential, (list, tuple)) else [potential]
    for component in components:
        component.isDissipative = False

    return potential


def main():
    potential = build_potential()
    save_potential(potential, POTENTIAL_FILE)

    # Reload the file deliberately: the plot therefore verifies that the saved
    # potential can immediately be reused by another script.

    print(f"Potential saved to: {POTENTIAL_FILE}")

if __name__ == "__main__":
    main()
