#!/usr/bin/env python3
# coding: utf-8
"""Nuclear-star-cluster mass prescriptions."""

import numpy as np


NSC_MASS_SCATTER_DEX = 0.6


def NSC_mass(Mstar, rng=None):
    """
    Infer the NSC mass from the host-galaxy stellar mass.

    The relation includes an intrinsic Gaussian scatter of 0.6 dex
    in log10(M_NSC).

    Parameters
    ----------
    Mstar : float or array_like
        Host-galaxy stellar mass in Msun.
    rng : numpy.random.Generator, optional
        Random-number generator. If None, a new generator is created.

    Returns
    -------
    float or ndarray
        NSC mass in Msun.
    """
    stellar_mass = np.asarray(Mstar, dtype=float)

    if np.any(~np.isfinite(stellar_mass)) or np.any(stellar_mass <= 0.0):
        raise ValueError(
            "Stellar mass must be finite and strictly positive."
        )

    if rng is None:
        rng = np.random.default_rng()

    log10_nsc_mass = (
        0.48 * np.log10(stellar_mass / 1.0e9)
        + 6.51
        + rng.normal(
            loc=0.0,
            scale=NSC_MASS_SCATTER_DEX,
            size=stellar_mass.shape,
        )
    )

    nsc_mass = 10.0**log10_nsc_mass

    return float(nsc_mass) if nsc_mass.ndim == 0 else nsc_mass


def resolve_nsc_mass(configured_mass, stellar_mass, rng=None):
    """
    Return a fixed NSC mass or infer it from the stellar mass.

    NSC_MASS = 0 uses the stellar-mass relation and its intrinsic
    scatter, whereas NSC_MASS > 0 imposes a fixed mass in Msun.
    """
    configured_mass = float(configured_mass)

    if not np.isfinite(configured_mass) or configured_mass < 0.0:
        raise ValueError(
            "NSC_MASS must be finite and non-negative."
        )

    if configured_mass > 0.0:
        return configured_mass

    return NSC_mass(stellar_mass, rng=rng)