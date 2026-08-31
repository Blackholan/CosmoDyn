#!/usr/bin/env python3
# coding: utf-8
"""Nuclear-star-cluster mass prescriptions."""

import numpy as np


def NSC_mass(Mstar):
    """Return the NSC mass inferred from stellar mass, in solar masses."""
    stellar_mass = np.asarray(Mstar, dtype=float)
    if np.any(~np.isfinite(stellar_mass)) or np.any(stellar_mass <= 0.0):
        raise ValueError("Stellar mass must be finite and strictly positive.")

    result = 10.0 ** (
        0.48 * np.log10(stellar_mass / 1.0e9) + 6.51
    )
    return float(result) if result.ndim == 0 else result


def resolve_nsc_mass(configured_mass, stellar_mass):
    """Resolve a fixed NSC mass or infer it from the stellar mass."""
    configured_mass = float(configured_mass)
    if not np.isfinite(configured_mass) or configured_mass < 0.0:
        raise ValueError("NSC_MASS must be finite and non-negative.")
    if configured_mass > 0.0:
        return configured_mass
    return NSC_mass(stellar_mass)
