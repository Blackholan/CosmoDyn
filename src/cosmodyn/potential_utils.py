#!/usr/bin/env python3
# coding: utf-8
"""Compatibility helpers for galpy potential containers."""


def potential_components(potential):
    """Return components from a list, tuple, CompositePotential, or scalar."""
    if isinstance(potential, (list, tuple)):
        return list(potential)
    class_name = type(potential).__name__
    if class_name.endswith("CompositePotential"):
        try:
            return list(potential)
        except TypeError as error:
            raise TypeError(
                f"Unable to extract components from {class_name}."
            ) from error
    return [potential]


def set_non_dissipative(potential):
    """Mark every conservative potential component as non-dissipative."""
    components = potential_components(potential)
    for component in components:
        component.isDissipative = False
    return components


def combine_potential_and_force(potential, force):
    """Return a flat galpy-compatible list containing an optional force."""
    if force is None:
        return potential
    return potential_components(potential) + [force]


def vcirc_compatible_potential(potential):
    """Return a potential container from which galpy can infer its scales.

    CompositePotential objects are preserved. Legacy lists and tuples are
    combined using galpy's addition interface before being passed to vcirc.
    """
    if type(potential).__name__.endswith("CompositePotential"):
        return potential

    components = potential_components(potential)
    if not components:
        raise ValueError("A potential must contain at least one component.")
    if len(components) == 1:
        return components[0]

    combined = components[0] + components[1]
    for component in components[2:]:
        if isinstance(combined, list):
            combined = combined + [component]
        else:
            combined = combined + component
    return combined
