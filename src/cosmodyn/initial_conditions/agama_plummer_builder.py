#!/usr/bin/env python3
# coding: utf-8

"""AGAMA subprocess used to generate equilibrium Plummer GC particles."""

import argparse
from pathlib import Path
import warnings

import agama
import h5py
import numpy as np


def build_plummer_particles(
    output_file,
    gc_mass,
    scale_radius,
    n_particles,
    n_iter,
):
    """Build and sample an equilibrium Plummer model with AGAMA."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    agama.setUnits(mass=1.0, length=1.0, velocity=1.0)
    warnings.filterwarnings("ignore")

    density_gc = agama.Density(
        Type="Plummer",
        mass=float(gc_mass),
        scaleRadius=float(scale_radius),
    )

    model = agama.SelfConsistentModel(
        rminSph=0.001,
        rmaxSph=1.0,
        sizeRadialSph=50,
        lmaxAngularSph=4,
        RminCyl=0.001,
        RmaxCyl=0.1,
        sizeRadialCyl=30,
        zminCyl=0.001,
        zmaxCyl=0.1,
        sizeVerticalCyl=30,
        useActionInterpolation=False,
    )

    model.components.append(
        agama.Component(density=density_gc, disklike=False)
    )

    for iteration in range(int(n_iter)):
        print(
            f"Starting AGAMA Plummer iteration "
            f"{iteration + 1}/{int(n_iter)}"
        )
        model.iterate()

    df_gc = agama.DistributionFunction(
        type="QuasiSpherical",
        potential=model.potential,
        density=density_gc,
    )

    model.components[0] = agama.Component(
        df=df_gc,
        disklike=False,
        rminSph=0.001,
        rmaxSph=0.1,
        sizeRadialSph=50,
        lmaxAngularSph=4,
    )

    galaxy_model_gc = agama.GalaxyModel(
        potential=model.potential,
        df=df_gc,
        af=model.af,
    )

    sampled_data = galaxy_model_gc.sample(int(n_particles))
    phase_space = np.asarray(sampled_data[0])

    if phase_space.ndim != 2 or phase_space.shape[1] < 6:
        raise RuntimeError(
            "AGAMA returned an unexpected phase-space array shape: "
            f"{phase_space.shape}"
        )

    x = phase_space[:, 0]
    y = phase_space[:, 1]
    z = phase_space[:, 2]
    vx = phase_space[:, 3]
    vy = phase_space[:, 4]
    vz = phase_space[:, 5]

    radius = np.sqrt(x**2 + y**2 + z**2)

    print(f"Sampled {len(x)} particles from the Plummer GC model.")

    with h5py.File(output_file, "w") as h5:
        group = h5.create_group("GCdata")
        group.create_dataset("PosX", data=x)
        group.create_dataset("PosY", data=y)
        group.create_dataset("PosZ", data=z)
        group.create_dataset("vX", data=vx)
        group.create_dataset("vY", data=vy)
        group.create_dataset("vZ", data=vz)
        group.create_dataset("Radius", data=radius)

        h5.attrs["gc_mass_msun"] = float(gc_mass)
        h5.attrs["plummer_scale_radius_kpc"] = float(scale_radius)
        h5.attrs["n_particles"] = int(n_particles)
        h5.attrs["n_iter"] = int(n_iter)

    print(f"Plummer particle distribution saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate an equilibrium Plummer GC particle distribution "
            "with AGAMA."
        )
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--mass", required=True, type=float)
    parser.add_argument("--scale-radius", required=True, type=float)
    parser.add_argument("--n-particles", required=True, type=int)
    parser.add_argument("--n-iter", default=30, type=int)
    args = parser.parse_args()

    build_plummer_particles(
        output_file=args.output,
        gc_mass=args.mass,
        scale_radius=args.scale_radius,
        n_particles=args.n_particles,
        n_iter=args.n_iter,
    )


if __name__ == "__main__":
    main()
