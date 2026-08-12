#!/usr/bin/env python3
# coding: utf-8

"""AGAMA-only subprocess for ex-situ GC initial conditions."""

import argparse
import os
import warnings

import agama
import h5py
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
agama.setUnits(mass=1.0, length=1.0, velocity=1.0)


def generate_ex_situ_agama_file(
    agama_file,
    mass_dm,
    mass_star,
    dm_scale_radius,
    stellar_half_mass_radius,
    tagging_radius_factor=1.0,
    n_iter=20,
    n_particles_per_component=1_000_000,
):
    """Build and sample one satellite stellar distribution with AGAMA."""

    if os.path.exists(agama_file):
        print(f"AGAMA satellite particle file already exists: {agama_file}")
        return

    if mass_dm <= 0.0 or mass_star <= 0.0:
        raise ValueError("Satellite masses must be strictly positive.")
    if dm_scale_radius <= 0.0 or stellar_half_mass_radius <= 0.0:
        raise ValueError("Satellite radii must be strictly positive.")
    if tagging_radius_factor <= 0.0:
        raise ValueError("tagging_radius_factor must be strictly positive.")

    stellar_scale_radius = (
        stellar_half_mass_radius * (np.sqrt(2.0) - 1.0)
    )

    density_halo = agama.Density(
        Type="NFW",
        mass=mass_dm,
        scaleRadius=dm_scale_radius,
    )

    density_star = agama.Density(
        Type="Dehnen",
        mass=mass_star,
        gamma=1.0,
        axisRatioY=1,
        axisRatioZ=1,
        scaleRadius=stellar_scale_radius,
    )

    model = agama.SelfConsistentModel(
        rminSph=0.01,
        rmaxSph=300.0,
        sizeRadialSph=50,
        lmaxAngularSph=4,
        RminCyl=0.2,
        RmaxCyl=50,
        sizeRadialCyl=30,
        zminCyl=0.04,
        zmaxCyl=10,
        sizeVerticalCyl=30,
        useActionInterpolation=False,
    )

    model.components.append(
        agama.Component(density=density_halo, disklike=False)
    )
    model.components.append(
        agama.Component(density=density_star, disklike=False)
    )

    for iteration in range(int(n_iter)):
        print(
            f"Starting AGAMA satellite iteration "
            f"{iteration + 1}/{int(n_iter)}",
            flush=True,
        )
        model.iterate()

    df_halo = agama.DistributionFunction(
        type="QuasiSpherical",
        potential=model.potential,
        density=density_halo,
    )

    df_star = agama.DistributionFunction(
        type="QuasiSpherical",
        potential=model.potential,
        density=density_star,
    )

    model.components[0] = agama.Component(
        df=df_halo,
        disklike=False,
        rminSph=0.1,
        rmaxSph=500.0,
        sizeRadialSph=50,
        lmaxAngularSph=4,
    )

    model.components[1] = agama.Component(
        df=df_star,
        disklike=False,
        rminSph=0.1,
        rmaxSph=500.0,
        sizeRadialSph=50,
        lmaxAngularSph=4,
    )

    galaxy_model_star = agama.GalaxyModel(
        potential=model.potential,
        df=df_star,
        af=model.af,
    )

    sampled = galaxy_model_star.sample(int(n_particles_per_component))

    dataframe = pd.DataFrame(
        sampled[0][:, :6],
        columns=["x", "y", "z", "u", "v", "w"],
    )

    dataframe["r"] = np.sqrt(
        dataframe["x"] ** 2
        + dataframe["y"] ** 2
        + dataframe["z"] ** 2
    )

    tagging_radius = tagging_radius_factor * stellar_half_mass_radius
    candidates = dataframe[dataframe["r"] < tagging_radius]

    output_directory = os.path.dirname(agama_file)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    with h5py.File(agama_file, "w") as h5:
        group = h5.create_group("GCdata")
        group.create_dataset("PosX", data=candidates["x"].values)
        group.create_dataset("PosY", data=candidates["y"].values)
        group.create_dataset("PosZ", data=candidates["z"].values)
        group.create_dataset("vX", data=candidates["u"].values)
        group.create_dataset("vY", data=candidates["v"].values)
        group.create_dataset("vZ", data=candidates["w"].values)

        h5.attrs["mass_dm_msun"] = float(mass_dm)
        h5.attrs["mass_star_msun"] = float(mass_star)
        h5.attrs["dm_scale_radius_kpc"] = float(dm_scale_radius)
        h5.attrs["stellar_half_mass_radius_kpc"] = float(
            stellar_half_mass_radius
        )
        h5.attrs["tagging_radius_factor"] = float(tagging_radius_factor)
        h5.attrs["n_particles_per_component"] = int(
            n_particles_per_component
        )
        h5.attrs["n_iter"] = int(n_iter)

    print(f"AGAMA satellite particle file created: {agama_file}")
    print(
        "Candidate stellar particles after spatial tagging: "
        f"{len(candidates)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--mass-dm", type=float, required=True)
    parser.add_argument("--mass-star", type=float, required=True)
    parser.add_argument("--dm-scale-radius", type=float, required=True)
    parser.add_argument("--stellar-half-mass-radius", type=float, required=True)
    parser.add_argument("--tagging-radius-factor", type=float, default=1.0)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--n-particles", type=int, default=1_000_000)

    args = parser.parse_args()

    generate_ex_situ_agama_file(
        agama_file=args.output,
        mass_dm=args.mass_dm,
        mass_star=args.mass_star,
        dm_scale_radius=args.dm_scale_radius,
        stellar_half_mass_radius=args.stellar_half_mass_radius,
        tagging_radius_factor=args.tagging_radius_factor,
        n_iter=args.n_iter,
        n_particles_per_component=args.n_particles,
    )


if __name__ == "__main__":
    main()
