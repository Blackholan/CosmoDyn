#!/usr/bin/env python3
# coding: utf-8
"""AGAMA-only process used to avoid loading AGAMA and galpy in the same Python process."""

import argparse
import os
import warnings

import agama
import h5py
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
agama.setUnits(mass=1.0, length=1.0, velocity=1.0)


def generate_agama_file(
    mwdata_path,
    agama_file,
    snapshot_index=8,
    tagging_radius_factor=1.0,
    n_iter=20,
    n_particles_per_component=1_000_000,
):
    if os.path.exists(agama_file):
        print(f"AGAMA particle file already exists: {agama_file}")
        return

    A = np.loadtxt(mwdata_path)

    r_hm1 = A[:, 10][snapshot_index]
    rc = r_hm1 * (np.sqrt(2.0) - 1.0)
    mass_bulge = A[:, 6][snapshot_index]
    mass_dm = A[:, 4][snapshot_index]
    rs_dm = A[:, 9][snapshot_index]

    density_halo = agama.Density(
        Type="NFW",
        mass=mass_dm,
        scaleRadius=rs_dm,
    )
    density_star = agama.Density(
        Type="Dehnen",
        mass=mass_bulge,
        gamma=1.0,
        axisRatioY=1,
        axisRatioZ=1,
        scaleRadius=rc,
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

    for i in range(n_iter):
        print(f"Starting AGAMA iteration {i + 1}/{n_iter}", flush=True)
        model.iterate()

    df_halo = agama.DistributionFunction(
        type="QuasiSpherical",
        potential=model.potential,
        density=density_halo,
    )
    df_star_agama = agama.DistributionFunction(
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
        df=df_star_agama,
        disklike=False,
        rminSph=0.1,
        rmaxSph=500.0,
        sizeRadialSph=50,
        lmaxAngularSph=4,
    )

    galaxy_model_star = agama.GalaxyModel(
        potential=model.potential,
        df=df_star_agama,
        af=model.af,
    )
    dat = galaxy_model_star.sample(n_particles_per_component)

    df_star = pd.DataFrame(
        dat[0][:, :6],
        columns=["x", "y", "z", "u", "v", "w"],
    )
    df_star["r"] = np.sqrt(
        df_star["x"] ** 2
        + df_star["y"] ** 2
        + df_star["z"] ** 2
    )

    tagging_radius = tagging_radius_factor * r_hm1

    df_gc = df_star[df_star["r"] < tagging_radius]

    output_directory = os.path.dirname(agama_file)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    with h5py.File(agama_file, "w") as f:
        g = f.create_group("GCdata")
        g.create_dataset("PosX", data=df_gc["x"].values)
        g.create_dataset("PosY", data=df_gc["y"].values)
        g.create_dataset("PosZ", data=df_gc["z"].values)
        g.create_dataset("vX", data=df_gc["u"].values)
        g.create_dataset("vY", data=df_gc["v"].values)
        g.create_dataset("vZ", data=df_gc["w"].values)

    print(f"AGAMA particle file created: {agama_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mwdata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snapshot-index", type=int, default=8)
    parser.add_argument("--tagging-radius-factor", type=int, default=1)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--n-particles", type=int, default=1_000_000)
    args = parser.parse_args()

    generate_agama_file(
        mwdata_path=args.mwdata,
        agama_file=args.output,
        snapshot_index=args.snapshot_index,
        tagging_radius_factor=args.tagging_radius_factor,
        n_iter=args.n_iter,
        n_particles_per_component=args.n_particles,
    )


if __name__ == "__main__":
    main()
