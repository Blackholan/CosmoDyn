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

    # np.loadtxt returns a one-dimensional array for a one-row file.
    # Converting it explicitly to 2D makes both one- and multi-row inputs work.
    A = np.atleast_2d(np.loadtxt(mwdata_path))

    if A.shape[1] < 11:
        raise ValueError(
            f"{mwdata_path} must contain at least 11 columns; "
            f"found {A.shape[1]}."
        )

    if snapshot_index < 0 or snapshot_index >= A.shape[0]:
        raise IndexError(
            f"Snapshot index {snapshot_index} is outside the available "
            f"row range 0-{A.shape[0] - 1} in {mwdata_path}."
        )

    mass_dm = A[snapshot_index, 4]
    rs_dm = A[snapshot_index, 9]
    mass_star = A[snapshot_index, 6]
    r_hm_star = A[snapshot_index, 10]

    if not np.isfinite(mass_dm) or mass_dm <= 0.0:
        raise ValueError(
            f"The dark-matter mass must be finite and positive; got {mass_dm}."
        )
    if not np.isfinite(rs_dm) or rs_dm <= 0.0:
        raise ValueError(
            f"The NFW scale radius must be finite and positive; got {rs_dm}."
        )

    has_stellar_component = (
        np.isfinite(mass_star)
        and mass_star > 0.0
        and np.isfinite(r_hm_star)
        and r_hm_star > 0.0
    )

    if has_stellar_component:
        reference_radius = r_hm_star
    else:
        reference_radius = rs_dm

    tagging_radius = (
        tagging_radius_factor * reference_radius
    )

    maximum_spherical_radius = (
        tagging_radius_factor + 1.0
    ) * reference_radius

    density_halo = agama.Density(
        Type="NFW",
        mass=mass_dm,
        scaleRadius=rs_dm,
    )

    model = agama.SelfConsistentModel(
        rminSph=0.01,
        rmaxSph=maximum_spherical_radius,
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

    if has_stellar_component:
        stellar_scale_radius = r_hm_star * (np.sqrt(2.0) - 1.0)
        density_star = agama.Density(
            Type="Dehnen",
            mass=mass_star,
            gamma=1.0,
            axisRatioY=1,
            axisRatioZ=1,
            scaleRadius=stellar_scale_radius,
        )
        model.components.append(
            agama.Component(density=density_star, disklike=False)
        )
        print("AGAMA model: NFW halo + stellar Dehnen component.")
    else:
        print("AGAMA model: NFW halo only.")
        print("No valid stellar mass and half-mass radius were found.")

    for i in range(n_iter):
        print(f"Starting AGAMA iteration {i + 1}/{n_iter}", flush=True)
        model.iterate()

    df_halo = agama.DistributionFunction(
        type="QuasiSpherical",
        potential=model.potential,
        density=density_halo,
    )
    model.components[0] = agama.Component(
        df=df_halo,
        disklike=False,
        rminSph=0.1,
        rmaxSph=maximum_spherical_radius,
        sizeRadialSph=50,
        lmaxAngularSph=4,
    )
    if has_stellar_component:
        df_sampling = agama.DistributionFunction(
            type="QuasiSpherical",
            potential=model.potential,
            density=density_star,
        )
        model.components[1] = agama.Component(
            df=df_sampling,
            disklike=False,
            rminSph=0.1,
            rmaxSph=maximum_spherical_radius,
            sizeRadialSph=50,
            lmaxAngularSph=4,
        )
        print("Sampling the stellar distribution function.")
    else:
        df_sampling = df_halo
        print("Sampling the NFW halo distribution function.")

    print(f"Tagging radius: {tagging_radius:.6g}")

    galaxy_model = agama.GalaxyModel(
        potential=model.potential,
        df=df_sampling,
        af=model.af,
    )
    dat = galaxy_model.sample(n_particles_per_component)

    df_particles = pd.DataFrame(
        dat[0][:, :6],
        columns=["x", "y", "z", "u", "v", "w"],
    )
    df_particles["r"] = np.sqrt(
        df_particles["x"] ** 2
        + df_particles["y"] ** 2
        + df_particles["z"] ** 2
    )

    df_gc = df_particles[df_particles["r"] < tagging_radius]

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
    parser.add_argument("--tagging-radius-factor", type=float, default=1.0)
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
