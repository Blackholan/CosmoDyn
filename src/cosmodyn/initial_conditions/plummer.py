#!/usr/bin/env python3
# coding: utf-8

"""Generate equilibrium Plummer GC particle initial conditions with AGAMA."""

from pathlib import Path
import subprocess

from .in_situ import find_agama_python


def generate_plummer_gc(
    output_file,
    gc_mass=1.0e6,
    gc_half_mass_radius=0.01,
    n_particles=100_000,
    n_iter=30,
    agama_builder_path=None,
    agama_python_executable=None,
    overwrite=False,
):
    """Generate one equilibrium Plummer particle distribution for a GC."""
    output_file = Path(output_file)

    if gc_mass <= 0:
        raise ValueError("gc_mass must be strictly positive.")
    if gc_half_mass_radius <= 0:
        raise ValueError("gc_half_mass_radius must be strictly positive.")
    if int(n_particles) <= 0:
        raise ValueError("n_particles must be strictly positive.")
    if int(n_iter) <= 0:
        raise ValueError("n_iter must be strictly positive.")

    if output_file.exists() and not overwrite:
        print(f"Plummer GC particle file already exists: {output_file}")
        print("AGAMA will not be run again.")
        return output_file

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if agama_builder_path is None:
        agama_builder_path = Path(__file__).with_name(
            "agama_plummer_builder.py"
        )
    else:
        agama_builder_path = Path(agama_builder_path)

    if not agama_builder_path.exists():
        raise FileNotFoundError(
            f"AGAMA Plummer builder script not found: {agama_builder_path}"
        )

    python_executable = find_agama_python(
        agama_python_executable=agama_python_executable,
    )

    command = [
        str(python_executable),
        str(agama_builder_path),
        "--output",
        str(output_file),
        "--mass",
        str(float(gc_mass)),
        "--scale-radius",
        str(float(gc_half_mass_radius)),
        "--n-particles",
        str(int(n_particles)),
        "--n-iter",
        str(int(n_iter)),
    ]

    print("Generating equilibrium Plummer GC particle distribution.")
    print(f"GC mass: {gc_mass:.6e} Msun")
    print(f"GC Plummer scale radius: {gc_half_mass_radius:.6e} kpc")
    print(f"Number of particles: {int(n_particles)}")
    print(f"AGAMA Python executable: {python_executable}")

    subprocess.run(command, check=True)

    if not output_file.exists():
        raise RuntimeError(
            "The AGAMA process finished without creating the expected "
            f"Plummer particle file: {output_file}"
        )

    print(f"Plummer GC particle file created: {output_file}")
    return output_file
