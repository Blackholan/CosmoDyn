#!/usr/bin/env python3
# coding: utf-8
"""
In-situ initial-condition pipeline compatible with macOS/OpenMP.

AGAMA is executed in a separate Python process so that AGAMA and galpy
never load their OpenMP runtimes in the same process.
"""

from __future__ import division

import json
import os
import pickle
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from astropy import units
from galpy.potential import evaluatePotentials, vcirc
from matplotlib.patches import Circle

def vcirc_fixed(r, pot):
    return vcirc(
        pot,
        np.asarray(r) * units.kpc)


def number_of_globular_clusters(m_vir):
    return 10 ** (-9.58 + 0.99 * np.log10(m_vir))



def _agama_config_file():
    """Return the user-level CosmoDyn configuration file."""
    return Path.home() / ".cosmodyn" / "config.json"


def _load_saved_agama_python():
    """Load a previously validated AGAMA Python interpreter, if available."""
    config_file = _agama_config_file()

    if not config_file.exists():
        return None

    try:
        with config_file.open("r", encoding="utf-8") as stream:
            config = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None

    executable = config.get("agama_python_executable")

    if not executable:
        return None

    return Path(executable).expanduser()


def _save_agama_python(python_executable):
    """Save a validated AGAMA Python interpreter for future runs."""
    config_file = _agama_config_file()
    config_file.parent.mkdir(parents=True, exist_ok=True)

    config = {}

    if config_file.exists():
        try:
            with config_file.open("r", encoding="utf-8") as stream:
                config = json.load(stream)
        except (OSError, json.JSONDecodeError):
            config = {}

    config["agama_python_executable"] = str(python_executable)

    with config_file.open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)


def _python_can_import_agama(python_executable):
    """Return True if the supplied Python interpreter can import AGAMA."""
    python_executable = Path(python_executable).expanduser()

    if not python_executable.exists():
        return False

    try:
        result = subprocess.run(
            [
                str(python_executable),
                "-c",
                "import agama",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False

    return result.returncode == 0


def _candidate_python_executables():
    """
    Return likely Python interpreters without duplicates.

    The search explicitly checks both the currently active interpreter
    (including a virtual environment) and the base/system interpreter.
    """
    candidates = []

    # 1. Current interpreter.
    #    If CosmoDyn is running inside a virtual environment, this is the
    #    virtual-environment Python and must always be tested first.
    candidates.append(Path(sys.executable))

    # 2. Previously validated AGAMA interpreter.
    saved_executable = _load_saved_agama_python()
    if saved_executable is not None:
        candidates.append(saved_executable)

    # 3. Base/system interpreter used to create the virtual environment.
    base_executable = getattr(sys, "_base_executable", None)
    if base_executable:
        candidates.append(Path(base_executable))

    # 4. Reconstruct possible base interpreters from sys.base_prefix.
    base_prefix = Path(sys.base_prefix)
    for executable_name in ("python3", "python"):
        candidates.append(base_prefix / "bin" / executable_name)

    # 5. Python executables visible through the current PATH.
    for command_name in ("python3", "python"):
        executable = shutil.which(command_name)
        if executable is not None:
            candidates.append(Path(executable))

    # 6. Common Unix/macOS system locations.
    for executable in (
        Path("/usr/bin/python3"),
        Path("/usr/local/bin/python3"),
        Path("/opt/homebrew/bin/python3"),
    ):
        candidates.append(executable)

    # 7. Common Python.org framework installations on macOS.
    framework_root = Path("/Library/Frameworks/Python.framework/Versions")
    if framework_root.exists():
        for executable in sorted(
            framework_root.glob("*/bin/python3"),
            reverse=True,
        ):
            candidates.append(executable)

    unique_candidates = []
    seen = set()

    for candidate in candidates:
        if candidate is None:
            continue

        candidate = Path(candidate).expanduser()

        if not candidate.exists():
            continue

        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate

        candidate_key = str(resolved)

        if candidate_key not in seen:
            seen.add(candidate_key)
            unique_candidates.append(resolved)

    return unique_candidates

def find_agama_python(agama_python_executable=None):
    """
    Find a Python interpreter capable of importing AGAMA.

    A manually supplied interpreter has priority. Otherwise, CosmoDyn
    checks the current Python interpreter, a previously saved interpreter,
    and common Python executables available on the system.
    """
    if agama_python_executable is not None:
        manual_executable = Path(
            agama_python_executable
        ).expanduser()

        if not manual_executable.exists():
            raise FileNotFoundError(
                "AGAMA Python executable not found: "
                f"{manual_executable}"
            )

        if not _python_can_import_agama(manual_executable):
            raise RuntimeError(
                "The supplied Python interpreter cannot import AGAMA: "
                f"{manual_executable}"
            )

        _save_agama_python(manual_executable)
        return manual_executable

    tested_executables = []

    print("Searching for a Python interpreter with AGAMA...")

    for candidate in _candidate_python_executables():
        tested_executables.append(str(candidate))
        print(f"Testing Python interpreter: {candidate}")

        if _python_can_import_agama(candidate):
            print(f"AGAMA found with Python: {candidate}")
            _save_agama_python(candidate)
            return candidate

    tested_text = "\n".join(
        f"  - {executable}"
        for executable in tested_executables
    )

    raise RuntimeError(
        "AGAMA could not be imported with any detected Python "
        "interpreter.\n\n"
        "Interpreters tested:\n"
        f"{tested_text}\n\n"
        "Install AGAMA separately, or provide the interpreter manually "
        "with:\n"
        "agama_python_executable='/path/to/python'\n\n"
        "To print the path of a Python interpreter, run:\n"
        "python3 -c \"import sys; print(sys.executable)\""
    )


def ensure_agama_file(
    mwdata_path,
    agama_file,
    snapshot_index,
    tagging_radius_factor,
    n_iter,
    n_particles_per_component,
    agama_builder_path=None,
    agama_python_executable=None,
):
    """Create the AGAMA HDF5 file in a separate Python process if it does not already exist."""
    if os.path.exists(agama_file):
        print(f"AGAMA particle file already exists: {agama_file}")
        print("AGAMA will not be run again.")
        return

    if agama_builder_path is None:
        agama_builder_path = Path(__file__).with_name("agama_builder.py")
    else:
        agama_builder_path = Path(agama_builder_path)

    if not agama_builder_path.exists():
        raise FileNotFoundError(
            f"AGAMA builder script not found: {agama_builder_path}"
        )

    python_executable = find_agama_python(
        agama_python_executable=agama_python_executable,
    )

    command = [
        str(python_executable),
        str(agama_builder_path),
        "--mwdata",
        str(mwdata_path),
        "--output",
        str(agama_file),
        "--snapshot-index",
        str(snapshot_index),
        "--tagging-radius-factor",
        str(tagging_radius_factor),
        "--n-iter",
        str(n_iter),
        "--n-particles",
        str(n_particles_per_component),
    ]

    print("AGAMA particle file not found.")
    print("Running AGAMA in a separate Python process.")
    print(f"AGAMA Python executable: {python_executable}")
    subprocess.run(command, check=True)

    if not os.path.exists(agama_file):
        raise RuntimeError(
            "The AGAMA process finished without creating the expected file: "
            f"{agama_file}"
        )


def generate_in_situ_gcs(
    galaxy_id,
    mwdata_path,
    mwpots_path,
    agama_file,
    output_file,
    plot_file=None,
    snapshot_index=8,
    ngc=0,
    alpha=3,
    tagging_radius_factor=1.0,
    circularity_threshold=0.6,
    n_iter=20,
    n_particles_per_component=1_000_000,
    random_seed=None,
    agama_builder_path=None,
    keep_agama_file=True,
    agama_python_executable=None,
):
    """
    Generate in-situ GC initial conditions.

    Parameters
    ----------
    agama_python_executable : str or pathlib.Path, optional
        Advanced override for the Python interpreter used by AGAMA.
        When omitted, CosmoDyn searches automatically and remembers a
        working interpreter in ``~/.cosmodyn/config.json``.
    """
    if not os.path.exists(mwdata_path):
        raise FileNotFoundError(f"File not found: {mwdata_path}")
    if not os.path.exists(mwpots_path):
        raise FileNotFoundError(f"File not found: {mwpots_path}")

    ensure_agama_file(
        mwdata_path=mwdata_path,
        agama_file=agama_file,
        snapshot_index=snapshot_index,
        tagging_radius_factor=tagging_radius_factor,
        n_iter=n_iter,
        n_particles_per_component=n_particles_per_component,
        agama_builder_path=agama_builder_path,
        agama_python_executable=agama_python_executable,
    )

    A = np.loadtxt(mwdata_path)
    m_vir = A[:, 4][snapshot_index]

    if ngc == 0:
        number_of_gcs = int(
            round(number_of_globular_clusters(m_vir) * alpha, 0)
        )
        print(
            "Number of GCs computed from the halo-mass relation "
            f"et alpha={alpha} : {number_of_gcs}"
        )
    elif ngc > 0:
        number_of_gcs = int(ngc)
        print(
            "Number of GCs set by the NGC parameter: "
            f"{number_of_gcs}"
        )
    else:
        raise ValueError("NGC must be greater than or equal to zero.")

    with h5py.File(agama_file, "r") as f:
        x = f["GCdata/PosX"][:]
        y = f["GCdata/PosY"][:]
        z = f["GCdata/PosZ"][:]
        vx = f["GCdata/vX"][:]
        vy = f["GCdata/vY"][:]
        vz = f["GCdata/vZ"][:]

    pos = np.sqrt(x**2 + y**2 + z**2)
    vel = np.sqrt(vx**2 + vy**2 + vz**2)
    R = np.sqrt(x**2 + y**2)
    safe_R = np.where(R == 0, np.nan, R)
    vR = (x * vx + y * vy) / safe_R
    vT = (x * vy - y * vx) / safe_R
    phi = np.arctan2(y, x)

    valid = np.isfinite(vR) & np.isfinite(vT)
    pos, vel, R, z, vz = (
        arr[valid] for arr in (pos, vel, R, z, vz)
    )
    vR, vT, phi = (arr[valid] for arr in (vR, vT, phi))

    with open(mwpots_path, "rb") as f:
        mwpots = pickle.load(f)

    potential = mwpots[snapshot_index]

    # Important for potentials loaded from older pickle files:
    # the attribute may not exist at all, so it must be created explicitly,
    # as in the original script.
    if isinstance(potential, (list, tuple)):
        for component in potential:
            component.isDissipative = False
    else:
        potential.isDissipative = False

    E_pot_corrected1 = evaluatePotentials(
        mwpots[snapshot_index][0],
        pos * units.kpc,
        0 * units.kpc,
    )
    E_pot_corrected2 = evaluatePotentials(
        mwpots[snapshot_index][1],
        pos * units.kpc,
        0 * units.kpc,
    )

    E2 = 0.5 * vel**2 + E_pot_corrected1 + E_pot_corrected2
    Lz2 = R * vT

    plt.scatter(Lz2 / 1e3, E2 / 1e5, s=1)

    u1 = np.linspace(0.01, 200, 10000)

    Lzm1 = u1 * vcirc_fixed(u1, mwpots[snapshot_index][0] + mwpots[snapshot_index][1])

    Etest1 = (
        evaluatePotentials(mwpots[snapshot_index][1], u1 * units.kpc, 0 * units.kpc)
        + evaluatePotentials(mwpots[snapshot_index][0], u1 * units.kpc, 0 * units.kpc)
    )

    Ego1 = 0.5 * vcirc_fixed(u1, mwpots[snapshot_index][0] + mwpots[snapshot_index][1])**2 + Etest1

    order = np.argsort(Ego1)

    Ecirc_sorted = Ego1[order]
    Lcirc_sorted = Lzm1[order]

    Ecirc_unique, ind = np.unique(
        Ecirc_sorted,
        return_index=True,
    )

    Lcirc_unique = Lcirc_sorted[ind]

    Lcirc_of_E = interp1d(
        Ecirc_unique,
        Lcirc_unique,
        bounds_error=False,
        fill_value=np.nan,
    )

    Lcirc_part = Lcirc_of_E(E2)

    sol1 = np.where(
        np.isfinite(Lcirc_part)
        & (Lz2 >= circularity_threshold * Lcirc_part)
        & (Lz2 <= Lcirc_part)
    )[0]

    print(
    f"Number of candidate particles available for GC tagging: "
    f"{len(sol1)} (requested: {number_of_gcs})")

    plt.scatter(Lz2[sol1] / 1e3, E2[sol1] / 1e5, s=1)

    plt.plot(Lzm1/1e3,Ego1/1e5,lw=1,color='k')
    #plt.plot(0.6*Lzm1/1e3,Ego1/1e5)

    rng = np.random.default_rng(random_seed)
    drawn = rng.choice(sol1, size=number_of_gcs, replace=False)

    plt.scatter(
    Lz2[drawn] / 1e3,
    E2[drawn] / 1e5,
    s=5,
    label="Tagged globular clusters")

    plt.ylim(min(E2 / 1e5)+(0.05*min(E2 / 1e5)),max(E2 / 1e5)+(0.05*min(E2 / 1e5)))
    plt.xlim(min(Lz2 / 1e3)-(0.05*min(Lz2 / 1e3)),max(Lz2 / 1e3)+(0.05*min(Lz2 / 1e3)))

    plt.xlabel(r'Angular momentum $L_z \; [10^3 \; kpc \; km/s ]$', fontsize=15, fontweight='bold')
    plt.ylabel(r'Energy $E \; [10^5 \; km^2/s^2]$', fontsize=15, fontweight='bold')


    plot_file_ELz = Path(plot_file).with_stem(Path(plot_file).stem + "_ELz")

    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(plot_file_ELz, format="png", dpi=200)
    plt.close()

    plot_file_xy = Path(plot_file).with_stem(Path(plot_file).stem + "_xy")

    r_hm1 = A[:, 10][snapshot_index]
    x_gc = x[drawn]
    y_gc = y[drawn]
    z_gc = z[drawn]

    ###########################

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # x-y plane
    axes[0].scatter(
        x_gc,
        y_gc,
        s=20,color='C2',
    )

    circle_xy = Circle(
        (0.0, 0.0),
        r_hm1,
        fill=False,
        linestyle="--",
        linewidth=1.5,label=r"Stellar half-mass radius",
    )

    axes[0].add_patch(circle_xy)
    axes[0].legend()
    axes[0].set_xlabel(r"$x$ [kpc]")
    axes[0].set_ylabel(r"$y$ [kpc]")
    axes[0].set_title(r"$x-y$ plane")
    axes[0].set_aspect("equal", adjustable="box")
    #axes[0].set_xlim(-1.1 * r_hm1, 1.1 * r_hm1)
    #axes[0].set_ylim(-1.1 * r_hm1, 1.1 * r_hm1)

    # x-z plane
    axes[1].scatter(
        x_gc,
        z_gc,
        s=20,color='C2',
    )

    circle_xz = Circle(
        (0.0, 0.0),
        r_hm1,
        fill=False,
        linestyle="--",
        linewidth=1.5,
    )

    axes[1].add_patch(circle_xz)
    axes[1].set_xlabel(r"$x$ [kpc]")
    axes[1].set_ylabel(r"$z$ [kpc]")
    axes[1].set_title(r"$x-z$ plane")
    axes[1].set_aspect("equal", adjustable="box")
    #axes[1].set_xlim(-1.1 * r_hm1, 1.1 * r_hm1)
    #axes[1].set_ylim(-1.1 * r_hm1, 1.1 * r_hm1)

    fig.tight_layout()
    fig.savefig(
        plot_file_xy,
        format="png",
        dpi=200,
    )
    plt.close(fig)

    print(
    f"Maximum galactocentric radius of tagged GCs: "
    f"{np.max(pos[drawn]):.2f} kpc "
    f"(stellar half-mass radius = {r_hm1:.2f} kpc)")

    ###########################

    data = np.column_stack(
        [R[drawn], vR[drawn], vT[drawn], z[drawn], vz[drawn], phi[drawn]]
    )

    output_directory = os.path.dirname(output_file)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    np.savetxt(output_file, data, fmt="%s")
    print(f"Initial conditions created: {output_file}")

    if not keep_agama_file:
        if os.path.exists(agama_file):
            os.remove(agama_file)
            print(f"AGAMA particle file removed: {agama_file}")
        else:
            print(
                f"AGAMA particle file could not be removed because "
                f"it does not exist: {agama_file}"
            )

    return data
