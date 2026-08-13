#!/usr/bin/env python3
# coding: utf-8

"""Computation-time utilities for CosmoDyn."""

from pathlib import Path
import time


def _file_size_bytes(path):
    """Return a file size, or zero if the file disappears during the scan."""
    try:
        return path.stat().st_size
    except (FileNotFoundError, OSError):
        return 0


def _directory_size_bytes(directory):
    """Return the recursive size of regular files within one directory."""
    total = 0
    try:
        for path in directory.rglob("*"):
            if path.is_file():
                total += _file_size_bytes(path)
    except (FileNotFoundError, OSError):
        return total
    return total


def _format_size_gb(size_bytes):
    """Format bytes in decimal gigabytes (1 Go = 1e9 bytes)."""
    return f"{size_bytes / 1.0e9:.9f} Go"


def _write_output_storage_report(stream, output_directory):
    """Write the recursive InSitu and ExSitu size for every galaxy."""
    if output_directory is None:
        return

    output_directory = Path(output_directory)
    stream.write("\nOUTPUT STORAGE\n")
    stream.write("-" * 112 + "\n")

    if not output_directory.exists():
        stream.write(f"Output directory not found: {output_directory}\n")
        return

    entries = [
        path
        for path in output_directory.rglob("*")
        if not any(
            part.startswith(".")
            for part in path.relative_to(output_directory).parts
        )
    ]
    directories = [path for path in entries if path.is_dir()]
    files = [path for path in entries if path.is_file()]
    directory_sizes = {
        output_directory: 0,
        **{directory: 0 for directory in directories},
    }
    file_sizes = {}

    # Read each file size once, then add it to every ancestor below the
    # output root. This remains fast even for large stream inventories.
    for path in files:
        size_bytes = _file_size_bytes(path)
        file_sizes[path] = size_bytes
        parent = path.parent
        while True:
            if parent in directory_sizes:
                directory_sizes[parent] += size_bytes
            if parent == output_directory:
                break
            parent = parent.parent

    stream.write(
        f"{'Galaxy':<14}"
        f"{'Directory':<14}"
        f"{'Size [Go]':>16}\n"
    )
    stream.write("-" * 112 + "\n")

    galaxy_directories = sorted(
        (
            path
            for path in output_directory.iterdir()
            if path.is_dir() and path.name.startswith("G")
        ),
        key=lambda path: path.name,
    )
    for galaxy_directory in galaxy_directories:
        galaxy_label = galaxy_directory.name[1:]
        for directory_name in ("InSitu", "ExSitu"):
            directory = galaxy_directory / directory_name
            size_bytes = directory_sizes.get(directory, 0)
            stream.write(
                f"{galaxy_label:<14}"
                f"{directory_name:<14}"
                f"{size_bytes / 1.0e9:>16.9f}\n"
            )


def start_stage_timer():
    """Return a high-resolution wall-clock timer value."""
    return time.perf_counter()


def record_stage_timing(
    records,
    galaxy_id,
    stage,
    start_time,
    n_gcs=None,
    n_particles=None,
    n_cpus=1,
):
    """
    Record wall-clock time and CPU-hours for one pipeline stage.

    CPU-hours are defined as:
        CPUh = wall-clock time [hours] x number of CPUs
    """
    wall_seconds = time.perf_counter() - start_time
    wall_hours = wall_seconds / 3600.0

    n_cpus = max(int(n_cpus), 1)
    cpu_hours = wall_hours * n_cpus

    record = {
        "galaxy_id": int(galaxy_id),
        "stage": str(stage),
        "wall_seconds": float(wall_seconds),
        "wall_hours": float(wall_hours),
        "cpu_hours": float(cpu_hours),
        "n_cpus": n_cpus,
        "n_particles": None if n_particles is None else int(n_particles),
        "n_gcs": None if n_gcs is None else int(n_gcs),
    }

    records.append(record)

    print(
        f"{stage} completed in {wall_seconds:.2f} s "
        f"({wall_hours:.6f} h; "
        f"{cpu_hours:.6f} CPUh with {n_cpus} CPU(s))."
    )

    return record


def write_timing_report(
    output_file,
    records,
    galaxy_ids,
    run_parameters,
    enabled_stages,
    successful_galaxies=None,
    failed_galaxies=None,
    output_directory=None,
):
    """Write a human-readable CosmoDyn computation-time report."""

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    successful_galaxies = successful_galaxies or []
    failed_galaxies = failed_galaxies or []

    # Preserve the pipeline order supplied by the launcher. This avoids a
    # hard-coded stage list silently omitting newly added in-situ or ex-situ
    # stages from the enabled-stage and summary sections.
    stages = list(enabled_stages)

    with output_file.open("w", encoding="utf-8") as stream:
        stream.write("CosmoDyn computation-time report\n")
        stream.write("=" * 112 + "\n\n")

        stream.write("RUN CONFIGURATION\n")
        stream.write("-" * 112 + "\n")
        stream.write(f"Number of galaxies requested: {len(galaxy_ids)}\n")
        stream.write(f"Successful galaxies: {len(successful_galaxies)}\n")
        stream.write(f"Failed galaxies: {len(failed_galaxies)}\n")

        if failed_galaxies:
            stream.write(
                "Failed galaxy IDs: "
                + ", ".join(str(g) for g in failed_galaxies)
                + "\n"
            )

        parameter_labels = [
            ("galaxy_ids", "Galaxy IDs"),
            ("continue_on_error", "Continue after a galaxy failure"),
            ("available_cpus", "Available CPUs"),
            ("snapshot_index", "Start snapshot index"),
            ("end_snapshot_index", "End snapshot index"),
            ("integration_method", "Integration method"),
            ("timestep_file", "Timestep file"),
            ("ngc_in_situ", "In-situ GCs (0 = scaling relation)"),
            ("alpha_in_situ", "In-situ GC-number scaling factor"),
            ("circularity_threshold", "In-situ circularity threshold"),
            ("tagging_radius_factor", "In-situ tagging radius factor"),
            ("minimum_tagging_radius", "In-situ minimum tagging radius [kpc]"),
            ("n_iter_in_situ", "In-situ AGAMA iterations"),
            ("n_particles_per_component", "AGAMA particles for in-situ GC ICs"),
            ("random_seed_in_situ", "In-situ random seed"),
            ("keep_agama_file", "Keep in-situ AGAMA file"),
            ("ngc_ex_situ", "Ex-situ GCs per satellite (0 = scaling relation)"),
            ("alpha_ex_situ", "Ex-situ GC-number scaling factor"),
            ("ex_situ_tagging_radius_factor", "Ex-situ tagging radius factor"),
            ("ex_situ_minimum_tagging_radius", "Ex-situ minimum tagging radius [kpc]"),
            ("ex_situ_circularity_threshold", "Ex-situ circularity threshold"),
            ("ex_situ_n_iter", "Ex-situ AGAMA iterations"),
            ("ex_situ_n_particles_per_component", "AGAMA particles for ex-situ GC ICs"),
            ("ex_situ_random_seed", "Ex-situ random seed"),
            ("keep_ex_situ_agama_files", "Keep ex-situ AGAMA files"),
            ("include_moving_satellites", "Include moving satellites in GC orbits"),
            ("maximum_satellite_radius", "Maximum satellite radius [kpc]"),
            ("potential_mode", "Potential mode"),
            ("static_potential_index", "Static potential index"),
            ("df_model", "Dynamical-friction model"),
            ("m22", "m22 [1e-22 eV]"),
            ("gc_mass", "GC mass [Msun]"),
            ("gc_half_mass_radius", "GC half-mass radius [kpc]"),
            ("reuse_df_cache", "Reuse DF cache"),
            ("central_capture_radius", "Central capture radius [kpc]"),
            ("generate_gc_moving_potentials", "Generate GC moving potentials"),
            ("gc_potential_scale_radius", "GC potential scale radius [kpc]"),
            ("mass_loss_mode", "Mass-loss mode"),
            ("mass_loss_gamma", "Mass-loss gamma"),
            ("tidal_strength_reference", "Tidal-strength reference"),
            ("dissolution_time_normalization", "Dissolution-time normalization"),
            ("release_energy_tolerance", "Ex-situ release-energy tolerance [km2 s-2]"),
            ("n_stream_particles", "Stream particles per GC"),
            ("n_stream_iter", "Stream AGAMA iterations"),
            ("overwrite_stream_ics", "Overwrite stream ICs"),
            ("stream_n_jobs", "Requested stream jobs"),
            ("stream_cpu_count", "Effective stream CPUs"),
            ("stream_batch_size", "Stream batch size"),
        ]

        for key, label in parameter_labels:
            if key not in run_parameters:
                continue
            if key == "m22" and run_parameters.get("df_model") != "fdm":
                continue

            value = run_parameters[key]

            if key in (
                "gc_mass",
                "gc_half_mass_radius",
                "central_capture_radius",
                "gc_potential_scale_radius",
            ):
                if value is not None:
                    value = f"{float(value):.6e}"

            stream.write(f"{label}: {value}\n")

        stream.write("\nEnabled stages:\n")
        for stage in stages:
            stream.write(
                f"  {stage} = "
                f"{bool(enabled_stages.get(stage, False))}\n"
            )

        stream.write("\nTIMING BY GALAXY AND STAGE\n")
        stream.write("-" * 120 + "\n")
        stream.write(
            f"{'Galaxy':<12}"
            f"{'Stage':<30}"
            f"{'Wall [s]':>14}"
            f"{'Wall [h]':>14}"
            f"{'N CPUs':>10}"
            f"{'CPUh':>14}"
            f"{'N particles':>16}"
            f"{'N GCs':>10}\n"
        )
        stream.write("-" * 120 + "\n")

        for record in records:
            n_particles = (
                "-"
                if record["n_particles"] is None
                else str(record["n_particles"])
            )
            n_gcs = (
                "-"
                if record["n_gcs"] is None
                else str(record["n_gcs"])
            )
            galaxy_label = (
                "GLOBAL"
                if record["galaxy_id"] == -1
                else str(record["galaxy_id"])
            )

            stream.write(
                f"{galaxy_label:<12}"
                f"{record['stage']:<30}"
                f"{record['wall_seconds']:>14.2f}"
                f"{record['wall_hours']:>14.6f}"
                f"{record['n_cpus']:>10}"
                f"{record['cpu_hours']:>14.6f}"
                f"{n_particles:>16}"
                f"{n_gcs:>10}\n"
            )

        stream.write("\nSUMMARY BY STAGE\n")
        stream.write("-" * 112 + "\n")

        for stage in stages:
            stage_records = [
                record for record in records if record["stage"] == stage
            ]
            if not stage_records:
                continue

            total_wall_seconds = sum(
                record["wall_seconds"] for record in stage_records
            )
            total_wall_hours = total_wall_seconds / 3600.0
            total_cpu_hours = sum(
                record["cpu_hours"] for record in stage_records
            )
            total_gcs = sum(
                record["n_gcs"]
                for record in stage_records
                if record["n_gcs"] is not None
            )

            if any(record["galaxy_id"] == -1 for record in stage_records):
                galaxy_text = "GLOBAL"
            else:
                galaxy_text = str(len(stage_records))

            stream.write(
                f"{stage:<30}"
                f"Galaxies={galaxy_text:<8}"
                f"GCs={total_gcs:<8}"
                f"Wall={total_wall_seconds:.2f} s "
                f"({total_wall_hours:.6f} h)   "
                f"CPUh={total_cpu_hours:.6f}\n"
            )

        total_wall_seconds = sum(
            record["wall_seconds"] for record in records
        )
        total_wall_hours = total_wall_seconds / 3600.0
        total_cpu_hours = sum(
            record["cpu_hours"] for record in records
        )

        stream.write("\nTOTAL RECORDED PIPELINE COST\n")
        stream.write("-" * 112 + "\n")
        stream.write(
            "Wall time summed over all recorded stages: "
            f"{total_wall_seconds:.2f} s "
            f"({total_wall_hours:.6f} h)\n"
        )
        stream.write(
            "CPU-hours summed over all recorded stages: "
            f"{total_cpu_hours:.6f} CPUh\n"
        )

        _write_output_storage_report(
            stream,
            output_directory,
        )

    print(f"Computation-time report saved to: {output_file}")
