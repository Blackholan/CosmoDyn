#!/usr/bin/env python3
# coding: utf-8

"""Computation-time utilities for CosmoDyn."""

from pathlib import Path
import time


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
):
    """Write a human-readable CosmoDyn computation-time report."""

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    successful_galaxies = successful_galaxies or []
    failed_galaxies = failed_galaxies or []

    stages = [
        "RUN_ICS",
        "RUN_STREAM_ICS",
        "RUN_DYNAMICS",
        "RUN_STREAMS",
    ]

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
            ("snapshot_index", "Snapshot index"),
            ("circularity_threshold", "Circularity threshold"),
            ("tagging_radius_factor", "Tagging radius factor"),
            ("n_particles_per_component", "AGAMA particles for GC ICs"),
            ("potential_mode", "Potential mode"),
            ("df_model", "Dynamical-friction model"),
            ("m22", "m22"),
            ("gc_mass", "GC mass [Msun]"),
            ("gc_half_mass_radius", "GC half-mass radius [kpc]"),
            ("central_capture_radius", "Central capture radius [kpc]"),
            ("mass_loss_mode", "Mass-loss mode"),
            ("n_stream_particles", "Stream particles per GC"),
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
            ):
                value = f"{float(value):.6e}"

            stream.write(f"{label}: {value}\n")

        stream.write("\nEnabled stages:\n")
        for stage in stages:
            stream.write(
                f"  {stage} = "
                f"{bool(enabled_stages.get(stage, False))}\n"
            )

        stream.write("\nTIMING BY GALAXY AND STAGE\n")
        stream.write("-" * 112 + "\n")
        stream.write(
            f"{'Galaxy':<12}"
            f"{'Stage':<20}"
            f"{'Wall [s]':>14}"
            f"{'Wall [h]':>14}"
            f"{'N CPUs':>10}"
            f"{'CPUh':>14}"
            f"{'N particles':>16}"
            f"{'N GCs':>10}\n"
        )
        stream.write("-" * 112 + "\n")

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
                f"{record['stage']:<20}"
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
                f"{stage:<20}"
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
        stream.write(
            "\nCPUh definition: "
            "wall-clock time [hours] x number of CPUs.\n"
        )

    print(f"Computation-time report saved to: {output_file}")
