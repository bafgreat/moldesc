#!/usr/bin/env python3
"""
Extract ORCA ligand calculation folders into one HDF5 file per ligand.

Each ligand folder is assumed to be named by its InChIKey, for example:

WOAHJDHKFWSLKE-UHFFFAOYSA-N/
├── SOKVAQ.out
├── SOKVAQ.hess
├── SOKVAQ.xyz
├── SOKVAQ_trj.xyz
├── SOKVAQ.inp
├── SOKVAQ.property.txt
└── ...

Usage
-----
python extract_orca_to_hdf5.py /path/to/orca_folders /path/to/hdf5_output

With overwrite:
python extract_orca_to_hdf5.py /path/to/orca_folders /path/to/hdf5_output --overwrite

Without storing raw text files:
python extract_orca_to_hdf5.py /path/to/orca_folders /path/to/hdf5_output --no-raw-files
"""

from pathlib import Path
import argparse
import csv
import json
import traceback
from datetime import datetime

import h5py
import numpy as np

from moldesc.orca_parser.parse_output import OrcaOutputParser, OrcaHessParser


TEXT_SUFFIXES = {
    ".out",
    ".inp",
    ".opt",
    ".xyz",
    ".bibtex",
    ".engrad",
    ".log",
    ".txt",
    ".sh",
}

TEXT_NAME_ENDINGS = {
    "_slurm.out",
    "_slurm.log",
    "_trj.xyz",
    ".property.txt",
}

LARGE_BINARY_SUFFIXES = {
    ".gbw",
    ".densities",
    ".densitiesinfo",
}


def safe_call(func):
    try:
        return func()
    except Exception as exc:
        return {
            "__error__": str(exc),
            "__function__": getattr(func, "__name__", "unknown"),
        }


def is_error_value(value):
    return isinstance(value, dict) and "__error__" in value


def to_json_string(value):
    return json.dumps(value, default=str, ensure_ascii=False)


def write_value(group, name, value):
    """
    Recursively write Python data to HDF5.

    - dict -> group
    - numeric/list/array -> dataset
    - string/scalar -> attribute
    - unsupported objects -> JSON string dataset
    """

    name = str(name)

    if value is None:
        group.attrs[name] = "None"
        return

    if isinstance(value, dict):
        subgrp = group.create_group(name)
        for key, val in value.items():
            write_value(subgrp, str(key), val)
        return

    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "O"}:
            group.create_dataset(name, data=to_json_string(value.tolist()))
        else:
            group.create_dataset(name, data=value)
        return

    if isinstance(value, (list, tuple)):
        try:
            arr = np.asarray(value)

            if arr.dtype.kind in {"i", "u", "f", "b"}:
                group.create_dataset(name, data=arr)
            else:
                group.create_dataset(name, data=to_json_string(value))

        except Exception:
            group.create_dataset(name, data=to_json_string(value))

        return

    if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
        group.attrs[name] = value
        return

    group.create_dataset(name, data=to_json_string(value))


def write_text_file(group, filepath):
    try:
        text = filepath.read_text(errors="replace")
        group.create_dataset(filepath.name, data=text)
    except Exception as exc:
        group.attrs[f"{filepath.name}__error"] = str(exc)


def should_store_as_raw_text(filepath):
    name = filepath.name

    if filepath.suffix in TEXT_SUFFIXES:
        return True

    for ending in TEXT_NAME_ENDINGS:
        if name.endswith(ending):
            return True

    return False


def find_first_file(folder, suffix):
    files = sorted(folder.glob(f"*{suffix}"))
    return files[0] if files else None


def get_output_data(parser):
    return {
        "calculation_settings": safe_call(parser.get_cal_setting),

        "formula": safe_call(parser.get_formula),
        "mass": safe_call(parser.get_mass),
        "smiles": safe_call(parser.get_smile),
        "cheminformatics": safe_call(parser.get_cheminformatic),

        "thermal_energy_kcal_mol": safe_call(parser.get_thermal_energy),
        "enthalpy_kcal_mol": safe_call(parser.get_enthalpy),
        "zero_point_energy_kcal_mol": safe_call(parser.get_zero_point_energy),
        "entropy_cal_mol_K": safe_call(parser.get_total_enthropy),
        "gibbs_free_energy_kcal_mol": safe_call(parser.get_gibbs_free_energy),
        "energy_correction": safe_call(parser.energy_correction),

        "symmetry": safe_call(parser.get_symmetry),
        "dipole_moment": safe_call(parser.get_dipole_moment),
        "quadrupole_moment": safe_call(parser.get_quadrupole_moment),

        "static_polarizability_dipole_dipole": safe_call(
            parser.get_static_polarizability_dipole_dipole
        ),
        "static_polarizability_dipole_quadrupole": safe_call(
            parser.get_static_polarizability_dipole_quadrupole
        ),
        "static_traceless_polarizability_dipole_quadrupole": safe_call(
            parser.get_static_traceless_polarizability_dipole_quadrupole
        ),

        "nmr_efg": safe_call(parser.get_nmr_efg),
        "scf_correction": safe_call(parser.get_scf_correction),

        "mayer_analysis": safe_call(parser.get_mayer_analyis),
        "mayer_bond_order": safe_call(parser.get_mayer_bond_order),

        "frontier_orbital": safe_call(parser.get_frontier_orbital),
        "orbital_energy": safe_call(parser.get_orbital_energy),

        "ir_spectrum": safe_call(parser.get_ir_spectrum),
        "raman_spectrum": safe_call(parser.get_raman_spectrum),
        "vcd_spectrum": safe_call(parser.get_vcd_spectrum),

        "normal_modes": safe_call(parser.get_normal_modes),

        "loewdin_charges": safe_call(parser.get_loewdin_charges),
        "mulliken_charges": safe_call(parser.get_mulliken),

        "trajectory": safe_call(parser.get_trajectory),
        "trajectory_energies_kcal_mol": safe_call(parser.get_traj_energies),
        "forces": safe_call(parser.get_forces),
        "dispersion_forces": safe_call(parser.get_dispersion_forces),
    }


def get_hess_data(hess_parser):
    return {
        "hessian": safe_call(hess_parser.get_hessian),
        "dipole_derivatives": safe_call(hess_parser.get_dipole_derivative),
        "polarizability_derivatives": safe_call(
            hess_parser.get_polarization_derivative
        ),
    }


def extract_index_row(inchikey, h5_path, out_file, hess_file, output_data):
    settings = output_data.get("calculation_settings", {})
    if is_error_value(settings):
        settings = {}

    return {
        "inchikey": inchikey,
        "hdf5_path": str(h5_path),
        "out_file": out_file.name if out_file else "",
        "hess_file": hess_file.name if hess_file else "",
        "formula": output_data.get("formula", ""),
        "smiles": output_data.get("smiles", ""),
        "mass": output_data.get("mass", ""),
        "method": settings.get("method", ""),
        "functional_name": settings.get("functional_name", ""),
        "functional_kind": settings.get("functional_kind", ""),
        "charge": settings.get("total_charge", ""),
        "multiplicity": settings.get("multiplicity", ""),
        "gibbs_free_energy_kcal_mol": output_data.get(
            "gibbs_free_energy_kcal_mol", ""
        ),
        "enthalpy_kcal_mol": output_data.get("enthalpy_kcal_mol", ""),
        "zero_point_energy_kcal_mol": output_data.get(
            "zero_point_energy_kcal_mol", ""
        ),
    }


def extract_ligand_folder(
    folder,
    output_root,
    overwrite=False,
    store_raw_files=True,
):
    inchikey = folder.name
    h5_path = output_root / f"{inchikey}.hdf5"

    if h5_path.exists() and not overwrite:
        return {
            "status": "skipped",
            "inchikey": inchikey,
            "hdf5_path": str(h5_path),
            "message": "HDF5 already exists",
            "index_row": None,
        }

    out_file = find_first_file(folder, ".out")
    hess_file = find_first_file(folder, ".hess")

    if out_file is None:
        return {
            "status": "failed",
            "inchikey": inchikey,
            "hdf5_path": str(h5_path),
            "message": "No .out file found",
            "index_row": None,
        }

    tmp_h5_path = output_root / f"{inchikey}.hdf5.tmp"

    if tmp_h5_path.exists():
        tmp_h5_path.unlink()

    parser = OrcaOutputParser(str(out_file))
    output_data = get_output_data(parser)

    with h5py.File(tmp_h5_path, "w") as h5:
        h5.attrs["inchikey"] = inchikey
        h5.attrs["source_folder"] = str(folder.resolve())
        h5.attrs["created_at"] = datetime.now().isoformat()
        h5.attrs["orca_out_file"] = out_file.name

        if hess_file:
            h5.attrs["orca_hess_file"] = hess_file.name

        status_group = h5.create_group("status")
        status_group.attrs["success"] = True

        files_group = h5.create_group("files")
        for file in sorted(folder.iterdir()):
            if file.is_file():
                files_group.attrs[file.name] = str(file.resolve())

        write_value(h5, "orca_output", output_data)

        try:
            atoms = parser.get_ase_atoms()
            structure_group = h5.create_group("final_structure")
            write_value(structure_group, "symbols", atoms.get_chemical_symbols())
            write_value(structure_group, "positions", atoms.get_positions())
            write_value(structure_group, "atomic_numbers", atoms.get_atomic_numbers())
        except Exception as exc:
            status_group.attrs["final_structure_error"] = str(exc)

        if hess_file:
            hess_parser = OrcaHessParser(str(hess_file))
            hess_data = get_hess_data(hess_parser)
            write_value(h5, "orca_hess", hess_data)

        if store_raw_files:
            raw_group = h5.create_group("raw_text_files")

            for file in sorted(folder.iterdir()):
                if not file.is_file():
                    continue

                if file.suffix in LARGE_BINARY_SUFFIXES:
                    continue
                if (
                    file.name == "submit.sh"
                    or file.name.startswith("slurm")
                    or file.name.endswith("_slurm.out")
                    or file.name.endswith("_slurm.log")
                ):
                    continue

                if should_store_as_raw_text(file):
                    write_text_file(raw_group, file)

    tmp_h5_path.replace(h5_path)

    index_row = extract_index_row(
        inchikey=inchikey,
        h5_path=h5_path,
        out_file=out_file,
        hess_file=hess_file,
        output_data=output_data,
    )

    return {
        "status": "done",
        "inchikey": inchikey,
        "hdf5_path": str(h5_path),
        "message": "success",
        "index_row": index_row,
    }


def write_index_csv(index_rows, output_root):
    if not index_rows:
        return

    index_path = output_root / "ligand_index.csv"

    fieldnames = list(index_rows[0].keys())

    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)


def append_log(log_path, message):
    with log_path.open("a") as handle:
        handle.write(message.rstrip() + "\n")


def main():
    argparser = argparse.ArgumentParser(
        description="Convert ORCA ligand folders into HDF5 files."
    )

    argparser.add_argument(
        "input_root",
        help="Folder containing ligand subfolders named by InChIKey.",
    )

    argparser.add_argument(
        "output_root",
        help="Folder where HDF5 files and ligand_index.csv will be written.",
    )

    argparser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing HDF5 files.",
    )

    argparser.add_argument(
        "--no-raw-files",
        action="store_true",
        help="Do not store raw text files inside the HDF5.",
    )

    args = argparser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    log_path = output_root / "extraction.log"
    failure_log_path = output_root / "failures.log"

    ligand_folders = sorted([p for p in input_root.iterdir() if p.is_dir()])

    print(f"Input root:  {input_root}")
    print(f"Output root: {output_root}")
    print(f"Found {len(ligand_folders)} ligand folders")

    append_log(log_path, f"\nStarted: {datetime.now().isoformat()}")
    append_log(log_path, f"Input root: {input_root}")
    append_log(log_path, f"Output root: {output_root}")
    append_log(log_path, f"Number of ligand folders: {len(ligand_folders)}")

    index_rows = []

    n_done = 0
    n_skip = 0
    n_fail = 0

    for i, folder in enumerate(ligand_folders, start=1):
        print(f"[{i}/{len(ligand_folders)}] {folder.name}")

        try:
            result = extract_ligand_folder(
                folder=folder,
                output_root=output_root,
                overwrite=args.overwrite,
                store_raw_files=not args.no_raw_files,
            )

            status = result["status"]

            if status == "done":
                n_done += 1
                index_rows.append(result["index_row"])
                print(f"  DONE -> {result['hdf5_path']}")

            elif status == "skipped":
                n_skip += 1
                print(f"  SKIPPED -> {result['message']}")

            else:
                n_fail += 1
                print(f"  FAILED -> {result['message']}")
                append_log(
                    failure_log_path,
                    f"{folder.name}: {result['message']}",
                )

            append_log(
                log_path,
                f"{folder.name}: {status}: {result['message']}",
            )

        except Exception:
            n_fail += 1
            tb = traceback.format_exc()
            print(f"  FAILED -> unexpected error")
            append_log(failure_log_path, f"\n{folder.name}\n{tb}")
            append_log(log_path, f"{folder.name}: failed: unexpected error")

    write_index_csv(index_rows, output_root)

    append_log(log_path, f"Finished: {datetime.now().isoformat()}")
    append_log(log_path, f"Done: {n_done}")
    append_log(log_path, f"Skipped: {n_skip}")
    append_log(log_path, f"Failed: {n_fail}")

    print("\nSummary")
    print(f"Done:    {n_done}")
    print(f"Skipped: {n_skip}")
    print(f"Failed:  {n_fail}")
    print(f"Index:   {output_root / 'ligand_index.csv'}")
    print(f"Log:     {log_path}")


if __name__ == "__main__":
    main()