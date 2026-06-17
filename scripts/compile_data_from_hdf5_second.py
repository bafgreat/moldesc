"""
orca_to_hdf5.py
---------------
Walk a directory of ORCA calculations and write one HDF5 file per
calculation.  A "calculation" is identified by a shared stem, e.g.
    ethanol.out  +  ethanol.hess  →  ethanol.h5

Layout inside each HDF5 file
-----------------------------
/metadata/
    cal_setting         (attrs: method, functional_*, hartree_fock_type, …)
    symmetry            (attrs: point_group, symmetry_number, …)
    formula             (scalar string dataset)
    smiles              (scalar string dataset)
    mass                (scalar float dataset)

/geometry/
    symbols             (N,)   string array
    positions           (N, 3) float64

/charges/
    mulliken            (N,)   float64
    loewdin             (N,)   float64

/energies/
    thermal_energy      scalar float64  [kcal/mol]
    enthalpy            scalar float64  [kcal/mol]
    zero_point_energy   scalar float64  [kcal/mol]
    gibbs_free_energy   scalar float64  [kcal/mol]
    total_entropy       scalar float64  [cal/(mol·K)]
    scf/                group  (nuclear_repulsion, electronic_energy, …)
    corrections/        group  (thermal_vibrational_correction, …)

/multipoles/
    dipole/             (electronic, nuclear, total, magnitude, along_axis)
    quadrupole/         (electronic, nuclear, total_au, total_buckingham,
                         principal_au, principal_buckingham, isotropic)

/polarizability/
    dipole_dipole/      (cartesian_au, principal_au, isotropic_au)
    dipole_quadrupole/  (raw tensor as dataset with component labels in attrs)
    traceless_dipole_quadrupole/

/spectra/
    ir/                 datasets: mode_index, frequency, epsilon, intensity, T2, Tx, Ty, Tz
    raman/              datasets: mode_index, frequency, activity, depolarization
    vcd/                datasets: mode_index, frequency, intensity

/normal_modes/
    modes               (n_modes, n_atoms, 3) float64

/orbital_energies/
    orbital             (M,) int32
    occupation          (M,) float64
    energy_Eh           (M,) float64
    energy_eV           (M,) float64

/frontier_orbitals/
    homo_index          scalar int32
    lumo_index          scalar int32
    homo_population/    atom_index, element, mulliken, loewdin
    lumo_population/    atom_index, element, mulliken, loewdin

/mayer/
    analysis/           atom_index, gross_atomic_population, …
    bond_orders/        atom1, atom2, bond_order

/nmr_efg/
    <atom_index>/       tensor (3,3), principal (3,), asymmetry_parameter

/trajectory/
    step_<i>/
        symbols         (N,)   string
        positions       (N, 3) float64
        energy          scalar float64  [kcal/mol]
    forces/
        step_<i>        (N, 3) float64
    dispersion_forces/
        step_<i>        (N, 3) float64

/hessian/               (only if .hess file is present)
    hessian             (n_modes, n_atoms, 3) float64
    dipole_derivatives  (3N, 3) float64
    polarization_derivatives (3N, 6) float64
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Adjust this import to wherever your parsers actually live
# ---------------------------------------------------------------------------
from orca_parser import OrcaOutputParser, OrcaHessParser   # ← change if needed

log = logging.getLogger(__name__)


# ============================================================
# Tiny helpers
# ============================================================

def _safe(fn, *args, default=None, label=""):
    """Call fn(*args), return default on any exception."""
    try:
        return fn(*args)
    except Exception as exc:
        if label:
            log.debug("Skipping %s: %s", label, exc)
        return default


def _write_scalar(group: h5py.Group, name: str, value: Any):
    """Store a Python scalar as a scalar dataset."""
    if value is None:
        return
    if isinstance(value, str):
        dt = h5py.string_dtype()
        group.create_dataset(name, data=value, dtype=dt)
    else:
        group.create_dataset(name, data=value)


def _write_attrs(group: h5py.Group, d: dict):
    """Store a flat dict as HDF5 attributes on *group*."""
    for k, v in d.items():
        if v is None:
            continue
        try:
            group.attrs[k] = v
        except Exception as exc:
            log.debug("Could not write attr %s: %s", k, exc)


def _string_array(strings: list[str]) -> np.ndarray:
    return np.array(strings, dtype=h5py.string_dtype())


# ============================================================
# Section writers
# ============================================================

def write_metadata(f: h5py.File, p: OrcaOutputParser):
    grp = f.require_group("metadata")

    cal = _safe(p.get_cal_setting, default={}, label="cal_setting")
    if cal:
        _write_attrs(grp.require_group("cal_setting"), cal)

    sym = _safe(p.get_symmetry, default={}, label="symmetry")
    if sym:
        sg = grp.require_group("symmetry")
        for k, v in sym.items():
            if isinstance(v, list):
                sg.create_dataset(k, data=np.array(v, dtype=float))
            else:
                sg.attrs[k] = v

    for name, fn in [("formula", p.get_formula),
                     ("smiles",  p.get_smile)]:
        val = _safe(fn, default=None, label=name)
        if val is not None:
            _write_scalar(grp, name, val)

    mass = _safe(p.get_mass, default=None, label="mass")
    _write_scalar(grp, "mass", mass)

    cheminf = _safe(p.get_cheminformatic, default={}, label="cheminformatic")
    if cheminf:
        _write_attrs(grp.require_group("cheminformatics"), cheminf)


def write_geometry(f: h5py.File, p: OrcaOutputParser):
    atoms = _safe(p.get_ase_atoms, default=None, label="ase_atoms")
    if atoms is None:
        return
    grp = f.require_group("geometry")
    grp.create_dataset("symbols",   data=_string_array(atoms.get_chemical_symbols()))
    grp.create_dataset("positions", data=np.array(atoms.get_positions(), dtype=float))


def write_charges(f: h5py.File, p: OrcaOutputParser):
    grp = f.require_group("charges")
    for name, fn in [("mulliken", p.get_mulliken),
                     ("loewdin",  p.get_loewdin_charges)]:
        val = _safe(fn, default=None, label=name)
        if val:
            grp.create_dataset(name, data=np.array(val, dtype=float))


def write_energies(f: h5py.File, p: OrcaOutputParser):
    grp = f.require_group("energies")

    scalars = {
        "thermal_energy":    p.get_thermal_energy,
        "enthalpy":          p.get_enthalpy,
        "zero_point_energy": p.get_zero_point_energy,
        "gibbs_free_energy": p.get_gibbs_free_energy,
        "total_entropy":     p.get_total_enthropy,
    }
    for name, fn in scalars.items():
        _write_scalar(grp, name, _safe(fn, default=None, label=name))

    scf = _safe(p.get_scf_correction, default={}, label="scf_correction")
    if scf:
        sg = grp.require_group("scf")
        for k, v in scf.items():
            _write_scalar(sg, k, v)

    corr = _safe(p.energy_correction, default={}, label="energy_correction")
    if corr:
        cg = grp.require_group("corrections")
        for k, v in corr.items():
            _write_scalar(cg, k, v)


def write_multipoles(f: h5py.File, p: OrcaOutputParser):
    grp = f.require_group("multipoles")

    # ---- dipole ----
    dip = _safe(p.get_dipole_moment, default={}, label="dipole_moment")
    if dip:
        dg = grp.require_group("dipole")
        for key, val in dip.items():
            if isinstance(val, list):
                dg.create_dataset(key, data=np.array(val, dtype=float))
            elif isinstance(val, dict):
                sg = dg.require_group(key)
                for unit, v in val.items():
                    if isinstance(v, list):
                        sg.create_dataset(unit, data=np.array(v, dtype=float))
                    else:
                        _write_scalar(sg, unit, v)

    # ---- quadrupole ----
    quad = _safe(p.get_quadrupole_moment, default={}, label="quadrupole_moment")
    if quad:
        qg = grp.require_group("quadrupole")
        component_keys = ["XX", "YY", "ZZ", "XY", "XZ", "YZ"]

        def _write_tensor_dict(parent, name, d):
            sg = parent.require_group(name)
            for k in component_keys:
                if k in d:
                    _write_scalar(sg, k, d[k])

        for section, value in quad.items():
            if isinstance(value, dict):
                for unit, tensor in value.items():
                    if isinstance(tensor, dict):
                        _write_tensor_dict(qg.require_group(section), unit, tensor)
            elif value is not None:
                _write_scalar(qg, section, value)


def write_polarizability(f: h5py.File, p: OrcaOutputParser):
    grp = f.require_group("polarizability")

    dd = _safe(p.get_static_polarizability_dipole_dipole,
               default={}, label="dipole_dipole_polarizability")
    if dd:
        ddg = grp.require_group("dipole_dipole")
        for k, v in dd.items():
            if isinstance(v, dict):
                sg = ddg.require_group(k)
                for comp, val in v.items():
                    _write_scalar(sg, comp, val)
            else:
                _write_scalar(ddg, k, v)

    for attr, fn, name in [
        ("dipole_quadrupole",          p.get_static_polarizability_dipole_quadrupole,          "dipole_quadrupole"),
        ("traceless_dipole_quadrupole", p.get_static_traceless_polarizability_dipole_quadrupole, "traceless_dipole_quadrupole"),
    ]:
        val = _safe(fn, default={}, label=attr)
        if val:
            sg = grp.require_group(name)
            keys = list(val.keys())
            vals = [val[k] for k in keys]
            sg.create_dataset("values", data=np.array(vals, dtype=float))
            sg.create_dataset("components", data=_string_array(keys))


def write_spectra(f: h5py.File, p: OrcaOutputParser):
    grp = f.require_group("spectra")

    # IR
    ir = _safe(p.get_ir_spectrum, default={}, label="ir_spectrum")
    if ir:
        ig = grp.require_group("ir")
        modes = sorted(ir.keys(), key=int)
        fields = ["frequency", "epsilon", "intensity", "T2", "Tx", "Ty", "Tz"]
        ig.create_dataset("mode_index", data=np.array([int(m) for m in modes], dtype=np.int32))
        for field in fields:
            ig.create_dataset(field, data=np.array([ir[m][field] for m in modes], dtype=float))

    # Raman
    raman = _safe(p.get_raman_spectrum, default={}, label="raman_spectrum")
    if raman:
        rg = grp.require_group("raman")
        modes = sorted(raman.keys(), key=int)
        rg.create_dataset("mode_index",    data=np.array([int(m) for m in modes], dtype=np.int32))
        rg.create_dataset("frequency",     data=np.array([raman[m]["frequency"]   for m in modes], dtype=float))
        rg.create_dataset("activity",      data=np.array([raman[m]["activity"]    for m in modes], dtype=float))
        rg.create_dataset("depolarization", data=np.array([raman[m]["depolarization"] for m in modes], dtype=float))

    # VCD
    vcd = _safe(p.get_vcd_spectrum, default={}, label="vcd_spectrum")
    if vcd:
        vg = grp.require_group("vcd")
        modes = sorted(vcd.keys(), key=int)
        vg.create_dataset("mode_index", data=np.array([int(m) for m in modes], dtype=np.int32))
        vg.create_dataset("frequency", data=np.array([vcd[m]["frequency"] for m in modes], dtype=float))
        vg.create_dataset("intensity", data=np.array([vcd[m]["intensity"] for m in modes], dtype=float))


def write_normal_modes(f: h5py.File, p: OrcaOutputParser):
    modes = _safe(p.get_normal_modes, default=None, label="normal_modes")
    if modes is not None and modes.size:
        grp = f.require_group("normal_modes")
        grp.create_dataset("modes", data=modes.astype(float),
                           compression="gzip", compression_opts=4)
        grp["modes"].attrs["shape_description"] = "(n_modes, n_atoms, 3)"


def write_orbital_energies(f: h5py.File, p: OrcaOutputParser):
    orbs = _safe(p.get_orbital_energy, default=[], label="orbital_energies")
    if not orbs:
        return
    grp = f.require_group("orbital_energies")
    grp.create_dataset("orbital",    data=np.array([o["orbital"]    for o in orbs], dtype=np.int32))
    grp.create_dataset("occupation", data=np.array([o["occupation"] for o in orbs], dtype=float))
    grp.create_dataset("energy_Eh",  data=np.array([o["energy_Eh"]  for o in orbs], dtype=float))
    grp.create_dataset("energy_eV",  data=np.array([o["energy_eV"]  for o in orbs], dtype=float))


def write_frontier_orbitals(f: h5py.File, p: OrcaOutputParser):
    fo = _safe(p.get_frontier_orbital, default={}, label="frontier_orbitals")
    if not fo:
        return
    grp = f.require_group("frontier_orbitals")
    orbs = fo.get("orbitals", {})

    for label in ("HOMO", "LUMO"):
        info = orbs.get(label, {})
        key = label.lower()
        _write_scalar(grp, f"{key}_index", info.get("index"))
        pop = info.get("population", {})
        if pop:
            pg = grp.require_group(f"{key}_population")
            indices = sorted(pop.keys())
            pg.create_dataset("atom_index", data=np.array(indices, dtype=np.int32))
            pg.create_dataset("element",    data=_string_array([pop[i]["element"]  for i in indices]))
            pg.create_dataset("mulliken",   data=np.array([pop[i]["Mulliken"] for i in indices], dtype=float))
            pg.create_dataset("loewdin",    data=np.array([pop[i]["Loewdin"]  for i in indices], dtype=float))


def write_mayer(f: h5py.File, p: OrcaOutputParser):
    grp = f.require_group("mayer")

    analysis = _safe(p.get_mayer_analyis, default=[], label="mayer_analysis")
    if analysis:
        ag = grp.require_group("analysis")
        fields = ["atom_index", "gross_atomic_population", "total_nuclear_charge",
                  "gross_atomic_charge", "total_valence", "bonded_valence", "free_valence"]
        for field in fields:
            ag.create_dataset(field, data=np.array([row[field] for row in analysis], dtype=float))

    bo = _safe(p.get_mayer_bond_order, default={}, label="mayer_bond_order")
    if bo:
        bg = grp.require_group("bond_orders")
        pairs = list(bo.keys())
        bg.create_dataset("atom1",      data=np.array([a for a, _ in pairs], dtype=np.int32))
        bg.create_dataset("atom2",      data=np.array([b for _, b in pairs], dtype=np.int32))
        bg.create_dataset("bond_order", data=np.array([bo[pair] for pair in pairs], dtype=float))


def write_nmr_efg(f: h5py.File, p: OrcaOutputParser):
    nmr = _safe(p.get_nmr_efg, default={}, label="nmr_efg")
    if not nmr:
        return
    grp = f.require_group("nmr_efg")
    for atom_idx, data in nmr.items():
        ag = grp.require_group(str(atom_idx))
        ag.attrs["atom_label"] = data.get("atom_label", "")
        ag.attrs["units"]      = data.get("units", "")

        tensor = data.get("tensor", {})
        if tensor:
            mat = np.array([
                [tensor["XX"], tensor["XY"], tensor["XZ"]],
                [tensor["YX"], tensor["YY"], tensor["YZ"]],
                [tensor["ZX"], tensor["ZY"], tensor["ZZ"]],
            ], dtype=float)
            ag.create_dataset("tensor", data=mat)

        principal = data.get("principal", {})
        if principal:
            ag.create_dataset("principal",
                              data=np.array([principal.get("Vxx"), principal.get("Vyy"), principal.get("Vzz")],
                                            dtype=float))

        eta = data.get("asymmetry_parameter")
        if eta is not None:
            _write_scalar(ag, "asymmetry_parameter", eta)


def write_trajectory(f: h5py.File, p: OrcaOutputParser):
    traj    = _safe(p.get_trajectory,       default=[], label="trajectory")
    energies= _safe(p.get_traj_energies,    default=[], label="traj_energies")
    forces  = _safe(p.get_forces,           default=[], label="forces")
    disp_f  = _safe(p.get_dispersion_forces,default=[], label="dispersion_forces")

    if not traj:
        return

    grp = f.require_group("trajectory")

    for i, (symbols, positions) in enumerate(traj):
        sg = grp.require_group(f"step_{i}")
        sg.create_dataset("symbols",   data=_string_array(symbols))
        sg.create_dataset("positions", data=np.array(positions, dtype=float))
        if i < len(energies):
            _write_scalar(sg, "energy", energies[i])

    if forces:
        fg = grp.require_group("forces")
        for i, step_forces in enumerate(forces):
            fg.create_dataset(f"step_{i}", data=np.array(step_forces, dtype=float))

    if disp_f:
        dg = grp.require_group("dispersion_forces")
        for i, step_forces in enumerate(disp_f):
            dg.create_dataset(f"step_{i}", data=np.array(step_forces, dtype=float))


def write_hessian(f: h5py.File, hp: OrcaHessParser):
    grp = f.require_group("hessian")

    hess = _safe(hp.get_hessian, default=None, label="hessian")
    if hess is not None and hess.size:
        grp.create_dataset("hessian", data=hess.astype(float),
                           compression="gzip", compression_opts=4)
        grp["hessian"].attrs["shape_description"] = "(n_modes, n_atoms, 3)"

    dd = _safe(hp.get_dipole_derivative, default=None, label="dipole_derivatives")
    if dd:
        grp.create_dataset("dipole_derivatives", data=np.array(dd, dtype=float))

    pd = _safe(hp.get_polarization_derivative, default=None, label="polarization_derivatives")
    if pd:
        grp.create_dataset("polarization_derivatives", data=np.array(pd, dtype=float))


# ============================================================
# Main converter
# ============================================================

def convert_calculation(out_file: Path, hess_file: Path | None, output_dir: Path):
    stem = out_file.stem
    h5_path = output_dir / f"{stem}.h5"

    log.info("Converting %s → %s", out_file.name, h5_path.name)

    p = OrcaOutputParser(str(out_file))

    with h5py.File(h5_path, "w") as f:
        f.attrs["source_out"]  = str(out_file)
        f.attrs["source_hess"] = str(hess_file) if hess_file else ""
        f.attrs["orca_parser_version"] = "1.0"

        write_metadata(f, p)
        write_geometry(f, p)
        write_charges(f, p)
        write_energies(f, p)
        write_multipoles(f, p)
        write_polarizability(f, p)
        write_spectra(f, p)
        write_normal_modes(f, p)
        write_orbital_energies(f, p)
        write_frontier_orbitals(f, p)
        write_mayer(f, p)
        write_nmr_efg(f, p)
        write_trajectory(f, p)

        if hess_file is not None:
            hp = OrcaHessParser(str(hess_file))
            write_hessian(f, hp)

    log.info("  ✓ Wrote %s (%.1f KB)", h5_path.name, h5_path.stat().st_size / 1024)
    return h5_path


def find_calculations(search_dir: Path) -> dict[str, dict]:
    """
    Scan *search_dir* recursively and group .out / .hess files by stem.
    Returns {stem: {"out": Path, "hess": Path | None}}.
    """
    out_files  = {p.stem: p for p in search_dir.rglob("*.out")}
    hess_files = {p.stem: p for p in search_dir.rglob("*.hess")}

    calcs = {}
    for stem, out_path in out_files.items():
        calcs[stem] = {
            "out":  out_path,
            "hess": hess_files.get(stem),
        }
    return calcs


def main():
    parser = argparse.ArgumentParser(
        description="Convert a folder of ORCA calculations to HDF5 (one file per calculation)."
    )
    parser.add_argument("input_dir",  type=Path, help="Directory containing .out / .hess files")
    parser.add_argument("output_dir", type=Path, nargs="?", default=None,
                        help="Directory for .h5 output files (default: same as input_dir)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    input_dir  = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    calcs = find_calculations(input_dir)
    if not calcs:
        log.error("No .out files found in %s", input_dir)
        sys.exit(1)

    log.info("Found %d calculation(s) in %s", len(calcs), input_dir)

    ok, failed = 0, []
    for stem, paths in calcs.items():
        try:
            convert_calculation(paths["out"], paths["hess"], output_dir)
            ok += 1
        except Exception as exc:
            log.error("FAILED %s: %s", stem, exc, exc_info=args.verbose)
            failed.append(stem)

    log.info("Done — %d succeeded, %d failed.", ok, len(failed))
    if failed:
        log.warning("Failed stems: %s", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()