"""
orca_to_hdf5.py
---------------
Convert a directory of ORCA calculations to HDF5 — one file per calculation.

Expected layout on disk
-----------------------
Each calculation lives in its own sub-folder named by an InChIKey or any
identifier.  The HDF5 file takes that folder name as its stem:

    ZZYASVWWDLJXIM-UHFFFAOYSA-N/
        WEWNES.out
        WEWNES.hess
        WEWNES_slurm.out    ← ignored
        WEWNES_slurm.log    ← ignored
        WEWNES_trj.xyz      ← ignored
    →  ZZYASVWWDLJXIM-UHFFFAOYSA-N.h5

When a folder contains more than one .out file (e.g. LOSDAA.out and
LOSDAA_atom47.out), the file whose stem is shared by the most other files
in that folder is chosen as the primary output.

HDF5 structure
--------------
/metadata/
    cal_setting/        attrs: method, functional_*, hartree_fock_type, …
    symmetry/           attrs: point_group, symmetry_number
                        datasets: rotational_constants_cm**-1, rotational_constants_MHz
    formula             string scalar
    smiles              string scalar
    mass                float64 scalar
    cheminformatics/    attrs: TPSA, logP, n_aromatic_bonds, …

/geometry/
    symbols             (N,)    string
    positions           (N, 3)  float64  [Å]

/charges/
    mulliken            (N,)    float64
    loewdin             (N,)    float64

/energies/
    thermal_energy      float64  [kcal/mol]
    enthalpy            float64  [kcal/mol]
    zero_point_energy   float64  [kcal/mol]
    gibbs_free_energy   float64  [kcal/mol]
    total_entropy       float64  [cal/(mol·K)]
    scf/                nuclear_repulsion, electronic_energy, …
    corrections/        thermal_vibrational_correction, …

/multipoles/
    dipole/             electronic_contribution, nuclear_contribution,
                        total_dipole_moment, magnitude/{au,Debye},
                        along_rotational_axis/{au,Debye}
    quadrupole/
        electronic/     XX YY ZZ XY XZ YZ
        nuclear/        XX YY ZZ XY XZ YZ
        total/au/       XX YY ZZ XY XZ YZ
        total/buckingham/ XX YY ZZ XY XZ YZ
        principal/au/   XX YY ZZ
        principal/buckingham/ XX YY ZZ
        isotropic       float64 scalar

/polarizability/
    dipole_dipole/
        cartesian_au/   XX XY XZ YX YY YZ ZX ZY ZZ
        principal_au/   XX YY ZZ
        isotropic_au    float64 scalar
    dipole_quadrupole/
        components      string array of labels
        values          float64 array
    traceless_dipole_quadrupole/
        components      string array of labels
        values          float64 array

/spectra/
    ir/     mode_index  frequency  epsilon  intensity  T2  Tx  Ty  Tz
    raman/  mode_index  frequency  activity  depolarization
    vcd/    mode_index  frequency  intensity

/normal_modes/
    modes               (n_modes, n_atoms, 3)  float64

/orbital_energies/
    orbital             (M,)  int32
    occupation          (M,)  float64
    energy_Eh           (M,)  float64
    energy_eV           (M,)  float64

/frontier_orbitals/
    homo_index          int32 scalar
    lumo_index          int32 scalar
    homo_population/    atom_index  element  mulliken  loewdin
    lumo_population/    atom_index  element  mulliken  loewdin

/mayer/
    analysis/           atom_index  gross_atomic_population  total_nuclear_charge
                        gross_atomic_charge  total_valence  bonded_valence  free_valence
    bond_orders/        atom1  atom2  bond_order

/nmr_efg/
    <atom_index>/
        tensor              (3, 3)  float64
        principal           (3,)    float64   [Vxx, Vyy, Vzz]
        asymmetry_parameter float64 scalar
        attrs: atom_label, units

/trajectory/
    step_<i>/
        symbols             (N,)    string
        positions           (N, 3)  float64  [Å]
        energy              float64 scalar   [kcal/mol]
    forces/
        step_<i>            (N, 3)  float64
    dispersion_forces/
        step_<i>            (N, 3)  float64

/hessian/               (present only when a .hess file exists)
    hessian                 (n_modes, n_atoms, 3)  float64
    dipole_derivatives      (3N, 3)  float64
    polarization_derivatives (3N, 6)  float64
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from moldesc.orca_parser.parse_output import OrcaOutputParser, OrcaHessParser

log = logging.getLogger(__name__)

# ── File-name patterns that are never the primary ORCA output ────────────────
_EXCLUDED_STEM_SUFFIXES = ("_slurm", "_trj", "_atom")
_EXCLUDED_EXTENSIONS    = {".log"}


# ============================================================
# Low-level HDF5 helpers
# ============================================================

def _safe(fn, *args, default=None, label: str = ""):
    """Call fn(*args); on any exception return *default* and log at DEBUG."""
    try:
        return fn(*args)
    except Exception as exc:
        if label:
            log.debug("Skipping %s: %s", label, exc)
        return default


def _str_ds(group: h5py.Group, name: str, value: str):
    """Write a scalar string dataset."""
    group.create_dataset(name, data=value, dtype=h5py.string_dtype())


def _scalar(group: h5py.Group, name: str, value: Any):
    """Write a scalar numeric or string dataset; silently skip None."""
    if value is None:
        return
    if isinstance(value, str):
        _str_ds(group, name, value)
    else:
        group.create_dataset(name, data=value)


def _attrs(group: h5py.Group, d: dict):
    """Write a flat dict as HDF5 attributes; skip None values."""
    for k, v in d.items():
        if v is None:
            continue
        try:
            group.attrs[k] = v
        except Exception as exc:
            log.debug("Could not write attr %s: %s", k, exc)


def _str_arr(strings: list[str]) -> np.ndarray:
    return np.array(strings, dtype=h5py.string_dtype())


def _f64(data) -> np.ndarray:
    return np.array(data, dtype=np.float64)


def _i32(data) -> np.ndarray:
    return np.array(data, dtype=np.int32)


# ============================================================
# Section writers  (one function per HDF5 top-level group)
# ============================================================

def _write_metadata(f: h5py.File, p: OrcaOutputParser):
    grp = f.require_group("metadata")

    cal = _safe(p.get_cal_setting, default={}, label="cal_setting")
    if cal:
        _attrs(grp.require_group("cal_setting"), cal)

    sym = _safe(p.get_symmetry, default={}, label="symmetry")
    if sym:
        sg = grp.require_group("symmetry")
        for k, v in sym.items():
            if isinstance(v, list):
                sg.create_dataset(k, data=_f64(v))
            else:
                sg.attrs[k] = v

    for name, fn in [("formula", p.get_formula), ("smiles", p.get_smile)]:
        val = _safe(fn, default=None, label=name)
        if val is not None:
            _scalar(grp, name, val)

    _scalar(grp, "mass", _safe(p.get_mass, default=None, label="mass"))

    cheminf = _safe(p.get_cheminformatic, default={}, label="cheminformatics")
    if cheminf:
        _attrs(grp.require_group("cheminformatics"), cheminf)


def _write_geometry(f: h5py.File, p: OrcaOutputParser):
    atoms = _safe(p.get_ase_atoms, default=None, label="geometry")
    if atoms is None:
        return
    grp = f.require_group("geometry")
    grp.create_dataset("symbols",   data=_str_arr(atoms.get_chemical_symbols()))
    grp.create_dataset("positions", data=_f64(atoms.get_positions()))


def _write_charges(f: h5py.File, p: OrcaOutputParser):
    grp = f.require_group("charges")
    for name, fn in [("mulliken", p.get_mulliken),
                     ("loewdin",  p.get_loewdin_charges)]:
        val = _safe(fn, default=None, label=name)
        if val:
            grp.create_dataset(name, data=_f64(val))


def _write_energies(f: h5py.File, p: OrcaOutputParser):
    grp = f.require_group("energies")

    for name, fn in [
        ("thermal_energy",    p.get_thermal_energy),
        ("enthalpy",          p.get_enthalpy),
        ("zero_point_energy", p.get_zero_point_energy),
        ("gibbs_free_energy", p.get_gibbs_free_energy),
        ("total_entropy",     p.get_total_enthropy),
    ]:
        _scalar(grp, name, _safe(fn, default=None, label=name))

    scf = _safe(p.get_scf_correction, default={}, label="scf_correction")
    if scf:
        sg = grp.require_group("scf")
        for k, v in scf.items():
            _scalar(sg, k, v)

    corr = _safe(p.energy_correction, default={}, label="energy_corrections")
    if corr:
        cg = grp.require_group("corrections")
        for k, v in corr.items():
            _scalar(cg, k, v)


def _write_tensor_components(group: h5py.Group, d: dict,
                              keys=("XX", "YY", "ZZ", "XY", "XZ", "YZ")):
    """Write a dict of tensor components as scalar datasets."""
    for k in keys:
        if k in d:
            _scalar(group, k, d[k])


def _write_multipoles(f: h5py.File, p: OrcaOutputParser):
    grp = f.require_group("multipoles")

    # ── dipole ──────────────────────────────────────────────────────────
    dip = _safe(p.get_dipole_moment, default={}, label="dipole_moment")
    if dip:
        dg = grp.require_group("dipole")
        for key, val in dip.items():
            if isinstance(val, list):
                dg.create_dataset(key, data=_f64(val))
            elif isinstance(val, dict):
                sg = dg.require_group(key)
                for unit, v in val.items():
                    if isinstance(v, list):
                        sg.create_dataset(unit, data=_f64(v))
                    else:
                        _scalar(sg, unit, v)

    # ── quadrupole ───────────────────────────────────────────────────────
    quad = _safe(p.get_quadrupole_moment, default={}, label="quadrupole_moment")
    if quad:
        qg = grp.require_group("quadrupole")

        for section_key in ("electronic_contribution_to_quadrupole_moment",
                            "nuclear_contribution_to_quadrupole_moment"):
            short = "electronic" if "electronic" in section_key else "nuclear"
            d = quad.get(section_key, {})
            if d:
                _write_tensor_components(qg.require_group(short), d)

        tot = quad.get("total_quadrupole_moment", {})
        if tot:
            tg = qg.require_group("total")
            for unit, td in tot.items():
                if isinstance(td, dict):
                    _write_tensor_components(tg.require_group(unit), td)

        pri = quad.get("principal_quadrupole_moment", {})
        if pri:
            pg = qg.require_group("principal")
            for unit, pd in pri.items():
                if isinstance(pd, dict):
                    _write_tensor_components(pg.require_group(unit), pd,
                                             keys=("XX", "YY", "ZZ"))

        iso = quad.get("isotropic_quadrupole")
        if iso is not None:
            _scalar(qg, "isotropic", iso)


def _write_polarizability(f: h5py.File, p: OrcaOutputParser):
    grp = f.require_group("polarizability")

    # ── dipole/dipole ────────────────────────────────────────────────────
    dd = _safe(p.get_static_polarizability_dipole_dipole,
               default={}, label="dipole_dipole_polarizability")
    if dd:
        ddg = grp.require_group("dipole_dipole")
        cart = dd.get("cartesian_tensor_au", {})
        if cart:
            _write_tensor_components(
                ddg.require_group("cartesian_au"), cart,
                keys=("XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"),
            )
        prin = dd.get("principal_polarizability_au", {})
        if prin:
            _write_tensor_components(
                ddg.require_group("principal_au"), prin,
                keys=("XX", "YY", "ZZ"),
            )
        _scalar(ddg, "isotropic_au", dd.get("isotropic_polarizability_au"))

    # ── dipole/quadrupole (raw and traceless) ────────────────────────────
    for fn, group_name in [
        (p.get_static_polarizability_dipole_quadrupole,          "dipole_quadrupole"),
        (p.get_static_traceless_polarizability_dipole_quadrupole, "traceless_dipole_quadrupole"),
    ]:
        val = _safe(fn, default={}, label=group_name)
        if val:
            sg = grp.require_group(group_name)
            keys = list(val.keys())
            sg.create_dataset("components", data=_str_arr(keys))
            sg.create_dataset("values",     data=_f64([val[k] for k in keys]))


def _write_spectra(f: h5py.File, p: OrcaOutputParser):
    grp = f.require_group("spectra")

    # IR
    ir = _safe(p.get_ir_spectrum, default={}, label="ir_spectrum")
    if ir:
        ig = grp.require_group("ir")
        modes = sorted(ir, key=int)
        ig.create_dataset("mode_index", data=_i32([int(m) for m in modes]))
        for field in ("frequency", "epsilon", "intensity", "T2", "Tx", "Ty", "Tz"):
            ig.create_dataset(field, data=_f64([ir[m][field] for m in modes]))

    # Raman
    raman = _safe(p.get_raman_spectrum, default={}, label="raman_spectrum")
    if raman:
        rg = grp.require_group("raman")
        modes = sorted(raman, key=int)
        rg.create_dataset("mode_index",     data=_i32([int(m) for m in modes]))
        rg.create_dataset("frequency",      data=_f64([raman[m]["frequency"]      for m in modes]))
        rg.create_dataset("activity",       data=_f64([raman[m]["activity"]       for m in modes]))
        rg.create_dataset("depolarization", data=_f64([raman[m]["depolarization"] for m in modes]))

    # VCD
    vcd = _safe(p.get_vcd_spectrum, default={}, label="vcd_spectrum")
    if vcd:
        vg = grp.require_group("vcd")
        modes = sorted(vcd, key=int)
        vg.create_dataset("mode_index", data=_i32([int(m) for m in modes]))
        vg.create_dataset("frequency",  data=_f64([vcd[m]["frequency"]  for m in modes]))
        vg.create_dataset("intensity",  data=_f64([vcd[m]["intensity"]  for m in modes]))


def _write_normal_modes(f: h5py.File, p: OrcaOutputParser):
    modes = _safe(p.get_normal_modes, default=None, label="normal_modes")
    if modes is not None and modes.size:
        ds = f.require_group("normal_modes").create_dataset(
            "modes", data=modes.astype(np.float64),
            compression="gzip", compression_opts=4,
        )
        ds.attrs["shape_description"] = "(n_modes, n_atoms, 3)"


def _write_orbital_energies(f: h5py.File, p: OrcaOutputParser):
    orbs = _safe(p.get_orbital_energy, default=[], label="orbital_energies")
    if not orbs:
        return
    grp = f.require_group("orbital_energies")
    grp.create_dataset("orbital",    data=_i32([o["orbital"]    for o in orbs]))
    grp.create_dataset("occupation", data=_f64([o["occupation"] for o in orbs]))
    grp.create_dataset("energy_Eh",  data=_f64([o["energy_Eh"]  for o in orbs]))
    grp.create_dataset("energy_eV",  data=_f64([o["energy_eV"]  for o in orbs]))


def _write_frontier_orbitals(f: h5py.File, p: OrcaOutputParser):
    fo = _safe(p.get_frontier_orbital, default={}, label="frontier_orbitals")
    if not fo:
        return
    grp  = f.require_group("frontier_orbitals")
    orbs = fo.get("orbitals", {})

    for label in ("HOMO", "LUMO"):
        info = orbs.get(label, {})
        key  = label.lower()
        _scalar(grp, f"{key}_index", info.get("index"))
        pop = info.get("population", {})
        if pop:
            pg      = grp.require_group(f"{key}_population")
            indices = sorted(pop)
            pg.create_dataset("atom_index", data=_i32(indices))
            pg.create_dataset("element",    data=_str_arr([pop[i]["element"]  for i in indices]))
            pg.create_dataset("mulliken",   data=_f64([pop[i]["Mulliken"] for i in indices]))
            pg.create_dataset("loewdin",    data=_f64([pop[i]["Loewdin"]  for i in indices]))


def _write_mayer(f: h5py.File, p: OrcaOutputParser):
    grp = f.require_group("mayer")

    analysis = _safe(p.get_mayer_analyis, default=[], label="mayer_analysis")
    if analysis:
        ag = grp.require_group("analysis")
        for field in ("atom_index", "gross_atomic_population", "total_nuclear_charge",
                      "gross_atomic_charge", "total_valence", "bonded_valence", "free_valence"):
            ag.create_dataset(field, data=_f64([row[field] for row in analysis]))

    bo = _safe(p.get_mayer_bond_order, default={}, label="mayer_bond_orders")
    if bo:
        bg    = grp.require_group("bond_orders")
        pairs = list(bo)
        bg.create_dataset("atom1",      data=_i32([a for a, _ in pairs]))
        bg.create_dataset("atom2",      data=_i32([b for _, b in pairs]))
        bg.create_dataset("bond_order", data=_f64([bo[pair] for pair in pairs]))


def _write_nmr_efg(f: h5py.File, p: OrcaOutputParser):
    nmr = _safe(p.get_nmr_efg, default={}, label="nmr_efg")
    if not nmr:
        return
    grp = f.require_group("nmr_efg")
    for atom_idx, data in nmr.items():
        ag = grp.require_group(str(atom_idx))
        ag.attrs["atom_label"] = data.get("atom_label", "")
        ag.attrs["units"]      = data.get("units", "")

        t = data.get("tensor", {})
        if t:
            mat = np.array([
                [t["XX"], t["XY"], t["XZ"]],
                [t["YX"], t["YY"], t["YZ"]],
                [t["ZX"], t["ZY"], t["ZZ"]],
            ], dtype=np.float64)
            ag.create_dataset("tensor", data=mat)

        pri = data.get("principal", {})
        if pri:
            ag.create_dataset(
                "principal",
                data=_f64([pri.get("Vxx"), pri.get("Vyy"), pri.get("Vzz")]),
            )

        eta = data.get("asymmetry_parameter")
        if eta is not None:
            _scalar(ag, "asymmetry_parameter", eta)


def _write_trajectory(f: h5py.File, p: OrcaOutputParser):
    traj     = _safe(p.get_trajectory,        default=[], label="trajectory")
    energies = _safe(p.get_traj_energies,     default=[], label="traj_energies")
    forces   = _safe(p.get_forces,            default=[], label="forces")
    disp_f   = _safe(p.get_dispersion_forces, default=[], label="dispersion_forces")

    if not traj:
        return

    grp = f.require_group("trajectory")

    for i, (symbols, positions) in enumerate(traj):
        sg = grp.require_group(f"step_{i}")
        sg.create_dataset("symbols",   data=_str_arr(symbols))
        sg.create_dataset("positions", data=_f64(positions))
        if i < len(energies):
            _scalar(sg, "energy", energies[i])

    if forces:
        fg = grp.require_group("forces")
        for i, step in enumerate(forces):
            fg.create_dataset(f"step_{i}", data=_f64(step))

    if disp_f:
        dg = grp.require_group("dispersion_forces")
        for i, step in enumerate(disp_f):
            dg.create_dataset(f"step_{i}", data=_f64(step))


def _write_hessian(f: h5py.File, hp: OrcaHessParser):
    grp = f.require_group("hessian")

    hess = _safe(hp.get_hessian, default=None, label="hessian")
    if hess is not None and hess.size:
        ds = grp.create_dataset(
            "hessian", data=hess.astype(np.float64),
            compression="gzip", compression_opts=4,
        )
        ds.attrs["shape_description"] = "(n_modes, n_atoms, 3)"

    dd = _safe(hp.get_dipole_derivative, default=None, label="dipole_derivatives")
    if dd:
        grp.create_dataset("dipole_derivatives", data=_f64(dd))

    pd = _safe(hp.get_polarization_derivative, default=None, label="polarization_derivatives")
    if pd:
        grp.create_dataset("polarization_derivatives", data=_f64(pd))


# ============================================================
# File discovery
# ============================================================

def _is_excluded(path: Path) -> bool:
    """True for slurm logs, trajectory files, and atom-specific outputs."""
    if path.suffix in _EXCLUDED_EXTENSIONS:
        return True
    return any(path.stem.endswith(s) for s in _EXCLUDED_STEM_SUFFIXES)


def _pick_primary_out(subdir: Path, candidates: list[Path]) -> Path:
    """
    From a list of .out candidates, return the one whose stem is shared
    by the most other files in the folder.  Ties are broken by shortest stem.
    """
    all_stems = [f.stem for f in subdir.iterdir() if f.is_file()]
    def score(p: Path) -> tuple:
        count = sum(1 for s in all_stems if s == p.stem or s.startswith(p.stem + "_"))
        return (-count, len(p.stem))   # most matches first, then shortest
    candidates.sort(key=score)
    chosen = candidates[0]
    ignored = [c.name for c in candidates[1:]]
    if ignored:
        log.info("%s: multiple .out files — using %s (ignoring %s)",
                 subdir.name, chosen.name, ignored)
    return chosen


def find_calculations(search_dir: Path) -> list[dict]:
    """
    Return a list of dicts, each describing one calculation:
        {"out": Path, "hess": Path|None, "h5_stem": str}

    Sub-folder mode (primary): each immediate sub-directory is one
    calculation; the HDF5 stem is the sub-folder name.

    Flat mode (fallback): if no sub-directories are found, treat .out
    files directly inside search_dir as individual calculations.
    """
    subdirs = [d for d in sorted(search_dir.iterdir()) if d.is_dir()]

    calcs = []
    for subdir in subdirs:
        candidates = [f for f in subdir.glob("*.out") if not _is_excluded(f)]
        if not candidates:
            log.debug("%s: no .out file found — skipping", subdir.name)
            continue

        out_file  = _pick_primary_out(subdir, candidates)
        hess_file = out_file.with_suffix(".hess")

        calcs.append({
            "out":     out_file,
            "hess":    hess_file if hess_file.exists() else None,
            "h5_stem": subdir.name,
        })

    # Flat fallback
    if not calcs:
        for out_file in sorted(search_dir.glob("*.out")):
            if _is_excluded(out_file):
                continue
            hess_file = out_file.with_suffix(".hess")
            calcs.append({
                "out":     out_file,
                "hess":    hess_file if hess_file.exists() else None,
                "h5_stem": out_file.stem,
            })

    return calcs


# ============================================================
# Top-level converter
# ============================================================

def convert(out_file: Path, hess_file: Path | None,
            output_dir: Path, h5_stem: str) -> Path:
    """Parse one ORCA calculation and write a single HDF5 file."""
    h5_path = output_dir / f"{h5_stem}.h5"
    log.info("%-55s → %s", str(out_file), h5_path.name)

    p = OrcaOutputParser(str(out_file))

    with h5py.File(h5_path, "w") as f:
        f.attrs["calculation_id"]      = h5_stem
        f.attrs["source_out"]          = str(out_file)
        f.attrs["source_hess"]         = str(hess_file) if hess_file else ""
        f.attrs["orca_parser_version"] = "1.0"

        _write_metadata(f, p)
        _write_geometry(f, p)
        _write_charges(f, p)
        _write_energies(f, p)
        _write_multipoles(f, p)
        _write_polarizability(f, p)
        _write_spectra(f, p)
        _write_normal_modes(f, p)
        _write_orbital_energies(f, p)
        _write_frontier_orbitals(f, p)
        _write_mayer(f, p)
        _write_nmr_efg(f, p)
        _write_trajectory(f, p)

        if hess_file is not None:
            _write_hessian(f, OrcaHessParser(str(hess_file)))

    log.info("  ✓ %.1f KB", h5_path.stat().st_size / 1024)
    return h5_path


# ============================================================
# CLI entry point
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Convert ORCA output folders to HDF5 (one .h5 per folder)."
    )
    ap.add_argument("input_dir",  type=Path,
                    help="Root directory containing per-calculation sub-folders")
    ap.add_argument("output_dir", type=Path, nargs="?", default=None,
                    help="Destination for .h5 files (default: same as input_dir)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Enable DEBUG logging")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    input_dir  = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    calcs = find_calculations(input_dir)
    if not calcs:
        log.error("No ORCA .out files found in %s", input_dir)
        sys.exit(1)

    log.info("Found %d calculation(s)", len(calcs))

    succeeded, failed = [], []
    for c in calcs:
        try:
            convert(c["out"], c["hess"], output_dir, c["h5_stem"])
            succeeded.append(c["h5_stem"])
        except Exception as exc:
            log.error("FAILED %s: %s", c["h5_stem"], exc, exc_info=args.verbose)
            failed.append(c["h5_stem"])

    log.info("Done — %d succeeded, %d failed", len(succeeded), len(failed))
    if failed:
        log.warning("Failed: %s", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()