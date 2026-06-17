from __future__ import print_function

__author__ = "Dr. Dinga Wonanke"
__status__ = "production"

import json
from typing import Dict, Any, Optional

from rdkit import Chem
from rdkit.Chem import (
    Descriptors,
    rdMolDescriptors,
    rdFingerprintGenerator,
    MACCSkeys,
)
from moldesc import loader


_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2,
                                                        fpSize=2048
                                                        )

rdkit_resource = loader.load_data("rdkit_desc.json")
smart_search = rdkit_resource.get("smart_search")
DESCRIPTOR_LABELS = rdkit_resource.get("labels")
SMART_SEARCH_LABELS = rdkit_resource.get("smart_search_labels", {})
FR_TO_SMART_SEARCH_MAP = rdkit_resource.get("smart_search_map", {})
FR_STANDALONE = rdkit_resource.get("fr_standalone", {})


class RDKitDescriptors:
    """
    Compute RDKit molecular descriptors, fingerprints, atom/bond data,
    and functional-group annotations from a SMILES string.

    Parameters
    ----------
    smile : str
        Input SMILES string representing the molecule.
    name : str, optional
        Optional molecule name or identifier.

    Attributes
    ----------
    smile : str
        Original input SMILES string.
    name : str or None
        Optional molecule name.
    mol : rdkit.Chem.Mol
        RDKit molecule object generated from the input SMILES.
    mol_h : rdkit.Chem.Mol
        Hydrogen-expanded RDKit molecule.

    Raises
    ------
    ValueError
        If the input SMILES string cannot be parsed by RDKit.

    Notes
    -----
    The output is designed to be JSON serialisable and suitable for
    storage in document databases or PostgreSQL JSON/JSONB columns.
    """

    def __init__(self, smile: str, name: Optional[str] = None):
        """
        Initialise the descriptor calculator from a SMILES string.

        Parameters
        ----------
        smile : str
            SMILES representation of the molecule.
        name : str, optional
            Optional molecule name or external identifier.

        Raises
        ------
        ValueError
            If RDKit fails to parse the supplied SMILES string.
        """
        self.smile = smile
        self.name = name
        self.mol = Chem.MolFromSmiles(smile)

        if self.mol is None:
            raise ValueError(f"Invalid SMILES: {smile}")

        self.mol_h = Chem.AddHs(self.mol)

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """
        Convert values to JSON-safe Python scalar types where possible.

        Parameters
        ----------
        value : Any
            Value returned by RDKit or another numerical backend.

        Returns
        -------
        Any
            JSON-serialisable value. NumPy scalar values are converted to
            native Python scalar types. Other values are returned unchanged.
        """
        try:
            import numpy as np

            if isinstance(value, np.generic):
                return value.item()
        except ImportError:
            pass

        return value

    def identifiers(self) -> Dict[str, Any]:
        """
        Generate standard molecular identifiers.

        Returns
        -------
        dict
            Dictionary containing canonical SMILES, isomeric SMILES, InChI,
            InChIKey, and molecular formula.
        """
        return {
            "canonical_smiles": Chem.MolToSmiles(
                self.mol,
                canonical=True,
            ),
            "isomeric_smiles": Chem.MolToSmiles(
                self.mol,
                canonical=True,
                isomericSmiles=True,
            ),
            "inchi": Chem.MolToInchi(self.mol),
            "inchi_key": Chem.MolToInchiKey(self.mol),
            "molecular_formula": rdMolDescriptors.CalcMolFormula(self.mol),
        }

    def ring_properties(self) -> Dict[str, Any]:
        """
        Calculate molecular ring statistics.

        Returns
        -------
        dict
            Ring-related descriptors including total rings, aromatic rings,
            aliphatic rings, saturated rings, heterocycles, ring atom count,
            and individual ring sizes.
        """
        ring_info = self.mol.GetRingInfo()

        return {
            "num_rings": rdMolDescriptors.CalcNumRings(self.mol),
            "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(self.mol),
            "num_aliphatic_rings": rdMolDescriptors.CalcNumAliphaticRings(self.mol),
            "num_saturated_rings": rdMolDescriptors.CalcNumSaturatedRings(self.mol),
            "num_aromatic_heterocycle": rdMolDescriptors.CalcNumAromaticHeterocycles(self.mol),
            "num_aliphatic_heterocycles": rdMolDescriptors.CalcNumAliphaticHeterocycles(self.mol),
            "num_saturated_heterocycles": rdMolDescriptors.CalcNumSaturatedHeterocycles(self.mol),
            "ring_atom_count": len(set(a for ring in ring_info.AtomRings() for a in ring)),
            "ring_sizes": [len(ring) for ring in ring_info.AtomRings()],
        }

    def all_rdkit_descriptors(self) -> Dict[str, Any]:
        """
        Calculate all descriptors available from ``rdkit.Chem.Descriptors``.

        Returns
        -------
        dict
            Mapping of RDKit descriptor names to calculated values. If a
            descriptor fails, its value is set to ``None``.
        """
        values = {}

        for name, func in Descriptors.descList:
            try:
                values[name] = func(self.mol)
            except Exception:
                values[name] = None

        return values

    def labelled_rdkit_descriptors(self) -> Dict[str, Any]:
        """
        Calculate labelled RDKit descriptors, excluding ``fr_`` descriptors.

        Returns
        -------
        dict
            Dictionary where each descriptor contains a human-readable label
            and calculated value.

        Notes
        -----
        RDKit descriptors whose names start with ``fr_`` are excluded because
        functional-group descriptors are handled separately in
        :meth:`unified_functional_groups`.
        """
        raw = self.all_rdkit_descriptors()
        labelled = {}

        for key, value in raw.items():
            if key.startswith("fr_"):
                continue

            labelled[key] = {
                "label": DESCRIPTOR_LABELS.get(key, key),
                "value": self._json_safe(value),
            }

        return labelled

    def unified_functional_groups(self) -> Dict[str, Any]:
        """
        Build a unified functional-group annotation dictionary.

        This combines SMARTS-based substructure searches with selected RDKit
        ``fr_`` functional-group descriptors.

        Returns
        -------
        dict
            Functional-group dictionary. Each entry contains source, category,
            label, description, count, optional RDKit ``fr_`` count, atom
            indices, and a Boolean presence flag.

        Notes
        -----
        If a functional group is detected by both SMARTS and an RDKit ``fr_``
        descriptor, the SMARTS-based result is preferred because it provides
        atom-level match indices.
        """
        results = {}

        reversed_map = {v: k for k, v in FR_TO_SMART_SEARCH_MAP.items()}

        for smart_key, smarts in smart_search.items():
            patt = Chem.MolFromSmarts(smarts)

            if patt is None:
                continue

            matches = self.mol.GetSubstructMatches(patt)
            label_info = SMART_SEARCH_LABELS.get(smart_key, {})
            fr_key = reversed_map.get(smart_key)
            fr_value = None

            if fr_key:
                try:
                    func = dict(Descriptors.descList).get(fr_key)
                    if func:
                        fr_value = int(func(self.mol))
                except Exception:
                    pass

            results[smart_key] = {
                "source": "smarts",
                "category": label_info.get("category", "Uncategorised"),
                "label": label_info.get("label", smart_key),
                "description": label_info.get("description", ""),
                "count": len(matches),
                "fr_count": fr_value,
                "atom_indices": [list(m) for m in matches],
                "present": len(matches) > 0,
            }

        fr_suppressed = set(FR_TO_SMART_SEARCH_MAP.keys())
        raw_descriptors = dict(Descriptors.descList)

        for fr_key, category in FR_STANDALONE.items():
            if fr_key in fr_suppressed:
                continue

            func = raw_descriptors.get(fr_key)

            if func is None:
                continue

            try:
                value = int(func(self.mol))
            except Exception:
                value = None

            label_info = DESCRIPTOR_LABELS.get(fr_key, {})

            if isinstance(label_info, str):
                label_info = {"label": label_info, "description": ""}

            results[fr_key] = {
                "source": "fr_descriptor",
                "category": category,
                "label": label_info.get("label", fr_key),
                "description": label_info.get("description", ""),
                "fr_count": value,
                "count": value,
                "atom_indices": [],
                "present": bool(value) if value is not None else False,
            }

        return results

    def fingerprints(self) -> Dict[str, Any]:
        """
        Generate molecular fingerprint bitstrings.

        Returns
        -------
        dict
            Dictionary containing Morgan radius-2 2048-bit fingerprint and
            MACCS keys fingerprint as bitstrings.
        """
        morgan = _MORGAN_GEN.GetFingerprint(self.mol)
        maccs = MACCSkeys.GenMACCSKeys(self.mol)

        return {
            "morgan_radius2_2048_bitstring": morgan.ToBitString(),
            "maccs_keys_bitstring": maccs.ToBitString(),
        }

    def atom_data(self) -> Dict[str, Any]:
        """
        Extract atom-level molecular information.

        Returns
        -------
        dict
            Dictionary containing a list of atom records. Each atom record
            includes index, symbol, atomic number, formal charge,
            hybridisation, aromaticity, degree, total valence, and implicit
            hydrogen count.
        """
        atoms = []

        for atom in self.mol.GetAtoms():
            atoms.append(
                {
                    "index": atom.GetIdx(),
                    "symbol": atom.GetSymbol(),
                    "atomic_number": atom.GetAtomicNum(),
                    "formal_charge": atom.GetFormalCharge(),
                    "hybridization": str(atom.GetHybridization()),
                    "is_aromatic": atom.GetIsAromatic(),
                    "degree": atom.GetDegree(),
                    "total_valence": atom.GetTotalValence(),
                    "implicit_hydrogens": atom.GetNumImplicitHs(),
                }
            )

        return {"atoms": atoms}

    def bond_data(self) -> Dict[str, Any]:
        """
        Extract bond-level molecular information.

        Returns
        -------
        dict
            Dictionary containing a list of bond records. Each bond record
            includes begin atom index, end atom index, bond type, aromaticity,
            conjugation status, and ring membership.
        """
        bonds = []

        for bond in self.mol.GetBonds():
            bonds.append(
                {
                    "begin_atom": bond.GetBeginAtomIdx(),
                    "end_atom": bond.GetEndAtomIdx(),
                    "bond_type": str(bond.GetBondType()),
                    "is_aromatic": bond.GetIsAromatic(),
                    "is_conjugated": bond.GetIsConjugated(),
                    "is_in_ring": bond.IsInRing(),
                }
            )

        return {"bonds": bonds}

    def to_dict(self, include_fingerprints: bool = True) -> Dict[str, Any]:
        """
        Convert all calculated molecular information into a dictionary.

        Parameters
        ----------
        include_fingerprints : bool, default=True
            Whether to include molecular fingerprint bitstrings in the output.

        Returns
        -------
        dict
            JSON-serialisable dictionary containing identifiers, ring
            properties, descriptors, atom data, bond data, functional groups,
            and optionally fingerprints.
        """
        data = {}
        data["identifiers"] = self.identifiers()
        data["ring_properties"] = self.ring_properties()
        data["descriptors"] = self.labelled_rdkit_descriptors()
        data["atom_data"] = self.atom_data()["atoms"]
        data["bond_data"] = self.bond_data()["bonds"]
        data["functional_groups"] = self.unified_functional_groups()

        if include_fingerprints:
            data["fingerprints"] = self.fingerprints()

        return data

    def to_json(self, path: Optional[str] = None) -> str:
        """
        Serialise the molecular descriptor dictionary to JSON.

        Parameters
        ----------
        path : str, optional
            Optional file path. If supplied, the JSON string is also written
            to this file.

        Returns
        -------
        str
            Pretty-printed JSON string containing the molecular descriptor
            data.
        """
        data = self.to_dict()
        text = json.dumps(data, indent=2)

        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)

        return text