
from __future__ import print_function
import re
__author__ = "Dr. Dinga Wonanke"
__status__ = "production"

import re
import ase
from ase import Atoms
import numpy as np
from mofstructure import filetyper, mofdeconstructor

EH2KCALMOL = 627.509

class OrcaOutputParser:
    def __init__(self, filename):
        self.filename = filename
        self.content = filetyper.get_contents(filename)

    def get_cal_setting(self):
        """
        """
        calculation_setting = {}
        section = get_section(
            self.content,
            "SCF SETTINGS",
            "Convergence Acceleration:"
        )
        for line in section:
            if "Method" in line:
                calculation_setting['method'] = line.split("....")[1]
            if "Functional kind" in line:
                calculation_setting['functional_kind'] = line.split("....")[1]
            if "Functional name" in line:
                calculation_setting['functional_name'] = line.split("....")[1]
            if "Functional family" in line:
                calculation_setting['functional_family'] = line.split("....")[1]
            if "Hartree-Fock type" in line:
                calculation_setting['hartree_fock_type'] = line.split("....")[1]
            if "Multiplicity" in line:
                calculation_setting['multiplicity'] = float(line.split("....")[1])
            if "Total Charge" in line:
                calculation_setting['total_charge'] = float(line.split("....")[1])
        return calculation_setting

    def get_ase_atoms(self):
        """
        Parses the ORCA output file and returns an ASE Atoms object.
        Returns:
        ---------
        ase.Atoms
        """
        # Initialize variables to store atomic symbols and positions
        symbols = []
        positions = []
        atom_section = get_section(self.content,
                                   "CARTESIAN COORDINATES (ANGSTROEM)",
                                   'CARTESIAN COORDINATES (A.U.)',
                                   2,
                                   -3
                                   )
        for line in atom_section:
            parts = line.split()
            if len(parts) >= 4:
                symbols.append(parts[0])
                positions.append([float(x) for x in parts[1:4]])

        return Atoms(symbols=symbols, positions=positions)

    def get_pybel(self):
        """
        Function to get pybel object
        """

        return mofdeconstructor.\
            ase_2_pybel(self.get_ase_atoms())

    def get_formula(self):
        return self.get_pybel().formula

    def get_mass(self):
        return self.get_pybel().exactmass

    def get_OBMol(self):
        return self.get_pybel().OBMol

    def get_smile(self):
        return mofdeconstructor.compute_smi(self.get_OBMol())

    def get_calcfp_f2(self):
        return self.get_pybel().calcfp()

    def get_cheminformatic(self):
        cheminformatics = {}
        desc = self.get_pybel().calcdesc()
        cheminformatics["TPSA"] = desc.get("TPSA", 0.0)
        cheminformatics["logP"] = desc.get('logP', 0.0)
        cheminformatics["n_aromatic_bonds"] = desc.get('abonds', 0.0)
        cheminformatics["n_double_bonds"] = desc.get('dbonds', 0.0)
        cheminformatics["n_triple_bonds"] = desc.get('tbonds', 0.0)
        cheminformatics["n_donor_hydrogen_bonds"] = desc.get('HBD', 0.0)
        cheminformatics["n_acceptor_hydrogen_bonds"] = desc.get('HBA2', 0.0)
        cheminformatics["n_of_rotatable_bonds"] = desc.get('rotors', 0.0)
        cheminformatics["molar_refractivity"] = desc.get('MR', 0.0)
        return cheminformatics

    def get_thermal_energy(self):
        """
        Parses the ORCA output file and returns the total
        thermal energy in kcal/mol.
         U= E(el) + E(ZPE) + E(vib) + E(rot) + E(trans)

         Returns:
         ---------
            float
                Total thermal energy in kcal/mol.
        """
        for line in self.content:
            if "Total thermal energy" in line:
                energy = float(line.split()[3])
                return energy * EH2KCALMOL
        return None

    def get_enthalpy(self):
        """
        Parses the ORCA output file and returns the total enthalpy in kcal/mol.
         H = U + kB*T

         Returns:
         ---------
         float
                Total enthalpy in kcal/mol.
         """
        for line in self.content:
            if "Total Enthalpy" in line:
                energy = float(line.split()[3])
                return energy * EH2KCALMOL
        return None

    def get_zero_point_energy(self):
        """
        Parses the ORCA output file and returns
        the zero-point energy in kcal/mol.
        Not coverting because it is already in kcal/mol in the output file.

        Returns:
        ---------
        float
            Zero-point energy in kcal/mol.
         ZPE = E(ZPE)
         where E(ZPE) is the sum of the zero-point vibrational energies of all
        """
        for line in self.content:
            if "Zero point energy" in line:
                energy = float(line.split()[6])
                return energy
        return None

    def get_total_enthropy(self):
        """
        Parses the ORCA output file and returns
        the total entropy in cal/(mol*K).
        T*S = T*(S(el)+S(vib)+S(rot)+S(trans))

        Returns:
        ---------
        float
            Total entropy in cal/(mol*K).
         Not coverting because it is already in cal/(mol*K) in the output file.
        """
        for line in self.content:
            if "Final entropy term" in line:
                entropy = float(line.split()[6])
                return entropy
        return None

    def get_gibbs_free_energy(self):
        """
        Parses the ORCA output file and returns the Gibbs
        free energy in kcal/mol.
         G = H - T*S
         Returns:
         ---------
         float
             Gibbs free energy in kcal/mol.
        """
        for line in self.content:
            if "Final Gibbs free energy" in line:
                energy = float(line.split()[5])
                return energy * EH2KCALMOL
        return None

    def energy_correction(self):
        """
        Parses the ORCA output file and returns the energy corrections.

        Returns:
        ---------
        dict[str, float]
            Dictionary containing the following energy corrections:

            - thermal_vibrational_correction : float
                Thermal vibrational correction in kcal/mol.

            - thermal_rotational_correction : float
                Thermal rotational correction in kcal/mol.

            - thermal_translational_correction : float
                Thermal translational correction in kcal/mol.

            - thermal_enthalpy_correction : float
                Thermal enthalpy correction in kcal/mol.

            - electronic_entropy : float
                Electronic entropy contribution to the
                Gibbs free energy in cal/(mol*K).

            - rotational_entropy : float
                Rotational entropy contribution to the
                Gibbs free energy in cal/(mol*K).

            - vibrational_entropy : float
                Vibrational entropy contribution to the
                Gibbs free energy in cal/(mol*K).

            - translational_entropy : float
                Translational entropy contribution to
                the Gibbs free energy in cal/(mol*K).
        """
        energ_corrections = {}

        for line in self.content:
            if "Thermal vibrational correction" in line:
                energ_corrections["thermal_vibrational_correction"] = float(line.split()[6])
            if "Thermal rotational correction" in line:
                energ_corrections["thermal_rotational_correction"] = float(line.split()[6])
            if "Thermal translational correction" in line:
                energ_corrections["thermal_translational_correction"] = float(line.split()[6])
            if "Thermal Enthalpy correction" in line:
                energ_corrections["thermal_enthalpy_correction"] = float(line.split()[6])
            if "Electronic entropy" in line:
                energ_corrections["electronic_entropy"] = float(line.split()[5])
            if "Rotational entropy" in line and line.startswith("Rotational entropy"):
                energ_corrections["rotational_entropy"] = float(line.split()[5])
            if "Vibrational entropy" in line and len(line.split()) == 7:
                energ_corrections["vibrational_entropy"] = float(line.split()[5])
            if "Translational entropy" in line:
                energ_corrections["translational_entropy"] = float(line.split()[5])
        return energ_corrections


    def get_vcd_spectrum(self):
        """
        Extract the vibrational circular dichroism (VCD) spectrum from an ORCA
        output file.

        Returns
        -------
        dict[str, dict[str, float]]
            Dictionary mapping each vibrational mode
            number to its VCD properties.

            Each mode contains:

            - frequency : float
                Harmonic vibrational frequency in cm⁻¹.

            - intensity : float
                Vibrational circular dichroism (VCD) intensity in
                10⁻⁴⁴ esu²·cm². Positive and negative values correspond to
                opposite rotational strengths and determine the sign of the VCD
                absorption band.

        Notes
        -----
        VCD measures the difference in infrared absorption of left- and
        right-circularly polarized light by chiral molecules.
        Unlike conventional
        IR spectroscopy, VCD intensities may be either positive or negative.

        Examples
        --------
        >>> vcd = parser.get_vcd_spectrum()
        >>> vcd["12"]
        {
            "frequency": 3131.4,
            "intensity": 0.00
        }
        """
        vcd = {}

        vcd_section = get_section(
            self.content,
            "VCD SPECTRUM CALCULATION",
            "SUGGESTED CITATIONS FOR THIS RUN",
            9,
            -5,
        )

        for line in vcd_section:
            data = line.split()

            if len(data) == 3:
                mode = data[0].strip(":")
                vcd[mode] = {
                    "frequency": float(data[1]),
                    "intensity": float(data[2]),
                }

        return vcd

    def get_raman_spectrum(self):
        """
        Extract the Raman vibrational
        spectrum from an ORCA output file.

        Returns
        -------
        dict[str, dict[str, float]]
            Dictionary mapping each
            vibrational mode number to Raman properties.

            Each mode contains:

            - frequency : float
                Harmonic vibrational
                frequency in cm⁻¹.

            - activity : float
                Raman activity. Larger values indicate
                stronger Raman scattering
                due to a larger change in molecular polarizability.

            - depolarization : float
                Raman depolarization ratio. Values near 0 indicate polarized,
                usually totally symmetric vibrations. Values near 0.75 indicate
                depolarized, usually non-totally symmetric vibrations.

        Examples
        --------
        >>> raman = parser.get_raman_spectrum()
        >>> raman["11"]
        {
            "frequency": 3029.33,
            "activity": 161.869956,
            "depolarization": 0.0
        }
        """
        raman = {}

        raman_section = get_section(
            self.content,
            "RAMAN SPECTRUM",
            "THERMOCHEMISTRY AT 298.15K",
            5,
            -7,
        )

        for line in raman_section:
            data = line.split()

            if len(data) == 4:
                mode = data[0].strip(":")
                raman[mode] = {
                    "frequency": float(data[1]),
                    "activity": float(data[2]),
                    "depolarization": float(data[3]),
                }

        return raman

    def get_ir_spectrum(self):
        """
        Extract the infrared (IR) vibrational spectrum
        from an ORCA output file.

        Returns
        -------
        dict[str, dict[str, float]]
            Dictionary mapping each vibrational mode number to IR properties.

            Each mode contains:

            - frequency : float
                Harmonic vibrational frequency in cm⁻¹.

            - epsilon : float
                Molar absorptivity in L mol⁻¹ cm⁻¹.

            - intensity : float
                Integrated IR intensity in km/mol.

            - T2 : float
                Squared magnitude of the dipole moment derivative vector.

            - Tx : float
                x-component of the dipole moment derivative.

            - Ty : float
                y-component of the dipole moment derivative.

            - Tz : float
                z-component of the dipole moment derivative.

        Examples
        --------
        >>> ir = parser.get_ir_spectrum()
        >>> ir["6"]
        {
            "frequency": 1339.88,
            "epsilon": 0.002849,
            "intensity": 14.40,
            "T2": 0.000663,
            "Tx": -0.001412,
            "Ty": 0.018168,
            "Tz": -0.018205
        }
        """
        ir = {}

        ir_section = get_section(
            self.content,
            "IR SPECTRUM",
            "* The epsilon (eps)",
            6,
            -2,
        )

        for line in ir_section:
            cleaned_line = line.replace("(", "").replace(")", "")
            data = cleaned_line.split()

            if len(data) == 8:
                mode = data[0].strip(":")
                ir[mode] = {
                    "frequency": float(data[1]),
                    "epsilon": float(data[2]),
                    "intensity": float(data[3]),
                    "T2": float(data[4]),
                    "Tx": float(data[5]),
                    "Ty": float(data[6]),
                    "Tz": float(data[7]),
                }

        return ir

    def get_symmetry(self):
        """
        Extract symmetry information from the ORCA output file.

        Returns
        -------
        dict[str, float | int | list[float]]
            Dictionary containing symmetry information.
        """
        symmetry = {}
        for line in self.content:
            if "Point Group:" in line:
                symmetry["point_group"] = line.split()[2].strip(",")
                symmetry["symmetry_number"] = int(line.split()[5])
            if "Rotational constants in cm-1:" in line:
                symmetry["rotational_constants_cm**-1"] = [float(x) for x in line.split()[4:7]]
            if "Rotational constants in MHz :" in line:
                symmetry["rotational_constants_MHz"] = [float(x) for x in line.split()[5:8]]
        return symmetry

    def get_dipole_moment(self):
        """
        Extract the dipole moment from the ORCA output file.

        Returns
        -------
        dict[str, list[float] | float]
            Dictionary containing the following dipole moment information:
            - electronic_contribution_to_dipole_moment : list[float]
                x, y, z components of the electronic contribution to the dipole moment in Debye.
            - nuclear_contribution_to_dipole_moment : list[float]
                x, y, z components of the nuclear contribution to the dipole moment in Debye.
            - total_dipole_moment : list[float]
                x, y, z components of the total dipole moment in Debye.
            - dipole_magnitude_au : float
                Magnitude of the dipole moment in atomic units.
            - dipole_magnitude_debye : float
                Magnitude of the dipole moment in Debye.
        """
        dipole_moment = {}
        dipole_moment["dipole_magnitude"] = {}
        dipole_moment["dipole_along_rotational_axis"] = {}
        dipole_section = get_section(
            self.content,
            "DIPOLE MOMENT",
            "QUADRUPOLE MOMENT"
        )
        for line in dipole_section:
            if "Electronic contribution:" in line:
                dipole_moment["electronic_contribution_to_dipole_moment"] = [float(i) for i in line.split()[2:5]]
            if "Nuclear contribution" in line:
                dipole_moment["nuclear_contribution_to_dipole_moment"] = [float(i) for i in line.split()[3:6]]
            if "Total Dipole Moment" in line:
                dipole_moment["total_dipole_moment"] = [float(i) for i in line.split()[4:7]]
            if "Magnitude (a.u.)" in line:
                dipole_moment["dipole_magnitude"]['au'] = float(line.split()[3])
            if "Magnitude (Debye)" in line:
                dipole_moment["dipole_magnitude"]['Debye'] = float(line.split()[3])
            if "x,y,z [a.u.] :" in line:
                dipole_moment["dipole_along_rotational_axis"]['au'] = [float(i) for i in line.split()[3:6]]
            if "x,y,z [Debye] :" in line:
                dipole_moment["dipole_along_rotational_axis"]['Debye'] = [float(i) for i in line.split()[3:6]]
        return dipole_moment

    def get_quadrupole_moment(self):
        """
        Extract the quadrupole moment from the ORCA output file.

        Returns
        -------
        dict[str, dict[str, float]]
            Dictionary containing the following quadrupole moment information:
            - electronic_contribution_to_quadrupole_moment : dict[str, float]
                XX, YY, ZZ, XY, XZ, YZ components of the electronic contribution to the quadrupole moment in atomic units.
            - nuclear_contribution_to_quadrupole_moment : dict[str, float]
                XX, YY, ZZ, XY, XZ, YZ components of the nuclear contribution to the quadrupole moment in atomic units.
            - total_quadrupole_moment : dict[str, dict[str, float]]
                XX, YY, ZZ, XY, XZ, YZ components of the total quadrupole moment in atomic units and Buckingham units.
            - principal_quadrupole_moment : dict[str, dict[str, float]]
                XX, YY, ZZ  components of the diagonalized tensor in atomic units and Buckingham units.
            - isotropic_quadrupole : float
                Isotropic quadrupole moment in atomic units.
        NB:
        ---
        The traceless quadrupole moment can be computed from the total quadrupole moment as follows:
        Q = np.array([
                [xx, xy, xz],
                [xy, yy, yz],
                [xz, yz, zz],
            ])

            Q_iso = np.trace(Q) / 3.0
            Q_traceless = Q - np.eye(3) * Q_iso
        """
        quadrupole_moment = {}
        quadrupole_moment["electronic_contribution_to_quadrupole_moment"] = {}
        quadrupole_moment["nuclear_contribution_to_quadrupole_moment"] = {}

        quadrupole_section = get_section(
            self.content,
            "QUADRUPOLE MOMENT",
            "Quadrupole moment calculation",
            7,
            -1
        )
        for index, line in enumerate(quadrupole_section):
            if " NUC" in line:
                data = line.split()
                quadrupole_moment["nuclear_contribution_to_quadrupole_moment"]["XX"] = float(data[1])
                quadrupole_moment["nuclear_contribution_to_quadrupole_moment"]["YY"] = float(data[2])
                quadrupole_moment["nuclear_contribution_to_quadrupole_moment"]["ZZ"] = float(data[3])
                quadrupole_moment["nuclear_contribution_to_quadrupole_moment"]["XY"] = float(data[4])
                quadrupole_moment["nuclear_contribution_to_quadrupole_moment"]["XZ"] = float(data[5])
                quadrupole_moment["nuclear_contribution_to_quadrupole_moment"]["YZ"] = float(data[6])
            if "EL" in line:
                data = line.split()
                quadrupole_moment["electronic_contribution_to_quadrupole_moment"]["XX"] = float(data[1])
                quadrupole_moment["electronic_contribution_to_quadrupole_moment"]["YY"] = float(data[2])
                quadrupole_moment["electronic_contribution_to_quadrupole_moment"]["ZZ"] = float(data[3])
                quadrupole_moment["electronic_contribution_to_quadrupole_moment"]["XY"] = float(data[4])
                quadrupole_moment["electronic_contribution_to_quadrupole_moment"]["XZ"] = float(data[5])
                quadrupole_moment["electronic_contribution_to_quadrupole_moment"]["YZ"] = float(data[6])
            if "TOT" in line:
                data = line.split()
                au = {}
                au["XX"] = float(data[1])
                au["YY"] = float(data[2])
                au["ZZ"] = float(data[3])
                au["XY"] = float(data[4])
                au["XZ"] = float(data[5])
                au["YZ"] = float(data[6])
                quadrupole_moment["total_quadrupole_moment"] = {"au": au}
                data2 = quadrupole_section[index + 1].split()
                buckingham = {}
                buckingham["XX"] = float(data2[0])
                buckingham["YY"] = float(data2[1])
                buckingham["ZZ"] = float(data2[2])
                buckingham["XY"] = float(data2[3])
                buckingham["XZ"] = float(data2[4])
                buckingham["YZ"] = float(data2[5])
                quadrupole_moment["total_quadrupole_moment"]["buckingham"] = buckingham
            if "diagonalized tensor:" in line:
                data = quadrupole_section[index + 1].split()
                diagonalized_tensor = {}
                diagonalized_tensor["XX"] = float(data[0])
                diagonalized_tensor["YY"] = float(data[1])
                diagonalized_tensor["ZZ"] = float(data[2])
                quadrupole_moment["principal_quadrupole_moment"] = {"au": diagonalized_tensor}

                data2 = quadrupole_section[index + 2].split()
                diagonalized_tensor_buckingham = {}
                diagonalized_tensor_buckingham["XX"] = float(data2[0])
                diagonalized_tensor_buckingham["YY"] = float(data2[1])
                diagonalized_tensor_buckingham["ZZ"] = float(data2[2])
                quadrupole_moment["principal_quadrupole_moment"]["buckingham"] = diagonalized_tensor_buckingham

            if "Isotropic quadrupole :" in line:
                data = line.split()
                isotropic_quadrupole = float(data[3])
                quadrupole_moment["isotropic_quadrupole"] = isotropic_quadrupole
        return quadrupole_moment

    def get_static_polarizability_dipole_dipole(self):
        """
        Extract the static dipole-dipole polarizability from the ORCA output file.
        Returns
        -------
        dict[str, dict[str, float] | float]
            Dictionary containing the following static dipole-dipole polarizability information:
            - cartesian_tensor_au : dict[str, float]
                XX, XY, XZ, YX, YY, YZ, ZX, Z
                components of the static dipole-dipole polarizability tensor in atomic units.
            - principal_polarizability_au : dict[str, float]
                XX, YY, ZZ components of the diagonalized static dipole-dipole polarizability tensor in atomic units.
            - isotropic_polarizability_au : float
                Isotropic static dipole-dipole polarizability in atomic units.
        """

        pole_section = get_section(
            self.content,
            "STATIC POLARIZABILITY TENSOR (Dipole/Dipole)",
            "STATIC POLARIZABILITY TENSOR (Dipole/Quadrupole)",
            9,
            -5
        )
        static_polarizability = {}
        for index, line in enumerate(pole_section):
            if "The raw cartesian tensor (atomic units)" in line:
                data = {}
                data["XX"] = float(pole_section[index + 1].split()[0])
                data["XY"] = float(pole_section[index + 1].split()[1])
                data["XZ"] = float(pole_section[index + 1].split()[2])
                data["YX"] = float(pole_section[index + 2].split()[0])
                data["YY"] = float(pole_section[index + 2].split()[1])
                data["YZ"] = float(pole_section[index + 2].split()[2])
                data["ZX"] = float(pole_section[index + 3].split()[0])
                data["ZY"] = float(pole_section[index + 3].split()[1])
                data["ZZ"] = float(pole_section[index + 3].split()[2])
                static_polarizability["cartesian_tensor_au"] = data
            if "diagonalized tensor:" in line:
                data = {}
                data["XX"] = float(pole_section[index + 1].split()[0])
                data["YY"] = float(pole_section[index + 1].split()[1])
                data["ZZ"] = float(pole_section[index + 1].split()[2])
                static_polarizability["principal_polarizability_au"] = data
            if "Isotropic polarizability :" in line:
                data = float(line.split()[3])
                static_polarizability["isotropic_polarizability_au"] = data
        return static_polarizability

    def get_static_polarizability_dipole_quadrupole(self):
        """
        Extract static dipole quadrupole polarizability tensor
        from orca.

        Return
        ------

        """

        static_polarizability_dipole_quadrupole = {}
        section = get_section(
            self.content,
            "STATIC POLARIZABILITY TENSOR (Dipole/Quadrupole)",
            "STATIC TRACELESS POLARIZABILITY TENSOR (Dipole/Quadrupole)",
            11,
            -4
            )
        for line in section:
            data = line.split()
            xyz = ''.join(data[0:3])
            static_polarizability_dipole_quadrupole[xyz] = float(data[4])
        return static_polarizability_dipole_quadrupole


    def get_static_traceless_polarizability_dipole_quadrupole(self):
        """
        Extract traceless static dipole quadrupole polarizability tensor
        from orca.

        Return
        ------

        """

        static_traceless_polarizability_dipole_quadrupole = {}
        section = get_section(
            self.content,
            "STATIC TRACELESS POLARIZABILITY TENSOR (Dipole/Quadrupole",
            "ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE (5 nuclei)",
            11,
            -5
            )
        for line in section:
            data = line.split()
            xyz = ''.join(data[0:3])
            static_traceless_polarizability_dipole_quadrupole[xyz] = float(data[4])
        return static_traceless_polarizability_dipole_quadrupole

    def nmr_atom_mapping(self):
        atom_map = {}
        section = get_section(
            self.content,
            "BASIS SET INFORMATION",
            "---------------------------------",
            7,
            -1
        )
        for i, line in enumerate(section):
            atom_map[line.split()[1]] = i
        return atom_map

    def get_nmr_efg(self):
        """
        """
        atom_map = self.nmr_atom_mapping()
        section = get_section(
            self.content,
            "ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE",
            "Hyperfine and quadrupole coupling calculation done"
        )

        def compute_asymmetry_parameter(principal, eps=1e-10):
            # Convention: |Vzz| >= |Vyy| >= |Vxx|
            vals = sorted(principal, key=lambda x: abs(x))
            Vxx, Vyy, Vzz = vals[0], vals[1], vals[2]

            if abs(Vzz) < eps:
                return None

            return abs((Vxx - Vyy) / Vzz), Vxx, Vyy, Vzz

        nmr_data = {}

        for i, line in enumerate(section):
            if "Nucleus" not in line:
                continue

            atom = line.split()[1]
            atom_index = atom_map[atom]

            tensor = {}

            # Find Raw EFG matrix after this nucleus line
            raw_idx = None
            vtot_idx = None

            for j in range(i, len(section)):
                if "Raw EFG matrix" in section[j]:
                    raw_idx = j
                if "V(Tot)" in section[j]:
                    vtot_idx = j
                    break

            if raw_idx is None or vtot_idx is None:
                continue

            # Matrix values are usually 2 lines after "Raw EFG matrix"
            matrix_lines = section[raw_idx + 2: raw_idx + 5]

            row1 = [float(x) for x in matrix_lines[0].split()]
            row2 = [float(x) for x in matrix_lines[1].split()]
            row3 = [float(x) for x in matrix_lines[2].split()]

            tensor["XX"], tensor["XY"], tensor["XZ"] = row1
            tensor["YX"], tensor["YY"], tensor["YZ"] = row2
            tensor["ZX"], tensor["ZY"], tensor["ZZ"] = row3

            parts = section[vtot_idx].split()
            principal_raw = [float(parts[1]), float(parts[2]), float(parts[3])]

            eta_result = compute_asymmetry_parameter(principal_raw)

            if eta_result is None:
                eta = None
                Vxx = Vyy = Vzz = None
            else:
                eta, Vxx, Vyy, Vzz = eta_result

            nmr_data[atom_index] = {
                "atom_label": atom,
                "atom_index": atom_map[atom],
                "tensor": tensor,
                "principal_raw": principal_raw,
                "principal": {
                    "Vxx": Vxx,
                    "Vyy": Vyy,
                    "Vzz": Vzz
                },
                "asymmetry_parameter": eta,
                "units": "a.u.^-3"
            }

        return nmr_data

    def get_scf_correction(self):
        """

        """
        scf = {}
        for i, line in enumerate(self.content):
            if "Nuclear Repulsion  : " in line:
                scf['nuclear_repulsion'] = float(line.split()[3])*EH2KCALMOL
            if "Electronic Energy  : " in line:
                scf["electronic_energy"] = float(line.split()[3])*EH2KCALMOL
            if "One Electron Energy:" in line:
                scf["one_electron_energy"] = float(line.split()[3])*EH2KCALMOL
            if "Two Electron Energy:" in line:
                scf["two_electron_energy"] = float(line.split()[3])*EH2KCALMOL
            if "Potential Energy   :" in line:
                scf["potential_energy"] = float(line.split()[3])*EH2KCALMOL
            if "Kinetic Energy     :" in line:
                scf["kinetic_energy"] = float(line.split()[3])*EH2KCALMOL
            if "Virial Ratio       :" in line:
                scf["virial_ratio"] = float(line.split()[3])*EH2KCALMOL
            if "N(Alpha)           :" in line:
                scf["n_alpha"] = float(line.split()[2])*EH2KCALMOL
            if "N(Beta)            :" in line:
                scf["n_beta"] = float(line.split()[2])*EH2KCALMOL
            if "N(Total)           :" in line:
                scf["n_total"] = float(line.split()[2])*EH2KCALMOL
            if "E(XC)              :" in line:
                scf["exchange_energy"] = float(line.split()[2])*EH2KCALMOL
            if " DFT DISPERSION CORRECTION" in line:
                dat = self.content[i+8].split()
                if len(dat) == 3:
                    scf["dispersion_correction"] = float(dat[2])*EH2KCALMOL
        return scf

    def get_mayer_analyis(self):
        mayer = []
        section = get_section(
            self.content,
            "* MAYER POPULATION ANALYSIS *",
            "Mayer bond orders",
            11,
            -2
            )
        for line in section:
            data = line.split()
            mayer.append(
                {
                    "atom_index": float(data[0]),
                    "gross_atomic_population": float(data[2]),
                    "total_nuclear_charge": float(data[3]),
                    "gross_atomic_charge": float(data[4]),
                    "total_valence": float(data[5]),
                    "bonded_valence": float(data[6]),
                    "free_valence": float(data[7]),
                }
            )
        return mayer

    def get_mayer_bond_order(self):
        bond_pattern = re.compile(r'B\(\s*(\d+)-\w+\s*,\s*(\d+)-\w+\s*\)')
        section = get_section(
            self.content,
            "Mayer bond orders larger than 0.100000",
            "Environment variable NBOEXE",
            1,
            -2
        )

        bond_orders = {}

        for line in section:

            parts = line.split(":")

            match = bond_pattern.search(parts[0])
            if not match:
                continue

            atom1 = int(match.group(1))
            atom2 = int(match.group(2))

            for item in parts[1:]:

                value_match = re.match(r"\s*([\d.]+)", item)
                bond_match = bond_pattern.search(item)

                if value_match is None:
                    continue

                value = float(value_match.group(1))

                bond_orders[(atom1, atom2)] = value

                if bond_match:
                    atom1 = int(bond_match.group(1))
                    atom2 = int(bond_match.group(2))

        return bond_orders

    def get_frontier_orbital(self):
        frontier = {
            "orbitals": {
                "HOMO": {
                    "index": None,
                    "population": {}
                },
                "LUMO": {
                    "index": None,
                    "population": {}
                }
            }
        }

        # Extract HOMO/LUMO orbital indices
        for line in self.content:
            match = re.search(
                r"ANALYZING ORBITALS:\s*HOMO=\s*(\d+)\s*LUMO=\s*(\d+)",
                line
            )
            if match:
                frontier["orbitals"]["HOMO"]["index"] = int(match.group(1))
                frontier["orbitals"]["LUMO"]["index"] = int(match.group(2))
                break

        section = get_section(
            self.content,
            "Atom        Q(Mulliken)     Q(Loewdin)     Q(Mulliken)     Q(Loewdin)",
            "*****************************",
            3,
            -3
        )

        for line in section:
            data = line.split()

            if len(data) < 5 or "-" not in data[0]:
                continue

            atom_index_str, element = data[0].split("-", 1)
            atom_index = int(atom_index_str)

            frontier["orbitals"]["HOMO"]["population"][atom_index] = {
                "element": element,
                "Mulliken": float(data[1]),
                "Loewdin": float(data[2]),
            }

            frontier["orbitals"]["LUMO"]["population"][atom_index] = {
                "element": element,
                "Mulliken": float(data[3]),
                "Loewdin": float(data[4]),
            }

        return frontier

    def get_orbital_energy(self):
        orbitals = []
        section = get_section(
            self.content,
            "ORBITAL ENERGIES",
            "*Only the first 10",
            4,
            -1
            )
        for line in section:
            data = line.split()
            tmp = {
                'orbital': int(data[0]),
                'occupation': float(data[1]),
                "energy_Eh": float(data[2]),
                "energy_eV": float(data[3]),
            }
            orbitals.append(tmp)

        return orbitals

    def get_normal_modes(self):
        """
        Parse ORCA NORMAL MODES section.

        Returns
        -------
            - np.ndarray
                ndarray with shape (n_modes, n_atoms, 3)
        """

        section = get_section(
            self.content,
            "NORMAL MODES",
            "IR SPECTRUM",
            7,
            -4
        )

        # Temporary storage:
        # coordinate_index -> {mode_index: value}
        coordinate_rows = {}

        current_modes = []

        for line in section:
            parts = line.split()

            if not parts:
                continue

            # Header line, e.g.
            # 0 1 2 3 4 5
            # 6 7 8 9 10 11
            # 12 13 14
            if all(re.fullmatch(r"\d+", p) for p in parts):
                current_modes = [int(p) for p in parts]
                continue

            # Data line, e.g.
            # 0 0.006856 -0.124801 ...
            if current_modes and re.fullmatch(r"\d+", parts[0]):
                coord_index = int(parts[0])
                values = [float(x) for x in parts[1:]]

                if len(values) != len(current_modes):
                    continue

                if coord_index not in coordinate_rows:
                    coordinate_rows[coord_index] = {}

                for mode_index, value in zip(current_modes, values):
                    coordinate_rows[coord_index][mode_index] = value

        if not coordinate_rows:
            return np.array([])

        n_coordinates = max(coordinate_rows) + 1

        if n_coordinates % 3 != 0:
            raise ValueError(
                f"Number of Cartesian coordinates ({n_coordinates}) is not divisible by 3."
            )

        n_atoms = n_coordinates // 3

        all_modes = sorted({
            mode
            for row in coordinate_rows.values()
            for mode in row
        })

        n_modes = max(all_modes) + 1

        mode_tensor = np.zeros((n_modes, n_atoms, 3), dtype=float)

        for coord_index, mode_values in coordinate_rows.items():
            atom_index = coord_index // 3
            xyz_index = coord_index % 3

            for mode_index, value in mode_values.items():
                mode_tensor[mode_index, atom_index, xyz_index] = value
        # print(" mode_flexibility", np.linalg.norm(mode_tensor, axis=2).sum(axis=1))
        # print("atomic_flexibility", np.linalg.norm(mode_tensor, axis=2).sum(axis=0))
        # print("mode_strength",  np.sum(mode_tensor**2, axis=(1, 2)))
        # print("atom_strength", np.sum(mode_tensor**2, axis=(0,2)))
        return mode_tensor

    def get_loewdin_charges(self):
        charges = []
        sections = get_section(
            self.content,
            "LOEWDIN ATOMIC CHARGES",
            "LOEWDIN REDUCED ORBITAL CHARGES",
            2,
            -3
            )
        for line in sections:
            data = line.split()
            charges.append(float(data[3]))
        return charges

    def get_mulliken(self):
        charges = []
        sections = get_section(
            self.content,
            "MULLIKEN ATOMIC CHARGES",
            "Sum of atomic charges:",
            2,
            -1
            )
        for line in sections:
            data = line.split()
            charges.append(float(data[3]))
        return charges

    def get_trajectory(self):
        geometry = []
        all_sections = get_all_sections(
            self.content,
            "CARTESIAN COORDINATES (ANGSTROEM)",
            "CARTESIAN COORDINATES (A.U.)",
            2,
            -3
            )
        for section in all_sections:
            symbol, position = [], []
            for line in section:
                data = line.split()
                symbol.append(data[0])
                position.append([float(i) for i in data[1:]])
            geometry.append([symbol, position])
        return geometry

    def get_traj_energies(self):
        """
        """

        energies = []
        for line in self.content:
            if "Total Energy       :" in line:
                data = line.split()
                energies.append(float(data[3])*EH2KCALMOL)
        return energies

    def get_forces(self):
        all_forces = []
        all_sections = get_all_sections(
            self.content,
            "CARTESIAN GRADIENT",
            "Difference to translation invariance:",
            3,
            -2
            )
        for section in all_sections:
            forces = []
            for line in section:
                data = line.split()
                forces.append([-1*float(i) for i in data[3:]])
            all_forces.append(forces)
        return all_forces

    def get_dispersion_forces(self):
        all_forces = []
        all_sections = get_all_sections(
            self.content,
            "DISPERSION GRADIENT",
            "Difference to translation invariance:",
            3,
            -2
            )
        for section in all_sections:
            forces = []
            for line in section:
                data = line.split()
                forces.append([-1*float(i) for i in data[3:]])
            all_forces.append(forces)
        return all_forces


class OrcaHessParser:
    def __init__(self, filename):
        self.filename = filename
        self.content = filetyper.get_contents(filename)

    def get_hessian(self):
        """
        Parse ORCA hessian section.

        Returns
        -------
            - np.ndarray
                ndarray with shape (n_modes, n_atoms, 3)
        """
        section = get_section(
            self.content,
            "$hessian",
            "$vibrational_frequencies",
            2,
            -2
        )

        # Temporary storage:
        # coordinate_index -> {mode_index: value}
        coordinate_rows = {}

        current_modes = []

        for line in section:
            parts = line.split()

            if not parts:
                continue

            if all(re.fullmatch(r"\d+", p) for p in parts):
                current_modes = [int(p) for p in parts]
                continue

            # Data line, e.g.
            # 0 0.006856 -0.124801 ...
            if current_modes and re.fullmatch(r"\d+", parts[0]):
                coord_index = int(parts[0])
                values = [float(x) for x in parts[1:]]

                if len(values) != len(current_modes):
                    continue

                if coord_index not in coordinate_rows:
                    coordinate_rows[coord_index] = {}

                for mode_index, value in zip(current_modes, values):
                    coordinate_rows[coord_index][mode_index] = value

        if not coordinate_rows:
            return np.array([])

        n_coordinates = max(coordinate_rows) + 1

        if n_coordinates % 3 != 0:
            raise ValueError(
                f"Number of Cartesian coordinates ({n_coordinates}) is not divisible by 3."
            )

        n_atoms = n_coordinates // 3

        all_modes = sorted({
            mode
            for row in coordinate_rows.values()
            for mode in row
        })

        n_modes = max(all_modes) + 1

        hessian_tensor = np.zeros((n_modes, n_atoms, 3), dtype=float)

        for coord_index, mode_values in coordinate_rows.items():
            atom_index = coord_index // 3
            xyz_index = coord_index % 3

            for mode_index, value in mode_values.items():
                hessian_tensor[mode_index, atom_index, xyz_index] = value
        return hessian_tensor

    def get_dipole_derivative(self):
        dipole_derivatives = []
        section = get_section(
            self.content,
            "$dipole_derivatives",
            "#",
            2,
            -2
            )
        for line in section:
            data = line.split()
            dipole_derivatives.append([float(i) for i in data])
        return dipole_derivatives

    def get_polarization_derivative(self):
        polarization_derivative = []
        section = get_section(
            self.content,
            "$polarizability_derivatives",
            "#",
            2,
            -2
            )
        for line in section:
            data = line.split()
            polarization_derivative.append([float(i) for i in data])
        return polarization_derivative

def get_section(contents,
                start_key,
                stop_key,
                start_offset=0,
                stop_offset=0
                ):
    """
    Extract a section of text from the ORCA output file based on start and stop keys.
    Parameters
    ----------
     - contents : list[str]
        List of lines from the ORCA output file.
     - start_key : str
        Key to identify the start of the section.
     - stop_key : str
        Key to identify the end of the section.
     - start_offset : int, optional
        Offset for the start index, by default 0.
     - stop_offset : int, optional
        Offset for the stop index, by default 0.
    """

    all_start_indices = []
    for i, line in enumerate(contents):
        if start_key in line:
            all_start_indices.append(i + start_offset)
    start_index = all_start_indices[-1]
    for i in range(start_index, len(contents)):
        line = contents[i]
        if stop_key in line:
            stop_index = i + 1 + stop_offset
            break
    data = contents[start_index:stop_index]
    return data


def get_all_sections(contents,
                     start_key,
                     stop_key,
                     start_offset=0,
                     stop_offset=0
                     ):
    """
    Extract all sections between start_key and stop_key.

    Returns
    -------
    list[list[str]]
        A list containing every matched section.
    """

    sections = []

    for i, line in enumerate(contents):

        if start_key in line:

            start = i + start_offset

            stop = None
            for j in range(start, len(contents)):
                if stop_key in contents[j]:
                    stop = j + 1 + stop_offset
                    break

            if stop is not None:
                sections.append(contents[start:stop])

    return sections
