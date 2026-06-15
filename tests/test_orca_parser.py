from moldesc.orca_parser.parse_output import OrcaOutputParser, OrcaHessParser

data = OrcaOutputParser('tests/data/CH4.out')

assert len(data.get_cal_setting()) == 7
assert len(data.get_ase_atoms()) == 5
assert data.get_thermal_energy() == -25386.28208504071
assert data.get_zero_point_energy() == 27.95
assert data.get_total_enthropy() == 13.26
assert data.get_gibbs_free_energy() == -25398.94795259995
assert len(data.get_vcd_spectrum()) == 9
assert len(data.get_raman_spectrum()) == 9
assert len(data.get_ir_spectrum()) == 9
assert len(data.energy_correction()) == 8
assert len(data.get_symmetry()) == 4
assert len(data.get_dipole_moment()) == 5
assert len(data.get_quadrupole_moment()) == 5
assert len(data.get_static_polarizability_dipole_dipole()) == 3
assert len(data.get_static_polarizability_dipole_quadrupole()) == 18
assert len(data.get_static_traceless_polarizability_dipole_quadrupole()) == 18
assert len(data.nmr_atom_mapping()) == 5
assert len(data.get_nmr_efg()) == 5
assert len(data.get_scf_correction()) == 12
assert len(data.get_mayer_analyis()) == 5
assert len(data.get_mayer_bond_order()) == 4
assert len(data.get_frontier_orbital()) == 1
assert len(data.get_orbital_energy()) == 16
assert len(data.get_normal_modes()) == 15
assert len(data.get_loewdin_charges()) == 5
assert len(data.get_mulliken()) == 5
assert len(data.get_trajectory()) == 3
assert len(data.get_traj_energies()) == 3
assert len(data.get_forces()) == 2
assert len(data.get_dispersion_forces()) == 2
print(data.get_cheminformatic())

hess_data = OrcaHessParser('tests/data/CH4.hess')
assert len(hess_data.get_hessian()) == 15
assert len(hess_data.get_dipole_derivative()) == 15
assert len(hess_data.get_polarization_derivative()) == 15
