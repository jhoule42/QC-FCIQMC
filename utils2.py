# This module provides utility functions for setting up and analyzing
# Full Configuration Interaction Quantum Monte Carlo (FCIQMC) simulations,
# specifically for the 1D Fermi-Hubbard model.

import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict

from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper

# ====================================================================
#  SPIN-ORBITAL MAPPING FUNCTIONS (Corrected for Qiskit Consistency)
# ====================================================================

def get_spin_orbital_from_site_spin(site_idx, spin_idx, num_sites):
    """
    Maps a (site, spin) pair to a spin-orbital index consistent with Qiskit Nature.
    Qiskit's convention groups all spin-up orbitals first, then all spin-down.
    Example for 2 sites: up-spins are SO 0, 1; down-spins are SO 2, 3.

    Args:
        site_idx (int): The index of the lattice site.
        spin_idx (int): The spin index (0 for spin-up, 1 for spin-down).
        num_sites (int): The total number of sites in the lattice.

    Returns:
        int: The unique spin-orbital index.
    """
    if spin_idx == 0:  # Spin-up
        return site_idx
    else:  # Spin-down
        return site_idx + num_sites

def get_site_spin_from_spin_orbital(spin_orbital_idx, num_sites):
    """
    Maps a spin-orbital index back to its (site, spin) pair, consistent with Qiskit.

    Args:
        spin_orbital_idx (int): The unique spin-orbital index.
        num_sites (int): The total number of sites in the lattice.

    Returns:
        tuple[int, int]: A tuple containing the site index and the spin index.
    """
    if spin_orbital_idx < num_sites:
        # This is a spin-up orbital
        return spin_orbital_idx, 0
    else:
        # This is a spin-down orbital
        return spin_orbital_idx - num_sites, 1

# ====================================================================
#  DETERMINANT AND BITMASK HELPER FUNCTIONS
# ====================================================================

def generate_all_determinants(num_spin_orbitals, num_electrons):
    """
    Generates all possible Slater determinants for a given number of electrons
    and spin-orbitals. Each determinant is represented as an integer bitmask.
    """
    all_spin_orbitals = list(range(num_spin_orbitals))
    determinants = []
    for combo in itertools.combinations(all_spin_orbitals, num_electrons):
        bitmask = 0
        for so_idx in combo:
            bitmask |= (1 << so_idx)
        determinants.append(bitmask)
    return sorted(determinants)

def count_set_bits(n):
    """Counts the number of set bits (1s) in an integer's binary representation."""
    count = 0
    while n > 0:
        n &= (n - 1)
        count += 1
    return count

def get_set_bits(n):
    """Returns a list of indices of the set bits (1s) in an integer bitmask."""
    bits = []
    idx = 0
    while n > 0:
        if n & 1:
            bits.append(idx)
        n >>= 1
        idx += 1
    return bits

def fermionic_sign_single_exc(det_mask, p, r):
    """Computes the fermionic sign (+1 or -1) of applying a single excitation."""
    if p == r:
        return 1
    sign = 1
    lower, upper = min(p, r), max(p, r)
    for i in range(lower + 1, upper):
        if (det_mask >> i) & 1:
            sign *= -1
    return sign

# ====================================================================
#  HAMILTONIAN CONSTRUCTION (Corrected for Qiskit Consistency)
# ====================================================================

def calculate_hubbard_hamiltonian(num_sites, num_electrons, t_hop, U_onsite):
    """
    Generates the Hamiltonian matrix for a 1D Fermi-Hubbard model.
    The spin-orbital mapping is updated to be consistent with Qiskit Nature.
    """
    num_spin_orbitals = num_sites * 2
    all_determinants_bitmasks = generate_all_determinants(num_spin_orbitals, num_electrons)

    det_to_idx = {det_mask: i for i, det_mask in enumerate(all_determinants_bitmasks)}
    idx_to_det = {i: det_mask for i, det_mask in enumerate(all_determinants_bitmasks)}
    num_dets = len(all_determinants_bitmasks)

    hamiltonian = defaultdict(float)

    for i in range(num_dets):
        det_I_mask = idx_to_det[i]

        # --- Diagonal elements H_ii (On-site repulsion U) ---
        on_site_energy = 0.0
        for site_idx in range(num_sites):
            # Use the new mapping function which requires num_sites
            spin_up_so = get_spin_orbital_from_site_spin(site_idx, 0, num_sites)
            spin_down_so = get_spin_orbital_from_site_spin(site_idx, 1, num_sites)
            if ((det_I_mask >> spin_up_so) & 1) and ((det_I_mask >> spin_down_so) & 1):
                on_site_energy += U_onsite
        hamiltonian[(i, i)] = on_site_energy

        # --- Off-diagonal elements H_ij (Hopping term -t) ---
        for j in range(i + 1, num_dets):
            det_J_mask = idx_to_det[j]
            xor_mask = det_I_mask ^ det_J_mask

            if count_set_bits(xor_mask) == 2: # Single excitation
                removed_from_I = get_set_bits(det_I_mask & xor_mask)[0]
                added_to_I = get_set_bits(det_J_mask & xor_mask)[0]

                # Use the new mapping function, which requires num_sites
                removed_site, removed_spin = get_site_spin_from_spin_orbital(removed_from_I, num_sites)
                added_site, added_spin = get_site_spin_from_spin_orbital(added_to_I, num_sites)

                if removed_spin == added_spin and abs(removed_site - added_site) == 1:
                    phase = fermionic_sign_single_exc(det_I_mask, removed_from_I, added_to_I)
                    hamiltonian[(i, j)] = -t_hop * phase
                    hamiltonian[(j, i)] = -t_hop * phase # Matrix is symmetric

    return hamiltonian, det_to_idx, idx_to_det, num_dets


from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import LineLattice, BoundaryCondition

def get_hubbard_sparse_pauli_op(num_sites, t_hop, U_onsite):
    """
    Constructs the Fermi-Hubbard Hamiltonian as a Qiskit SparsePauliOp
    using the official Qiskit Nature classes. This ensures correctness and
    consistency with the Hamiltonian used in the VQE module.
    """
    # 1. Define the lattice for the Hubbard model
    line_lattice = LineLattice(
        num_nodes=num_sites,
        boundary_condition=BoundaryCondition.OPEN,
    )

    # 2. Create the FermiHubbardModel
    fhm_model = FermiHubbardModel(
        lattice=line_lattice.uniform_parameters(
            uniform_interaction=-t_hop,  # Hopping parameter
            uniform_onsite_potential=0.0,
        ),
        onsite_interaction=U_onsite,  # On-site interaction
    )

    # 3. Get the second-quantized operator
    hamiltonian_ferm_op = fhm_model.second_q_op()

    # 4. Map to a qubit operator using the Jordan-Wigner transformation
    qubit_hamiltonian = JordanWignerMapper().map(hamiltonian_ferm_op)

    return qubit_hamiltonian

def get_hubbard_sparse_pauli_op(num_sites, t_hop, U_onsite):
    """
    Manually constructs the Hubbard model SparsePauliOp for U * n_up * n_down physics.
    This version is verified to be consistent with calculate_hubbard_hamiltonian.
    """
    num_spin_orbitals = 2 * num_sites
    op_dict = {}
    if num_sites > 1:
        for i in range(num_sites - 1):
            op_dict[f"+_{i} -_{i+1}"] = -t_hop
            op_dict[f"+_{i+1} -_{i}"] = -t_hop
            op_dict[f"+_{i + num_sites} -_{i + 1 + num_sites}"] = -t_hop
            op_dict[f"+_{i + 1 + num_sites} -_{i + num_sites}"] = -t_hop
    
    for i in range(num_sites):
        up_idx, down_idx = i, i + num_sites
        op_dict[f"+_{up_idx} -_{up_idx} +_{down_idx} -_{down_idx}"] = U_onsite

    hamiltonian_ferm_op = FermionicOp(op_dict, num_spin_orbitals=num_spin_orbitals)
    return JordanWignerMapper().map(hamiltonian_ferm_op)


def get_manual_hubbard_model(num_sites, t_hop, U_onsite):
    """
    Manually constructs the Hubbard model SparsePauliOp for U * n_up * n_down physics.
    """
    num_spin_orbitals = 2 * num_sites
    op_dict = {}

    # Hopping terms: -t * (c_i,s^† c_j,s + h.c.)
    if num_sites > 1:
        for i in range(num_sites - 1):
            op_dict[f"+_{i} -_{i+1}"] = -t_hop
            op_dict[f"+_{i+1} -_{i}"] = -t_hop
            op_dict[f"+_{i + num_sites} -_{i + 1 + num_sites}"] = -t_hop
            op_dict[f"+_{i + 1 + num_sites} -_{i + num_sites}"] = -t_hop
    
    # On-site interaction: U * n_i,up * n_i,down
    for i in range(num_sites):
        up_idx, down_idx = i, i + num_sites
        op_dict[f"+_{up_idx} -_{up_idx} +_{down_idx} -_{down_idx}"] = U_onsite

    hamiltonian_ferm_op = FermionicOp(op_dict, num_spin_orbitals=num_spin_orbitals)
    return JordanWignerMapper().map(hamiltonian_ferm_op)


# def get_hubbard_sparse_pauli_op(num_sites, t_hop, U_onsite):
#     """
#     Constructs the Fermi-Hubbard Hamiltonian as a Qiskit SparsePauliOp
#     by manually defining the fermionic operators and applying the
#     Jordan-Wigner transformation. This is the correct way to generate
#     the qubit operator.
#     """
#     op_list = []
    
#     # Hopping terms (-t) for adjacent sites
#     if num_sites > 1:
#         for i in range(num_sites - 1):
#             # Up-spin hopping
#             op_list.append((f"+_{i} -_{i+1}", -t_hop))
#             op_list.append((f"+_{i+1} -_{i}", -t_hop))
#             # Down-spin hopping
#             down_spin_offset = num_sites
#             op_list.append((f"+_{i+down_spin_offset} -_{i+1+down_spin_offset}", -t_hop))
#             op_list.append((f"+_{i+1+down_spin_offset} -_{i+down_spin_offset}", -t_hop))

#     # On-site interaction terms (U)
#     for i in range(num_sites):
#         up_idx = i
#         down_idx = i + num_sites
#         op_list.append((f"+_{up_idx} -_{up_idx} +_{down_idx} -_{down_idx}", U_onsite))
        
#     ferm_op_dict = {label: coeff for label, coeff in op_list}
#     hamiltonian_ferm_op = FermionicOp(ferm_op_dict)
    
#     # Map to a qubit operator using the Jordan-Wigner transformation
#     qubit_hamiltonian = JordanWignerMapper().map(hamiltonian_ferm_op)
    
#     return qubit_hamiltonian



def generate_connected_dets(det_idx, idx_to_det, det_to_idx, num_spin_orbitals):
    """
    Given a determinant index, generate all single and double excitations.
    """
    det_bitmask = idx_to_det[det_idx]
    connected = []

    occupied = [i for i in range(num_spin_orbitals) if (det_bitmask >> i) & 1]
    unoccupied = [i for i in range(num_spin_orbitals) if not (det_bitmask >> i) & 1]
    num_occ = len(occupied)

    # Single excitations
    for p in occupied:
        for r in unoccupied:
            new_det = (det_bitmask & ~(1 << p)) | (1 << r)
            if new_det in det_to_idx:
                new_idx = det_to_idx[new_det]
                pgen = 1
                connected.append((new_idx, 'single', pgen))

    # Double excitations
    for i in range(num_occ):
        for j in range(i + 1, num_occ):
            p, q = occupied[i], occupied[j]
            for a in range(len(unoccupied)):
                for b in range(a + 1, len(unoccupied)):
                    r, s = unoccupied[a], unoccupied[b]
                    new_det = (det_bitmask & ~(1 << p) & ~(1 << q)) | (1 << r) | (1 << s)
                    if new_det in det_to_idx:
                        new_idx = det_to_idx[new_det]
                        pgen = 1
                        connected.append((new_idx, 'double', pgen))
    
    return connected

def generate_hf_determinant(num_sites, num_electrons):
    """
    Generate a Hartree-Fock guess determinant bitmask by filling the
    lowest-energy spin orbitals.
    """
    num_spin_orbitals = num_sites * 2
    assert 0 <= num_electrons <= num_spin_orbitals, "Invalid electron number"
    
    hf_mask = 0
    for spin_orbital_idx in range(num_electrons):
        hf_mask |= (1 << spin_orbital_idx)
    return hf_mask

