#%% Simulation of the Hybrid QC-FCIQMC Algorithm
import numpy as np
import random
import math
from tqdm import tqdm
from collections import defaultdict

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.quantum_info import Operator
from utils2 import *
from VQE_1D_hubbard import *
from visualize import *


#%%
def get_H_prime_element(hamiltonian_op: SparsePauliOp, U_circuit: QuantumCircuit,
                        i_idx: int, j_idx: int, idx_to_det: dict) -> complex:
    """
    Computes the transformed Hamiltonian matrix element H'_{ij} = ⟨i|U† H U|j⟩ = ⟨φ_i|H|φ_j⟩
    
    Args:
        hamiltonian_op: The problem Hamiltonian as SparsePauliOp
        U_circuit: Variational quantum circuit from VQE
        i_idx, j_idx: Indices of determinants i and j
        idx_to_det: Mapping from indices to determinant bit masks
    
    Returns:
        Complex matrix element H'_{ij}
    """
    num_qubits = U_circuit.num_qubits
    
    # Create |ψ_j⟩ = U |j⟩
    det_j_bitmask = idx_to_det[j_idx]
    psi_j_sv = Statevector.from_int(det_j_bitmask, dims=2**num_qubits).evolve(U_circuit)
    
    # Create |ψ_i⟩ = U |i⟩
    det_i_bitmask = idx_to_det[i_idx]
    psi_i_sv = Statevector.from_int(det_i_bitmask, dims=2**num_qubits).evolve(U_circuit)
    
    # Compute ⟨ψ_i| H |ψ_j⟩ = ⟨i|U† H U|j⟩
    h_psi_j_sv = psi_j_sv.evolve(hamiltonian_op)
    h_prime_ij = psi_i_sv.inner(h_psi_j_sv)
    
    return h_prime_ij


def annihilate_walkers(walkers):
    """
    Performs the annihilation step by summing signs of walkers on the same determinant.
    walkers: list of (determinant_index, sign) tuples
    """
    # Create dict of all the occup. determinants and their total sign.
    occupied_determinants = defaultdict(int)
    for det_idx, sign in walkers:
        occupied_determinants[det_idx] += sign

    annihilated_walkers = []
    for det_idx, signed_sum in occupied_determinants.items():
        if signed_sum != 0:
            # Add walkers with the final sign
            for _ in range(abs(signed_sum)):
                annihilated_walkers.append((det_idx, int(np.sign(signed_sum))))
    return annihilated_walkers


def run_qcfciqmc(hamiltonian_op, H_prime_matrix, idx_to_det, ref_idx=0,
                 num_steps=10000, dt=0.001, N_W_initial=10, N_W_target=1000, 
                 E_trial=-1.0, shift_update_time=50, damping=0.1):
    """
    Main QC-FCIQMC simulation function implementing the hybrid quantum-classical algorithm.
    This version uses a unified walker list, similar to the classical implementation.
    """
    
    # Cache diagonal elements for efficiency
    num_determinants = len(idx_to_det)
    H_diag_cache = {i: H_prime_matrix[i, i] for i in range(num_determinants)}

    # ====================================================================
    # FCIQMC INITIALIZATION
    # ====================================================================

    # Initialize walker population as a list of (determinant_index, sign) tuples
    walkers = [(ref_idx, 1)] * N_W_initial

    # Initialize tracking arrays
    total_walkers_history = [len(walkers)]
    shift_history = []
    estimated_energies = []

    # Initialize energy shift S (controls walker population growth)
    S = E_trial
    shift_history.append(S)
    shift_update_enabled = False # Population control variables
    
    # ====================================================================
    # MAIN FCIQMC TIME EVOLUTION LOOP
    # ====================================================================
    print(f"Starting QC-FCIQMC simulation with {num_steps} steps...")
    
    for step in tqdm(range(num_steps)):
        
        # Annihilate walkers at the start of the step to get the current state
        walkers = annihilate_walkers(walkers)
        
        if not walkers:
            print(f"\n--- Step {step}: All walkers have died out. Simulation stopped. ---")
            break

        # ================================================================
        # STEP 1: SPAWNING
        # ================================================================

        # List of newly spawned walkers and survivors from the death step
        spawned_children = []

        for det_idx_parent, sign_parent in walkers:

            # Attempt spawning to all connected determinants j
            # For j with non-zero |H_{ij}|
            for j in range(num_determinants):
                if det_idx_parent == j:
                    continue  # No self-spawning
                
                H_ji = H_prime_matrix[j, det_idx_parent]  # Matrix element H'_{ji}
                
                # Make sure the matrix element is non-zero
                if abs(H_ji) < 1e-12:
                    continue

                # Spawning probability proportional to |H'_{ji}| × dt
                # The number of spawned walkers is stochastic
                prob_spawn = dt * abs(H_ji)
                
                num_spawned_here = 0
                if prob_spawn > 1.0: 
                    num_spawned_here = int(math.floor(prob_spawn))
                    prob_spawn -= num_spawned_here
                
                if random.random() < prob_spawn: 
                    num_spawned_here += 1
                
                if num_spawned_here > 0:
                    sign_child = sign_parent * (-np.sign(H_ji))

                    for _ in range(num_spawned_here):
                        spawned_children.append((j, int(sign_child)))
        
        # ================================================================
        # STEP 2: DEATH/CLONING (Diagonal Evolution)
        # ================================================================
        survivors_after_death_clone = []
        for det_idx, sign in walkers:
            H_ii = H_diag_cache[det_idx]
            
            # Probability of death/cloning event
            prob_death_clone = dt * (H_ii - S)

            if prob_death_clone > 0: # Death event
                if random.random() >= prob_death_clone:
                    survivors_after_death_clone.append((det_idx, sign)) # Walker survives
            elif prob_death_clone < 0: # Cloning event
                abs_prob_clone = abs(prob_death_clone)
                num_clones = int(math.floor(abs_prob_clone))
                if random.random() < (abs_prob_clone - num_clones):
                    num_clones += 1
                
                # Add original and clones
                for _ in range(num_clones + 1):
                    survivors_after_death_clone.append((det_idx, sign))
            else: # prob_death_clone == 0
                survivors_after_death_clone.append((det_idx, sign))

        # ================================================================
        # STEP 3: COMBINE AND PREPARE FOR NEXT ITERATION
        # ================================================================
        # The new walker list is the combination of spawned children and survivors
        walkers = spawned_children + survivors_after_death_clone
        total_walkers_new = len(walkers) # get total walker number

        # ================================================================
        # POPULATION CONTROL AND SHIFT UPDATE
        # ================================================================
        # Enable shift updates when target population is reached
        if not shift_update_enabled and total_walkers_new >= N_W_target:
            print(f"\n--- Step {step}: Target walkers reached ({total_walkers_new}). Enabling dynamic shift. ---")
            shift_update_enabled = True
            
        # Update energy shift S to control population growth
        if shift_update_enabled and (step + 1) % shift_update_time == 0:
            update_term = (damping / (shift_update_time * dt)) * math.log(max(total_walkers_new, 1) / N_W_target)
            S -= update_term

        # ================================================================
        # PROJECTED ENERGY CALCULATION
        # ================================================================
        # First, get the net populations after annihilation for an accurate energy reading
        current_populations_dict = defaultdict(int)
        for det_idx, sign in walkers:
            current_populations_dict[det_idx] += sign
        
        projected_energy = np.nan
        N_ref = current_populations_dict.get(ref_idx, 0)

        if N_ref != 0:
            # E_proj = H'_{00} + Σ_{j≠0} H'_{0j} * (N_j / N_0)
            H_00 = H_diag_cache[ref_idx]
            energy_sum = H_00
            for j, N_j in current_populations_dict.items():
                if j != ref_idx:
                    H_0j = H_prime_matrix[ref_idx, j]
                    energy_sum += H_0j * (N_j / N_ref)
            projected_energy = energy_sum
            estimated_energies.append(projected_energy)
        elif estimated_energies: # If ref walker disappears, reuse last energy
            estimated_energies.append(estimated_energies[-1])

        # Record history for plotting
        shift_history.append(S)
        total_walkers_history.append(total_walkers_new)

    return total_walkers_history, shift_history, estimated_energies


#%%
if __name__ == '__main__':

    # ====================================================================
    # SYSTEM AND SIMULATION PARAMETERS
    # ====================================================================
    t_hop = 1.0          # Hopping parameter
    U_coulomb = 4.0      # Coulomb repulsion parameter
    num_sites = 2        # Number of sites in 1D Hubbard model
    num_electrons = 1    # Number of electrons

    # Simulation Parameters
    N_W_initial = 10         # Initial walker population
    N_W_target = 5000        # Target walker population for shift updates
    dt = 0.001               # Time step
    num_steps = 10000        # Total simulation steps
    shift_update_time = 20   # Update shift every 20 steps
    damping = 0.1            # Damping factor for shift updates

    # ====================================================================
    # RUNNING VQE PREPROCESSING TO GET THE UNITARY CIRCUIT (U)
    # ====================================================================
    U_circuit_VQE, VQE_energy = run_vqe_hubbard_model(
        num_sites=num_sites,
        num_electrons=num_electrons,
        t_hop=t_hop,
        U_onsite=U_coulomb)
    
    print(f"VQE completed. Final VQE energy: {VQE_energy:.8f}")


    # ====================================================================
    # SETUP THE CORRECT HAMILTONIAN AND BASIS
    # ====================================================================

    # Generate Hubbard Hamiltonian Pauli Operators
    H_q_hubbard = get_hubbard_sparse_pauli_op(num_sites, t_hop, U_coulomb)
    H_full_matrix = H_q_hubbard.to_matrix() # convert to matrix

    # Get ground state energy by exact diagonalization
    exact_gs_energy = np.linalg.eigh(H_full_matrix)[0][0]
    print(f"Exact ground state energy: {exact_gs_energy:.4f}")

    # Get the determinant-to-index mappings
    num_spin_orbitals = num_sites * 2
    all_determinants_bitmasks = generate_all_determinants(num_spin_orbitals, num_electrons)

    det_to_idx = {det_mask: i for i, det_mask in enumerate(all_determinants_bitmasks)}
    idx_to_det = {i: det_mask for i, det_mask in enumerate(all_determinants_bitmasks)}
    num_dets = len(all_determinants_bitmasks)


    # ====================================================================
    #  APPLY THE BASIS CHANGED 
    # ====================================================================

    # Create Identity Circuit for debugging
    U_circuit_id = QuantumCircuit(U_circuit_VQE.num_qubits)
    U_matrix = Operator(U_circuit_id).to_matrix()

    # Apply the basis change
    U_matrix = Operator(U_circuit_VQE).to_matrix()
    H_prime_full = U_matrix.conj().T @ H_full_matrix @ U_matrix

    # Project the full H' into the smaller FCI basis
    # The FCI basis only includes those states from the full space
    # that have the correct particle number.
    #  <φ_i|H|φ_j⟩ = (⟨i|U†) H (U|j⟩) = ⟨i|U†HU|j⟩
    fci_indices = list(idx_to_det.values())

    H_prime_matrix = np.zeros((num_dets, num_dets), dtype=complex)
    for i in range(num_dets):
        for j in range(num_dets):
            H_prime_matrix[i, j] = H_prime_full[fci_indices[i], fci_indices[j]]

    H_prime_matrix = H_prime_matrix.real
    # np.set_printoptions(linewidth=np.inf)
    # print("\nH_full_matrix (Real Part):")
    # print(H_full_matrix.real)

    # print("\n H_prime_matrix:")
    # print(H_prime_matrix.real)

    # Find the ground state of H' to determine the reference state
    eigenvalues_prime, e_vecs_prime = np.linalg.eigh(H_prime_matrix)
    print(f"\nGround state energy of the new H' matrix: {eigenvalues_prime[0].real:.8f}")


    ground_state_vector_H_prime = e_vecs_prime[:, 0]
    new_ref_idx = np.argmax(np.abs(ground_state_vector_H_prime))
    print(f"ANALYSIS: The new reference determinant index is: {new_ref_idx}")

    # The diagonal of a Hermitian matrix is always real.
    initial_shift = H_prime_matrix[new_ref_idx, new_ref_idx].real
    print(f"Using initial shift S = H'_{{{new_ref_idx},{new_ref_idx}}} = {initial_shift:.8f}")

    # ====================================================================
    # RUN THE QC-FCIQMC SIMULATION
    # ====================================================================
    print("\n--- Starting QC-FCIQMC Simulation ---")
    total_walkers, shift, estimated_energies = run_qcfciqmc(
        hamiltonian_op=H_q_hubbard,
        H_prime_matrix=H_prime_matrix,
        idx_to_det=idx_to_det,
        ref_idx=new_ref_idx,
        num_steps=num_steps,
        dt=dt,
        N_W_initial=N_W_initial,
        N_W_target=N_W_target,
        E_trial=initial_shift,
        shift_update_time=shift_update_time,
        damping=damping
    )

    print("\n--- Simulation Finished. Plotting Results ---")
    plot_results(total_walkers, shift, estimated_energies,
                 dt, N_W_target, exact_gs_energy, shift_update_time)
# %%
