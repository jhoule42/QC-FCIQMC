#%% Implementation of the Full Configuration Interaction Quantum Monte Carlo Algorithm.
# Author: Julien-Pierre Houle
# Revised by AI for Appendix B spawning and scoping.

import numpy as np
import random
import math
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import eigsh
from utils import *

#%%
class FCIQMCSimulator:
    """
    Basic FCIQMC simulation in a discrete Slatter space.
    """

    def __init__(self, hamiltonian, num_dets, idx_to_det, det_to_idx, num_spin_orbitals,
                 delta_tau=1e-3, initial_nb_walker=1,
                 target_walkers=1000, damping=0.05, update_interval=10,
                 initial_shift=0.0, max_iters=10000, hf_determinant_idx=0):
        
        self.hamiltonian = hamiltonian
        self.num_dets = num_dets
        self.idx_to_det = idx_to_det
        self.det_to_idx = det_to_idx
        self.num_spin_orbitals = num_spin_orbitals
        self.delta_tau = delta_tau
        self.shift = initial_shift
        self.target_walkers = target_walkers
        self.damping = damping
        self.update_interval = update_interval
        self.max_iters = max_iters
        
        if not (0 <= hf_determinant_idx < self.num_dets):
            print(f"Warning: HF determinant index {hf_determinant_idx} is out of bounds for num_dets={self.num_dets}. Resetting to 0.")
            self.hf_determinant_idx = 0
        else:
            self.hf_determinant_idx = hf_determinant_idx

        # List of (det_index, sign)
        self.walkers = [(self.hf_determinant_idx, 1)] * initial_nb_walker
        self.iteration = 0
        self.history = []
    
    
    def spawn_step(self, parent_walkers):
        """
        Performs the spawning step using Appendix B-style excitation generation.
        Computes pgen(j ← i) explicitly and uses it to normalize spawning probability.
        """
        spawned_children = []
        if not parent_walkers:
            return spawned_children

        for det_idx_parent, sign_parent in parent_walkers:
            # Generate all connected determinants single & double exctitation from parents.
            # Uses instance attributes for idx_to_det, det_to_idx, and num_spin_orbitals
            connected_dets = generate_connected_dets(det_idx_parent,
                                                     self.idx_to_det,
                                                     self.det_to_idx,
                                                     self.num_spin_orbitals)

            # Apply spawning rules for all the determinants
            for det_idx_child, _, pgen in connected_dets:

                # H_val for spawning from parent i to child j is <D_j|H|D_i> = H_ji
                # If self.hamiltonian stores H_row,col = <row|H|col>, then we need H_child,parent
                H_val = self.hamiltonian.get((det_idx_child, det_idx_parent), 0.0) 
                
                if H_val == 0 or pgen == 0: # pgen can be zero if no excitations of a type are possible
                    continue

                # Spawning probability using Booth et al. Eq. (15) / (B1)
                # P_spawn = delta_tau * |H_ji| / p_gen(j|i)
                prob_spawn = self.delta_tau * abs(H_val) / pgen

                num_spawned_here = 0
                if prob_spawn > 1.0: 
                    num_spawned_here = int(math.floor(prob_spawn))
                    prob_spawn -= num_spawned_here
                
                if random.random() < prob_spawn: 
                    num_spawned_here += 1

                if num_spawned_here > 0:
                    # Sign of child: sign_parent * sign(-H_ji)
                    sign_child = sign_parent * (-1 * np.sign(H_val))
                    for _ in range(num_spawned_here):
                        spawned_children.append((det_idx_child, int(sign_child)))

        return spawned_children


    def death_cloning_step(self, parent_walkers):
        """
        Performs the diagonal death/cloning step on parent_walkers.
        Returns a new list of walkers (survivors and their clones).
        """
        new_walker_list_after_death_clone = []
        if not parent_walkers:
            return new_walker_list_after_death_clone

        for det_idx, sign in parent_walkers:
            H_ii = self.hamiltonian.get((det_idx, det_idx), 0.0) 
            prob_death_clone = self.delta_tau * (H_ii - self.shift) 

            if prob_death_clone > 0: # Death event
                if random.random() < prob_death_clone:
                    continue
                else: # Walker survives
                    new_walker_list_after_death_clone.append((det_idx, sign))

            elif prob_death_clone < 0: # Cloning event
                abs_prob_clone = abs(prob_death_clone)
                num_additional_clones = int(math.floor(abs_prob_clone)) 
                if random.random() < (abs_prob_clone - num_additional_clones): 
                    num_additional_clones += 1
                
                for _ in range(num_additional_clones + 1): 
                    new_walker_list_after_death_clone.append((det_idx, sign))
            else: # prob_death_clone == 0
                new_walker_list_after_death_clone.append((det_idx, sign))
                
        return new_walker_list_after_death_clone


    def annihilate_walkers(self, walkers):
        """
        Performs the annihilation step.
        """
        occupied_determinants = defaultdict(int)
        for det_idx, sign in walkers:
            occupied_determinants[det_idx] += sign

        annihilated_walkers = []
        for det_idx, signed_sum in occupied_determinants.items():
            if signed_sum != 0:
                for _ in range(abs(signed_sum)):
                    annihilated_walkers.append((det_idx, int(np.sign(signed_sum))))
        return annihilated_walkers


    def run(self):
        """ Runs the FCIQMC simulation. """

        startup_phase_completed = False
        E_HF_D0 = self.hamiltonian.get((self.hf_determinant_idx, 
                                        self.hf_determinant_idx), 0.0)

        # Loop for each time step delta tau
        for self.iteration in tqdm(range(self.max_iters), desc="FCIQMC Iterations"):

            # Construct the population dict: {determinant idx: sign}
            current_populations_dict = defaultdict(int)
            for det_idx, sign in self.walkers:
                current_populations_dict[det_idx] += sign

            parent_walkers_for_steps = []
            for det_idx, signed_sum in current_populations_dict.items():
                if signed_sum != 0:
                    sign_val = int(np.sign(signed_sum))
                    for _ in range(abs(signed_sum)):
                        parent_walkers_for_steps.append((det_idx, sign_val))
            
            # Make sure there are still walkers
            if not parent_walkers_for_steps and self.iteration > 0: 
                print(f"\nIteration {self.iteration}: No walkers left to process. Terminating.")
                break


            # Combine the 3 steps for the Algorithm
            spawned_children = self.spawn_step(parent_walkers_for_steps)
            walkers_after_death_clone = self.death_cloning_step(parent_walkers_for_steps)
            all_walkers_for_annihilation = spawned_children + walkers_after_death_clone
            self.walkers = self.annihilate_walkers(all_walkers_for_annihilation)

            # Re-construct the updated population dict by determinant
            current_populations_dict.clear()
            for det_idx, sign in self.walkers:
                current_populations_dict[det_idx] += sign
            num_total_walkers = sum(abs(count) for count in current_populations_dict.values())
            
            # Deal with startup phase to initially increase the number of walker.
            if not startup_phase_completed:
                if num_total_walkers >= self.target_walkers:
                    startup_phase_completed = True
                    tqdm.write(f"\nStartup phase completed at iteration {self.iteration+1}. Num walkers: {num_total_walkers}. Starting adaptive shift.")
            
            # Update the shift parameter (S)
            if startup_phase_completed and \
                (self.iteration + 1) % self.update_interval == 0:
                self.shift -= (self.damping / (self.update_interval * self.delta_tau)) * math.log(num_total_walkers / self.target_walkers)


            # ------------------ Projected Energy Calculation ------------------
            # E_proj = E_HF + ∑ <D_j|H|D_0> * N_j(τ)/ N_0(τ)  [eq. 20]
    
            projected_energy = np.nan # initialize energy

            # Make sure the Hartree-Fock determinant is in the population dict
            if self.hf_determinant_idx in current_populations_dict and current_populations_dict[self.hf_determinant_idx] != 0:
                
                N_0_tau = current_populations_dict[self.hf_determinant_idx]
                projected_energy = E_HF_D0  # Start with H_00 (E_HF for the reference)
                
                # Sum over off-diagonal connections from HF determinant
                for (i, j), H_val in self.hamiltonian.items():
                    # Look for H_j0 terms
                    if j == self.hf_determinant_idx and i != self.hf_determinant_idx:
                        if i in current_populations_dict and current_populations_dict[i] != 0:
                            # Add H_i0 * (N_i / N_0)
                            projected_energy += H_val * (current_populations_dict[i] / N_0_tau)
            

            self.history.append((self.iteration, num_total_walkers, self.shift, projected_energy))
            
            if num_total_walkers == 0 and self.iteration > 2 * self.update_interval : 
                 tqdm.write(f"\nIteration {self.iteration+1}: Population collapsed. Terminating.")
                 break
        
        if self.iteration == self.max_iters - 1:
            tqdm.write("\nMaximum iterations reached.")
        return self.history
    


#%% =========================  Set Simulation Parameters  =========================
if __name__ == "__main__":

    # 1D Hubbard Model Parameters
    num_sites_val = 4
    num_electrons_val = 4  # Half-filling for 2 sites
    t_hop_val = 1.0
    U_onsite_val = 4.0 # Example

    # Generate the Hamiltonian
    ham_dict, det_to_idx_map, idx_to_det_map, total_dets = \
        calculate_hubbard_hamiltonian(num_sites_val, num_electrons_val, t_hop_val, U_onsite_val)
    
    num_spin_orbitals_val = num_sites_val * 2
    print(f"System: {num_sites_val} sites, {num_electrons_val} electrons, U/t = {U_onsite_val/t_hop_val if t_hop_val != 0 else 'inf'}")
    print(f"Total number of determinants: {total_dets}")

    # Fill lowest-energy spin orbitals for the given number of electrons (Hartree-Fock)
    hf_mask = generate_hf_determinant(num_sites_val, num_electrons_val)
    hf_idx = det_to_idx_map.get(hf_mask) # get corresponding Hamiltonian index
    print(f"\nHartree-Fock determinant Bitmask: {bin(hf_mask)}, Index: {hf_idx}")

    # Set the initial shift to the Hartree-Fock Energy
    initial_s_val = ham_dict.get((hf_idx, hf_idx))
    print(f"Initial shift (E_HF for D_0): {initial_s_val:.2f}")


    # Simulation parameters
    delta_tau_sim = 0.001 
    initial_walkers_sim = 10     # Increased initial walkers
    target_walkers_sim = 1e4    # Reduced target for small system
    max_iters_sim = 3000        # Longer simulation time
    update_interval_sim = 10    # Slower shift updates
    damping_sim = 0.1           # Damping factor

    print(f"\nStarting FCIQMC simulation for {num_sites_val}-site Hubbard model...")
    simulator = FCIQMCSimulator(
        hamiltonian=ham_dict,
        num_dets=total_dets,
        idx_to_det=idx_to_det_map,
        det_to_idx=det_to_idx_map,
        num_spin_orbitals=num_spin_orbitals_val,
        delta_tau=delta_tau_sim,
        initial_nb_walker=initial_walkers_sim,
        target_walkers=target_walkers_sim,
        damping=damping_sim,
        update_interval=update_interval_sim,
        initial_shift=initial_s_val, # Start shift at E_HF of D_0
        max_iters=max_iters_sim,
        hf_determinant_idx=hf_idx
    )

    sim_history = simulator.run()
    print('FCIQMC simulation done.')


    # Exact Solution by Diagonalization
    E_gs_exact = np.nan
    if total_dets > 0:
        H_matrix_sparse = dok_matrix((total_dets, total_dets), dtype=np.float64)
        for (r, c), val in ham_dict.items():
            H_matrix_sparse[r, c] = val
        
        if H_matrix_sparse.nnz > 0: 
            try:
                if total_dets < 200: 
                    eigenvalues_exact = np.linalg.eigh(H_matrix_sparse.todense())[0]
                else:
                    eigenvalues_exact, _ = eigsh(H_matrix_sparse.tocsr(),
                                                 k=min(10, total_dets-1),
                                                 which='SA')
                E_gs_exact = eigenvalues_exact[0]
                print(f"Exact ground state energy (from diagonalization): {E_gs_exact:.6f}")
            except Exception as e:
                print(f"Error during exact diagonalization: {e}")
        else:
            print("Hamiltonian matrix is empty, skipping exact diagonalization.")
    else:
        print("No determinants generated, skipping exact diagonalization.")


    #%% ===========================  Plotting Results  ===========================
    if sim_history:
        iters_plot, walkers_plot, shifts_plot, energies_plot = zip(*sim_history)

        plt.figure(figsize=(14, 10))
        plt.subplot(2, 1, 1)
        plt.plot(iters_plot, energies_plot, label="Projected Energy ($E_{proj}$)", marker='.', linestyle='-', markersize=2, alpha=0.6)
        plt.plot(iters_plot, shifts_plot, label="Shift ($S$)", linestyle='--', alpha=0.8)
        if not np.isnan(E_gs_exact):
            plt.axhline(E_gs_exact, color='r', linestyle=':', linewidth=2, label=f"Exact $E_0$ ({E_gs_exact:.4f})")
        plt.xlabel("Iteration")
        plt.ylabel("Energy")
        plt.ylim(min(E_gs_exact - 2, np.nanmin(energies_plot)-1) if not np.isnan(E_gs_exact) else np.nanmin(energies_plot)-1, 
                 max(initial_s_val + 2, np.nanmax(energies_plot)+1)) # Dynamic y-lim
        plt.title(f"FCIQMC: Energy ({num_sites_val} sites, {num_electrons_val} e, U={U_onsite_val:.1f}, $\\tau=${delta_tau_sim})")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.subplot(2, 1, 2)
        plt.plot(iters_plot, walkers_plot, label="Number of Walkers ($N_w$)", color='g', marker='.', linestyle='-', markersize=2, alpha=0.6)
        plt.axhline(target_walkers_sim, color='orange', linestyle=':', linewidth=2, label=f"Target Walkers ({target_walkers_sim})")
        plt.xlabel("Iteration")
        plt.ylabel("Number of Walkers")
        plt.yscale('log') 
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()
    else:
        print("No simulation history to plot.")
# %%
