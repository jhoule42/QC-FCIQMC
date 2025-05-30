# Utilitary functions, especially to generate the 1D Hubbard Hamiltonian.
# Author: Julien-Pierre Houle

from collections import defaultdict
import itertools

def get_spin_orbital_from_site_spin(site_idx, spin_idx):
    """ Maps (site_idx, spin_idx) to a unique spin orbital index."""
    return site_idx * 2 + spin_idx

def get_site_spin_from_spin_orbital(spin_orbital_idx):
    """ Maps a spin orbital index back to (site_idx, spin_idx)."""
    site_idx = spin_orbital_idx // 2
    spin_idx = spin_orbital_idx % 2
    return site_idx, spin_idx

def generate_all_determinants(num_spin_orbitals, num_electrons):
    """ Generates all possible Slatter determinants as bitmasks.
    Each bit correspond to a spin orbital.
    Bit 1 means the spin-orbital is occupied.
    Bit 0 means the spin-orbital is not occupied.
    """
    all_spin_orbitals = list(range(num_spin_orbitals)) # [0, 1, 2, ..., N]
    determinants = []
    for combo in itertools.combinations(all_spin_orbitals, num_electrons):
        bitmask = 0
        for so_idx in combo:
            bitmask |= (1 << so_idx) # Turn on bit at position i, and leave others unchanged.
        determinants.append(bitmask)
    return sorted(determinants) # Sort for consistent indexing

def count_set_bits(n):
    """ Counts the number of set bits (1s) in a bitmask. """
    count = 0
    while n > 0:
        n &= (n - 1)
        count += 1
    return count

def get_set_bits(n):
    """ Returns a list of indices of set bits in a bitmask. """
    bits = []
    idx = 0
    while n > 0:
        if n & 1:
            bits.append(idx)
        n >>= 1
        idx += 1
    return bits


def generate_hf_determinant(num_sites, num_electrons):
    """
    Generate a Hartree-Fock guess determinant bitmask by filling the
    lowest-energy spin orbitals for the given number of electrons.
    
    Assumes num_spin_orbitals = num_sites * 2.
    
    Returns an integer bitmask representing the occupied spin orbitals.

    """
    num_spin_orbitals = num_sites * 2
    assert 0 <= num_electrons <= num_spin_orbitals, "Invalid electron number"
    
    hf_mask = 0

    # Fill the lowest num_electrons spin orbitals (index 0 to num_electrons-1)
    for spin_orbital_idx in range(num_electrons):
        # 1 << spin_orbital_idx: decalage binaire vers la gauche
        # |= : Ajoute un 1 au bit correspondant sans modifiers les autres bits
        # ie: Pour 3 electrons:
        # spin_orbital_idx = 0 → hf_mask = 0b0001
        # spin_orbital_idx = 1 → hf_mask = 0b0011
        # spin_orbital_idx = 2 → hf_mask = 0b0111
        hf_mask |= (1 << spin_orbital_idx)
    return hf_mask



def fermionic_sign_single_exc(det_mask, p, r):
    """
    Computes the fermionic sign of a single excitation: p → r.
    Returns +1 or -1.
    """
    if p == r:
        return 1

    sign = 1
    lower = min(p, r)
    upper = max(p, r)

    # Count number of occupied orbitals between p and r
    for i in range(lower + 1, upper):
        if (det_mask >> i) & 1:
            sign *= -1
    return sign


def calculate_hubbard_hamiltonian(num_sites, num_electrons, t_hop, U_onsite):
    """
    Generates the Hamiltonian for a 1D Hubbard model with open boundary conditions.
    Returns the Hamiltonian dictionary, a mapping from bitmask to index,
    a mapping from index to bitmask, and the total number of determinants.
    """

    # Get all the occupied determinant bitmask (expressed as integer)
    num_spin_orbitals = num_sites * 2
    all_determinants_bitmasks = generate_all_determinants(num_spin_orbitals, num_electrons)
    
    # Create correspondance dictionary to map bitmasks to integer indices
    det_to_idx = {det_mask: i for i, det_mask in enumerate(all_determinants_bitmasks)}
    idx_to_det = {i: det_mask for i, det_mask in enumerate(all_determinants_bitmasks)}

    hamiltonian = defaultdict(float)
    num_dets = len(all_determinants_bitmasks)

    # Loop through each determinants
    for i in range(num_dets):
        det_I_mask = idx_to_det[i]

        # Diagonal elements H_II (On-site repulsion U)
        on_site_energy = 0.0
        for site_idx in range(num_sites):

            # check for a given Slatter determinant whether both spin up & down
            # orbitals are occupied on the same site.
            spin_up_spin_orbital = get_spin_orbital_from_site_spin(site_idx, 0)
            spin_down_spin_orbital = get_spin_orbital_from_site_spin(site_idx, 1)
            
            # Check if both spin are occupied
            is_spin_up_occupied = (det_I_mask >> spin_up_spin_orbital) & 1
            is_spin_down_occupied = (det_I_mask >> spin_down_spin_orbital) & 1
            
            if is_spin_up_occupied and is_spin_down_occupied:
                on_site_energy += U_onsite
        hamiltonian[(i, i)] = on_site_energy

        # Off-diagonal elements H_IJ (hopping term -t)
        # Loop over all pairs of determinant once
        for j in range(i + 1, num_dets): # Only calculate upper triangle, then add lower
            det_J_mask = idx_to_det[j]
            
            # Check if determinants differ by exactly two orbitals (single excitation)
            xor_mask = det_I_mask ^ det_J_mask  # finds which bits are different between 2 determinants. 

            if count_set_bits(xor_mask) == 2:
                # Find the differing orbitals: one removed from I, one added to I to get J
                removed_from_I = get_set_bits(det_I_mask & xor_mask)[0]
                added_to_I = get_set_bits(det_J_mask & xor_mask)[0]

                # Check if it's a valid hopping event (same spin, adjacent sites)
                removed_site, removed_spin = get_site_spin_from_spin_orbital(removed_from_I)
                added_site, added_spin = get_site_spin_from_spin_orbital(added_to_I)

                if removed_spin == added_spin: # Same spin
                    # Check for adjacent sites (open boundary conditions)
                    if abs(removed_site - added_site) == 1:
                        # This is a hopping term. The matrix element is -t_hop.
                        phase = fermionic_sign_single_exc(det_I_mask, removed_from_I, added_to_I)
                        hamiltonian[(i, j)] = -t_hop * phase
                        hamiltonian[(j, i)] = -t_hop * phase # Symmetric (Hermitian operator)

    return hamiltonian, det_to_idx, idx_to_det, num_dets


def generate_connected_dets(det_idx, idx_to_det, det_to_idx, num_spin_orbitals):
    """
    Given a determinant index, generate all single and double excitations.
    Returns a list of tuples (new_det_idx, excitation_type, pgen_value).
    This is a simplified version (without symmetry constraints).
    """
    det_bitmask = idx_to_det[det_idx]
    connected = []

    occupied = [i for i in range(num_spin_orbitals) if (det_bitmask >> i) & 1]
    unoccupied = [i for i in range(num_spin_orbitals) if not (det_bitmask >> i) & 1]
    num_occ = len(occupied)

    # Single excitations
    for p in occupied:
        for r in unoccupied:
            # New determinant: remove the electron corresponding to bit p 
            new_det = (det_bitmask & ~(1 << p)) | (1 << r)
            if new_det in det_to_idx:
                new_idx = det_to_idx[new_det]
                # pgen = 1 / (num_occ * (num_spin_orbitals - num_occ))  # uniform for now
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
                        denom = (num_occ * (num_occ - 1) / 2) * (len(unoccupied) * (len(unoccupied) - 1) / 2)
                        # pgen = 1 / denom if denom > 0 else 0.0
                        pgen = 1
                        connected.append((new_idx, 'double', pgen))
    
    return connected



#%%

