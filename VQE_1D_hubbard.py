#%%
import numpy as np
from qiskit.circuit.library import EfficientSU2

from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD, UCC
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import LineLattice, BoundaryCondition
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SPSA, SLSQP, COBYLA
from qiskit_aer.primitives import Estimator as AerEstimator

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_nature.second_q.operators import FermionicOp
from utils2 import get_hubbard_sparse_pauli_op

#%%

def run_vqe_hubbard_model(num_sites, num_electrons, t_hop, U_onsite):
    """
    Runs a VQE simulation for a 1D Fermi-Hubbard model to find an
    approximate ground state circuit and energy.
    """

    # # --- 1. System Setup ---
    # line_lattice = LineLattice(num_nodes=num_sites, boundary_condition=BoundaryCondition.OPEN)
    # fhm_model = FermiHubbardModel(
    #     lattice=line_lattice.uniform_parameters(
    #         uniform_interaction=-t_hop,
    #         uniform_onsite_potential=0.0
    #     ),
    #     onsite_interaction=U_onsite
    # )
    # hamiltonian_2nd_q = fhm_model.second_q_op()
    # mapper = JordanWignerMapper()

    qubit_hamiltonian = get_hubbard_sparse_pauli_op(num_sites, t_hop, U_onsite)
    mapper = JordanWignerMapper() # Keep the mapper for use in the ansatz
    print(f"Second quantized qubit Hamiltonian:\n{qubit_hamiltonian}")

    num_alpha = num_electrons // 2
    num_beta = num_electrons - num_alpha
    num_particles = (num_alpha, num_beta)

    # --- 2. Define the Ansatz and Initial State ---
    initial_state = HartreeFock(num_spatial_orbitals=num_sites,
                                num_particles=num_particles,
                                qubit_mapper=mapper)

    # 1. Define the single excitation...
    excitation_op = FermionicOp({"+_3 -_2": 1.0}, num_spin_orbitals=num_sites * 2)

    # 2. Create the anti-hermitian operator...
    cluster_op = excitation_op - excitation_op.adjoint()

    # 3. Map to an anti-hermitian qubit operator (with complex coefficients).
    qubit_op = mapper.map(cluster_op)

    # 4. Convert to a Hermitian operator with real coefficients by multiplying by 1j.
    hermitian_op = (1j * qubit_op)

    # 5. Define a single variational parameter.
    theta = Parameter("θ")

    # 6. Create the evolution circuit using the real Hermitian operator.
    evolution_circuit = QuantumCircuit(initial_state.num_qubits)
    evolution_gate = PauliEvolutionGate(hermitian_op, time=theta) # Use the corrected operator
    evolution_circuit.append(evolution_gate, range(initial_state.num_qubits))

    ansatz = initial_state.compose(evolution_circuit)

    # ansatz = EfficientSU2()

    optimizer = COBYLA(maxiter=1000)
    
    # Use the Aer Estimator for simulation
    estimator = AerEstimator(run_options={"shots": None}, approximation=True)

    # Instantiate the VQE solver
    vqe_solver = VQE(estimator=estimator,
                     ansatz=ansatz,
                     optimizer=optimizer)

    # Run the VQE optimization
    vqe_result = vqe_solver.compute_minimum_eigenvalue(operator=qubit_hamiltonian)
    E_vqe = vqe_result.eigenvalue.real

    optimal_circuit = ansatz.assign_parameters(vqe_result.optimal_point)

    return optimal_circuit, E_vqe


# %%
if __name__ == '__main__':
    # Set your simulation parameters
    num_sites = 2
    num_electrons = 1
    t_hop = 1.0
    U_onsite = 4.0

    # --- Run the VQE simulation ---
    optimized_circuit, ground_state_energy, hamiltonian, n_orbitals = run_vqe_hubbard_model(
        num_sites=num_sites,
        num_electrons=num_electrons,
        t_hop=t_hop,
        U_onsite=U_onsite,
        # reps=ansatz_repetitions
    )

    print("\n--- VQE Results ---")
    print(f"Number of sites: {num_sites}")
    print(f"Number of electrons: {num_electrons}")
    print(f"Hopping parameter (t): {t_hop}")
    print(f"On-site interaction (U): {U_onsite}")
    print(f"\nVQE Ground State Energy: {ground_state_energy:.6f}\n")

# %%
