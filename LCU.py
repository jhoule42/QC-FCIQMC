#%% Script to compute the Linear Combination of Unitaries (LCU)

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator
import math

#%%
def create_prep_circuit(alpha_k: list, n_ancilla: int):
    """
    Creates the PREP circuit and its inverse (PREP_dag) for a general LCU.

    Args:
        alpha_k (list or np.ndarray): A list of the alpha_k coefficients.
        n_ancilla (int): The number of ancilla qubits.

    Returns:
        tuple: A tuple containing:
            - QuantumCircuit: The circuit for the PREP operation.
            - QuantumCircuit: The circuit for the PREP_dag operation.
    """
    # Convert to a numpy array for easier calculations.
    alpha_k = np.array(alpha_k)
    num_terms = len(alpha_k)
    
    # Lambda (λ) is the sum of all alpha_k coefficients.
    lam = np.sum(alpha_k)
    if lam == 0:
        raise ValueError("Sum of alpha_k (lambda) cannot be zero.")
        
    # Calculate the amplitudes for the state: |ψ> = Σ sqrt(α_k / λ) |k>.
    amplitudes = np.sqrt(alpha_k / lam)
    
    # The full state vector has a size of 2^n_ancilla.
    state_vector_size = 2**n_ancilla
    prep_vector = np.zeros(state_vector_size, dtype=complex)
    prep_vector[0:num_terms] = amplitudes

    # Create the PREP circuit using state preparation.
    prep_circuit = QuantumCircuit(n_ancilla, name='PREP')
    prep_circuit.prepare_state(prep_vector, prep_circuit.qubits)
    
    # The inverse circuit is simply the dagger of the PREP circuit.
    prep_dag_circuit = prep_circuit.inverse().to_gate()
    prep_dag_circuit.name = 'PREP_dag'
    
    return prep_circuit.to_gate(), prep_dag_circuit

def create_sel_circuit(unitary_list: list, n_system: int, n_ancilla: int):
    """
    Creates the SEL (select) circuit for a given list of unitaries.

    Args:
        unitary_list (list): A list of QuantumCircuit objects for each U_k.
        n_system (int): The number of qubits in the system register.
        n_ancilla (int): The number of qubits in the ancilla register.

    Returns:
        QuantumCircuit: The circuit implementing the SEL operation.
    """
    system_reg = QuantumRegister(n_system, name='system')
    ancilla_reg = QuantumRegister(n_ancilla, name='ancilla')
    sel_circuit = QuantumCircuit(system_reg, ancilla_reg, name='SEL')

    for k, unit_circ in enumerate(unitary_list):
        # Format k as a binary string for the control state.
        ctrl_state = f"{k:0{n_ancilla}b}"
        
        # Create a controllable gate from the unitary circuit.
        custom_gate = unit_circ.to_gate(label=f"U_{k}").control(
            num_ctrl_qubits=n_ancilla,
            label=f"C-U_{k}",
            ctrl_state=ctrl_state
        )
        
        # Apply the controlled gate.
        sel_circuit.append(custom_gate, ancilla_reg[:] + system_reg[:])

    return sel_circuit.to_gate()

def create_lcu_circuit(unitary_list: list, alpha_k: list):
    """
    Constructs the full LCU circuit U_A = PREP_dag * SEL * PREP.

    Args:
        unitary_list (list): A list of QuantumCircuit objects for each U_k.
        alpha_k (list): A list of the corresponding alpha_k coefficients.

    Returns:
        QuantumCircuit: The full LCU block-encoding circuit.
    """
    if len(unitary_list) != len(alpha_k):
        raise ValueError("Length of unitary_list and alpha_k must be the same.")
        
    n_system = unitary_list[0].num_qubits
    num_terms = len(unitary_list)
    n_ancilla = math.ceil(math.log2(num_terms))

    # Create the registers.
    system = QuantumRegister(n_system, name='sys')
    ancilla = QuantumRegister(n_ancilla, name='anc')
    lcu_circuit = QuantumCircuit(system, ancilla, name='LCU_A')

    # 1. Build the PREP and PREP_dag circuits.
    prep_gate, prep_dag_gate = create_prep_circuit(alpha_k, n_ancilla)

    # 2. Build the SEL circuit.
    sel_gate = create_sel_circuit(unitary_list, n_system, n_ancilla)

    # 3. Assemble the full LCU circuit: PREP, SEL, PREP_dag.
    lcu_circuit.append(prep_gate, ancilla)
    lcu_circuit.append(sel_gate, system[:] + ancilla[:])
    lcu_circuit.append(prep_dag_gate, ancilla)

    return lcu_circuit



#%% ============== DEMONSTRATION AND VERIFICATION ==============
# Define a system with 2 qubits.
N_system_qubits = 2

# Define a list of unitaries (U_k) for our Hamiltonian A.
u0 = QuantumCircuit(N_system_qubits, name='Z0')
u0.z(0)
u1 = QuantumCircuit(N_system_qubits, name='X1')
u1.x(1)
u2 = QuantumCircuit(N_system_qubits, name='Y0Y1')
u2.y(0)
u2.y(1)
unitary_list = [u0, u1, u2]

# Define the corresponding coefficients (alpha_k).
alpha_coefficients = [0.8, 0.3, 1.5]

# Build the LCU circuit
lcu_circuit = create_lcu_circuit(unitary_list, alpha_coefficients)
print("--- Full LCU Circuit (U_A = PREP_dag * SEL * PREP) ---")
lcu_circuit.draw('mpl')



#%% ============== Verification Step ==============
# We verify that the top-left block of the U_A matrix equals A/λ.

# Get the unitary matrix of the full LCU circuit.
print("\n--- Verifying the block encoding ---")
full_lcu_op = Operator(lcu_circuit)
full_lcu_matrix = full_lcu_op.data

# Extract the top-left block.
num_terms = len(alpha_coefficients)
n_ancilla = math.ceil(math.log2(num_terms)) if num_terms > 0 else 0
n_system = lcu_circuit.num_qubits - n_ancilla
block_size = 2**n_system
encoded_A_matrix = full_lcu_matrix[0:block_size, 0:block_size]

# Construct the target matrix A/λ directly.
lam = np.sum(alpha_coefficients)
# The shape of A_matrix must match the system size.
A_matrix = np.zeros((block_size, block_size), dtype=complex)
for alpha, u_circ in zip(alpha_coefficients, unitary_list):
    A_matrix += alpha * Operator(u_circ).data
target_matrix = A_matrix / lam

# Compare the matrices.
are_matrices_close = np.allclose(encoded_A_matrix, target_matrix)

print(f"Lambda (λ) = {lam:.4f}")
print(f"Verification successful: {are_matrices_close}")

if not are_matrices_close:
    print("Mismatch detected!")
    print("Encoded A/λ from circuit:\n", np.round(encoded_A_matrix, 4))
    print("\nTarget A/λ from definition:\n", np.round(target_matrix, 4))

# %%
