from numpy import pi
import qiskit.circuit.library
from qiskit.circuit.library import UnitaryGate
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import QasmSimulator, StatevectorSimulator
import random
import numpy as np
from qiskit.circuit.library import MCPhaseGate
from gate_gene import GateGene
from node_type import Node_type
from possible_qubits import Possible_qubits
from qubit_type import Qubit_type

def create_deutsch_josza_oracle_gate(num_input_qubits, oracle_case):
    """
    Creates a quantum gate implementing a Deutsch-Jozsa oracle.

    Args:
        oracle_type (int): Specifies the case of oracle to create. 
                           0 = constant 0 function
                           1 = constant 1 function
                           2-8 = different balanced functions
        num_input_qubits (int): The number of input qubits (1 or 2). Defaults to 2.

    Returns:
        Gate: A quantum gate representing the chosen oracle.
    """

    if num_input_qubits not in [1, 2]:
        raise ValueError("Invalid num_input_qubits. Choose 1 or 2.")

    total_qubits = num_input_qubits + 1  # input qubits + 1 output qubit
    qr = QuantumRegister(total_qubits, 'q')
    qc = QuantumCircuit(qr)

    if oracle_case == 0:
        pass  

    elif oracle_case == 1:
        qc.x(qr[total_qubits-1])  # Flip the output qubit to 1

    elif oracle_case in range(2, 8):
        if num_input_qubits == 1:
            if(oracle_case > 3):
                raise ValueError("Invalid oracle_type. For the 1 input qubit variant there are only 4 possible functions.")
            outputs_map = [[0, 1], [1, 0]]  # For 1 input qubit
            outputs = outputs_map[oracle_case - 2]
            if outputs[0] == 1:
                qc.cx(qr[0], qr[total_qubits - 1]) 
        else:  
            outputs_map = [
                [0, 0, 1, 1], 
                [0, 1, 0, 1], 
                [0, 1, 1, 0], 
                [1, 0, 0, 1], 
                [1, 0, 1, 0], 
                [1, 1, 0, 0]]
            outputs = outputs_map[oracle_case - 2]

        for i, output in enumerate(outputs):
            if output == 1:
                for j in range(num_input_qubits):
                    if i & (1 << j):  # Check if the j-th bit of 'i' is 1
                        qc.cx(qr[j], qr[total_qubits-1])

    else:
        raise ValueError("Invalid oracle_type. Must be an integer between 0 and 8.")

    return qc

def prepare_initial_state(qc, initial_bit_state):
    for j, bit in enumerate(initial_bit_state):
        if bit == '1':
            qc.x(j)

def gene_individual_to_qiskit_circuit(individual, num_qubits, initial_bit_state = "0", oracle_case = 0):
    qc = QuantumCircuit(num_qubits)
    prepare_initial_state(qc, initial_bit_state)
    for gate in individual:
        gate_name, *args = gate.to_args_list()
        if gate_name == "oracle":
            oracle_qc = create_deutsch_josza_oracle_gate(num_qubits - 1, oracle_case)
            qc.compose(oracle_qc, qubits=range(num_qubits), inplace=True)
        else:
            if gate.is_inverted:
                     getattr(qc, gate_name)(*args).inverse()
            else:
                getattr(qc, gate_name)(*args)  
    return qc

def neat_individual_to_qiskit_circuit(individual, num_qubits, initial_bit_state = "0", oracle_case = 0):
    qc = QuantumCircuit(num_qubits)
    prepare_initial_state(qc, initial_bit_state)
    appended_multi_qubit_gates = []
    qubits_arrived_at_multi_qubit_gates = {}
    start_nodes = [node for node in individual.start_nodes]
    curr_nodes = []
    for i in range(num_qubits):
        curr_nodes.append([connection.output_node for connection in start_nodes[i].connections_out if connection.qubit == i][0])
    qubits_done = []
    while len(qubits_done) < num_qubits:
        for i in [qubit for qubit in range(num_qubits) if qubit not in qubits_done]:
            hit_break = False
            while not curr_nodes[i].node_type == Node_type.END:
                curr_node = curr_nodes[i]
                if not curr_node.is_controlled_gate() and not curr_node.node_type == Node_type.ORACLE:
                    args = [i]
                    if curr_node.gate_type.has_angle():
                        args.insert(0, curr_node.angle)
                    getattr(qc, curr_node.gate_type.value)(*args)
                elif curr_node.is_controlled_gate() or curr_node.node_type == Node_type.ORACLE:
                    if curr_node.is_valid_gate() and not curr_node in appended_multi_qubit_gates:
                        if curr_node not in qubits_arrived_at_multi_qubit_gates:
                            qubits_arrived_at_multi_qubit_gates[curr_node] = [i]
                            hit_break = True
                            break
                        else:
                            qubits_arrived_at_multi_qubit_gates[curr_node].append(i)
                            if len(qubits_arrived_at_multi_qubit_gates[curr_node]) == len(curr_node.connections_in):
                                appended_multi_qubit_gates.append(curr_node)
                                control_qubits = [connection.qubit for connection in curr_node.connections_in if connection.type == Qubit_type.CONTROL]
                                target_qubits = [connection.qubit for connection in curr_node.connections_in if connection.type == Qubit_type.TARGET]
                                if curr_node.gate_type.control_qubit_capability() == Possible_qubits.SINGLE:
                                    getattr(qc, curr_node.gate_type.value)(control_qubits[0], target_qubits[0]) # ignore all qubits except the first ones
                                elif curr_node.gate_type.control_qubit_capability() == Possible_qubits.MULTIPLE:
                                    for target_qubit in target_qubits:
                                        args = [control_qubits, target_qubit] 
                                        if curr_node.gate_type.has_angle():
                                            args.insert(0, curr_node.angle) 
                                        getattr(qc, curr_node.gate_type.value)(*args)
                                else: # oracle gate
                                    oracle_qc = create_deutsch_josza_oracle_gate(num_qubits - 1, oracle_case)
                                    qc.compose(oracle_qc, qubits=range(num_qubits), inplace=True)
                            else:
                                hit_break = True
                                break 
                curr_nodes[i] = [connection.output_node for connection in curr_node.connections_out if connection.qubit == i][0]
            if not hit_break:
                qubits_done.append(i)
    return qc

def run_circuit(qc, simulator_type, num_shots = 1024):
    num_qubits = qc.num_qubits
    if simulator_type == "qasm":
        qc.measure(range(num_qubits), range(num_qubits))
        simulator = QasmSimulator()
        job = simulator.run(qc, shots=num_shots)
        counts = job.result().get_counts(qc) 
        for key in counts:
            counts[key] /= num_shots
        return counts
    elif simulator_type == "statevector":
        simulator = StatevectorSimulator()
        job = simulator.run(qc)
        statevector = job.result().get_statevector(qc) 
        return statevector

def generate_000_statevector():
    qc = QuantumCircuit(3)
    simulator = StatevectorSimulator()
    job = simulator.run(qc)
    return job.result().get_statevector(qc) 

def generate_3_qubit_fourier_transform_statevectors():
    num_qubits = 3
    statevectors = {}
    for i in range(2**num_qubits):
        # Convert the integer 'i' to its binary representation
        initial_state_binary = format(i, '0' + str(num_qubits) + 'b')
        qc = QuantumCircuit(num_qubits)

        # Initialize the qubits to the corresponding initial state
        for j, bit in enumerate(initial_state_binary):
            if bit == '1':
                qc.x(j)

        qc.h(2)
        qc.cp(pi/2, 1, 2)
        qc.h(1)
        qc.cp(pi/4, 0, 2)
        qc.cp(pi/2, 0, 1)
        qc.h(0)
        qc.swap(0, 2) 

        simulator = StatevectorSimulator()
        job = simulator.run(qc)
        statevector = job.result().get_statevector(qc) 
        statevectors[initial_state_binary] = statevector
    return statevectors

def generate_3_qubit_quantum_full_adder_statevectors():
    num_qubits = 4
    statevectors = {}
    for i in range(2**(num_qubits - 1)): # The last qubit is always 0
        # Convert the integer 'i' to its binary representation
        initial_state_binary = format(i, '0' + str(num_qubits - 1) + 'b') + '0'
        qc = QuantumCircuit(num_qubits)

        # Initialize the qubits to the corresponding initial state
        for j, bit in enumerate(initial_state_binary):
            if bit == '1':
                qc.x(j)

        qc.ccx(0, 1, 3)
        qc.cx(0, 1)
        qc.ccx(1, 2, 3)
        qc.cx(1, 2)
        qc.cx(0, 1)
        simulator = StatevectorSimulator()
        job = simulator.run(qc)
        statevector = job.result().get_statevector(qc)
        statevectors[initial_state_binary] = statevector
    return statevectors
        