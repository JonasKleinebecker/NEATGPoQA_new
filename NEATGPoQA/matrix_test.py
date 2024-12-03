import copy
import numpy as np
from qiskit import QuantumCircuit
from gate_gene import GateGene
from quantum_simulation import run_circuit_old

unitary_matrix = np.eye(4)
unitary_matrix[[0,1,2,3]] = unitary_matrix[[1,0,3,2]]
#print(unitary_matrix)

list = [1, 2, 3, 4, 5]
list.insert(len(list), 6)
#print(list)

qc = QuantumCircuit(3)
gates = []
gates.append(GateGene(gate_name="h", target_qubit=0))
gates.append(GateGene(gate_name="mcx", control_qubits=[0], target_qubit=1))
gates.append(GateGene(gate_name="mcx", control_qubits=[0,1], target_qubit=2))
gates.append(GateGene(gate_name="h", target_qubit=1))
gates.append(GateGene(gate_name="rx", angle=0.345, target_qubit=2))
gates.append(GateGene(gate_name="swap", control_qubits=[0], target_qubit=1))
gates.append(GateGene(gate_name="mcp", angle=0.123, control_qubits=[0, 1], target_qubit=2))
inverse_gates = copy.deepcopy(gates)
inverse_gates.reverse()
for gate in inverse_gates:
    gate.__invert__()
gates.extend(inverse_gates)
statevector = run_circuit_old("statevector", gates, 3, 0)
print(statevector)

class Test:
    def __init__(self, name, index):
        self.name = name
        self.index = index
    def __str__(self):
        return f"{self.name}({self.index})"

testlist = []
for i in range(5):
    testlist.append(Test(f"test{i}", i))

for test in testlist:
    print(test)

move_point = 3
move_length = 2
move_to_point = 1

if move_point < move_to_point:
    if move_point + move_length < move_to_point: # other cases result in no change
        for test in testlist:
            if move_point <= test.index < move_point + move_length: # move the elements that are in the range of the move
                test.index += move_to_point - move_point - move_length
            elif move_point + move_length <= test.index < move_to_point: # move that can indirectly affected by the move
                test.index -= move_length
elif move_point > move_to_point:
        for test in testlist:
            if move_point <= test.index < move_point + move_length: # move the elements that are in the range of the move
                test.index += move_to_point - move_point
            elif move_to_point <= test.index < move_point: # move that can indirectly affected by the move
                test.index += move_length

for test in testlist:
    print(test)