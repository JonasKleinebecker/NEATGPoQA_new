from numpy import pi
from qiskit.circuit.library import UnitaryGate
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import QasmSimulator, StatevectorSimulator
from qiskit.circuit.library.standard_gates import SGate, TGate

qc = QuantumCircuit(3)
qc2 = QuantumCircuit(4)
qc3 = QuantumCircuit(2)

qc3.h(0)
qc3.h(1)
qc3.cx(0, 1)
qc3.draw('mpl', filename='neat_example_circuit.png')

##the 3-qubit QFT
#cs_gate = SGate().control(1)
#ct_gate = TGate().control(1)
#qc.h(0)
#qc.append(cs_gate, [1, 0])
#qc.append(ct_gate, [2, 0])
#qc.h(1)
#qc.append(cs_gate, [2, 1])
#qc.h(2)
#qc.swap(0, 2) 
#qc.draw('latex_source', filename='qft.tex')

##the Full addder
#qc2.ccx(0, 1, 3)
#qc2.cx(0, 1)
#qc2.ccx(1, 2, 3)
#qc2.cx(1, 2)
#qc2.cx(0, 1)
#qc2.draw('latex_source', filename='full_adder.tex')