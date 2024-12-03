import random
import numpy as np

from possible_qubits import Possible_qubits

class GateGene:
    def __init__(self, gate_type, angle = None, control_qubits = [], target_qubit = None, is_inverted = False):
        self.gate_type = gate_type
        self.angle = angle
        self.control_qubits = control_qubits
        self.target_qubit = target_qubit
        self.is_inverted = is_inverted

    def to_args_list(self):
        args_list = []
        args_list.append(self.gate_type.value)
        if self.gate_type.has_angle():
            args_list.append(self.angle)
        if self.gate_type.control_qubit_capability() != Possible_qubits.NONE:
            args_list.append(self.control_qubits)
        if self.gate_type.target_qubit_capability() != Possible_qubits.NONE:
            args_list.append(self.target_qubit)
        return args_list
    def __str__(self):
        return f"{self.gate_type.value}({self.angle}, {self.control_qubits}, {self.target_qubit}, {self.is_inverted})"

    def invert(self):
        self.is_inverted = not self.is_inverted

    def mutate_qubits(self, num_qubits):
        if random.random() < 0.5:
            if self.gate_type.control_qubit_capability() == Possible_qubits.SINGLE:
                #random qubit that is not the target qubit
                self.control_qubits = [random.choice([q for q in range(num_qubits) if q != self.target_qubit])]
            elif self.gate_type.control_qubit_capability() == Possible_qubits.MULTIPLE:
                num_control_qubits = random.randint(1, num_qubits - 1)
                self.control_qubits = random.sample([q for q in range(num_qubits) if q != self.target_qubit], num_control_qubits)
        else:
            if self.gate_type.target_qubit_capability() != Possible_qubits.NONE:
                self.target_qubit = random.choice([q for q in range(num_qubits) if q not in self.control_qubits])
                

    def mutate_angle(self, angle_change, num_qubits):
        if self.gate_type.has_angle():
            self.angle += angle_change
            self.angle %= 2 * np.pi
        else:
            self.mutate_qubits(num_qubits)
            
