from possible_qubits import Possible_qubits
from qubit_type import Qubit_type
from node_type import Node_type
class Node:
    def __init__(self, inno_num, node_type = Node_type.GATE, gate_type = None, angle = None):
        self.inno_num = inno_num
        self.gate_type = gate_type
        self.node_type = node_type
        self.angle = angle
        self.connections_in = []
        self.connections_out = []

    def add_connection_in(self, connection):
        self.connections_in.append(connection)
    
    def add_connection_out(self, connection):
        self.connections_out.append(connection)
    
    def remove_connection_in(self, connection):
        self.connections_in.remove(connection)
    
    def remove_connection_out(self, connection):
        self.connections_out.remove(connection)

    def is_controlled_gate(self):
        if self.gate_type == None:
            return False
        else:
            return self.gate_type.control_qubit_capability() != Possible_qubits.NONE

    def is_valid_gate(self):
        if self.gate_type == None:
            return True
        if self.gate_type.control_qubit_capability() == Possible_qubits.NONE:
            return True
        else:
            has_target = False
            has_control = False
            for connection in self.connections_in:
                if connection.type == Qubit_type.TARGET:
                    has_target = True
                    if has_control:
                        return True
                elif connection.type == Qubit_type.CONTROL:
                    has_control = True
                    if has_target:
                        return True
            return False
