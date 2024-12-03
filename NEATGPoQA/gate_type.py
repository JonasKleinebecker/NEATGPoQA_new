from possible_qubits import Possible_qubits
from enum import Enum

class GateType(Enum):
    X = "x"
    Y = "y"
    Z = "z"
    H = "h"
    RX = "rx"
    RY = "ry"
    RZ = "rz"
    MCX = "mcx"
    MCZ = "mcz"
    MCP = "mcp"
    SWAP = "swap"
    ORACLE = "oracle"

    def create_gate_type(gate_str):
        return GateType(gate_str.lower()) # case insensitive matching
    
    def target_qubit_capability(self):
        if self in [GateType.ORACLE]:
            return Possible_qubits.NONE
        elif self in [GateType.X, GateType.Y, GateType.Z, GateType.H, GateType.RX, GateType.RY, GateType.RZ, GateType.MCX, GateType.MCZ, GateType.MCP]:
            return Possible_qubits.MULTIPLE
        elif self in [GateType.SWAP]:
            return Possible_qubits.SINGLE
        else:
            raise ValueError(f"Gate {self} not defined.")

    def control_qubit_capability(self):
        if self in [GateType.X, GateType.Y, GateType.Z, GateType.H, GateType.RX, GateType.RY, GateType.RZ, GateType.ORACLE]:
            return Possible_qubits.NONE
        elif self in [GateType.MCX, GateType.MCZ, GateType.MCP]:
            return Possible_qubits.MULTIPLE
        elif self in [GateType.SWAP]:
            return Possible_qubits.SINGLE
        else:
            raise ValueError(f"Gate {self} not defined.")

    def has_angle(self):
        if self in [GateType.RX, GateType.RY, GateType.RZ, GateType.MCP]:
            return True
        else:
            return False

