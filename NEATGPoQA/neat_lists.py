from node import Node
from node_type import Node_type
class Neat_lists:
    def __init__(self):
        self.nodes = []
        self.gate_nodes = []
        self.start_nodes = []
        self.end_nodes = []
        self.oracle_node = None
        self.connections = []
        self.connections_by_qubit = {}
    
    def add_node(self, node):
        self.nodes.append(node)
        if node.node_type == Node_type.START:
            self.start_nodes.append(node)
        elif node.node_type == Node_type.ORACLE:
            self.oracle_node = node
        elif node.node_type == Node_type.GATE:
            self.gate_nodes.append(node)
        elif node.node_type == Node_type.END:
            self.end_nodes.append(node)

    def add_connection(self, connection):
        self.connections.append(connection)
        self.connections_by_qubit[connection.qubit] = connection
        connection.input_node.add_connection_out(connection)
        connection.output_node.add_connection_in(connection)

    def remove_connection(self, connection):
        self.connections.remove(connection)
        del self.connections_by_qubit[connection.qubit]
        connection.input_node.remove_connection_out(connection)
        connection.output_node.remove_connection_in(connection)

    