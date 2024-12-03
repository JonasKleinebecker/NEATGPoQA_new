from collections import deque
import copy
import csv
import math
import random
import time
import os
import numpy as np
from deap import creator, base, tools
from check_direction import CheckDirection
from connection import Connection
from gate_type import GateType
from innovation_type import Innovation_type
from neat_lists import Neat_lists
from node import Node
from node_type import Node_type
from qiskit_aer.backends.compatibility import Statevector
from possible_qubits import Possible_qubits
from quantum_simulation import gene_individual_to_qiskit_circuit, generate_3_qubit_quantum_full_adder_statevectors, run_circuit, generate_3_qubit_fourier_transform_statevectors
from gate_gene import GateGene
from qubit_type import Qubit_type
from quantum_simulation import neat_individual_to_qiskit_circuit
from parameter_classes import NeatState, SetupParams, NeatParams, HyperParams, BaseGPParams, Statistics
import yaml

class ConfigurableGP:
    def __init__(self, config_file):
        config = self.load_yaml_config(config_file)
        config = self.parse_unsupported_types(config)

        self.stats = Statistics()
        self.setup_params = SetupParams(**config["setup_params"])
        self.hyper_params = HyperParams(**config["hyper_params"])
        if self.setup_params.use_neat:
            self.neat_params = NeatParams(**config["neat_params"])
            self.neat_state = NeatState()
        else:
            self.base_gp_params = BaseGPParams(**config["base_gp_params"])

        self.folder_path = f"results/{self.setup_params.computer_id}/"
        if self.setup_params.use_neat:
            self.folder_path += "neat_"
        else:
            self.folder_path += "base_"
        self.folder_path += f"{self.setup_params.test_problem}_{self.hyper_params.cross_prob}_{self.hyper_params.mut_prob}_{self.hyper_params.pop_size}/"
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        
        self.run_folder_path = self.folder_path + f"run_{self.setup_params.run_id}/"

        if not os.path.exists(self.run_folder_path):
            os.makedirs(self.run_folder_path)

        self.toolbox = base.Toolbox()
        creator.create("Fitness", base.Fitness, weights = self.setup_params.fitness_weights, id=None) 
        if(self.setup_params.use_neat):
            creator.create("Individual", Neat_lists, fitness=creator.Fitness, base_fitness=creator.Fitness, id=None ,is_elitist=False, species_id=None, offspring_count=0) # fitness is the adjusted fitness, base_fitness is the raw fitness
        else:
            creator.create("Individual", list, fitness=creator.Fitness)
        self.register_toolbox_functions()

    def parse_unsupported_types(self, config):
        gate_set = config["setup_params"]["gate_set"]
        processed_gate_set = [GateType.create_gate_type(gate_str) for gate_str in gate_set]
        config["setup_params"]["gate_set"] = processed_gate_set
        fitness_weights_str = config["setup_params"]["fitness_weights"]
        processed_fitness_weights = eval(fitness_weights_str)
        config["setup_params"]["fitness_weights"] = processed_fitness_weights
        return config

    def load_yaml_config(self, filepath):
        with open(filepath, "r") as file:
            config = yaml.safe_load(file)
        return config

    def register_toolbox_functions(self):
        if self.setup_params.use_neat:
            self.toolbox.register("individual", self.create_neat_individual)
            self.toolbox.register("mate", self.neat_crossover)
            if self.setup_params.use_multi_objective:
                self.toolbox.register("select", tools.selNSGA2)
            else:
                self.toolbox.register("select", tools.selBest)
            self.toolbox.register("mutate", self.mutate_neat_individual)
            self.toolbox.register("evaluate_adj", self.calculate_adjusted_fitness)
        else:
            self.toolbox.register("individual", self.create_linear_individual)
            self.toolbox.register("mate", self.single_point_crossover)
            self.toolbox.register("mutate", self.mutate_linear_individual)
            if self.setup_params.use_multi_objective:
                self.toolbox.register("select", tools.selTournamentDCD)
            else:
                self.toolbox.register("select", tools.selTournament, tournsize=4)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual) 
        if self.setup_params.test_problem == "deutsch_josza":
            self.toolbox.register("evaluate", self.evaluate_deutsch_josza_2_input_qubits)
            self.toolbox.register("calculate_error", self.calculate_error_fitness_probabilities)
            self.setup_params.num_qubits = 3
        elif self.setup_params.test_problem == "qft":
            self.toolbox.register("evaluate", self.evaluate_quantum_fourier_3_qubits)
            self.toolbox.register("calculate_error", self.calculate_phase_aware_error_fitness)
            self.setup_params.num_qubits = 3
        elif self.setup_params.test_problem == "full_adder":
            self.toolbox.register("evaluate", self.evaluate_quantum_full_adder_3_qubits)
            self.toolbox.register("calculate_error", self.calculate_error_fitness)
            self.setup_params.num_qubits = 4
        self.toolbox.register("calculate_fitness", self.calculate_error_and_gate_count_fitness)
        if self.setup_params.use_neat:
            self.toolbox.register("geno_to_pheno", neat_individual_to_qiskit_circuit)
        else:
            self.toolbox.register("geno_to_pheno", gene_individual_to_qiskit_circuit)

    def create_gene_gate(self):
        gate_type = random.choice(self.setup_params.gate_set)
        control_qubits = []  # Need to be remembered to avoid duplicate control and target qubits

        gate_args = {}
        gate_args["gate_type"] = gate_type
        if gate_type.has_angle():
            angle = np.random.uniform(0, 2 * np.pi)
            gate_args["angle"] = angle
        if gate_type.control_qubit_capability() == Possible_qubits.MULTIPLE:
            num_control_qubits = random.randint(1, self.setup_params.num_qubits - 1)
            control_qubits = random.sample(range(self.setup_params.num_qubits), num_control_qubits)
            gate_args["control_qubits"] = control_qubits
        elif gate_type.control_qubit_capability() == Possible_qubits.SINGLE:
            control_qubits.append(random.randint(0, self.setup_params.num_qubits - 1))
            gate_args["control_qubits"] = control_qubits
        if gate_type.target_qubit_capability() != Possible_qubits.NONE:
            target_qubit = random.choice([q for q in range(self.setup_params.num_qubits) if q not in control_qubits])
            gate_args["target_qubit"] = target_qubit
        return GateGene(**gate_args)

    def create_linear_individual(self):
        individual = []

        length = random.randint(self.base_gp_params.min_ind_length, self.setup_params.max_ind_length)  

        for _ in range(length):
            gate = self.create_gene_gate()
            individual.append(gate)

        return creator.Individual(individual)

    def create_neat_individual(self):
        new_individual = creator.Individual()
        num_qubits = self.setup_params.num_qubits
        starting_inno_nums = 0
        if self.setup_params.use_oracle:
            oracle_node = Node(self.get_new_innovation_numbers([starting_inno_nums], Innovation_type.INIT)[0], Node_type.ORACLE, GateType.ORACLE)
            new_individual.add_node(oracle_node)
            starting_inno_nums += 1
        for i in range(num_qubits):
            start_node = Node(self.get_new_innovation_numbers([starting_inno_nums], Innovation_type.INIT)[0], Node_type.START)
            starting_inno_nums += 1
            end_node = Node(self.get_new_innovation_numbers([starting_inno_nums], Innovation_type.INIT)[0], Node_type.END)
            starting_inno_nums += 1
            new_individual.add_node(start_node)
            new_individual.add_node(end_node)
            if self.setup_params.use_oracle:
                connection_in = Connection(start_node, oracle_node, i, Qubit_type.TARGET, self.get_new_innovation_numbers([starting_inno_nums], Innovation_type.INIT)[0])
                starting_inno_nums += 1
                connection_out = Connection(oracle_node, end_node, i, Qubit_type.TARGET, self.get_new_innovation_numbers([starting_inno_nums], Innovation_type.INIT)[0])
                starting_inno_nums += 1
                new_individual.add_connection(connection_in)
                new_individual.add_connection(connection_out)
            else:
                connection = Connection(start_node, end_node, i, Qubit_type.TARGET, self.get_new_innovation_numbers([starting_inno_nums], Innovation_type.INIT)[0])
                starting_inno_nums += 1
                new_individual.add_connection(connection)
        for i in range(self.neat_params.init_mutations):
            self.mutate_neat_individual(new_individual)
        return new_individual

    def single_point_crossover(self, ind1, ind2):
        if len(ind1) < 2 or len(ind2) < 2: 
            return ind1, ind2

        crossover_point = random.randrange(1, min(len(ind1), len(ind2)))

        # Exchange segments after the crossover point
        ind1[crossover_point:], ind2[crossover_point:] = (
            ind2[crossover_point:],
            ind1[crossover_point:],
        )
        return ind1, ind2

    def neat_crossover(self, ind1, ind2):
        child = creator.Individual()
        nodes_map = {} #track which nodes have been added to the child for quick access when adding connections
        for node in ind1.start_nodes:
            start_node = Node(node.inno_num, node_type=Node_type.START, angle=node.angle)
            nodes_map[node.inno_num] = start_node
            child.add_node(start_node)
        for node in ind1.end_nodes:
            end_node = Node(node.inno_num, node_type=Node_type.END, angle=node.angle)
            nodes_map[node.inno_num] = end_node
            child.add_node(end_node)
        if self.setup_params.use_oracle:
            ind1_oracle = ind1.oracle_node
            oracle_node = Node(ind1_oracle.inno_num, node_type=Node_type.ORACLE, gate_type=ind1_oracle.gate_type)
            nodes_map[ind1_oracle.inno_num] = oracle_node
            child.add_node(oracle_node)
        more_fit_ind, less_fit_ind = tools.selBest([ind1, ind2], 2)
        for node1 in more_fit_ind.gate_nodes:
            node_to_add = node1
            for node2 in less_fit_ind.gate_nodes:
                if node1.inno_num == node2.inno_num:
                    if random.random() < 0.5:
                        node_to_add = node2
                    break
            new_node = Node(node_to_add.inno_num, node_type=Node_type.GATE, gate_type=node_to_add.gate_type, angle=node_to_add.angle)
            nodes_map[node_to_add.inno_num] = new_node
            child.add_node(new_node)
        for connection1 in more_fit_ind.connections:
            connection_to_add = connection1
            for connection2 in less_fit_ind.connections:
                if connection1.inno_num == connection2.inno_num:
                    if random.random() < 0.5:
                        connection_to_add = connection2
                    break
            from_node = nodes_map[connection_to_add.input_node.inno_num]
            to_node = nodes_map[connection_to_add.output_node.inno_num]
            child_connection = Connection(from_node, to_node, connection_to_add.qubit, connection_to_add.type, connection_to_add.inno_num)
            child.add_connection(child_connection)
        return child,

# ------- Mutation operators -------

    def insert_random_gate(self, lin_individual):	
        if len(lin_individual) >= self.setup_params.max_ind_length:	
            return lin_individual, 
        new_gate = self.create_gene_gate()	
        insert_point = random.randint(0, len(lin_individual))	
        lin_individual.insert(insert_point, new_gate)	
        return lin_individual,
    
    def insert_random_gate_sequence(self, lin_individual):
        new_gates = [self.create_gene_gate() for _ in range(np.random.geometric(0.5))]
        if len(lin_individual) + len(new_gates) > self.setup_params.max_ind_length:
            return lin_individual,
        insert_point = random.randint(0, len(lin_individual))
        lin_individual[insert_point:insert_point] = new_gates
        return lin_individual,

    def insert_random_gate_sequence_and_inverse(self, lin_individual):
        new_gates = [self.create_gene_gate() for _ in range(np.random.geometric(0.5))]
        if len(lin_individual) + 2 * len(new_gates) > self.setup_params.max_ind_length:
            return lin_individual,
        inverse_gates = copy.deepcopy(new_gates)
        inverse_gates.reverse()
        for gate in inverse_gates:
            gate.invert()
        insert_point = random.randint(0, len(lin_individual))
        reverse_insert_point = random.randint(insert_point, len(lin_individual)) + len(new_gates)
        lin_individual[insert_point:insert_point] = new_gates
        lin_individual[reverse_insert_point:reverse_insert_point] = inverse_gates
        return lin_individual,

    def delete_random_gate_sequence(self, lin_individual):
        sequence_length = int(np.random.geometric(0.5))
        if len(lin_individual) > sequence_length:
            delete_point = random.randint(0, len(lin_individual) - sequence_length)
            del lin_individual[delete_point:delete_point + sequence_length]
        return lin_individual,

    def move_random_gate_sequence(self, lin_individual):  
        move_length = int(np.random.geometric(0.5)) # Number of gates to move
        if len(lin_individual) > move_length:
            move_point = random.randint(0, len(lin_individual) - move_length)
            move_to_point = random.randint(0, len(lin_individual) - move_length)
            while move_point == move_to_point and len(lin_individual) > move_length + 1:
                move_to_point = random.randint(0, len(lin_individual) - move_length)
            lin_individual[move_point:move_to_point - move_length], lin_individual[move_to_point:move_to_point + move_length] = lin_individual[move_point + move_length:move_to_point], lin_individual[move_point:move_point+move_length] #Todo: check this!
        return lin_individual,

    def swap_random_gates(self, lin_individual):
        index_to_move = random.randrange(len(lin_individual))
        move_to_index = random.randrange(len(lin_individual))
        if(len(lin_individual) > 1):
            while(index_to_move == move_to_index):
                move_to_index = random.randrange(len(lin_individual))
            lin_individual[index_to_move], lin_individual[move_to_index] = lin_individual[move_to_index], lin_individual[index_to_move]
        return lin_individual,

    def substitute_random_gate(self, lin_individual):
        index_to_change = random.randrange(len(lin_individual))
        current_gate_gene = lin_individual[index_to_change]
        new_gate_type = random.choice(list(filter(lambda x: x != current_gate_gene.gate_type, self.setup_params.gate_set)) ) # random new gate that is != the old gate
        if new_gate_type.target_qubit_capability() == Possible_qubits.NONE:
            current_gate_gene.target_qubit = None
        if new_gate_type.control_qubit_capability() == Possible_qubits.NONE:
            current_gate_gene.control_qubits = [] 
        elif new_gate_type.control_qubit_capability() == Possible_qubits.MULTIPLE and current_gate_gene.gate_type.control_qubit_capability() == Possible_qubits.NONE:
            num_control_qubits = random.randint(1, self.setup_params.num_qubits - 1)
            current_gate_gene.control_qubits = random.sample([q for q in range(self.setup_params.num_qubits) if q != current_gate_gene.target_qubit], num_control_qubits)
        elif new_gate_type.control_qubit_capability() == Possible_qubits.SINGLE:
            if current_gate_gene.gate_type.control_qubit_capability() == Possible_qubits.MULTIPLE:
                current_gate_gene.control_qubits = [random.choice(current_gate_gene.control_qubits)]
            elif current_gate_gene.gate_type.control_qubit_capability() == Possible_qubits.NONE:
                current_gate_gene.control_qubits = [random.choice([q for q in range(self.setup_params.num_qubits) if q != current_gate_gene.target_qubit])] 
    
        if new_gate_type.target_qubit_capability() != Possible_qubits.NONE and current_gate_gene.gate_type.target_qubit_capability() == Possible_qubits.NONE:        
            current_gate_gene.target_qubit = random.choice([q for q in range(self.setup_params.num_qubits) if q not in current_gate_gene.control_qubits])
    
        if new_gate_type.has_angle() and not current_gate_gene.gate_type.has_angle():
            current_gate_gene.angle = np.random.uniform(0, 2 * np.pi)
        elif not new_gate_type.has_angle() and current_gate_gene.gate_type.has_angle():
            current_gate_gene.angle = None
    
        current_gate_gene.gate_type = new_gate_type
    
        return lin_individual,

    def change_random_angle(self, lin_individual):
        index_to_change = random.randrange(len(lin_individual))
        gate = lin_individual[index_to_change]
        gate.mutate_angle(random.gauss(0, 0.2), self.setup_params.num_qubits)
        return lin_individual,

    def change_random_qubits(self, lin_individual):
        index_to_change = random.randrange(len(lin_individual))
        gate = lin_individual[index_to_change]
        gate.mutate_qubits(self.setup_params.num_qubits)
        return lin_individual,

    def mutate_linear_individual(self, lin_individual):
        mutation_types = [
            (self.base_gp_params.insert_gate_prob, self.insert_random_gate),
            (self.base_gp_params.insert_gate_sequence_prob, self.insert_random_gate_sequence),
            (self.base_gp_params.insert_gate_sequence_and_inverse_prob, self.insert_random_gate_sequence_and_inverse),
            (self.base_gp_params.delete_gate_sequence_prob, self.delete_random_gate_sequence),
            (self.base_gp_params.move_gate_sequence_prob, self.move_random_gate_sequence),
            (self.base_gp_params.swap_gates_prob, self.swap_random_gates),
            (self.base_gp_params.substitute_gate_prob, self.substitute_random_gate),
            (self.base_gp_params.change_angle_prob, self.change_random_angle),
            (self.base_gp_params.change_qubits_prob, self.change_random_qubits),
        ]
        mutation_choice = np.random.choice(list(func for _, func in mutation_types), p=list(prob for prob, _ in mutation_types))
        mutation_choice(lin_individual)

        return lin_individual, # Comma is needed because the function should return a tuple due to DEAP requirements

    def check_nodes(self, nodes_to_check, qubit, valid_nodes, direction):
        while len(nodes_to_check) > 0:
            node_to_check = nodes_to_check.popleft()
            connection_list = []
            if direction == CheckDirection.FORWARDS:
                connection_list = node_to_check.connections_in
            elif direction == CheckDirection.BACKWARDS:
                connection_list = node_to_check.connections_out
                if node_to_check in valid_nodes:
                    valid_nodes.remove(node_to_check)
            for connection in connection_list:
                if direction == CheckDirection.FORWARDS:
                    next_node = connection.input_node
                elif direction == CheckDirection.BACKWARDS:
                    next_node = connection.output_node
                if next_node not in nodes_to_check:
                    nodes_to_check.append(next_node)
                if connection.qubit == qubit and next_node in valid_nodes:
                    valid_nodes.remove(next_node)      
        return valid_nodes

    def get_valid_from_nodes(self, neat_individual, to_node, qubit):
        valid_nodes = [node for node in neat_individual.nodes if qubit in [connection.qubit for connection in node.connections_out]]
    
        nodes_to_check_infront = [connection.input_node for connection in to_node.connections_in]
        nodes_to_check_behind = [connection.output_node for connection in to_node.connections_out]
        valid_nodes = self.check_nodes(deque(nodes_to_check_infront), qubit, valid_nodes, CheckDirection.FORWARDS)
        valid_nodes = self.check_nodes(deque(nodes_to_check_behind), qubit, valid_nodes, CheckDirection.BACKWARDS)
        return valid_nodes

    def get_next_innovation_number(self):
        self.neat_state.global_innovation_num += 1
        return self.neat_state.global_innovation_num

    def get_new_innovation_numbers(self, innovation_numbers, innovation_type):
        new_innovation_numbers = []
        lookup_string = ""
        for innovation_number in innovation_numbers:
            lookup_string += str(innovation_number) + "_"
        lookup_string += str(innovation_type)
        if lookup_string in self.neat_state.known_innovations:
            new_innovation_numbers = self.neat_state.known_innovations[lookup_string]
        else:
            if innovation_type == Innovation_type.SPLIT:
                new_innovation_numbers = [self.get_next_innovation_number(), self.get_next_innovation_number(), self.get_next_innovation_number()] # new in connection, new out connection, new node
            elif innovation_type == Innovation_type.ADD:
                new_innovation_numbers = [self.get_next_innovation_number()] # new connection
            elif innovation_type == Innovation_type.ADDEXTRA:
                new_innovation_numbers = [self.get_next_innovation_number()] # adding a connection in a quantum circuit makes the old connection incompatible, it has to be replaced by a new one
            elif innovation_type == Innovation_type.INIT:
                new_innovation_numbers = [self.get_next_innovation_number()] # initialized node or connection

            self.neat_state.known_innovations[lookup_string] = new_innovation_numbers
        return new_innovation_numbers

    def split_connection(self, neat_individual):
        old_connection = random.choice(neat_individual.connections)
        neat_individual.remove_connection(old_connection) # in the original NEAT paper, the connection is disabled, but in quantum algorithm this connection is uncompatible with the new ones and therefore deleted.
        inno_nums = self.get_new_innovation_numbers([old_connection.inno_num], Innovation_type.SPLIT)
        new_node = Node(inno_nums[2], gate_type=random.choice(self.setup_params.gate_set), angle=random.uniform(0, 2 * math.pi))
        connection1 = Connection(old_connection.input_node, new_node, old_connection.qubit, random.choice([Qubit_type.CONTROL, Qubit_type.TARGET]), inno_nums[0])
        connection2 = Connection(new_node, old_connection.output_node, old_connection.qubit, random.choice([Qubit_type.CONTROL, Qubit_type.TARGET]), inno_nums[1])
        neat_individual.add_connection(connection1)
        neat_individual.add_connection(connection2)
        neat_individual.add_node(new_node)
        return neat_individual,

    def add_connection(self, neat_individual):
        nodes_tried = []
        while len(nodes_tried) < len(neat_individual.gate_nodes):
            to_node = random.choice([node for node in neat_individual.gate_nodes if node not in nodes_tried])
            if len(to_node.connections_in) == self.setup_params.num_qubits: # if the node is already connected to all qubits, skip
                nodes_tried.append(to_node)
                continue
            else:
                qubit = random.choice([qubit for qubit in range(self.setup_params.num_qubits) if qubit not in [connection.qubit for connection in to_node.connections_in]])
                valid_from_nodes = self.get_valid_from_nodes(neat_individual, to_node, qubit)
                if len(valid_from_nodes) != 0:
                    from_node = random.choice(valid_from_nodes)
                    connection_to_replace = None
                    for connection in from_node.connections_out:
                        if connection.qubit == qubit:
                            connection_to_replace = connection
                            break
                    neat_individual.remove_connection(connection_to_replace)
                    inno_num_in = self.get_new_innovation_numbers([from_node.inno_num, to_node.inno_num, qubit], Innovation_type.ADD)
                    inno_num_out = self.get_new_innovation_numbers([from_node.inno_num, to_node.inno_num, connection_to_replace.output_node.inno_num, qubit], Innovation_type.ADDEXTRA)
                    new_connection_in = Connection(from_node, to_node, qubit, random.choice([Qubit_type.CONTROL, Qubit_type.TARGET]), inno_num_in[0])
                    new_connection_out = Connection(to_node, connection_to_replace.output_node, qubit, connection_to_replace.type, inno_num_out[0])
                    neat_individual.add_connection(new_connection_in)
                    neat_individual.add_connection(new_connection_out)
        return neat_individual,

    def change_angles_neat(self, neat_individual):
        for node in neat_individual.gate_nodes:
            if random.random() <self.neat_params.change_ind_angle_prob:
                if random.random() < self.neat_params.assign_random_angle_prob:
                    node.angle = random.uniform(0, 2 * math.pi)
                else:
                    node.angle += random.uniform(-self.neat_params.max_angle_change, self.neat_params.max_angle_change)
                    if node.angle < 0:
                        node.angle += 2 * math.pi
                    elif node.angle > 2 * math.pi:
                        node.angle -= 2 * math.pi
        return neat_individual,

    def change_gates_neat(self, individual):
        for node in individual.gate_nodes:
            if random.random() < self.neat_params.change_ind_gate_prob:
                node.gate_type = random.choice(self.setup_params.gate_set)
        return individual,

    def change_conn_types_neat(self, individual):
        for connection in individual.connections:
            if random.random() < self.neat_params.change_ind_conn_type_prob:
                connection.type = random.choice([Qubit_type.CONTROL, Qubit_type.TARGET])
        return individual,

    def mutate_neat_individual(self, neat_individual):
        mutation_types = [
            (self.neat_params.split_conn_prob, self.split_connection),
            (self.neat_params.add_conn_prob, self.add_connection),
            (self.neat_params.change_angles_prob, self.change_angles_neat),
            (self.neat_params.change_gates_prob, self.change_gates_neat),
            (self.neat_params.change_conn_types_prob, self.change_conn_types_neat),
        ]
        mutation_choice = np.random.choice(list(func for _, func in mutation_types), p=list(prob for prob, _ in mutation_types))
        mutation_choice(neat_individual)
        return neat_individual,

    #------- Helper functions -------
    def normalize_value_0_to_1(self, value, min_value, max_value):
        return (value - min_value) / (max_value - min_value)

    def convert_statevectors_to_probabilities_dict(self, statevectors):
        results_probabilities = []
        for statevector in statevectors:    
            statevector_array = np.asarray(statevector) # Working on the statevector directly is deprecated
            probabilities_dict = {}
            for i, amplitude in enumerate(statevector_array):
                probability = abs(amplitude) ** 2
                bitstring = format(i, f'0{self.setup_params.num_qubits}b')  # Convert index to bitstring
                probabilities_dict[bitstring] = probability
            results_probabilities.append(probabilities_dict)
        return results_probabilities

    #------- Single-objective fitness functions -------
    def calculate_spector_1998_fitness(self, simulation_results, correct_states, individual):
        hits = len(simulation_results)
        correctness = 0
        if(isinstance(simulation_results[0],Statevector)):
            results_probabilities = self.convert_statevectors_to_probabilities_dict(simulation_results)
            simulation_results = results_probabilities
        for simulation_result, correct_states_per_case in zip(simulation_results, correct_states):
            accuracy = 0
            for state in correct_states_per_case:
                accuracy += simulation_result.get(state, 0)
            if (accuracy >= 0.52):
                hits -= 1
            else:
                correctness += 0.52 - accuracy
        if hits > 1:
            correctness / hits
        if hits == 0:
            fitness = len(individual) / 1000
        else:
            fitness = correctness + hits
        return fitness,

    def calculate_custom_spector_1998_fitness(self, simulation_results, correct_states, individual):
        hits = len(simulation_results)
        error = 0
        if(isinstance(simulation_results[0],Statevector)):
            results_probabilities = self.convert_statevectors_to_probabilities_dict(simulation_results)
            simulation_results = results_probabilities
        for simulation_result, correct_states_per_case in zip(simulation_results, correct_states):
            accuracy = 0
            for state in correct_states_per_case:
                accuracy += simulation_result.get(state, 0)
            error += 1 - accuracy
            if (accuracy >= 0.52):
                hits -= 1
        error /= len(simulation_results)
        fitness = error + hits
        if fitness < 0.00001: # account for floating point errors
            fitness = len(individual) / 1000
        return fitness,

    def calculate_error_fitness(self, simulation_statevectors, correct_statevectors, _ = None):
        error = 0
        if len(simulation_statevectors) != len(correct_statevectors):
            raise ValueError("The number of simulation results and correct statevectors must be equal.")
        for initial_bit_string, correct_statevector in correct_statevectors.items():
            simulation_result = simulation_statevectors[initial_bit_string]
            inner_product = np.abs(np.dot(simulation_result, correct_statevector.conjugate()))
            error += (1 - inner_product**2) / len(simulation_statevectors)
        return error,


    def calculate_error_fitness_probabilities(self, simulation_results, correct_states, _ = None):
        correctness = 0
        if(isinstance(simulation_results[0],Statevector)):
            results_probabilities = self.convert_statevectors_to_probabilities_dict(simulation_results)
            simulation_results = results_probabilities
        for simulation_result, correct_states_per_case in zip(simulation_results, correct_states):
            for state in correct_states_per_case:
                correctness += simulation_result.get(state, 0)
        correctness /= len(simulation_results) 
        return 1 - correctness,

    def calculate_phase_aware_error_fitness(self, simulation_statevectors, correct_statevectors, _ = None): 
        error = 0
        summed_dot_products = 0
        if len(simulation_statevectors) != len(correct_statevectors):
            raise ValueError("The number of simulation results and correct statevectors must be equal.")
        for initial_bit_string, correct_statevector in correct_statevectors.items():
            simulation_result = simulation_statevectors[initial_bit_string]
            dot_product = np.dot(simulation_result, correct_statevector.conjugate())
            summed_dot_products += dot_product

        absolute_summed_dot_products = np.abs(summed_dot_products)
        normalized_summed_dot_products = absolute_summed_dot_products / len(simulation_statevectors)
        error = 1 - normalized_summed_dot_products
        return error,

    #------- Multi-objective fitness functions -------
    def calculate_error_and_gate_count_fitness(self, simulation_results, correct_states, num_gates):
        error = round(self.toolbox.calculate_error(simulation_results, correct_states)[0], 5)
        gate_count_error = round(self.normalize_value_0_to_1(num_gates, 1, self.setup_params.max_ind_length), 5)
        return error, max(self.setup_params.min_gate_error, gate_count_error)

    #------- Evaluation functions -------
    def evaluate_deutsch_josza_1_input_qubits(self, individual):
        simulation_results = []
        for i in range(4):
            qc = self.toolbox.geno_to_pheno(individual, 2, oracle_case=i)
            simulation_results.append(run_circuit(qc, self.setup_params.simulator_type))

        if self.setup_params.use_neat:
            num_gates = sum(qc.count_ops().values())
        else:
            num_gates = len(individual)
        correct_states = [["01", "11"], ["01", "11"], ["10","00"], ["10","00"]]
        return self.toolbox.calculate_fitness(simulation_results, correct_states, num_gates)

    def evaluate_deutsch_josza_2_input_qubits(self, individual):
        simulation_results = []
        for i in range(8):
            qc = self.toolbox.geno_to_pheno(individual, 3, oracle_case=i)
            simulation_results.append(run_circuit(qc, self.setup_params.simulator_type))

        if self.setup_params.use_neat:
            num_gates = sum(qc.count_ops().values())
        else:
            num_gates = len(individual)
        correct_states = [["100","000"], ["100","000"], ["001","010","101","110","011", "111"], ["001","010","101","110","011", "111"], ["001","010","101","110","011", "111"], ["001","010","101","110","011", "111"], ["001","010","101","110","011", "111"], ["001","010","101","110","011", "111"], ]
        return self.toolbox.calculate_fitness(simulation_results, correct_states, num_gates)

    def evaluate_quantum_fourier_3_qubits(self, individual):
        if self.setup_params.statevector_3_qubit_qft is None:
            self.setup_params.statevector_3_qubit_qft = generate_3_qubit_fourier_transform_statevectors()

        simulation_results = {}
        for i in range (8):
            initial_state_binary = format(i, '0' + str(3) + 'b')
            qc = self.toolbox.geno_to_pheno(individual, 3, initial_state_binary)
            simulation_results[initial_state_binary] = (run_circuit(qc, self.setup_params.simulator_type))
        if self.setup_params.use_neat:
            num_gates = sum(qc.count_ops().values())
        else:
            num_gates = len(individual)
        return self.toolbox.calculate_fitness(simulation_results, self.setup_params.statevector_3_qubit_qft, num_gates)

    def evaluate_quantum_full_adder_3_qubits(self, individual):
        if self.setup_params.statevector_3_qubit_quantum_full_adder is None:
            self.setup_params.statevector_3_qubit_quantum_full_adder = generate_3_qubit_quantum_full_adder_statevectors()

        simulation_results = {}
        for i in range (8):
            initial_state_binary = format(i, '0' + str(3) + 'b') + "0"
            qc = self.toolbox.geno_to_pheno(individual, 4, initial_state_binary)
            simulation_results[initial_state_binary] = (run_circuit(qc, self.setup_params.simulator_type))
        if self.setup_params.use_neat:
            num_gates = sum(qc.count_ops().values())
        else:
            num_gates = len(individual)
        return self.toolbox.calculate_fitness(simulation_results, self.setup_params.statevector_3_qubit_quantum_full_adder, num_gates)

    # ------- NEAT specific functions -------
    def calculate_adjusted_fitness(self, individual, species_length):
        adjusted_fitness = []
        if individual.is_elitist:
            for fitness_value in individual.base_fitness.values:
                adjusted_fitness.append(fitness_value * self.neat_params.adj_fitness_reg_term)
        else: 
            for fitness_value in individual.base_fitness.values:
                adjusted_fitness.append(fitness_value * (species_length + self.neat_params.adj_fitness_reg_term))
        return adjusted_fitness 

    def calculate_dissimilarity_score(self, ind1, ind2):
        if len(ind1.gate_nodes) == 0 and len(ind2.gate_nodes) == 0:
            return 0 # Both individuals are empty and therefore identical
        matching_node_genes = 0
        matching_connection_genes = 0
        matching_gate_types = 0
        angle_difference = 0
        for node1 in ind1.gate_nodes: # end_nodes and oracle nodes always match per definition and are excluded
            for node2 in ind2.gate_nodes:
                if node1.inno_num == node2.inno_num:
                    matching_node_genes += 1
                    if node1.gate_type == node2.gate_type:
                        matching_gate_types += 1
                    angle_difference += abs(node1.angle - node2.angle)
                    break
        for connection1 in ind1.connections:
            for connection2 in ind2.connections:
                if connection1.inno_num == connection2.inno_num:
                    matching_connection_genes += 1
                    break
        
        non_matching_node_genes = len(ind1.gate_nodes) + len(ind2.gate_nodes) - 2 * matching_node_genes
        non_matching_gate_types = matching_node_genes - matching_gate_types
        non_matching_connection_genes = len(ind1.connections) + len(ind2.connections) - 2 * matching_connection_genes

        gate_node_dissimilarity = self.neat_params.gate_node_coeff * (non_matching_node_genes / (len(ind1.gate_nodes) + len(ind2.gate_nodes))) 
        connection_dissimilarity = self.neat_params.conn_node_coeff * (non_matching_connection_genes / (len(ind1.connections) + len(ind2.connections)))
        angle_dissimilarity = self.neat_params.angle_coeff * (angle_difference / matching_node_genes) if matching_node_genes > 0 else 0
        gate_type_dissimilarity = self.neat_params.gate_type_coeff * (non_matching_gate_types / matching_node_genes) if matching_node_genes > 0 else 0
        return gate_node_dissimilarity + connection_dissimilarity + angle_dissimilarity + gate_type_dissimilarity
    
    def neat_speciate_initial(self, population):
        representatives = {}
        ind_by_species = {}
        for ind in population:
            species_match = False
            for id, representative in representatives.items():
                if self.calculate_dissimilarity_score(ind, representative) < self.neat_params.species_threshold:
                    species_match = True
                    ind_by_species[id].append(ind)
                    ind.species_id = id
                    break
            if not species_match:
                ind.species_id = self.neat_state.next_species_id
                ind_by_species[ind.species_id] = [ind]
                representatives[ind.species_id] = ind
                self.neat_state.next_species_id += 1
        return ind_by_species

    def neat_speciate(self, new_individuals, ind_by_species):
        representatives = {}
        for id, species_list in ind_by_species.items():
            representatives[id] = random.choice(species_list)
        for ind in new_individuals:
            species_match = False
            for id, representative in representatives.items():
                if self.calculate_dissimilarity_score(ind, representative) < self.neat_params.species_threshold:
                    species_match = True
                    ind_by_species[id].append(ind)
                    ind.species_id = id
                    break
            if not species_match:
                ind.species_id = self.neat_state.next_species_id
                self.neat_state.next_species_id += 1
                ind_by_species[ind.species_id] = [ind]
        return ind_by_species

    def assemble_parents_and_assign_offspring(self, ind_by_species, avg_base_fitness):
        parents_by_species = {species_id: [self.toolbox.clone(ind) for ind in species_list] for species_id, species_list in ind_by_species.items()}
        parents_list = []
        for species_id, species_list in parents_by_species.items():
            num_ind_to_keep = round(len(species_list) * self.neat_params.survival_perc)
            if num_ind_to_keep == 0:
                num_ind_to_keep = 1
            parents_by_species[species_id] = self.toolbox.select(species_list, num_ind_to_keep)
            for ind in species_list:
                aggregate_fitness = 0
                for i in range(self.setup_params.num_objectives):
                    aggregate_fitness_add = avg_base_fitness[i] / (ind.base_fitness.values[i] + 0.0001) # Add a small value to avoid division by zero
                    if aggregate_fitness_add > self.hyper_params.pop_size * self.neat_params.max_offspring_perc:
                        aggregate_fitness_add = self.hyper_params.pop_size * self.neat_params.max_offspring_perc
                    aggregate_fitness += aggregate_fitness_add
                ind.offspring_count = aggregate_fitness / self.setup_params.num_objectives
            parents_list.extend(parents_by_species[species_id])
        parents_list = self.toolbox.select(parents_list, len(parents_list)) # sorts the list 
        return parents_by_species, parents_list

    def create_new_generation_nsga2(self, population, new_individuals):
        population = tools.selNSGA2(population + new_individuals, self.hyper_params.pop_size)
        return population

    def create_new_generation_paper(self, population, new_individuals, elite_individuals):
        new_population = []
        elites_in_new_ind = 0
        new_population.extend(elite_individuals)
        #remove elite individuals from new individuals and population
        for elite in elite_individuals:
            if elite in new_individuals:
                new_individuals.remove(elite)
                elites_in_new_ind += 1
            if elite in population:
                population.remove(elite)
        num_to_remove = math.ceil(self.hyper_params.pop_size * self.neat_params.replace_p_worst)
        num_ind_to_replace = num_to_remove - elites_in_new_ind
        if num_ind_to_replace > len(new_individuals):
            print(f"Warning: The number of individuals({len(new_individuals)}) is smaller than the number of individuals to replace({num_ind_to_replace}).")
            num_ind_to_replace = len(new_individuals)
        elites_in_pop = len(elite_individuals) - elites_in_new_ind
        population = tools.selNSGA2(population, self.hyper_params.pop_size - (num_to_remove + elites_in_pop))
        new_individuals_to_keep = tools.selNSGA2(new_individuals, num_ind_to_replace)
        new_population.extend(population)
        new_population.extend(new_individuals_to_keep)
        return new_population
    
    def sync_dict_to_population(self, ind_by_species, population):
        for species_list in ind_by_species.copy().values(): # Copy is needed to avoid changing the dictionary during iteration
            for ind in species_list.copy():
                if ind.id not in [ind_p.id for ind_p in population]:
                    ind_by_species[ind.species_id].remove(ind)
                    if len(ind_by_species[ind.species_id]) == 0:
                        del ind_by_species[ind.species_id]
        return ind_by_species

    def save_individual(self, individual, filepath):
        best_ind_circuit = self.toolbox.geno_to_pheno(individual, self.setup_params.num_qubits)
        best_ind_circuit.draw(output='mpl', filename=filepath)

    def calculate_2d_hypervolume(self, pareto_front_fitnesses, reference_point):
        pareto_front_fitnesses = sorted(pareto_front_fitnesses, key=lambda x: (x[0], -x[1]))

        hypervolume = 0.0

        for i, (x, y) in enumerate(pareto_front_fitnesses):
            # For the last point, the rectangle extends to the reference point
            if i == 0:
                prev_y = reference_point[1]
            else:
                prev_y = pareto_front_fitnesses[i - 1][1]

            width = reference_point[0] - x
            height = prev_y - y

            # Ensure both width and height are positive
            if width > 0 and height > 0:
                hypervolume += width * height

        return hypervolume
    
    def calc_unique_pareto_front(self, pareto_front):
        unique_pareto_front = {}
        for ind in pareto_front:
            fitness_tuple = ind.fitness.values
            if fitness_tuple not in unique_pareto_front:
                unique_pareto_front[fitness_tuple] = ind
        return list(unique_pareto_front.values())

    def calc_statistics(self, population, unique_pareto_front, generation):
        if self.setup_params.use_neat:
            pareto_front_fitnesses = [ind.base_fitness.values for ind in unique_pareto_front]
        else:
            pareto_front_fitnesses = [ind.fitness.values for ind in unique_pareto_front]
        reference_point = np.array([1.0001, 1.0001])
        hypervolume = self.calculate_2d_hypervolume(pareto_front_fitnesses, reference_point)
        self.stats.hypervolumes.append(hypervolume)
        if hypervolume > self.stats.best_hypervolume:
            self.stats.best_hypervolume = hypervolume
            self.stats.best_hypervolume_gen = generation 
            self.stats.best_pareto_front_ind = unique_pareto_front
        for i in range(self.setup_params.num_objectives):
            if self.setup_params.use_neat:
                fits = [ind.base_fitness.values[i] for ind in population]
            else:
                fits = [ind.fitness.values[i] for ind in population]
            min_fit = min(fits)
            if min_fit < self.stats.best_fitness[i]:
                self.stats.best_fitness[i] = min_fit
                self.stats.best_fitness_gen[i] = generation 
            max_fit = max(fits)
            avg_fit = np.mean(fits)
            self.stats.min_fitnesses[i].append(min_fit)
            self.stats.max_fitnesses[i].append(max_fit)
            self.stats.avg_fitnesses[i].append(avg_fit)
            if self.setup_params.verbose:
                print(f"  Min error of objective {i + 1}: {min_fit}")
                print(f"  Max error of objective {i + 1}: {max_fit}")
                print(f"  Avg error of objective {i + 1}: {avg_fit}")
        if self.setup_params.verbose:
            print(f"  Hypervolume: {hypervolume}")

    def write_statistics_to_csv(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Generation", "Hypervolume", "Min error", "Max error", "Avg error", "Min gate fit", "Max gate fit", "Avg gate fit"])

            for i in range(len(self.stats.hypervolumes)):
                writer.writerow([i, self.stats.hypervolumes[i], self.stats.min_fitnesses[0][i], self.stats.max_fitnesses[0][i], self.stats.avg_fitnesses[0][i], self.stats.min_fitnesses[1][i], self.stats.max_fitnesses[1][i], self.stats.avg_fitnesses[1][i]])

            writer.writerow(["Best Pareto front"])
            writer.writerow(["Generation", "Hypervolume", "error", "gate fit"])
            if self.setup_params.use_neat:
                for i in range(len(self.stats.best_pareto_front_ind)):
                    writer.writerow([self.stats.best_hypervolume_gen, self.stats.best_hypervolume, self.stats.best_pareto_front_ind[i].base_fitness.values[0], self.stats.best_pareto_front_ind[i].base_fitness.values[1]])
            else:
                for i in range(len(self.stats.best_pareto_front_ind)):
                    writer.writerow([self.stats.best_hypervolume_gen, self.stats.best_hypervolume, self.stats.best_pareto_front_ind[i].fitness.values[0], self.stats.best_pareto_front_ind[i].fitness.values[1]])
            
            writer.writerow(["Best fitness"])
            writer.writerow(["Objective", "Generation", "Error"])
            for i in range(len(self.stats.best_fitness)):
                writer.writerow([i, self.stats.best_fitness_gen[i], self.stats.best_fitness[i]])

            writer.writerow(["Avg simulation runtime", "Avg generation runtime"])
            writer.writerow([np.mean(self.stats.simulation_runtimes), np.mean(self.stats.generation_runtimes)])

    def swap_fitness_values(self, individuals):
        for ind in individuals:
            base_fitness = ind.base_fitness
            ind.base_fitness = ind.fitness
            ind.fitness = base_fitness

    def run_base_gp(self):
        pop = self.toolbox.population(n=self.hyper_params.pop_size)
        next_ind_id = 0
        g = 1

        fitnesses = list(map(self.toolbox.evaluate, pop))
        start_time_gp = time.time()

        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
            ind.id = next_ind_id
            next_ind_id += 1

        while time.time() - start_time_gp < self.setup_params.max_time * 60:
            if self.setup_params.verbose:
                print(f"-- Generation {g} --")
            g += 1
            start_time_gen = time.time()

            # Assign Pareto ranks and crowding distances
            pop = tools.selNSGA2(pop, len(pop))

            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if child1.id == child2.id:
                    print("Error: Two individuals have the same ID.")
                child1.id = next_ind_id
                next_ind_id += 1
                child2.id = next_ind_id
                next_ind_id += 1
                if random.random() < self.hyper_params.cross_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
                if random.random() < self.hyper_params.mut_prob:
                    self.toolbox.mutate(child1)
                    del child1.fitness.values
                if random.random() < self.hyper_params.mut_prob:
                    self.toolbox.mutate(child2)
                    del child2.fitness.values
        
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            start_time_sim = time.time()
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit 

            time_taken_sim = round(time.time() - start_time_sim, 2)
            self.stats.simulation_runtimes.append(time_taken_sim)

            if self.setup_params.verbose:
                print(f"  Time taken for simulation: {time_taken_sim}s")

            combined_pop = pop + offspring
            pop = tools.selNSGA2(combined_pop, self.hyper_params.pop_size)

            pareto_front = tools.sortNondominated(pop, len(pop), True)[0]
            unique_pareto_front = self.calc_unique_pareto_front(pareto_front)

            self.calc_statistics(pop, unique_pareto_front, g)

            time_taken_gen = round(time.time() - start_time_gen, 2)
            self.stats.generation_runtimes.append(time_taken_gen)
            if self.setup_params.verbose:
                print(f"  Time taken for generation: {time_taken_gen}s")
        self.write_statistics_to_csv(self.folder_path + f"stats_{self.setup_params.run_id}.csv")
        for ind in self.stats.best_pareto_front_ind:
            self.save_individual(ind, self.run_folder_path + f"pareto_front_{ind.id}.png")

    def get_next_ind_id(self):
        self.neat_state.next_ind_id += 1
        return self.neat_state.next_ind_id

    def run_neat_gp(self):
        start_time_gp = time.time()
        g = 1

        initial_pop = self.toolbox.population(n=self.hyper_params.pop_size)

        base_fitnesses = list(map(self.toolbox.evaluate, initial_pop))
    
        for ind, fit in zip(initial_pop, base_fitnesses):
            ind.base_fitness.values = fit
            ind.id = self.get_next_ind_id()

        ind_by_species = self.neat_speciate_initial(initial_pop)

        if self.setup_params.verbose:
            print(f"  Number of initial species: {len(ind_by_species)}")

        while time.time() - start_time_gp < self.setup_params.max_time * 60:
            if self.setup_params.verbose:
                print(f"-- Generation {g} --")
            start_time_gen = time.time()
            g += 1

            pop = []
            avg_base_fitness = (0,0)
            start_time_evaluate_adj = time.time()
            for species_list in ind_by_species.values():
                self.swap_fitness_values(species_list)
                elites = tools.selNSGA2(species_list, self.hyper_params.num_elites)
                self.swap_fitness_values(species_list)
                for ind in elites:
                    ind.is_elitist = True
                for ind in species_list:
                    ind.fitness.values = self.toolbox.evaluate_adj(ind, len(species_list))
                    ind.is_elitist = False
                    avg_base_fitness = tuple(x + y for x, y in zip(avg_base_fitness, ind.base_fitness.values))
                pop.extend(species_list)
            avg_base_fitness = tuple(x / len(pop) for x in avg_base_fitness)
            time_taken_evaluate_adj = round(time.time() - start_time_evaluate_adj, 2)

            new_individuals = []
            start_time_assemble_parents = time.time()
            potential_parents_by_species, potential_parents_list = self.assemble_parents_and_assign_offspring(ind_by_species, avg_base_fitness)
            time_taken_assemble_parents = round(time.time() - start_time_assemble_parents, 2)
            
            if self.setup_params.verbose:
                print(f"  Num potential offsprings: {sum([ind.offspring_count for ind in potential_parents_list])}")

            start_time_evolution_ops = time.time()
            while len(potential_parents_list) > 0 and len(new_individuals) < math.ceil(self.hyper_params.pop_size * self.neat_params.replace_p_worst):
                if random.random() < 0.5:
                    parent_1 = potential_parents_list[0]
                else:
                    parent_1 = random.choice(potential_parents_list)
                if len(potential_parents_list) > 1 and random.random() < self.hyper_params.cross_prob:
                    parent_2 = potential_parents_by_species[parent_1.species_id][0]
                    if parent_1 == parent_2:
                        if len(potential_parents_by_species[parent_1.species_id]) > 1:
                            parent_2 = potential_parents_by_species[parent_1.species_id][1]
                        else:
                        #pick any other random individual
                            while parent_2 == parent_1:
                                parent_2 = random.choice(potential_parents_list)
                    parent_2.offspring_count -= 0.5
                    child = self.toolbox.mate(parent_1, parent_2)[0]
                    child.id = self.get_next_ind_id()
                    if parent_2.offspring_count <= 0:
                        potential_parents_by_species[parent_2.species_id].remove(parent_2)
                        potential_parents_list.remove(parent_2)
                    parent_1.offspring_count -= 0.5
                else:
                    child = self.toolbox.clone(parent_1)
                    child.id = self.get_next_ind_id()
                    parent_1.offspring_count -= 1
                if parent_1.offspring_count <= 0:
                    potential_parents_by_species[parent_1.species_id].remove(parent_1)
                    potential_parents_list.remove(parent_1)
                if random.random() < self.hyper_params.mut_prob:    
                    self.toolbox.mutate(child)
                new_individuals.append(child)
            time_taken_evolution_ops = round(time.time() - start_time_evolution_ops, 2)

            start_time_simulation = time.time()
            base_fitnesses = list(map(self.toolbox.evaluate, new_individuals))
            for ind, fit in zip(new_individuals, base_fitnesses):
                ind.base_fitness.values = fit
            
            time_taken_sim = round(time.time() - start_time_simulation, 2)
            self.stats.simulation_runtimes.append(time_taken_sim)

            start_time_speciation = time.time()
            ind_by_species = self.neat_speciate(new_individuals, ind_by_species)
            time_taken_speciation = round(time.time() - start_time_speciation, 2)

            start_time_evlauate_adj_children = time.time()
            for ind in new_individuals:
                ind.fitness.values = self.toolbox.evaluate_adj(ind, len(ind_by_species[ind.species_id]))
            time_taken_evaluate_adj_children = round(time.time() - start_time_evlauate_adj_children, 2)

            start_time_unique_pareto_front = time.time()
            start_time_sort_non_dom = time.time()
            self.swap_fitness_values(pop + new_individuals)
            pareto_front = tools.sortNondominated(pop + new_individuals, len(pop + new_individuals), first_front_only=True)[0]
            self.swap_fitness_values(pop + new_individuals)
            time_taken_sort_non_dom = round(time.time() - start_time_sort_non_dom, 2)
            unique_pareto_front = self.calc_unique_pareto_front(pareto_front)
            time_taken_unique_pareto_front = round(time.time() - start_time_unique_pareto_front, 2)

            start_time_new_gen = time.time()
            pop = self.create_new_generation_paper(pop, new_individuals, unique_pareto_front)
            ind_by_species = self.sync_dict_to_population(ind_by_species, pop)
            time_taken_new_gen = round(time.time() - start_time_new_gen, 2)

            species_with_1_ind = [species for species in ind_by_species.values() if len(species) == 1]

            if self.setup_params.verbose:
               print(f"  Species with 1 individual: {len(species_with_1_ind)}")

            if self.setup_params.verbose:
                print(f"  Number of species: {len(ind_by_species)}")

            if len(ind_by_species) <= 15:
                self.neat_params.species_threshold *= 0.93
            elif len(ind_by_species) >= 25:
                self.neat_params.species_threshold *= 1.07

            start_time_stats = time.time()
            self.calc_statistics(pop, unique_pareto_front, g)
            time_taken_stats = round(time.time() - start_time_stats, 2)

            time_taken_gen = round(time.time() - start_time_gen, 2)
            self.stats.generation_runtimes.append(time_taken_gen)
            if self.setup_params.verbose:
                print(f"  Time taken for:")
                print(f"    generation: {time_taken_gen}s")
                print(f"    evaluate_adj: {time_taken_evaluate_adj}s")
                print(f"    assemble_parents: {time_taken_assemble_parents}s")
                print(f"    evolution operators: {time_taken_evolution_ops}s")
                print(f"    simulation: {time_taken_sim}s")
                print(f"    speciation: {time_taken_speciation}s")
                print(f"    evaluate_adj_children: {time_taken_evaluate_adj_children}s")
                print(f"    unique_pareto_front: {time_taken_unique_pareto_front}s")
                print(f"    sort_non_dom: {time_taken_sort_non_dom}s")
                print(f"    new generation: {time_taken_new_gen}s")
                print(f"    statistics: {time_taken_stats}s")

        self.write_statistics_to_csv(self.folder_path + f"neat_stats_{self.setup_params.run_id}.csv")
        for ind in self.stats.best_pareto_front_ind:
            self.save_individual(ind,self.run_folder_path + f"pareto_front_{ind.id}.png")

    def run(self):
        if self.setup_params.use_neat:
            self.run_neat_gp()
        else:
            #deutsch_josza_3_individual = creator.Individual()
            #deutsch_josza_3_individual.append(GateGene(GateType.H, target_qubit=0))
            #deutsch_josza_3_individual.append(GateGene(GateType.H, target_qubit=1))
            #deutsch_josza_3_individual.append(GateGene(GateType.X, target_qubit=2))
            #deutsch_josza_3_individual.append(GateGene(GateType.H, target_qubit=2))
            #deutsch_josza_3_individual.append(GateGene(GateType.ORACLE))
            #deutsch_josza_3_individual.append(GateGene(GateType.H, target_qubit=0))
            #deutsch_josza_3_individual.append(GateGene(GateType.H, target_qubit=1))
            #print(f"Known deutsch_josza solution error: {self.evaluate_deutsch_josza_2_input_qubits(deutsch_josza_3_individual)}")
            #quantum_fourier_3_individual = creator.Individual()
            #quantum_fourier_3_individual.append(GateGene(GateType.H, target_qubit=2))
            #quantum_fourier_3_individual.append(GateGene(GateType.MCP, target_qubit=2, control_qubits=[1], angle=np.pi/2))
            #quantum_fourier_3_individual.append(GateGene(GateType.H, target_qubit=1))
            #quantum_fourier_3_individual.append(GateGene(GateType.MCP, target_qubit=2, control_qubits=[0], angle=np.pi/4))
            #quantum_fourier_3_individual.append(GateGene(GateType.MCP, target_qubit=1, control_qubits=[0], angle=np.pi/2))
            #quantum_fourier_3_individual.append(GateGene(GateType.H, target_qubit=0))
            #quantum_fourier_3_individual.append(GateGene(GateType.SWAP, target_qubit=0, control_qubits=[2]))
            #print(f"Known quantum fourier solution error: {self.evaluate_quantum_fourier_3_qubits(quantum_fourier_3_individual)}")
            #quantum_full_adder_ind = creator.Individual()
            #quantum_full_adder_ind.append(GateGene(GateType.MCX, target_qubit=3, control_qubits=[0,1]))
            #quantum_full_adder_ind.append(GateGene(GateType.MCX, target_qubit=1, control_qubits=[0]))
            #quantum_full_adder_ind.append(GateGene(GateType.MCX, target_qubit=3, control_qubits=[1,2]))
            #quantum_full_adder_ind.append(GateGene(GateType.MCX, target_qubit=2, control_qubits=[1]))
            #quantum_full_adder_ind.append(GateGene(GateType.MCX, target_qubit=1, control_qubits=[0]))
            #print(f"Known quantum full adder solution error: {self.evaluate_quantum_full_adder_3_qubits(quantum_full_adder_ind)}")
            self.run_base_gp()
#quantum_fourier_3_individual = [["h", 2], ["cp", np.pi/2, 1, 2], ["h", 1], ["cp", np.pi/4, 0, 2], ["cp",np.pi/2, 0, 1], ["h", 0], ["swap", 0, 2]]
#deutsch_josza_3_individual = [('h', 0), ('h', 1), ('x', 2), ('h', 2), ['oracle'], ('h', 0), ('h', 1)]
#print(f"Known solution error: {self.toolbox.evaluate(deutsch_josza_3_individual)}")