import math

from gate_type import GateType


class SetupParams:
    def __init__(self, fitness_weights, use_neat, gate_set, max_ind_length, simulator_type, min_gate_error, verbose, computer_id, run_id, test_problem, max_time):
        self.fitness_weights = fitness_weights
        self.use_neat = use_neat
        self.use_multi_objective = len(fitness_weights) > 1
        self.gate_set = gate_set
        self.num_qubits = None
        self.max_ind_length = max_ind_length # In the Neat variation this is not enforced but ony used as the max value for normalization
        self.statevector_3_qubit_qft = None
        self.statevector_3_qubit_quantum_full_adder = None
        self.statevector_000 = None
        self.simulator_type = simulator_type
        self.min_gate_error = min_gate_error
        self.verbose = verbose
        self.computer_id = computer_id
        self.run_id = run_id
        self.max_time = max_time
        self.test_problem = test_problem
        if self.test_problem == "deutsch_josza":
            self.use_oracle = True
        else:
            self.use_oracle = False
        self.num_objectives = len(fitness_weights)
        if self.use_oracle and not self.use_neat: #In NEAT version the oracle is fixed and should not be in the gate set
            self.gate_set.append(GateType.ORACLE)

class NeatParams:
    def __init__(self, replace_p_worst, survival_perc, min_init_mutations, max_init_mutations, conn_node_coeff, angle_coeff, gate_node_coeff, gate_type_coeff, species_threshold, adj_fitness_reg_term, add_conn_prob, split_conn_prob, change_angles_prob, assign_random_angle_prob, max_angle_change, change_gates_prob, change_conn_types_prob, max_offspring_perc):
        self.replace_p_worst = replace_p_worst
        self.survival_perc = survival_perc
        self.min_init_mutations = min_init_mutations
        self.max_init_mutations = max_init_mutations
        self.conn_node_coeff = conn_node_coeff
        self.angle_coeff = angle_coeff
        self.gate_node_coeff = gate_node_coeff
        self.gate_type_coeff = gate_type_coeff
        self.species_threshold = species_threshold
        self.adj_fitness_reg_term = adj_fitness_reg_term
        self.max_offspring_perc = max_offspring_perc

        #Mutation type probabilities
        self.add_conn_prob = add_conn_prob
        self.split_conn_prob = split_conn_prob
        self.change_angles_prob = change_angles_prob
        self.change_gates_prob = change_gates_prob
        self.change_conn_types_prob = change_conn_types_prob

        #Mutation action probabilities
        self.assign_random_angle_prob = assign_random_angle_prob
        self.max_angle_change = max_angle_change

        if not math.isclose(change_angles_prob + change_gates_prob + change_conn_types_prob + split_conn_prob + add_conn_prob, 1, rel_tol=1e-9):
            raise ValueError("The sum of the NEAT mutation type probabilities must be 1")

class BaseGPParams:
    def __init__(self, insert_gate_prob, insert_gate_sequence_prob, insert_gate_sequence_and_inverse_prob, delete_gate_sequence_prob, move_gate_sequence_prob, swap_gates_prob, substitute_gate_prob, change_angle_prob, change_qubits_prob, min_ind_length):
        self.min_ind_length = min_ind_length

        self.insert_gate_prob = insert_gate_prob
        self.insert_gate_sequence_prob = insert_gate_sequence_prob
        self.insert_gate_sequence_and_inverse_prob = insert_gate_sequence_and_inverse_prob
        self.delete_gate_sequence_prob = delete_gate_sequence_prob
        self.move_gate_sequence_prob = move_gate_sequence_prob
        self.swap_gates_prob = swap_gates_prob
        self.substitute_gate_prob = substitute_gate_prob
        self.change_angle_prob = change_angle_prob
        self.change_qubits_prob = change_qubits_prob

        if not math.isclose(insert_gate_sequence_prob + insert_gate_sequence_and_inverse_prob + delete_gate_sequence_prob + move_gate_sequence_prob + swap_gates_prob + substitute_gate_prob != 1, 1, rel_tol=1e-9):
            raise ValueError("The sum of the base GP mutation probabilities must be 1")
    
class HyperParams:
    def __init__(self,cross_prob, mut_prob, pop_size, num_elites):
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob
        self.pop_size = pop_size
        self.num_elites = num_elites

class NeatState:
    def __init__(self):
        self.known_innovations = {}
        self.global_innovation_num = 0
        self.next_species_id = 0
        self.next_ind_id = 0

class Statistics:
    def __init__(self):
        self.best_fitness = [1,1]
        self.best_fitness_gen = [0,0]
        self.max_fitnesses = [[],[]]
        self.avg_fitnesses = [[],[]]
        self.min_fitnesses = [[],[]]
        self.hypervolumes = []
        self.best_hypervolume = 0
        self.best_hypervolume_gen = None
        self.best_pareto_front_ind = []
        self.generation_runtimes = []
        self.simulation_runtimes = []

        