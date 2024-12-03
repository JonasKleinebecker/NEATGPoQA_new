import time
from evolutionary_algorithm import ConfigurableGP
import itertools
import multiprocessing as mp
import random
import os
import shutil
import yaml

def run_gp(config_file):
    try:
        start_time = time.time()
        print(f"Started running {config_file} at {time.strftime('%H:%M:%S', time.localtime(start_time))}")
        gp = ConfigurableGP(config_file)
        gp.run()
        end_time = time.time()
        print(f"Finished running {config_file} at {time.strftime('%H:%M:%S', time.localtime(end_time))}")
    except Exception as e:
        print(f"Error processing {config_file}: {e}")


def organize_config_files(config_dir):
    """
    Organizes config files into folders based on their computer_id.
    Includes a delay to prevent permission errors.

    Args:
        config_dir (str): The directory containing the config files.
    """
    for filename in os.listdir(config_dir):
        if filename.endswith(".yaml"):
            filepath = os.path.join(config_dir, filename)
            
            try:
                # Read the file and extract computer_id
                with open(filepath, 'r') as f:
                    data = yaml.safe_load(f)

                computer_id = data['setup_params']['computer_id']

                # Create the destination folder
                folder_name = f"computer_{computer_id}"
                folder_path = os.path.join(config_dir, folder_name)
                os.makedirs(folder_path, exist_ok=True)

                # Introduce a small delay to ensure no lingering file locks
                time.sleep(0.2)

                # Move the file
                shutil.move(filepath, folder_path)

            except yaml.YAMLError as e:
                print(f"Error reading YAML file {filename}: {e}")
            except KeyError as e:
                print(f"Error: 'computer_id' not found in {filename}")
            except PermissionError as e:
                print(f"Permission error with file {filename}: {e}")

def generate_config_files(param_ranges, output_dir, base_config_file, num_computers, runs, prefix):
    """
    Generates config files for all combinations of given parameter ranges, 
    randomly distributing them across multiple computers with an even distribution,
    and creating multiple runs for each configuration. Uses spaces for indentation.

    Args:
        param_ranges (dict): A dictionary where keys are parameter names 
                              and values are lists of parameter values.
        output_dir (str): The directory to save the generated config files.
        base_config_file (str): The path to the base config file.
        num_computers (int): The number of computers to distribute the configs to.
        runs (int): The number of runs to create for each configuration.
    """

    all_combinations = list(itertools.product(*param_ranges.values()))
    random.shuffle(all_combinations)

    num_configs = len(all_combinations)
    configs_per_computer = num_configs // num_computers

    computer_ids = list(range(1, num_computers + 1)) * configs_per_computer
    random.shuffle(computer_ids)

    with open(base_config_file, 'r') as f:
        base_config_lines = f.readlines()

    config_count = 3
    computer_id_index = 0

    for combination in all_combinations:
        for run_id in range(1, runs + 1):
            computer_id = computer_ids[computer_id_index]
            computer_id_index = (computer_id_index + 1) % len(computer_ids)

            config_filename = f"{output_dir}/{prefix}_config_{config_count}.yaml"
            with open(config_filename, 'w') as f:
                for line in base_config_lines:
                    if "setup_params:" in line:
                        f.write(line)
                        f.write(f"  computer_id: {computer_id}\n")  # 2 spaces for indentation
                        f.write(f"  run_id: {run_id}\n")  # 2 spaces for indentation
                    else:
                        for j, param_name in enumerate(param_ranges.keys()):
                            if param_name in line:
                                if ',' in line:
                                    current_value = line.split(',')[1].strip()
                                    if current_value.replace('.', '', 1).isdigit():
                                        line = line.replace(current_value, str(combination[j]) + '\n')
                                    else:
                                        line = f"{param_name},{combination[j]}\n"
                                else:
                                    line = f"  {param_name}: {combination[j]}\n"  # 2 spaces for indentation
                        f.write(line)
            config_count += 1

if __name__ == "__main__":
    param_ranges = {
        'pop_size': [500, 1000, 2000, 5000],
        'cross_prob': [0.8, 1.0],
        'mut_prob': [0.55, 0.7, 0.85],
        'test_problem': ['deutsch_josza', 'qft', 'full_adder']
    }

    output_dir = "config_files/generated"
    base_config_file = "config_files/neat_test.yaml"
    num_computers = 20
    runs = 10  
    #generate_config_files(param_ranges, output_dir, base_config_file, num_computers, runs, "neat")

    param_ranges = {
        'pop_size': [500, 1000, 2000, 5000],
        'cross_prob': [0.8, 1.0],
        'mut_prob': [0.15, 0.25, 0.35],
        'test_problem': ['deutsch_josza', 'qft', 'full_adder']
    }
    base_config_file = "config_files/base_test.yaml"
    #generate_config_files(param_ranges, output_dir, base_config_file, num_computers, runs, "base")
    #organize_config_files(output_dir)    

    computer_id = 1

    config_files = [f"config_files/generated/computer_{computer_id}/{f}" for f in os.listdir(f"config_files/generated/computer_{computer_id}")]
    #config_files = ["config_files/generated/computer_1/base_config_9.yaml"]

    #run_gp(config_files[0])

    with mp.Pool(processes=6) as pool:
        pool.map(run_gp, config_files)
