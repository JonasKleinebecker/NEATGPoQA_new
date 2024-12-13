import shutil
import pandas as pd
import os
from collections import defaultdict
import matplotlib.pyplot as plt

def load_experiment_data_from_folders(base_folder):
    """
    Load experiment data from a folder structure into a structured dictionary.

    Args:
        base_folder (str): Path to the base folder containing parameter subfolders.

    Returns:
        dict: Nested dictionary where keys are experiment parameters and values are lists of DataFrames.
    """
    experiment_data = defaultdict(lambda: defaultdict(list))

    computers_parsed = 0
    total_experiments_parsed = 0
    
    for computer_folder in os.listdir(base_folder):
        experiments_parsed_in_computer = 0
        for experiment_folder in os.listdir(os.path.join(base_folder, computer_folder)):
            folder_path = os.path.join(base_folder, computer_folder, experiment_folder)
        
            if os.path.isdir(folder_path):
                # Parse parameters from the folder name
                params = parse_folder_name(experiment_folder)
            
                # Load all CSV files in the folder
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.csv'):
                        file_path = os.path.join(folder_path, file_name)
                        df = load_csv_excluding_pareto(file_path)
                    
                        # Store the DataFrame in the nested dictionary
                        key = (
                            params['approach'], 
                            params['test_problem'], 
                            params['crossover_rate'], 
                            params['mutation_rate'], 
                            params['population_size']
                        )
                        experiment_data[key]['data'].append(df)
                        experiment_data[key]['params'] = params
                        total_experiments_parsed += 1
                        experiments_parsed_in_computer += 1
        computers_parsed += 1
        print(f"Experiments parsed in computer {computer_folder}: {experiments_parsed_in_computer}")
    print(f"Total experiments parsed: {total_experiments_parsed}")
    print(f"Total computers parsed: {computers_parsed}")
    return experiment_data

def parse_folder_name(folder_name):
    """
    Parse experiment parameters from a folder name.

    Args:
        folder_name (str): Folder name containing encoded parameters.

    Returns:
        dict: Dictionary of parameters extracted from the folder name.
    """
    params = {}
    parts = folder_name.split('_')
    params['approach'] = parts[0]
    params['test_problem'] = parts[1]
    params['crossover_rate'] = float(parts[2])
    params['mutation_rate'] = float(parts[3])
    params['population_size'] = int(parts[4])
    return params

def load_csv_excluding_pareto(file_path):
    """
    Load a CSV file and exclude rows starting from "Best Pareto Front" or similar markers.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing only valid rows above "Best Pareto Front".
    """
    with open(file_path, 'r') as file:
        rows = file.readlines()
    
    # Find the first occurrence of "Best Pareto Front"
    pareto_index = next((i for i, row in enumerate(rows) if "Best Pareto front" in row), None)

    # Keep only rows above the marker
    valid_rows = rows[:pareto_index] if pareto_index is not None else rows

    # Convert valid rows back into a DataFrame
    from io import StringIO
    valid_data = StringIO(''.join(valid_rows))
    df = pd.read_csv(valid_data, sep=',')

    # Drop rows where the generation or other key columns have NaN or invalid data
    df = df.dropna(subset=["Generation"])  # Keep only rows with valid generation data
    df = df[df["Generation"].apply(lambda x: str(x).isdigit())]  # Ensure "Generation" contains only numbers

    return df




def average_metrics_by_percentage(runs, percentages):
    """
    Calculate averaged metrics for specified percentages of total runtime.

    Args:
        runs (dict): Data structure containing runs (keyed by parameters) and their data as DataFrames.
        percentages (list of float): List of percentages (0-100) for which to compute metrics.

    Returns:
        pd.DataFrame: DataFrame where rows are percentages, and columns are the averaged metrics.
    """
    # Create a dictionary to store results for each percentage
    results = {perc: [] for perc in percentages}
    
    for run_key, run_data in runs.items():
        for df in run_data['data']:
            # Get total number of generations
            total_generations = len(df)
            
            # Identify the row indices corresponding to each percentage
            indices = [int(total_generations * perc / 100) - 1 for perc in percentages]
            indices = [max(0, min(idx, total_generations - 1)) for idx in indices]  # Clamp indices
            
            # Extract metrics at these indices and store them
            for perc, idx in zip(percentages, indices):
                results[perc].append(df.iloc[idx])
    
    # Compute averages for each percentage
    averaged_results = {}
    for perc, rows in results.items():
        metrics_df = pd.DataFrame(rows)  # Convert to DataFrame for easier aggregation
        averaged_results[perc] = metrics_df.mean()  # Compute column-wise averages
    
    # Convert the averaged results to a DataFrame for easier handling
    averaged_df = pd.DataFrame(averaged_results).T
    averaged_df.index.name = 'Percentage'
    return averaged_df

def plot_multiple_metrics_vs_percentage(averaged_results_list, approach_names, metric_name, title, filename=None, ymin=None, ymax=None):
    """
    Plots the specified metric against the percentages of total runtime for multiple averaged_results.

    Args:
        averaged_results_list (list of dicts): List containing multiple dictionaries of averaged results.
        approach_names (list of str): Names of the approaches corresponding to the averaged results.
        metric_name (str): The name of the metric to plot (e.g., "Hypervolume", "Min error").
        title (str): The title of the plot.
        filename (str, optional): If provided, saves the plot to the specified file.
        ymin (float, optional): The lower bound of the y-axis.
        ymax (float, optional): The upper bound of the y-axis.

    Returns:
        None: Displays or saves the plot.
    """
    plt.figure(figsize=(10, 6))  # Set up the plot figure
    
    # Loop over each averaged_results dictionary in the list
    for averaged_results, approach_name in zip(averaged_results_list, approach_names):
        metrics = list(averaged_results.keys())  # x values (percentages)

        if metric_name not in metrics:
            raise ValueError(f"Metric '{metric_name}' not found in the averaged results.")

        percentages = averaged_results[metrics[0]].index

        # Extract the metric values for each percentage
        metric_values = [averaged_results[metric_name][perc] for perc in percentages]

        # Plot the data for this particular set of results
        label = approach_name  # Default label (can be customized)
        plt.plot(percentages, metric_values, marker='o', linestyle='-', label=label)

    # Add labels and title
    plt.xlabel('Percentage of Total Runtime', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(title, fontsize=14, fontstyle='italic')

    # Set static y-axis limits if provided
    if ymin is not None or ymax is not None:
        plt.ylim(ymin, ymax)

    # Show the grid and plot
    plt.grid(True)
    plt.xticks(percentages)  # Ensure that all percentages are displayed on the x-axis
    plt.legend()
    plt.tight_layout()

    # If filename is provided, save the plot as a file
    if filename:
        plt.savefig(filename)
        print(f"Plot saved as {filename}")
    else:
        # Display the plot if no filename is provided
        plt.show()


def rename_folders(base_folder):
    terms_to_replace = ["deutsch_josza", "full_adder"]

    for computer_folder in os.listdir(base_folder):
        for experiment_folder in os.listdir(os.path.join(base_folder, computer_folder)):
            original_folder_path = os.path.join(base_folder, computer_folder, experiment_folder)
            if os.path.isdir(original_folder_path):
                new_folder_name = experiment_folder
                for term in terms_to_replace:
                    new_folder_name = new_folder_name.replace(term, term.replace("_", "-"))
                if new_folder_name != experiment_folder:
                    new_folder_path = os.path.join(base_folder, computer_folder, new_folder_name)
                    os.rename(original_folder_path, new_folder_path)                

def delete_old_folders(base_folder):
    terms_to_check = ["deutsch_josza", "full_adder"]

    for computer_folder in os.listdir(base_folder):
        for experiment_folder in os.listdir(os.path.join(base_folder, computer_folder)):
            folder_path = os.path.join(base_folder, computer_folder, experiment_folder)
            if os.path.isdir(folder_path):
                for term in terms_to_check:
                    if term in experiment_folder:
                        print(f"Deleting folder {folder_path}")
                        shutil.rmtree(folder_path)
                        break


base_folder = "results/exp_1"
#rename_folders(base_folder)
experiment_data = load_experiment_data_from_folders(base_folder)

values = experiment_data.values()
all_neat_runs = {k: v for k, v in experiment_data.items() if k[0] == "neat"}
all_base_runs = {k: v for k, v in experiment_data.items() if k[0] == "base"}

percentages = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55 ,60, 65, 70, 75, 80, 85, 90, 95, 100]


test_problems = {
    "deutsch-josza": "Deutsch-Josza Problem",
    "qft" : "Quantum Fourier Transform",
    "full-adder" : "Full Adder"
}

cross_rates = [0.8, 1.0]
mut_rates_base = [0.15, 0.25, 0.35]
mut_rates_neat = [0.55, 0.7, 0.85]
pop_sizes = [500, 1000, 2000, 5000]

for test_problem_key, test_problem_name in test_problems.items():
    approach_names = ["Base Approach", "NEAT Approach"]
    for cross_rate in cross_rates:
        neat_runs = {k: v for k, v in all_neat_runs.items() if k[1] == test_problem_key and k[2] == cross_rate}
        base_runs = {k: v for k, v in all_base_runs.items() if k[1] == test_problem_key and k[2] == cross_rate}

        averaged_metrics_base = average_metrics_by_percentage(base_runs, percentages)
        averaged_metrics_neat = average_metrics_by_percentage(neat_runs, percentages)

        metrics_list = [averaged_metrics_base, averaged_metrics_neat]

        plot_multiple_metrics_vs_percentage(metrics_list, approach_names, "Hypervolume", f"Hypervolume vs Percent Runtime for {test_problem_name} Crossover Rate = {cross_rate}", filename=f"plots/{test_problem_key}/Hypervolume_cross_rate_{cross_rate}.png", ymin=0.4, ymax=0.95)
    
    for pop_size in pop_sizes:
        neat_runs = {k: v for k, v in all_neat_runs.items() if k[1] == test_problem_key and k[4] == pop_size}
        base_runs = {k: v for k, v in all_base_runs.items() if k[1] == test_problem_key and k[4] == pop_size}

        averaged_metrics_base = average_metrics_by_percentage(base_runs, percentages)
        averaged_metrics_neat = average_metrics_by_percentage(neat_runs, percentages)

        metrics_list = [averaged_metrics_base, averaged_metrics_neat]

        plot_multiple_metrics_vs_percentage(metrics_list, approach_names, "Hypervolume", f"Hypervolume vs Percent Runtime for {test_problem_name} Population Size = {pop_size}", filename=f"plots/{test_problem_key}/Hypervolume_pop_size_{pop_size}.png", ymin=0.4, ymax=0.95)

    metrics_list = []
    approach_names = []
    for mut_rate in mut_rates_base:
        base_runs = {k: v for k, v in all_base_runs.items() if k[1] == test_problem_key and k[3] == mut_rate}
        averaged_metrics_base = average_metrics_by_percentage(base_runs, percentages)
        metrics_list.append(averaged_metrics_base)
        approach_names.append(f"Mutation Rate = {mut_rate}")
    
    plot_multiple_metrics_vs_percentage(metrics_list, approach_names, "Hypervolume", f"Hypervolume vs Percent Runtime for {test_problem_name} Base Approach", filename=f"plots/{test_problem_key}/Hypervolume_base_mut_rates.png", ymin=0.4, ymax=0.95)

    metrics_list = []
    approach_names = []
    for mut_rate in mut_rates_neat:
        neat_runs = {k: v for k, v in all_neat_runs.items() if k[1] == test_problem_key and k[3] == mut_rate}
        averaged_metrics_neat = average_metrics_by_percentage(neat_runs, percentages)
        metrics_list.append(averaged_metrics_neat)
        approach_names.append(f"Mutation Rate = {mut_rate}")
    
    plot_multiple_metrics_vs_percentage(metrics_list, approach_names, "Hypervolume", f"Hypervolume vs Percent Runtime for {test_problem_name} NEAT Approach", filename=f"plots/{test_problem_key}/Hypervolume_neat_mut_rates.png", ymin=0.4, ymax=0.95)