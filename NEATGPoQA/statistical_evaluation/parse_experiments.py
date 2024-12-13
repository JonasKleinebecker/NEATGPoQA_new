import random
import shutil
from matplotlib.patches import Polygon
import pandas as pd
import os
from collections import defaultdict
from io import StringIO
import matplotlib.pyplot as plt

import os
from collections import defaultdict

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
                        
                        # Load both the CSV data, Pareto information, and average runtime info
                        df, pareto_info, avg_runtime = load_csv_with_pareto(file_path)
                    
                        # Store the DataFrame, Pareto info, and avg runtime in the nested dictionary
                        key = (
                            params['approach'], 
                            params['test_problem'], 
                            params['crossover_rate'], 
                            params['mutation_rate'], 
                            params['population_size'],
                            file_name
                        )
                        # Append the data, Pareto info, and average runtime
                        experiment_data[key]['data'].append(df)
                        experiment_data[key]['params'] = params
                        experiment_data[key]['pareto_info'].append(pareto_info)
                        experiment_data[key]['avg_runtime'].append(avg_runtime)
                        
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

def load_csv_with_pareto(file_path):
    """
    Load a CSV file and separate rows into experiment data, Pareto front information,
    and average runtime statistics.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        tuple:
            - pd.DataFrame: DataFrame containing only valid rows above "Best Pareto Front".
            - pd.DataFrame: DataFrame containing the Pareto front data.
            - tuple: (avg_simulation_runtime, avg_generation_runtime) values.
    """
    with open(file_path, 'r') as file:
        rows = file.readlines()
    
    # Find the first occurrence of "Best Pareto front" and "Best fitness"
    pareto_index = next((i for i, row in enumerate(rows) if "Best Pareto front" in row), None)
    fitness_index = next((i for i, row in enumerate(rows) if "Best fitness" in row), None)
    
    # Separate valid rows above the Pareto marker
    if pareto_index is not None:
        valid_rows = rows[:pareto_index]
    else:
        valid_rows = rows

    # Extract Pareto information up to the "Best fitness" line
    if pareto_index is not None and fitness_index is not None:
        pareto_info = ''.join(rows[pareto_index+1:fitness_index])  # Skip the header line
    elif pareto_index is not None:
        pareto_info = ''.join(rows[pareto_index+1:])  # Skip the header line if no fitness marker found
    else:
        pareto_info = None

    # Parse Pareto data into a DataFrame
    if pareto_info:
        # Skip the first line after "Best Pareto front" which contains the header
        pareto_lines = pareto_info.strip().split('\n')[1:]  # Skip the first line (header)
        pareto_data = '\n'.join(pareto_lines)
        pareto_df = pd.read_csv(StringIO(pareto_data), header=None, names=["Generation", "Hypervolume", "Error", "Gate Fit"])
    else:
        pareto_df = pd.DataFrame()

    # Look for average runtime info (last lines in CSV)
    avg_runtime_info = None
    if "Avg simulation runtime,Avg generation runtime" in rows[-2]:
        avg_runtime_info = rows[-1].strip().split(',')
        avg_simulation_runtime = float(avg_runtime_info[0])
        avg_generation_runtime = float(avg_runtime_info[1])
    else:
        avg_simulation_runtime = None
        avg_generation_runtime = None

    # Convert valid rows back into a DataFrame
    valid_data = StringIO(''.join(valid_rows))
    df = pd.read_csv(valid_data, sep=',')
    
    # Drop rows where the generation or other key columns have NaN or invalid data
    df = df.dropna(subset=["Generation"])  # Keep only rows with valid generation data
    df = df[df["Generation"].apply(lambda x: str(x).isdigit())]  # Ensure "Generation" contains only numbers

    return df, pareto_df, (avg_simulation_runtime, avg_generation_runtime)


def average_metrics_by_percentage(runs, percentages):
    """
    Calculate averaged metrics and their deviations (one sigma above and below the mean) for specified percentages of total runtime.

    Args:
        runs (dict): Data structure containing runs (keyed by parameters) and their data as DataFrames.
        percentages (list of float): List of percentages (0-100) for which to compute metrics.

    Returns:
        dict: A dictionary with keys 'mean', 'plus_sigma', and 'minus_sigma', each containing a DataFrame
              where rows are percentages and columns are the metrics.
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

    # Compute averages and deviations for each percentage
    mean_results = {}
    plus_sigma_results = {}
    minus_sigma_results = {}

    for perc, rows in results.items():
        metrics_df = pd.DataFrame(rows)  # Convert to DataFrame for easier aggregation
        mean = metrics_df.mean()
        std = metrics_df.std()

        mean_results[perc] = mean
        plus_sigma_results[perc] = mean + std
        minus_sigma_results[perc] = mean - std

    # Convert the results to DataFrames for easier handling
    mean_df = pd.DataFrame(mean_results).T
    mean_df.index.name = 'Percentage'

    plus_sigma_df = pd.DataFrame(plus_sigma_results).T
    plus_sigma_df.index.name = 'Percentage'

    minus_sigma_df = pd.DataFrame(minus_sigma_results).T
    minus_sigma_df.index.name = 'Percentage'

    return {
        'mean': mean_df,
        'plus_sigma': plus_sigma_df,
        'minus_sigma': minus_sigma_df
    }

def plot_multiple_metrics_vs_percentage(averaged_results_list, approach_names, metric_name, title, filename=None, ymin=None, ymax=None):
    """
    Plots the specified metric against the percentages of total runtime for multiple averaged_results, including sigma deviations.

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
        percentages = averaged_results['mean'].index

        if metric_name not in averaged_results['mean'].columns:
            raise ValueError(f"Metric '{metric_name}' not found in the averaged results.")

        # Extract the metric values for mean and sigma deviations
        mean_values = averaged_results['mean'][metric_name]
        plus_sigma_values = averaged_results['plus_sigma'][metric_name]
        minus_sigma_values = averaged_results['minus_sigma'][metric_name]

        # Plot the mean values
        plt.plot(percentages, mean_values, marker='o', linestyle='-', label=f"{approach_name} (Mean)")

        # Plot the sigma deviations with weaker or dotted lines
        plt.plot(percentages, plus_sigma_values, linestyle='--', color=plt.gca().lines[-1].get_color(), alpha=0.7, label=f"{approach_name} (+1 Sigma)")
        plt.plot(percentages, minus_sigma_values, linestyle='--', color=plt.gca().lines[-1].get_color(), alpha=0.7, label=f"{approach_name} (-1 Sigma)")

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

def find_best_parameter_setup(experiment_data):
    """
    Find the best parameter setup for each approach and test problem based on average Hypervolume.

    Args:
        experiment_data (dict): Dictionary where keys are lists of parameters and values are DataFrames.

    Returns:
        dict: A dictionary summarizing the best parameter setups for each approach and test problem.
    """
    from collections import defaultdict

    # Group results by (approach, test_problem, crossover_rate, mutation_rate, population_size)
    parameter_results = defaultdict(list)

    for key, df in experiment_data.items():
        # Unpack the key: [approach, test_problem, crossover_rate, mutation_rate, population_size, file_name]
        approach, test_problem, crossover_rate, mutation_rate, population_size, _ = key

        # Group by the unique parameter setup (excluding file_name)
        param_key = (approach, test_problem, crossover_rate, mutation_rate, population_size)

        df_data = df['data']
        df_data_0 = df_data[0]
        df_hypervolume = df_data_0['Hypervolume']
        last_generation_hv = df_hypervolume.values[-1]
        parameter_results[param_key].append(last_generation_hv)
        
    averaged_max_hv = {}
    for key, hvs in parameter_results.items():
        avg_hv = sum(hvs) / len(hvs) if hvs else 0
        averaged_max_hv[key] = avg_hv
    # Find the best parameter setup for each (approach, test_problem)
    best_setups = {}
    for (approach, test_problem, crossover_rate, mutation_rate, population_size), hv in averaged_max_hv.items():
        setup_key = (approach, test_problem)
        if setup_key not in best_setups or hv > best_setups[setup_key]["average_hypervolume"]:
            best_setups[setup_key] = {
                "best_params": {
                    "crossover_rate": crossover_rate,
                    "mutation_rate": mutation_rate,
                    "population_size": population_size,
                },
                "average_hypervolume": hv,
            }

    return best_setups

def plot_pareto_front_hypervolume(pareto_df, file_name):
    """
    Plot the Pareto front and shade the area under the curve to represent the hypervolume.
    
    Args:
        pareto_df (pd.DataFrame): DataFrame containing the Pareto front data (including columns 'Error', 'Gate Fit').
        file_name (str): Path to save the plot.
    """
    # Ensure data is sorted by Error and Gate Fit
    pareto_df_sorted = pareto_df.sort_values(by=['Error', 'Gate Fit'])

    # Get the x and y values
    x_values = pareto_df_sorted['Error']
    y_values = pareto_df_sorted['Gate Fit']

    # Set up the plot
    plt.figure(figsize=(10, 8))

    # Plot the Pareto front as a line
    plt.plot(x_values, y_values, color='gray', linestyle='-', linewidth=2)

    # Fill the area under the Pareto front to represent the hypervolume
    plt.fill_between(
        x_values, 
        y_values, 
        1,  # Reference point (1, 1) for the bottom of the plot
        color='lightblue',  # Color of the hypervolume area
        alpha=0.5  # Transparency of the shaded area
    )

    # Ensure that the plot extends to the edges
    plt.fill_between(
        [x_values.min(), 1],  # Extend to the left edge and the reference point on x-axis
        [y_values.min(), 1],  # Extend to the top edge and the reference point on y-axis
        1,
        color='lightblue', 
        alpha=0.5
    )

    # Add labels and title
    plt.xlabel('Error')
    plt.ylabel('Gate Fit')
    plt.title('Pareto Front with Hypervolume Area')

    # Add a label for the hypervolume
    plt.text(
        0.5, 0.1,  # Position of the label in the plot
        'Hypervolume Area',  # Text to display
        fontsize=12, 
        color='black', 
        ha='center', 
        va='center'
    )

    # Set axis limits to make sure the edges are connected
    plt.xlim(0, 1)  # Adjust x-axis limit to be between 0 and 1
    plt.ylim(0, 1)  # Adjust y-axis limit to be between 0 and 1

    # Save the plot as a file
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()  # Close the plot to free memory

def plot_pareto_front(df, file_name, reference_point=(1, 1)):
    """
    Plots the Pareto front with shaded hypervolume and saves it to a file.
    Connects the leftmost and rightmost points to the edges, uses 
    vertical/horizontal lines, and improves visual elements like axis limits
    and label placement.

    Args:
      df: Pandas DataFrame with columns 'Error' and 'Gate Fit'.
      file_name: String specifying the file name to save the plot (e.g., 'pareto_front.png').
      reference_point: Tuple representing the reference point for hypervolume calculation.
    """

    # Eliminate duplicate rows
    df_unique = df.drop_duplicates(subset=['Error', 'Gate Fit'])

    # Sort by 'Error' for plotting and hypervolume calculation
    df_unique = df_unique.sort_values('Error')

    # Extract data
    error = df_unique['Error'].values
    gate_fit = df_unique['Gate Fit'].values

    # Add points to connect to the edges
    if len(error) > 0:
        points = [(error[0], 1)]  # Leftmost point connected to top
        for i in range(len(error) - 1):
            points.extend([(error[i], gate_fit[i]), (error[i + 1], gate_fit[i])])  # Vertical then horizontal
        points.extend([(error[-1], gate_fit[-1]), (1, gate_fit[-1])])  # Rightmost point connected to right edge
        points.append(reference_point)  # Add the reference point
    else:
        points = [reference_point]

    # Plot Pareto front points
    plt.scatter(error, gate_fit, marker='o', color='black', label='Pareto Front')  # Black markers
    plt.xlabel('Error')
    plt.ylabel('Gate Fit')
    plt.title('Pareto Front')

    # Shade hypervolume area
    polygon = Polygon(points, closed=True, alpha=0.2, label='Hypervolume Area')  # Label the polygon
    plt.gca().add_patch(polygon)

    # Set axis limits
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Place the legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Save the plot to a file
    plt.savefig(file_name, bbox_inches='tight')  # Use bbox_inches='tight' to include the legend
    plt.close()

def create_boxplots(data, filename):
  """
  Generates boxplots for a list of lists containing float values and saves it to a file.

  Args:
    data: A list of lists, where each inner list contains float values.
    filename: The name of the file to save the plot.

  Returns:
    None. Saves the boxplots to the specified file.
  """
  plt.figure()
  plt.boxplot(data)
  plt.xticks(range(1, len(data) + 1))
  plt.xlabel("List Index")
  plt.ylabel("Values")
  plt.title("Boxplots of Lists")
  plt.savefig(filename)

def get_best_run(run_dict):
    best_run = None
    best_hv = 0
    for key, run_data in run_dict.items():
        df = run_data["data"][0]
        hv = df["Hypervolume"].values[-1]
        if hv > best_hv:
            best_hv = hv
            best_run = run_data
    return best_run


base_folder = "results/exp2"
#rename_folders(base_folder)
experiment_data = load_experiment_data_from_folders(base_folder)
best_setups = find_best_parameter_setup(experiment_data)

values = experiment_data.values()
all_neat_runs = {k: v for k, v in experiment_data.items() if k[0] == "neat"}
all_base_runs = {k: v for k, v in experiment_data.items() if k[0] == "base"}

avg_runtimes_neat = [run["avg_runtime"][0][0] for run in all_neat_runs.values()]
avg_simulation_times_neat = [run["avg_runtime"][0][1] for run in all_neat_runs.values()]

create_boxplots([avg_runtimes_neat, avg_simulation_times_neat], "plots/avg_runtimes_neat.png")


percentages = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55 ,60, 65, 70, 75, 80, 85, 90, 95, 100]


test_problems = {
    "deutsch-josza": "Deutsch-Josza Problem",
    "qft" : "Quantum Fourier Transform",
    "full-adder" : "Full Adder"
}

cross_rates = [0.8, 1.0]
mut_rates_base = [0.25, 0.35, 0.45]
mut_rates_neat = [0.55, 0.7, 0.85]
pop_sizes = [500, 1000, 2000, 5000]

best_neat_deutsch_josza_setup_params = best_setups[("neat", "deutsch-josza")]["best_params"]
best_neat_deutsch_josza_setup_runs = {k: v for k, v in all_neat_runs.items() if k[1] == "deutsch-josza" and k[2] == best_neat_deutsch_josza_setup_params["crossover_rate"] and k[3] == best_neat_deutsch_josza_setup_params["mutation_rate"] and k[4] == best_neat_deutsch_josza_setup_params["population_size"]}

best_run = get_best_run(best_neat_deutsch_josza_setup_runs)
plot_pareto_front(best_run["pareto_info"][0], file_name="plots/deutsch-josza/pareto_front_neat.png")

best_base_deutsch_josza_setup_params = best_setups[("base", "deutsch-josza")]["best_params"]
best_base_deutsch_josza_setup_runs = {k: v for k, v in all_base_runs.items() if k[1] == "deutsch-josza" and k[2] == best_base_deutsch_josza_setup_params["crossover_rate"] and k[3] == best_base_deutsch_josza_setup_params["mutation_rate"] and k[4] == best_base_deutsch_josza_setup_params["population_size"]}
best_neat_qft_setup_params = best_setups[("neat", "qft")]["best_params"]
best_neat_qft_setup_runs = {k: v for k, v in all_neat_runs.items() if k[1] == "qft" and k[2] == best_neat_qft_setup_params["crossover_rate"] and k[3] == best_neat_qft_setup_params["mutation_rate"] and k[4] == best_neat_qft_setup_params["population_size"]}
best_base_qft_setup_params = best_setups[("base", "qft")]["best_params"]
best_base_qft_setup_runs = {k: v for k, v in all_base_runs.items() if k[1] == "qft" and k[2] == best_base_qft_setup_params["crossover_rate"] and k[3] == best_base_qft_setup_params["mutation_rate"] and k[4] == best_base_qft_setup_params["population_size"]}
best_neat_full_adder_setup_params = best_setups[("neat", "full-adder")]["best_params"]
best_neat_full_adder_setup_runs = {k: v for k, v in all_neat_runs.items() if k[1] == "full-adder" and k[2] == best_neat_full_adder_setup_params["crossover_rate"] and k[3] == best_neat_full_adder_setup_params["mutation_rate"] and k[4] == best_neat_full_adder_setup_params["population_size"]}
best_base_full_adder_setup_params = best_setups[("base", "full-adder")]["best_params"]
best_base_full_adder_setup_runs = {k: v for k, v in all_base_runs.items() if k[1] == "full-adder" and k[2] == best_base_full_adder_setup_params["crossover_rate"] and k[3] == best_base_full_adder_setup_params["mutation_rate"] and k[4] == best_base_full_adder_setup_params["population_size"]}

approach_names = ["Base Approach", "NEAT Approach"]
metrics_list = [average_metrics_by_percentage(best_base_deutsch_josza_setup_runs, percentages), average_metrics_by_percentage(best_neat_deutsch_josza_setup_runs, percentages)]
plot_multiple_metrics_vs_percentage(metrics_list, approach_names, "Hypervolume", "Hypervolume vs Percent Runtime for Deutsch-Josza Problem (Best Setups)", filename="plots/deutsch-josza/Hypervolume_best_setup.png", ymin=0.4, ymax=0.95)

metrics_list = [average_metrics_by_percentage(best_base_qft_setup_runs, percentages), average_metrics_by_percentage(best_neat_qft_setup_runs, percentages)]
plot_multiple_metrics_vs_percentage(metrics_list, approach_names, "Hypervolume", "Hypervolume vs Percent Runtime for Quantum Fourier Transform (Best Setups)", filename="plots/qft/Hypervolume_best_setup.png", ymin=0.4, ymax=0.95)

metrics_list = [average_metrics_by_percentage(best_base_full_adder_setup_runs, percentages), average_metrics_by_percentage(best_neat_full_adder_setup_runs, percentages)]
plot_multiple_metrics_vs_percentage(metrics_list, approach_names, "Hypervolume", "Hypervolume vs Percent Runtime for Full Adder Problem (Best Setups)", filename="plots/full-adder/Hypervolume_best_setup.png", ymin=0.4, ymax=0.95)

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