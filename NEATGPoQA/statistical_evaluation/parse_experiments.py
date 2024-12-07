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
    
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        
        if os.path.isdir(folder_path):
            # Parse parameters from the folder name
            params = parse_folder_name(folder_name)
            
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

def plot_multiple_metrics_vs_percentage(averaged_results_list, metric_name, filenames=None):
    """
    Plots the specified metric against the percentages of total runtime for multiple averaged_results.

    Args:
        averaged_results_list (list of dicts): List containing multiple dictionaries of averaged results.
        metric_name (str): The name of the metric to plot (e.g., "Hypervolume", "Min error").
        filenames (list of str or str, optional): If provided, saves each plot to the corresponding file(s).

    Returns:
        None: Displays or saves the plot.
    """
    plt.figure(figsize=(10, 6))  # Set up the plot figure
    
    # Loop over each averaged_results dictionary in the list
    for i, averaged_results in enumerate(averaged_results_list):
        metrics = list(averaged_results.keys())  # x values (percentages)

        if metric_name not in metrics:
            raise ValueError(f"Metric '{metric_name}' not found in the averaged results.")

        percentages = averaged_results[metrics[0]].index   

        # Extract the metric values for each percentage
        metric_values = [averaged_results[metric_name][perc] for perc in percentages]

        # Plot the data for this particular set of results
        label = f"Base Aproach"  # Default label (can be customized)
        plt.plot(percentages, metric_values, marker='o', linestyle='-', label=label)

    # Add labels and title
    plt.xlabel('Percentage of Total Runtime', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'{metric_name} vs Percentage of Total Runtime Base Full-Adder-Problem', fontsize=14)
    
    # Show the grid and plot
    plt.grid(True)
    plt.xticks(percentages)  # Ensure that all percentages are displayed on the x-axis
    plt.legend()
    plt.tight_layout()

    # If filenames are provided, save the plot(s) as separate files
    if filenames:
        if isinstance(filenames, list):
            for i, filename in enumerate(filenames):
                plt.savefig(filename)
                print(f"Plot saved as {filename}")
        else:
            # Save a single plot if one filename is provided
            plt.savefig(filenames)
            print(f"Plot saved as {filenames}")
    else:
        # Display the plot if no filenames are provided
        plt.show()


base_folder = "results/all"
experiment_data = load_experiment_data_from_folders(base_folder)

values = experiment_data.values()
neat_qft_runs = {k: v for k, v in experiment_data.items() if k[0] == "neat" and k[1] == "qft"}
base_qft_runs = {k: v for k, v in experiment_data.items() if k[0] == "base" and k[1] == "fulladder"}

percentages = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55 ,60, 65, 70, 75, 80, 85, 90, 95, 100]

averaged_metrics_base = average_metrics_by_percentage(base_qft_runs, percentages)
averaged_metrics_neat = average_metrics_by_percentage(neat_qft_runs, percentages)
metrics_list = [averaged_metrics_base]

plot_multiple_metrics_vs_percentage(metrics_list, "Min error", filenames="MinError.png")

print(averaged_metrics_base)