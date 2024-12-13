import os
from pathlib import Path
import pandas as pd



def find_csv_files(directory, suffixes):
    """
    Recursively searches through a directory for CSV files with specific suffixes.

    Parameters:
        directory (str or Path): The directory to search.
        suffixes (list of str): List of suffixes to match (e.g., ['_data.csv', '_results.csv']).

    Returns:
        list of Path: List of paths to matching CSV files.
    """
    directory = Path(directory)  # Ensure the directory is a Path object
    if not directory.is_dir():
        raise ValueError(f"The provided directory '{directory}' is not a valid directory.")
    
    matching_files = []
    
    # Traverse the directory recursively
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file ends with any of the specified suffixes
            if any(file.endswith(suffix) for suffix in suffixes):
                matching_files.append(Path(root) / file)
    
    return matching_files


def load_csv_as_dataframe(file_path):
    """
    Loads a CSV file into a pandas DataFrame.

    Parameters:
        file_path (str or Path): Path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    file_path = Path(file_path)  # Ensure the file path is a Path object
    if not file_path.is_file():
        raise ValueError(f"The file '{file_path}' does not exist.")
    
    # Load the CSV as a DataFrame
    try:
        df = pd.read_csv(file_path, usecols=lambda column: column not in ["Unnamed: 0"])
        print(f"Loaded file: {file_path}")
        return df
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None