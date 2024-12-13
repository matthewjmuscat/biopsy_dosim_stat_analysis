import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pingouin import compute_effsize


def calculate_wilcoxon_dataframe_all(dataframe, patient_id_col, bx_index_col, bx_id_col, trial_num_col, voxel_index_col, dose_col):
    """
    Calculate Wilcoxon signed-rank test p-values between voxel pairs across all biopsies and patients.
    
    Args:
        dataframe (pd.DataFrame): The input dataframe containing voxel-wise data.
        patient_id_col (str): Column name for Patient ID.
        bx_index_col (str): Column name for Bx index.
        bx_id_col (str): Column name for Bx ID.
        trial_num_col (str): Column name for MC trial number.
        voxel_index_col (str): Column name for voxel index.
        dose_col (str): Column name for dose values.

    Returns:
        pd.DataFrame: A DataFrame containing Wilcoxon p-values for all voxel pairs 
                      across all patients and biopsies, including Bx ID.
    """
    results = []
    
    # Group by Patient ID, Bx index, and Bx ID
    grouped = dataframe.groupby([patient_id_col, bx_index_col, bx_id_col])
    
    for (patient_id, bx_index, bx_id), group in grouped:
        # Pivot table to get a matrix of voxel doses indexed by trial number
        voxel_trial_matrix = group.pivot_table(index=trial_num_col, 
                                               columns=voxel_index_col, 
                                               values=dose_col)
        
        # Get unique voxel indices
        voxel_indices = voxel_trial_matrix.columns
        
        # Perform Wilcoxon signed-rank tests for each voxel pair
        for i, voxel1 in enumerate(voxel_indices):
            for j, voxel2 in enumerate(voxel_indices):
                if i >= j:  # Avoid redundant calculations (matrix is symmetric)
                    continue
                
                # Extract dose values for the two voxels across trials
                dose1 = voxel_trial_matrix[voxel1].dropna()
                dose2 = voxel_trial_matrix[voxel2].dropna()
                
                # Ensure both have data for all trials
                if len(dose1) == len(dose2) and len(dose1) > 1:
                    try:
                        _, p_value = wilcoxon(dose1, dose2)
                        results.append({
                            'Patient ID': patient_id,
                            'Bx index': bx_index,
                            'Bx ID': bx_id,
                            'Voxel 1': voxel1,
                            'Voxel 2': voxel2,
                            'P-value': p_value
                        })
                    except ValueError:
                        # Wilcoxon can fail if inputs are invalid (e.g., constant data)
                        continue
    
    # Combine all results into a single DataFrame
    return pd.DataFrame(results)




def create_heatmaps_by_biopsy(result_df, patient_id_col, bx_index_col, bx_id_col, save_dir=None):
    """
    Create heatmaps for all unique combinations of Patient ID and Bx index.
    Each heatmap visualizes Wilcoxon signed-rank test p-values between voxel pairs.
    
    Args:
        result_df (pd.DataFrame): DataFrame containing Wilcoxon test results with p-values.
        patient_id_col (str): Column name for Patient ID.
        bx_index_col (str): Column name for Bx index.
        bx_id_col (str): Column name for Bx ID (used for labeling).
        save_dir (str, optional): Directory where heatmaps will be saved. If None, heatmaps are only displayed.
    
    Returns:
        None: Displays and/or saves the heatmaps for all biopsies.
    """
    # Group by Patient ID and Bx index
    grouped = result_df.groupby([patient_id_col, bx_index_col])
    
    # Ensure the save directory exists if provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    for (patient_id, bx_index), group in grouped:
        # Get the unique Bx ID for this combination
        bx_id = group[bx_id_col].iloc[0]  # Assumes one-to-one mapping
        
        # Get unique voxel indices
        voxels = sorted(set(group['Voxel 1']).union(set(group['Voxel 2'])))
        
        # Initialize an empty matrix for p-values
        p_value_matrix = np.full((len(voxels), len(voxels)), np.nan)
        
        # Populate the matrix
        voxel_to_idx = {voxel: idx for idx, voxel in enumerate(voxels)}
        for _, row in group.iterrows():
            i, j = voxel_to_idx[row['Voxel 1']], voxel_to_idx[row['Voxel 2']]
            p_value_matrix[i, j] = row['P-value']
            p_value_matrix[j, i] = row['P-value']  # Ensure symmetry
        
        # Create a DataFrame for the heatmap
        heatmap_df = pd.DataFrame(p_value_matrix, index=voxels, columns=voxels)
        
        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            heatmap_df,
            annot=True,  # Display annotations in cells
            fmt=".2e",   # Use scientific notation
            cmap="coolwarm",
            cbar_kws={'label': 'p-value'},
            annot_kws={'size': 6},  # Set font size for annotations
            vmin=0,
            vmax=1
        )
        title = f"Heatmap of Wilcoxon p-values for Patient {patient_id}, Bx Index {bx_index}, Bx ID {bx_id}"
        plt.title(title)
        plt.xlabel("Voxel Index")
        plt.ylabel("Voxel Index")
        plt.tight_layout()
        
        if save_dir is not None:
            # Save the plot as PNG and SVG
            base_file_name = f"Heatmap_Patient_{patient_id}_BxIndex_{bx_index}_BxID_{bx_id}"
            png_path = os.path.join(save_dir, f"{base_file_name}.png")
            svg_path = os.path.join(save_dir, f"{base_file_name}.svg")
            
            plt.savefig(png_path)
            plt.savefig(svg_path)
            print(f"Saved heatmap as PNG: {png_path}")
            print(f"Saved heatmap as SVG: {svg_path}")
        
        # Display the plot
        plt.close()






####
####
####


# Cohens d between voxel pairs and heatmaps 


def create_eff_size_dataframe(result_df, patient_id_col, bx_index_col, bx_id_col, voxel_index_col, dose_col, eff_size="cohen"):
    """
    Create a single DataFrame containing effect size values (e.g., Cohen's d) for voxel pairs
    across all unique combinations of Patient ID, Bx index, and Bx ID.
    
    Args:
        result_df (pd.DataFrame): Input DataFrame with voxel data.
        patient_id_col (str): Column name for Patient ID.
        bx_index_col (str): Column name for Bx index.
        bx_id_col (str): Column name for Bx ID.
        voxel_index_col (str): Column name for voxel index.
        dose_col (str): Column name for dose values.
        eff_size (str): Effect size type, e.g., "cohen", "hedges", "glass". Defaults to "cohen".
    
    Returns:
        pd.DataFrame: A single DataFrame containing effect size values with columns for:
                      Patient ID, Bx index, Bx ID, Voxel 1, Voxel 2, and the effect size value.
    """
    results = []
    
    # Group by Patient ID, Bx index, and Bx ID
    grouped = result_df.groupby([patient_id_col, bx_index_col, bx_id_col])
    
    for (patient_id, bx_index, bx_id), group in grouped:
        # Get unique voxel indices
        voxels = sorted(group[voxel_index_col].unique())
        
        # Loop through voxel pairs
        for i, voxel1 in enumerate(voxels):
            for j, voxel2 in enumerate(voxels):
                if i >= j:  # Avoid redundant calculations
                    continue
                
                # Extract dose values for both voxels
                group1 = group[group[voxel_index_col] == voxel1][dose_col]
                group2 = group[group[voxel_index_col] == voxel2][dose_col]
                
                # Calculate effect size using pingouin
                if len(group1) > 1 and len(group2) > 1:  # Ensure sufficient data
                    eff_value = compute_effsize(group1, group2, eftype=eff_size)
                    results.append({
                        patient_id_col: patient_id,
                        bx_index_col: bx_index,
                        bx_id_col: bx_id,
                        "Voxel 1": voxel1,
                        "Voxel 2": voxel2,
                        "Effect Size": eff_value
                    })
    
    # Convert results to a DataFrame
    eff_size_df = pd.DataFrame(results)
    return eff_size_df






def plot_eff_size_heatmaps(eff_size_df, patient_id_col, bx_index_col, bx_id_col, eff_size_col, eff_size_type, save_dir=None):
    """
    Create and optionally save heatmaps for effect size values for all unique combinations 
    of Patient ID, Bx index, and Bx ID.
    
    Args:
        eff_size_df (pd.DataFrame): DataFrame containing effect size values with columns for:
                                    Patient ID, Bx index, Bx ID, Voxel 1, Voxel 2, Effect Size.
        patient_id_col (str): Column name for Patient ID.
        bx_index_col (str): Column name for Bx index.
        bx_id_col (str): Column name for Bx ID.
        eff_size_col (str): Column name for the effect size values.
        save_dir (str, optional): Directory to save heatmaps. If None, only displays the heatmaps.
    
    Returns:
        None
    """
    # Ensure the save directory exists if provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    # Group by Patient ID, Bx index, and Bx ID
    grouped = eff_size_df.groupby([patient_id_col, bx_index_col, bx_id_col])
    
    for (patient_id, bx_index, bx_id), group in grouped:
        # Pivot the DataFrame to create a matrix for heatmap plotting
        heatmap_data = group.pivot(index="Voxel 1", columns="Voxel 2", values=eff_size_col)
        
        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            heatmap_data,
            annot=True,  # Display annotations in cells
            fmt=".2f",   # Use decimal format for effect size
            cmap="coolwarm",
            cbar_kws={'label': "Effect Size"},
            annot_kws={'size': 8},  # Font size for annotations
            vmin=-2,  # Adjust range for effect size
            vmax=2
        )
        title = (f"Effect Size Heatmap for {patient_id_col}: {patient_id}, "
                 f"{bx_index_col}: {bx_index}, {bx_id_col}: {bx_id}")
        plt.title(title)
        plt.xlabel("Voxel 2")
        plt.ylabel("Voxel 1")
        plt.tight_layout()
        
        if save_dir is not None:
            # Save the plot as PNG and SVG
            base_file_name = (f"EffSize_{eff_size_type}_Heatmap_{patient_id_col}_{patient_id}_"
                              f"{bx_index_col}_{bx_index}_{bx_id_col}_{bx_id}")
            png_path = os.path.join(save_dir, f"{base_file_name}.png")
            svg_path = os.path.join(save_dir, f"{base_file_name}.svg")
            
            plt.savefig(png_path)
            plt.savefig(svg_path)
            print(f"Saved heatmap as PNG: {png_path}")
            print(f"Saved heatmap as SVG: {svg_path}")
        
        # Close the plot to avoid displaying in non-interactive environments
        plt.close()

