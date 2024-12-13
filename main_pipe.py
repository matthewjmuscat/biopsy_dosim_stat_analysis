import pandas as pd 
import load_files
from pathlib import Path
import statistical_tests_1
import os 

def main():
    
    # Main output directory
    main_output_path = Path("/home/mjm/Documents/UBC/Research/biopsylocalization-new/Data/Output data/MC_sim_out- Date-Dec-12-2024 Time-11,20,06")  # Ensure the directory is a Path object
    
    ### Load Dataframes 

    # Set csv directory
    csv_directory = main_output_path.joinpath("Output CSVs")
    cohort_csvs_directory = csv_directory.joinpath("Cohort")

    # Cohort: Global dosimetry dataframe
    cohort_global_dosim_path = cohort_csvs_directory.joinpath("Cohort: Global dosimetry.csv")  # Ensure the directory is a Path object
    cohort_global_dosim_df = load_files.load_csv_as_dataframe(cohort_global_dosim_path)

    # Cohort: Global dosimetry by voxel dataframe
    cohort_global_dosim_by_voxel_path = cohort_csvs_directory.joinpath("Cohort: Global dosimetry by voxel.csv")  # Ensure the directory is a Path object
    cohort_global_dosim_by_voxel_df = load_files.load_csv_as_dataframe(cohort_global_dosim_by_voxel_path)

    # Cohort: Bx DVH metrics dataframe
    cohort_bx_dvh_metrics_path = cohort_csvs_directory.joinpath("Cohort: Bx DVH metrics.csv")  # Ensure the directory is a Path object
    cohort_bx_dvh_metrics_df = load_files.load_csv_as_dataframe(cohort_bx_dvh_metrics_path)


    # MC Simulation directory 
    mc_simulation_directory = csv_directory.joinpath("MC simulation")

    # Find voxelwise all trials dataframes 
    suffixes = ["-Voxel-wise dose output by MC trial number.csv"]
    voxelwise_all_trials_dosim_csvs_list = load_files.find_csv_files(mc_simulation_directory, suffixes)

    voxelwise_all_trials_dosim_dfs_list = []
    for path in voxelwise_all_trials_dosim_csvs_list:
        df = load_files.load_csv_as_dataframe(path)
        voxelwise_all_trials_dosim_dfs_list.append(df)

    voxelwise_all_trials_dosim_df = pd.concat(voxelwise_all_trials_dosim_dfs_list, ignore_index=True)


    ## Create output directory
    # Output directory 
    output_dir = Path(__file__).parents[0].joinpath("output_data")
    os.makedirs(output_dir, exist_ok=True)


    ## STATISTICAL TESTS 1

    statistical_tests_1_dir = output_dir.joinpath("statistical_tests_1")
    os.makedirs(statistical_tests_1_dir, exist_ok=True)




    # Wilcoxon signed rank test between all voxel pairs, where pairs for the actual Wilcoxon test are by trial number
    wilcoxon_p_vals_dataframe = statistical_tests_1.calculate_wilcoxon_dataframe_all(voxelwise_all_trials_dosim_df, "Patient ID", "Bx index", "Bx ID", "MC trial num", "Voxel index", "Dose (Gy)")

    wilcoxon_heatmaps_dir = statistical_tests_1_dir.joinpath("wilcoxon_heatmaps")
    os.makedirs(wilcoxon_heatmaps_dir, exist_ok=True)

    statistical_tests_1.create_heatmaps_by_biopsy(wilcoxon_p_vals_dataframe, "Patient ID", "Bx index", "Bx ID", save_dir = wilcoxon_heatmaps_dir)
    



    # Cohens d between each voxel pair for each biopsy
    eff_sizes = ['cohen', 'hedges', 'r', 'pointbiserialr', 'eta-square']
    for eff_size in eff_sizes:
        eff_size_df = statistical_tests_1.create_eff_size_dataframe(voxelwise_all_trials_dosim_df, "Patient ID", "Bx index", "Bx ID", "Voxel index", "Dose (Gy)", eff_size=eff_size)

        eff_size_heatmaps_dir = statistical_tests_1_dir.joinpath(f"{eff_size}_heatmaps")
        os.makedirs(wilcoxon_heatmaps_dir, exist_ok=True)

        statistical_tests_1.plot_eff_size_heatmaps(eff_size_df, "Patient ID", "Bx index", "Bx ID", "Effect Size", eff_size, save_dir=eff_size_heatmaps_dir)

    
    print('test')


if __name__ == "__main__":
    main()


    