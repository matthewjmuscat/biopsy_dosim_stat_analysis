import pandas as pd 
import load_files
from pathlib import Path
import statistical_tests_1, statistical_tests_0
import os 

def main():
    
    # Main output directory
    main_output_path = Path("/home/mjm/Documents/UBC/Research/biopsylocalization-new/Data/Output data/MC_sim_out- Date-Jan-09-2025 Time-01,04,42")  # Ensure the directory is a Path object
    
    ### Load Dataframes 

    # Set csv directory
    csv_directory = main_output_path.joinpath("Output CSVs")
    cohort_csvs_directory = csv_directory.joinpath("Cohort")

    # Cohort: Global dosimetry dataframe
    cohort_global_dosim_path = cohort_csvs_directory.joinpath("Cohort: Global dosimetry.csv")  # Ensure the directory is a Path object
    cohort_global_dosim_df = load_files.load_csv_as_dataframe(cohort_global_dosim_path)

    # Cohort: Global dosimetry by voxel dataframe
    cohort_global_dosim_by_voxel_path = cohort_csvs_directory.joinpath("Cohort: Global dosimetry by voxel.csv")  # Ensure the directory is a Path object
    cohort_global_dosim_by_voxel_df = load_files.load_multiindex_csv(cohort_global_dosim_by_voxel_path, index_col=None, header_rows=[0,1])

    # Cohort: Bx DVH metrics dataframe
    cohort_bx_dvh_metrics_path = cohort_csvs_directory.joinpath("Cohort: Bx DVH metrics (generalized).csv")  # Ensure the directory is a Path object
    cohort_bx_dvh_metrics_df = load_files.load_csv_as_dataframe(cohort_bx_dvh_metrics_path)

    # Cohort: 3d radiomic features df
    cohort_3d_radiomic_features_path = cohort_csvs_directory.joinpath("Cohort: 3D radiomic features all OAR and DIL structures.csv")  # Ensure the directory is a Path object
    cohort_3d_radiomic_features_df = load_files.load_csv_as_dataframe(cohort_3d_radiomic_features_path)

    # Cohort: all MC structure shift vecs df
    cohort_all_mc_structure_shift_vecs_path = cohort_csvs_directory.joinpath("Cohort: All MC structure shift vectors.csv")  # Ensure the directory is a Path object
    cohort_all_mc_structure_shift_vecs_df = load_files.load_csv_as_dataframe(cohort_all_mc_structure_shift_vecs_path)

    # Cohort: all MC structure shift vecs df
    cohort_biopsy_basic_spatial_features_path = cohort_csvs_directory.joinpath("Cohort: Biopsy basic spatial features dataframe.csv")  # Ensure the directory is a Path object
    cohort_biopsy_basic_spatial_features_df = load_files.load_csv_as_dataframe(cohort_biopsy_basic_spatial_features_path)


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
    





    ## STATISTICAL TESTS 0

    # this one takes a few minutes
    statistical_tests_0_dir = output_dir.joinpath("statistical_tests_0")
    os.makedirs(statistical_tests_0_dir, exist_ok=True)

    print('Statistical tests 0')
    # only trying lognorm because this was found to be the best fit and it speeds up the code (for both dose and grad)
    dists_to_try = ['lognorm']
    statistical_tests_0.histogram_and_fit(voxelwise_all_trials_dosim_df, dists_to_try = dists_to_try, bin_size = 1, dose_col="Dose (Gy)", save_path = statistical_tests_0_dir, custom_name = "histogram_fit_all_voxels_dose")

    statistical_tests_0.histogram_and_fit(voxelwise_all_trials_dosim_df, dists_to_try = dists_to_try, bin_size = 1, dose_col="Dose grad (Gy/mm)", save_path = statistical_tests_0_dir, custom_name = "histogram_fit_all_voxels_dose_gradient")


    # dvh boxplot
    statistical_tests_0.dvh_boxplot(cohort_bx_dvh_metrics_df, save_path = statistical_tests_0_dir, custom_name = "dvh_boxplot")


    # 3d radiomic features non-bx structures
    columns_to_stats = ['Volume', 'Surface area', 'Surface area to volume ratio', 'Sphericity',
       'Compactness 1', 'Compactness 2', 'Spherical disproportion',
       'Maximum 3D diameter', 'PCA major', 'PCA minor', 'PCA least',
       'Major axis (equivalent ellipse)',
       'Minor axis (equivalent ellipse)', 'Least axis (equivalent ellipse)',
       'Elongation', 'Flatness', 'L/R dimension at centroid',
       'A/P dimension at centroid', 'S/I dimension at centroid',
       'DIL centroid (X, prostate frame)', 'DIL centroid (Y, prostate frame)',
       'DIL centroid (Z, prostate frame)',
       'DIL centroid distance (prostate frame)']
    file_name_3d_radiomics_csv = '3d_radiomics_statistics_cohort.csv'
    group_by_column = 'Structure type'
    statistical_tests_0.save_grouped_stats_to_csv(cohort_3d_radiomic_features_df, columns_to_stats, statistical_tests_0_dir, file_name_3d_radiomics_csv, group_by_column, filter_condition=None)
    
    # 
    columns_to_stats = [
       'Length (mm)', 'Volume (mm3)',
       'BX to DIL centroid (X)', 'BX to DIL centroid (Y)',
       'BX to DIL centroid (Z)', 'BX to DIL centroid distance',
       'NN surface-surface distance']
    file_name_biopsy_info_csv = 'biopsy_info_statistics_cohort.csv'
    group_by_column = 'Simulated type'
    statistical_tests_0.save_grouped_stats_to_csv(cohort_biopsy_basic_spatial_features_df, columns_to_stats, statistical_tests_0_dir, file_name_biopsy_info_csv, group_by_column, filter_condition=None)
    
    ## STATISTICAL TESTS 1

    statistical_tests_1_dir = output_dir.joinpath("statistical_tests_1")
    os.makedirs(statistical_tests_1_dir, exist_ok=True)




    # Wilcoxon signed rank test between all voxel pairs, where pairs for the actual Wilcoxon test are by trial number
    wilcoxon_p_vals_dataframe = statistical_tests_1.calculate_wilcoxon_dataframe_all(voxelwise_all_trials_dosim_df, "Patient ID", "Bx index", "Bx ID", "MC trial num", "Voxel index", "Dose (Gy)")

    wilcoxon_heatmaps_dir = statistical_tests_1_dir.joinpath("wilcoxon_heatmaps")
    os.makedirs(wilcoxon_heatmaps_dir, exist_ok=True)

    statistical_tests_1.create_heatmaps_by_biopsy(wilcoxon_p_vals_dataframe, "Patient ID", "Bx index", "Bx ID", save_dir = wilcoxon_heatmaps_dir)
    



    # Cohens d between each voxel pair for each biopsy
    eff_sizes = ['cohen', 'hedges', 'r', 'pointbiserialr', 'eta-square', 'CLES']
    for eff_size in eff_sizes:
        eff_size_df = statistical_tests_1.create_eff_size_dataframe(voxelwise_all_trials_dosim_df, "Patient ID", "Bx index", "Bx ID", "Voxel index", "Dose (Gy)", eff_size=eff_size)

        eff_size_heatmaps_dir = statistical_tests_1_dir.joinpath(f"{eff_size}_heatmaps")
        os.makedirs(wilcoxon_heatmaps_dir, exist_ok=True)

        statistical_tests_1.plot_eff_size_heatmaps(eff_size_df, "Patient ID", "Bx index", "Bx ID", "Effect Size", eff_size, save_dir=eff_size_heatmaps_dir)

    
    print('test')


if __name__ == "__main__":
    main()


    