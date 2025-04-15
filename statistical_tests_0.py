import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, truncnorm, lognorm, gamma, weibull_min, expon, pareto, rice, gengamma, kstest
import os


def histogram_and_fit(df, dists_to_try=None, bin_size=1, dose_col='Dose (Gy)', save_path=None, custom_name = None):
    """
    Histograms the values of the specified dose column, fits multiple distributions,
    and annotates the plot with best fit statistics and parameters including mean,
    standard deviation, mode, and quantiles.

    Args:
        df (pd.DataFrame): Input DataFrame with dose data.
        dose_col (str): Column name for dose values.
        save_path (str, optional): Base path to save the plot as PNG and SVG. If None, displays the plot.

    Returns:
        None
    """
    # Extract dose values
    doses = df[dose_col].dropna()
    num_trials = df['MC trial num'].max()
    num_voxels = len(df[(df['MC trial num'] == 0)])

    # Define distributions to fit
    distributions = {
        'truncnorm': truncnorm,
        'lognorm': lognorm,
        'gamma': gamma,
        'weibull_min': weibull_min,
        'expon': expon,
        'pareto': pareto,
        'rice': rice,
        'gengamma': gengamma
    }
    if dists_to_try is not None:
        distributions = {k: distributions[k] for k in dists_to_try if k in distributions}

    def perform_fits(data, distributions):
        """Fit data to multiple distributions and select the best one."""
        best_fit = None
        best_stat = float('inf')
        best_p = 0
        best_dist_name = None

        fit_results = {}
        for dist_name, dist_func in distributions.items():
            print(f'Fitting: {dist_name}')
            if dist_name == 'truncnorm':
                mean, std = np.mean(data), np.std(data)
                a, b = (0 - mean) / std, (np.inf - mean) / std
                fit = (a, b, mean, std)
                cdf_func = lambda x: truncnorm.cdf(x, a, b, loc=mean, scale=std)
            else:
                fit = dist_func.fit(data)
                cdf_func = lambda x: dist_func.cdf(x, *fit)

            # Perform KS test
            stat, p_value = kstest(data, cdf_func)
            fit_results[dist_name] = (stat, p_value, fit)

            # Update the best fit
            if stat < best_stat:
                best_stat, best_p, best_fit = stat, p_value, fit
                best_dist_name = dist_name

        return best_dist_name, best_stat, best_p, best_fit, fit_results

    # Perform fits
    best_dist, best_stat, best_p, best_fit, fit_results = perform_fits(doses, distributions)

    # Generate data for the best fit distribution
    x = np.linspace(doses.min(), doses.max(), 1000)
    if best_dist == 'truncnorm':
        pdf = truncnorm.pdf(x, *best_fit)
    else:
        pdf = distributions[best_dist].pdf(x, *best_fit)

    # Define bin edges based on bin size
    bin_edges = np.arange(start=doses.min(), stop=doses.max() + bin_size, step=bin_size)

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(doses, bins=bin_edges, density=True, alpha=0.6, color='blue', label='Histogram of Doses')

    # Plot the best fit distribution
    plt.plot(x, pdf, 'r-', label=f'Best Fit: {best_dist}\nKS stat: {best_stat:.2f}, P-value: {best_p:.2e}')

    # Annotate fit parameters, mean, and standard deviation
    mean = np.mean(doses)
    std = np.std(doses)
    
    # Calculate argmax from the density plot
    argmax_x = x[np.argmax(pdf)]

    # Calculate quantiles before annotations
    quantiles = np.percentile(doses, [5, 25, 50, 75, 95])
    quantile_labels = ['Q5', 'Q25', 'Median', 'Q75', 'Q95']
    quantiles_text = ', '.join([f'{label}: {q:.2f}' for label, q in zip(quantile_labels, quantiles)])

    # Generate annotation text
    stats_text = f'Mean: {mean:.2f}, Std: {std:.2f}, Mode: {argmax_x:.2f}\n{quantiles_text}\n$N_{{trials}}$: {num_trials}, $N_{{voxels}}$: {num_voxels}'
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))



    # Add quantiles
    quantiles = np.percentile(doses, [5, 25, 50, 75, 95])
    quantile_colors = ['red', 'blue', 'black', 'blue', 'red']
    quantile_labels = ['Q5', 'Q25', 'Q50', 'Q75', 'Q95']
    for q, color, label in zip(quantiles, quantile_colors, quantile_labels):
        plt.axvline(q, color=color, linestyle='--', label=f'{label}', linewidth=0.75)



    # Add labels and legend
    plt.xlabel(dose_col)
    plt.ylabel('Density')
    plt.title('Histogram of Dose with Best Fit Distribution')
    plt.legend()

    # Save or display the plot
    if save_path:
        if custom_name == None:
            png_path = save_path.joinpath("all_voxels_dose_histogram_fit.png")
            svg_path = save_path.joinpath("all_voxels_dose_histogram_fit.svg")
        else:
            png_path = save_path.joinpath(f"{custom_name}.png")
            svg_path = save_path.joinpath(f"{custom_name}.svg")
        plt.savefig(png_path, format='png', bbox_inches='tight')
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"Plot saved as PNG: {png_path}")
        print(f"Plot saved as SVG: {svg_path}")
    else:
        plt.show()

# Ensure all necessary libraries like scipy.stats are properly imported and used within the function.




# Example usage
# histogram_and_fit(df, save_path="dose_histogram")




def dvh_boxplot(cohort_bx_dvh_metrics_df, save_path=None, custom_name=None):
    # Filter data for D_x and V_x metrics
    d_metrics = cohort_bx_dvh_metrics_df[cohort_bx_dvh_metrics_df['Metric'].str.startswith('D_')]
    v_metrics = cohort_bx_dvh_metrics_df[cohort_bx_dvh_metrics_df['Metric'].str.startswith('V_')]

    # Function to create boxplot
    def create_plot(data, metric_type):
        plt.figure(figsize=(12, 8))
        boxplot = data.boxplot(by='Metric', column=['Mean'], grid=True)
        plt.title(f'Boxplot of (MC) Mean Values for {metric_type} Metrics')
        plt.suptitle('')
        plt.xlabel('Metric')
        if 'V' in metric_type:
            plt.ylabel('Percent volume')
        elif 'D' in metric_type:
            plt.ylabel('Dose (Gy)')
        plt.xticks(rotation=45)

        if save_path:
            filename_suffix = metric_type.lower().replace(' ', '_')
            if custom_name:
                png_path = save_path.joinpath(f"{custom_name}_{filename_suffix}.png")
                svg_path = save_path.joinpath(f"{custom_name}_{filename_suffix}.svg")
            else:
                png_path = save_path.joinpath(f"{filename_suffix}.png")
                svg_path = save_path.joinpath(f"{filename_suffix}.svg")

            plt.savefig(png_path, format='png', bbox_inches='tight')
            plt.savefig(svg_path, format='svg', bbox_inches='tight')
            print(f"Plot saved as PNG: {png_path}")
            print(f"Plot saved as SVG: {svg_path}")
        else:
            plt.show()
        plt.close()

    # Create plots for D_x and V_x metrics
    create_plot(d_metrics, 'D_x')
    create_plot(v_metrics, 'V_x')



def save_filtered_stats_to_csv(df, columns, savepath, filename, filter_condition=None):
    """
    Save statistics for selected columns based on a filter condition to a CSV file.

    Args:
    df (pd.DataFrame): The dataframe to process.
    columns (list): List of column names to include in the statistics.
    filepath (str): The full file path and filename to save the CSV.
    filter_condition (pd.Series, optional): Boolean series to filter the dataframe, defaults to None.

    Returns:
    None
    """
    # Apply the filter condition to the dataframe
    if filter_condition is None:
        filtered_df = df
    else:
        filtered_df = df[filter_condition]

    # Calculate the required statistics
    stats = filtered_df[columns].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    stats.loc['mean'] = filtered_df[columns].mean()
    stats.loc['max'] = filtered_df[columns].max()
    stats.loc['min'] = filtered_df[columns].min()
    stats.loc['std'] = filtered_df[columns].std()
    stats.loc['sem'] = filtered_df[columns].sem()
    

    # Save the statistics to a CSV file
    filepath = os.path.join(savepath, filename)
    stats.to_csv(filepath)



def save_grouped_stats_to_csv(df, columns, savepath, file_name, group_by_column, filter_condition=None):
    """
    Save statistics for selected columns based on a filter condition and group by a specific column to a CSV file.

    Args:
    df (pd.DataFrame): The dataframe to process.
    columns (list): List of column names to include in the statistics.
    savepath (str): The directory path where the CSV will be saved.
    file_name (str): The filename to save the CSV.
    group_by_column (str): Column name to group the dataframe by.
    filter_condition (pd.Series, optional): Boolean series to filter the dataframe, defaults to None.

    Returns:
    None
    """
    # Apply the filter condition to the dataframe
    if filter_condition is not None:
        df = df[filter_condition]

    # Ensure the save path exists
    os.makedirs(savepath, exist_ok=True)

    # Group by the specified column and calculate statistics for each group
    grouped_df = df.groupby(group_by_column)
    result = pd.DataFrame()

    for name, group in grouped_df:
        stats = group[columns].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        stats.loc['mean'] = group[columns].mean()
        stats.loc['max'] = group[columns].max()
        stats.loc['min'] = group[columns].min()
        stats.loc['std'] = group[columns].std()
        stats.loc['sem'] = group[columns].sem()

        # Optional: Create a multi-level index to include group name if combining all groups into one file
        stats.columns = pd.MultiIndex.from_product([[name], stats.columns])

        # Append to the result dataframe
        if result.empty:
            result = stats
        else:
            result = pd.concat([result, stats], axis=1)

    # Save the statistics to a CSV file
    filepath = os.path.join(savepath, file_name)
    result.to_csv(filepath)