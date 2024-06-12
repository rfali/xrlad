import os, shutil, time, sys
import pandas as pd
import numpy as np
rootpath = os.path.join(os.getcwd())
sys.path.insert(0, rootpath)
from utils.patient_filter import filter_patient

# Add parent directory to the system path so that the utils module can be imported for standalone execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.plot import plot_comparison, plot_adni, plot_adni_mean, plot_adni_3in1, plot_trajectories, plot_patient, plot_all_methods

def process_rl_plots(algos, score='MMSE'):
    """
    Process and call the plot functions on passed RL methods.

    It reads individual results files, combines them into a single dataframe per algorithm, and then generates various plots
    for each algorithm. These plots include comparisons of RL vs Ground Truth trajectories, common data points, all data points,
    RL trajectories for each variable, and per-patient trajectory plots. It also saves the combined dataframe and common data points
    dataframe to Excel files.

    Parameters:
    algos (list): A list of strings representing the names of the RL algorithms to process.
    score (str): The score type to be used in the plots. It can be 'MMSE' or 'ADAS13'.

    Returns:
    None
    """

    results_path = 'results/'
    plots_dir = 'plots_rl'
    directories = ['', '/all', '/common', '/comparison', '/rl_trajectories', '/per_patient']
    files_read = 0

    combined_df = {algo: pd.DataFrame() for algo in algos}
    common_df = {algo: pd.DataFrame() for algo in algos}
    rows = {algo: 0 for algo in algos}

    if os.path.isdir(plots_dir):
        shutil.rmtree(plots_dir)
    os.makedirs(plots_dir)

    print('Reading all individual results files and combining them into single dataframe per algo')
    for root, dirs, files in os.walk(results_path):
        for file in files:
            if file.endswith('.xlsx'):
                file_path = os.path.join(root, file)
                try:
                    temp_df = pd.read_excel(file_path, sheet_name='test')
                    print(f"Successfully read Excel file: {file_path}")
                    files_read += 1
                    for algo in algos:
                        algo_in_filename = file.split('_')[0]       # to deal with algo names that overlap e.g. PPO-LSTM and PPO
                        if algo == algo_in_filename:
                            combined_df[algo] = combined_df[algo].append(temp_df, ignore_index=True)
                            rows[algo] += temp_df.shape[0]
                except Exception as e:
                    print(f"Error reading Excel file {file_path}: {e}")

    print(f'\nRead {files_read} files in total\n')

    for algo in algos:
        print(f'\n********** Processing data for {algo} **********')

        for directory in directories:
            os.makedirs(f'{plots_dir}/{algo}/{directory}', exist_ok=True)

        assert rows[algo] == combined_df[algo].shape[0], f"Number of rows in combined dataframe ({combined_df[algo].shape[0]}) does not match number of rows in the sum of all dataframes ({rows[algo]})"
        print(f'sum of rows {algo}: {rows[algo]}, rows in combined dataframe {algo}: {combined_df[algo].shape[0]}')

        print(f'Saving combined df to an excel file at {plots_dir}/{algo}/combined_results_{algo}.xlsx')            
        combined_df[algo].to_excel(f'{plots_dir}/{algo}/combined_results_{algo}.xlsx', sheet_name='test', index=False)

        # ********** Create DF with Common Data Points **********
        # Create a new df with common data points (cognition scores), that are available in ground truth data (not all patients have cog score for 11 years) and RL predictions (RL has all year predictions)
        # RL predicts for every year, but ground truth is not available for each year. Simply remove rows with no 'cogsc' value.
        common_df[algo] = combined_df[algo].copy(deep=True)                             # create a deep copy
        common_df[algo]['cogsc'].replace(r'^\s*$', pd.NA, regex=True, inplace=True)   # Replace blank values of 'cogsc' with NaN
        common_df[algo].dropna(subset=['cogsc'], inplace=True)                        # use dropna to drop these rows
        common_df[algo].to_excel(f'{plots_dir}/{algo}/common_results_{algo}.xlsx', sheet_name='test', index=False)

        # ********** Plotting Common Data Points **********
        # print(f'{algo}: Common data points: plotting Ground Truth...')
        # plot_adni(common_df[algo], f'{plots_dir}/{algo}/common/{algo}_{score}_groundtruth.pdf', score_type=score)
        # print(f'{algo}: Common data points: plotting RL predictions...')
        # plot_adni(common_df[algo], f'{plots_dir}/{algo}/common/{algo}_{score}_rl_pred.pdf', type='_rl', score_type=score)
        # print(f'{algo}: Common data points: plotting Mean Ground Truth vs RL Predictions...')
        # plot_adni_mean(common_df[algo], f'{plots_dir}/{algo}/common/{algo}_{score}_mean.pdf', score_type=score)
        print(f'{algo}: Common data points: plotting 3in1...')
        plot_adni_3in1(common_df[algo], f'{plots_dir}/{algo}/common/{algo}_{score}_common_3in1.pdf', method=algo, score_type=score)
        
        # ********** Plotting RL vs Ground Truth Trajectories using Common Data Points **********
        print(f'{algo}: Plotting Comparison of RL vs Ground Truth comparisons using common data points...')
        plot_comparison(common_df[algo], method=algo, name="Cognition", var_rl="cogsc_rl", var_gt=score + "_norm", y_min= 0, y_max=11, filepath=f'{plots_dir}/{algo}/comparison/{algo}_{score}_comp_cognition.pdf')
        plot_comparison(common_df[algo], method=algo, name="Hippocampus Size", var_rl="reg1_mri_rl", var_gt="mri_HIPPO_norm", y_min= 0, y_max=5, filepath=f'{plots_dir}/{algo}/comparison/{algo}_{score}_comp_size-hc.pdf')
        plot_comparison(common_df[algo], method=algo, name="Prefrontal Cortex Size", var_rl="reg2_mri_rl", var_gt="mri_FRONT_norm", y_min= 0, y_max=5, filepath=f'{plots_dir}/{algo}/comparison/{algo}_{score}_comp_size-pfc.pdf')
        plot_comparison(common_df[algo], method=algo, name="Hippocampus Amyloid", var_rl="reg1_D_rl", var_gt="HIPPOCAMPAL_SUVR", y_min= 0, y_max=2, filepath=f'{plots_dir}/{algo}/comparison/{algo}_{score}_comp_amyloid-hc.pdf')
        plot_comparison(common_df[algo], method=algo, name="Prefrontal Cortex Amyloid", var_rl="reg2_D_rl", var_gt="FRONTAL_SUVR", y_min= 0, y_max=2, filepath=f'{plots_dir}/{algo}/comparison/{algo}_{score}_comp_amyloid-pfc.pdf')

        # ********** Plotting All Data Points **********
        # print(f'\n{algo}: All data points: plotting Ground Truth...')
        # plot_adni(combined_df[algo], f'{plots_dir}/{algo}/all/{algo}_{score}_groundtruth.pdf', score_type=score)
        # print(f'{algo}: All data points: plotting RL predictions...')
        # plot_adni(combined_df[algo], f'{plots_dir}/{algo}/all/{algo}_{score}_rl_pred.pdf', type='_rl', score_type=score)
        # print(f'{algo}: All data points: plotting Mean Ground Truth vs RL Predictions...')
        # plot_adni_mean(combined_df[algo], f'{plots_dir}/{algo}/all/{algo}_{score}_mean.pdf', score_type=score)
        print(f'{algo}: All data points: plotting 3in1...')
        plot_adni_3in1(combined_df[algo], f'{plots_dir}/{algo}/all/{algo}_{score}_all_3in1.pdf', method=algo, score_type=score)

        # ********** Plotting RL Trajectories for each variable **********
        num_years = 11
        years = list(range(num_years))
        feature_list = ['reg1_info_rl', 'reg2_info_rl', 'reg1_fdg_rl', 'reg2_fdg_rl', 'reg1_mri_rl', 'reg2_mri_rl']

        feature_dict = {}
        # intialize empty numpy arrays for each feature
        for feature in feature_list:
            feature_dict[feature] = np.zeros(((rows[algo] // num_years), num_years))

        # Loop over each year in the 'years' list
        for year in years:
            # Filter the dataframe for the current algorithm to include only the data for the current year
            year_data = combined_df[algo][combined_df[algo]["Years"] == year]
            # Loop over each feature in the 'feature_list'
            for feature in feature_list:
                # Loop over each row in the 'year_data' dataframe. The number of rows in 'year_data' is the number of entries for the current year
                for i in range(0, len(year_data)):
                    # Update the dictionary 'feature_dict' for the current feature
                    # The cell at the ith row and 'year' column of the array is set to the value of the current feature for the ith patient in the 'year' year
                    # The expression 'i * 11 + year' is used to calculate the index of the row in 'year_data' that corresponds to the ith patient in the 'year' year
                    feature_dict[feature][i, year] = year_data[feature][i * 11 + year]

        print(f'\n{algo}: Plotting RL trajectories for each variable...')
        plot_trajectories(feature_dict['reg1_info_rl'],   # MTL/HC cognition load 
                        feature_dict['reg2_info_rl'],     # FTL/PFC cognition load 
                        feature_dict['reg1_fdg_rl'],      # MTL/HC activity
                        feature_dict['reg2_fdg_rl'],      # FTL/PFC activity
                        feature_dict['reg1_mri_rl'],      # MTL/HC size
                        feature_dict['reg2_mri_rl'],      # FTL/PFC size 
                        f'{plots_dir}/{algo}/rl_trajectories',
                        method=algo, 
                        score_type=score)


        # ********** Plotting RL Plots for Patients filtered on a given criteria **********
        print(f'********* {algo}: Plotting per-patient trajectory plots...**********\n')

        seeds = np.unique(pd.read_csv('results/summary_adni.csv')["seed"])

        def per_patient_evaluation(filter_feature, num_patients, num_seeds=5, years_of_data=6):
            """
            Collects patient (RID) data based on a specified filter feature. 
            Plots RL predictions vs Ground Truth curves (plot_patient). Generates SHAP values and calls SHAP plots.

            Parameters:
            filter_feature (str): The feature to filter data on. Can be 
                'range' : MMSE decline in a certain range, sort by least MAE (most accurate prediction)
                'mae', : sort by least MAE (most accurate prediction) 
                cognition score: 'MMSE', 'ADAS13'.
            num_patients (int): The number of top patients to select for filtering.
            num_seeds (int): The number of seeds/experiments per patient.
            years_of_data (int): The number of years of data that the patient must have.

            Returns:
            None
            """

            sorted_rids = filter_patient(combined_df, algo, filter_feature, num_patients, num_seeds, years_of_data)

            # Evaluate the selected patients
            for rid in sorted_rids:
                # Plotting RL curves for selected patients
                df_plot = combined_df[algo].loc[combined_df[algo]['RID'] == rid]
                print(f'Plotting Patient (RL vs Ground Truth) trajectory plots for RID {rid}, feature {filter_feature}...')
                plot_dir = f'{plots_dir}/{algo}/per_patient'
                # Temporary for generating plots only for this RID (already filtered in filter_patient function)
                if rid == 4294:
                    plot_patient(df_plot, f'{plot_dir}/RID-{rid}_{algo}_{score}_{filter_feature}', method=algo, score=score)

        # Plot Ground Truth vs RL Prediction curves (Cognition, Size, and Amyloid) for the selected filter and num_patients 
        # per_patient_evaluation(f'{score}_norm', num_patients=1, num_seeds=len(seeds))     # select the top num_patients with the most decline in MMSE score
        # per_patient_evaluation('mae', num_patients=1, num_seeds=len(seeds))               # select the top num_patients with the least MAE (best prediction)
        per_patient_evaluation('range', num_patients=3, num_seeds=len(seeds))             # select the top num_patients with the MMSE decline in a certain range

    print(f'\nAll data points: Plotting baselines...')
    plot_all_methods(combined_df, algos, plot_best=True, filepath=f'{plots_dir}/{score}_baselines_all.pdf', score_type=score)
    plot_all_methods(combined_df, algos, plot_best=False, filepath=f'{plots_dir}/{score}_rl_all.pdf', score_type=score)

    print(f'\nCommon data points: Plotting baselines...')
    plot_all_methods(common_df, algos, plot_best=True, filepath=f'{plots_dir}/{score}_baselines_common.pdf', score_type=score)
    plot_all_methods(common_df, algos, plot_best=False, filepath=f'{plots_dir}/{score}_rl_common.pdf', score_type=score)

    return combined_df

if __name__ == '__main__':
    algos = ["TRPO", "PPO", "DDPG", "SAC"] 
    plot_start_time = time.time()
    process_rl_plots(algos)
    plot_end_time = time.time()
    print(f"\nTotal Plotting time: {round((plot_end_time - plot_start_time)/60, 2)} minutes")