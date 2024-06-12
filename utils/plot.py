import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import numpy as np
import os


plt.rcParams['figure.dpi'] = 300 
plt.rcParams['font.size'] = 16
#plt.rcParams['axes.titlesize'] = 12    # only for title
label_fontsize=22
plt.rcParams['axes.labelsize'] = label_fontsize
fig_w, fig_h = 6, 4     # custome width and height for plots

def error_plot(mat, ylab, is_patient):
    """
    Create an error plot.

    Parameters:
        - mat (numpy.ndarray): The data matrix to be plotted.
        - ylab (str): The label for the y-axis.
        - is_patient (bool): whether this is per-patient data (for plotting correct legend)

    Returns:
        None
    """
    # Create x values from 1 to the number of columns in the transposed data matrix
    x = np.arange(0, mat.T.shape[0])

    # Calculate mean, maximum, and minimum values along the columns of the data matrix
    mat_mean = mat.mean(axis=0)
    mat_max = mat.max(axis=0)
    mat_min = mat.min(axis=0)

    # Plot the transposed data matrix with alpha transparency
    plt.plot(x, mat.T, alpha=0.3)

    # Plot the mean line in black with a specified linewidth
    plt.plot(x, mat_mean, color='black', linewidth=2)

    # Fill the area between the minimum and maximum values with gray color and specified transparency
    plt.fill_between(x, mat_min, mat_max, color='gray', alpha=0.1)

    # Set labels and adjust layout
    plt.ylabel(ylab, fontsize=label_fontsize)
    plt.xlabel("Years", fontsize=label_fontsize)
    plt.tight_layout()

    # Set x-axis tick positions
    plt.xticks(range(0, mat.T.shape[0]))

    # Create custom legend lines
    patient_line = mlines.Line2D([], [], color='blue', label='Patients (color)')
    mean_line = mlines.Line2D([], [], color='black', label='Mean')
    if is_patient:
        plt.legend(handles=[mean_line], loc='best') # just use mean line for legend
    else:
        plt.legend(handles=[patient_line, mean_line], loc='best')


def plot_trajectories(mtl_load, ftl_load, mtl_energy, ftl_energy, mtl_h, ftl_h, output_dir, method, score_type):
    """
    This function creates a multi-subplot figure, each subplot representing a different data category.
    It uses the `error_plot` function to plot each data category and then saves each subplot as a separate PNG file.

    Args:
        mtl_load (numpy.ndarray): MTL/HC (Medial Temporal Lobe) cognition load i.e. I(t-1)_HC
        ftl_load (numpy.ndarray): FTL/PFC (Frontal Temporal Lobe) cognition load. i.e. I(t-1)_PFC
        mtl_energy (numpy.ndarray): MTL energy.
        ftl_energy (numpy.ndarray): Frontal energy.
        mtl_h (numpy.ndarray): MTL health (size).
        ftl_h (numpy.ndarray): Frontal health (size).
        output_dir (str): Directory where the generated plots will be saved.

    Returns:
        None
    """
    is_patient = False
    if 'RID' in output_dir:
        # print("This is per-patient data")
        is_patient = True
    else:
        # print("This is train/valid/test/per-year data")
        pass

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    verbose = False
    
    # Create a new figure with a specified size
    # fig = plt.figure(figsize=(fig_w, fig_h)))
    fig_w, fig_h  = 4, 4
    label_fontsize = 20

    # Subplot 1: Plot cognition using the error_plot function
    cognition = mtl_load + ftl_load
    plt.figure(figsize=(fig_w, fig_h)) 
    error_plot(cognition[:, :], "Cognition $C(t)$", is_patient)
    stats_cog = f"Mean Total Cognition:{np.round(cognition.sum(axis=1).mean()/11, 2)}"
    #plt.title(stats_cog)
    if verbose: print(stats_cog)
    plt.xticks(range(11))
    plt.ylim([0, 10])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, f"{method}_{score_type}_cognition.pdf")
    ##plt.savefig(save_path)
    plt.close()

    # Subplot 2: Plot total energy
    mtl_energy = np.where(mtl_energy > 6, 5, mtl_energy)
    ftl_energy = np.where(ftl_energy > 6, 5, ftl_energy)
    total_energy = mtl_energy + ftl_energy
    plt.figure(figsize=(fig_w, fig_h))  # Create a new figure for the next subplot
    error_plot(total_energy, "Energy Cost $M(t)$", is_patient)
    stats_energy = f"Mean Energy Cost:{np.round(total_energy.sum(axis=1).mean()/11, 2)}"
    #plt.title(stats_energy)
    if verbose: print(stats_energy)
    plt.xticks(range(11))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, f"{method}_{score_type}_energy_cost.pdf")
    ##plt.savefig(save_path)
    plt.close()

    # Subplot 3: Plot activity
    plt.figure(figsize=(fig_w, fig_h))  # Create a new figure for the next subplot
    sns.lineplot(data=np.mean(mtl_energy, axis=0), marker="o", label="$Y_{HC}(t)$")
    sns.lineplot(data=np.mean(ftl_energy, axis=0), marker="o", label="$Y_{PFC}(t)$")
    stats_activity = f"Mean Brain Activity, HC:{np.round(np.mean(mtl_energy), 2)} PFC:{np.round(np.mean(ftl_energy), 2)}"
    #plt.title(stats_activity)
    if verbose: print(stats_activity)
    plt.legend(fontsize=14)
    plt.xlabel("Years", fontsize=label_fontsize)
    plt.xticks(range(11))
    plt.ylabel("Activity $Y_{v}(t)$", fontsize=label_fontsize)
    plt.ylim([0, 5])
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{method}_{score_type}_region_activity.pdf")
    plt.savefig(save_path)
    plt.close()

    # Subplot 4: Plot Information Processed
    plt.figure(figsize=(fig_w, fig_h))  # Create a new figure for the next subplot
    sns.lineplot(data=np.mean(mtl_load, axis=0), marker="o", label="$I_{HC}(t)$")
    sns.lineplot(data=np.mean(ftl_load, axis=0), marker="o", label="$I_{PFC}(t)$")
    stats_info = f"Mean Information Processed, HC:{np.round(np.mean(mtl_load), 2)} PFC:{np.round(np.mean(ftl_load), 2)}"
    #plt.title(stats_info)
    if verbose: print(stats_info)
    plt.legend()
    plt.xlabel("Years", fontsize=label_fontsize)
    plt.xticks(range(11))
    plt.ylabel("Info Processing $I_{v}(t)$", fontsize=label_fontsize)
    plt.ylim([0, 10])
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{method}_{score_type}_region_information.pdf")
    plt.savefig(save_path)
    plt.close()

    # Subplot 5: Plot Brain Size
    plt.figure(figsize=(fig_w, fig_h))  # Create a new figure for the next subplot
    sns.lineplot(data=np.mean(mtl_h, axis=0), marker="o", label="$X_{HC}(t)$")
    sns.lineplot(data=np.mean(ftl_h, axis=0), marker="o", label="$X_{PFC}(t)$")
    stats_brain_size = f"Mean Brain Size, HC:{np.round(np.mean(mtl_h), 2)} PFC:{np.round(np.mean(ftl_h), 2)}"
    #plt.title(stats_brain_size)
    if verbose: print(stats_brain_size)
    plt.legend()
    plt.xlabel("Years", fontsize=label_fontsize)
    plt.xticks(range(11))
    plt.ylabel("Brain Size $X_{v}(t)$", fontsize=label_fontsize)
    plt.ylim([0, 5])
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{method}_{score_type}_region_size.pdf")
    ##plt.savefig(save_path)
    plt.close()


def plot_comparison(df, method, name, var_rl, var_gt, y_min, y_max, filepath):
    """
    Plot predicted cognition scores against actual cognition scores.

    Parameters:
        df (pandas.DataFrame): DataFrame containing the columns 'cogsc' and 'cogsc_rl'.

    Returns:
        None
    """
    labelpad = 1

    plt.figure(figsize=(4, 6))
    sns.lineplot(data=df, x="Years", y=var_rl, color="darkorange", marker="o", label=method)
    sns.lineplot(data=df, x="Years", y=var_gt, color="black", marker="X", linestyle="--", label='Ground Truth')
    plt.legend(loc='best', fontsize=18)
    #plt.title('Actual vs Predicted: {name}')
    #plt.title(f"Mean GT:{np.round(df[var_gt].mean(), 2)}, Mean RL:{np.round(df[var_rl].mean(), 2)}")
    plt.ylim([y_min, y_max])
    plt.xlabel('Years', fontsize=label_fontsize) 
    plt.ylabel(f"{name}", fontsize=label_fontsize, labelpad=labelpad)
    plt.xticks(range(11))
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

def plot_adni(df, filepath, type='', score_type='MMSE'):
    """
    Generate individual plots for ADNI data.

    Parameters:
        - df (pandas.DataFrame): Input DataFrame containing ADNI data.
        - filepath (str): File path for saving the plot.
        - type (str, optional): Whether to use ground truth data (type='') or RL predicted data (type='_rl').
        - score_type (str, optional): Which cogsc score to use ('MMSE', 'ADAS11', 'ADAS13')

    Returns:
        None
    """
    # Calculate the number of unique participants
    n = len(np.unique(df['RID']))

    # Create a color palette for plotting
    palette = sns.color_palette("Paired", n)

    # Create a figure with a specific size
    plt.figure(figsize=(fig_w, fig_h))

    # Subplot 1: Plot cognitive scores over time for each participant (uses the hue='RID' parameter, creates a separate line for each unique value in the 'RID' column of the df)
    # plt.subplot(1, 2, 1)
    if type == '_rl':
        sns.lineplot(data=df, x="Years", y="cogsc_rl", hue='RID', palette=palette, marker="o", errorbar=None) 
        #plt.title(f"Mean Cognition:{np.round(df['cogsc_rl'].mean(), 2)}")
        plt.ylabel("Cognition Score (RL Predicted)", fontsize=label_fontsize) 
    else: # ground truth
        sns.lineplot(data=df, x="Years", y="cogsc", hue='RID', palette=palette, marker="o")
        #plt.title(f"Mean Cognition:{np.round(df['cogsc'].mean(), 2)}")
        plt.ylabel(f'Cognition Score ({score_type})', fontsize=label_fontsize)
    plt.legend([], [], frameon=False)
    plt.xticks(range(11))
    plt.ylim([0, 11])
    #plt.ylim([df['cogsc'].min() + 1, df['cogsc'].max()])

    # # Subplot 2: Plot the mean total cognition score over time for all participants
    # plt.subplot(1, 2, 2)
    # if type == "_rl":
    #     sns.lineplot(data=df, x="Years", y="cogsc_rl", marker="o") 
    #     plt.title(f"Mean Cognition:{np.round(df['cogsc_rl'].mean(), 2)}")
    #     plt.ylabel("Cognition Score (RL Predicted)") 
    # else: # ground truth
    #     sns.lineplot(data=df, x="Years", y="cogsc", marker="o")
    #     plt.title(f"Mean Cognition:{np.round(df['cogsc'].mean(), 2)}")
    #     plt.ylabel(f'Cognition Score ({score_type})')
    # plt.legend([], [], frameon=False)
    # plt.xticks(range(11))
    # plt.ylim([0, 11])
    
    plt.tight_layout()    
    # Save the plot to the specified file path
    plt.savefig(filepath)
    plt.close()

def plot_adni_mean(df, filepath, method, score_type='MMSE'):
    """
    Generate mean plots for ADNI data.

    Parameters:
        - df (pandas.DataFrame): Input DataFrame containing ADNI data.
        - filepath (str): File path for saving the plot.
        - score_type (str, optional): Which cogsc score to use ('MMSE', 'ADAS11', 'ADAS13')

    Returns:
        None
    """
    # Create a figure with a specific size
    plt.figure(figsize=(fig_w, fig_h))

    # Plot mean ground truth vs rl predictions
    sns.lineplot(data=df, x="Years", y="cogsc_rl", color="darkorange", marker="o", label=method)
    sns.lineplot(data=df, x="Years", y="cogsc", color="black", marker="X", linestyle="--", label='Ground Truth')
    #plt.title(f"Mean Cognition - GT:{np.round(df['cogsc'].mean(), 2)}, RL:{np.round(df['cogsc_rl'].mean(), 2)}")
    plt.ylabel(f'Cognition Score ({score_type})', fontsize=label_fontsize)
    plt.xticks(range(11))
    plt.ylim([0, 11])
    plt.legend()
    plt.tight_layout() 
    # Save the plot to the specified file path
    plt.savefig(filepath)
    plt.close()

def plot_adni_3in1(df, filepath, method, score_type='MMSE'):
    """
    Generates 3 plots for ADNI data. Individual ground truth trajectories, mean RL vs Ground Truth, and individual RL predicted trajectories.

    Parameters:
        - df (pandas.DataFrame): Input DataFrame containing ADNI data.
        - filepath (str): File path for saving the plot.
        - score_type (str, optional): Which cogsc score to use ('MMSE', 'ADAS11', 'ADAS13')

    Returns:
        None
    """
    # Calculate the number of unique participants
    n = len(np.unique(df['RID']))

    # Create a color palette for plotting
    palette = sns.color_palette("Paired", n)

    # Create a figure with a specific size
    plt.figure(figsize=(15, 4))
    labelpad = 1
    max_y = 10.5
    plt.subplots_adjust(wspace=0.5)

    # Define the font size for axes and legend
    label_fontsize = 16
    ticks_fontsize = 10
    legend_fontsize = 14

    # Subplot 1: Plot cognitive scores for Ground Truth over time for each participant (uses the hue='RID' parameter, creates a separate line for each unique value in the 'RID' column of the df)
    plt.subplot(1, 3, 1)
    sns.lineplot(data=df, x="Years", y="cogsc", hue='RID', palette=palette, marker="o")
    #plt.title(f"Mean Cognition:{np.round(df['cogsc'].mean(), 2)}")
    plt.xlabel('Years', fontsize=label_fontsize)
    plt.ylabel(f'Ground Truth ({score_type})', fontsize=label_fontsize, labelpad=labelpad)
    plt.legend([], [], frameon=False, fontsize=legend_fontsize)
    plt.xticks(range(11), fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.ylim([0, max_y])
    #plt.ylim([df['cogsc'].min() + 1, df['cogsc'].max()])

    # Subplot 2: Plot mean of cognitive scores for RL vs Ground Truth
    plt.subplot(1, 3, 2)
    sns.lineplot(data=df, x="Years", y="cogsc_rl", color="darkorange", marker="o", label=method)
    sns.lineplot(data=df, x="Years", y="cogsc", color="black", marker="X", linestyle="--", label='Ground Truth')
    #plt.title(f"Mean Cognition - GT:{np.round(df['cogsc'].mean(), 2)}, RL:{np.round(df['cogsc_rl'].mean(), 2)}")
    plt.xlabel('Years', fontsize=label_fontsize)
    plt.ylabel(f'Mean Cognition', fontsize=label_fontsize, labelpad=labelpad)
    plt.xticks(range(11), fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.ylim([0, max_y])
    plt.legend(fontsize=legend_fontsize)

    # Subplot 3: Plot cognitive scores for RL over time for each participant (uses the hue='RID' parameter, creates a separate line for each unique value in the 'RID' column of the df)
    plt.subplot(1, 3, 3)
    sns.lineplot(data=df, x="Years", y="cogsc_rl", hue='RID', palette=palette, marker="o", errorbar=None) 
    #plt.title(f"Mean Cognition:{np.round(df['cogsc_rl'].mean(), 2)}")
    plt.xlabel('Years', fontsize=label_fontsize)
    plt.ylabel("Predicted Cognition (RL)", fontsize=label_fontsize, labelpad=labelpad)
    plt.legend([], [], frameon=False, fontsize=legend_fontsize)
    plt.xticks(range(11), fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.ylim([0, max_y])
    
    plt.tight_layout()    
    # Save the plot to the specified file path
    plt.savefig(filepath, dpi=150)
    plt.close()

def plot_synthetic(df, filepath, type=''):
    """
    Generate individual plots for Synthetic data for ground truth and RL predictions.

    Parameters:
        - df (pandas.DataFrame): DataFrame containing the synthetic data.
        - filepath (str): Filepath to save the generated plot.
        - type (str, optional): Whether to use ground truth data (type='') or RL predicted data (type='_rl').

    Returns:
        None
    """
    # Calculate 'reward' based on specific data columns
    # R(t) = −[λ|Ctask − C(t)| + M(t)] where C_task is set to 5 for synthetic data (10 for ADNI data)
    df['reward'] = (-np.abs(df['reg1_info'+type] + df['reg2_info'+type] - 5) - (df['reg1_fdg'+type] + df['reg2_fdg'])).values
    
    # Define a color palette
    palette = sns.color_palette("Spectral", as_cmap=True)
    
    # Create a new figure with a specified size
    fig = plt.figure(figsize=(fig_w, fig_h))
    
    # Subplot 1: Plot the combined cognition
    plt.subplot(2, 2, 1)
    sns.lineplot(data=df, x="Years", y="cogsc"+type, hue='RID', palette=palette, marker="o")
    sns.lineplot(data=df, x="Years", y="cogsc"+type, marker="o", color="black", linewidth="2.5")
    field = "cogsc"+type
    plt.title("Mean Total Cognition:" + str(np.round(df[field].mean(), 2)))
    print("Mean Total Cognition:" + str(np.round(df[field].mean(), 2)))
    plt.legend([], [], frameon=False)
    
    # Subplot 2: Plot MTL/HC (reg1) information load 
    plt.subplot(2, 2, 2)
    sns.lineplot(data=df, x="Years", y="reg1_info"+type, hue='RID', palette=palette, marker="o")
    sns.lineplot(data=df, x="Years", y="reg1_info"+type, marker="o", color="black", linewidth="2.5")
    plt.title("Mean Total HC Load:" + str(np.round(df['reg1_info' + type].mean(), 2)))
    print("Mean Total HC Load:" + str(np.round(df['reg1_info' + type].mean(), 2)))
    plt.legend([], [], frameon=False)
    #plt.ylim([0, 11])

    # Subplot 3: Plot FTL/PFC information load 
    plt.subplot(2, 2, 3)
    #plt.plot(ftl_load.T)
    sns.lineplot(data=df, x="Years", y="reg2_info"+type, hue='RID', palette=palette, marker="o")
    sns.lineplot(data=df, x="Years", y="reg2_info"+type, marker="o", color="black", linewidth="2.5")
    plt.title("Mean Total PFC Load:" + str(np.round(df['reg2_info' + type].mean(), 2)))
    print("Mean Total PFC Load:" + str(np.round(df['reg2_info' + type].mean(), 2)))
    plt.legend([], [], frameon=False)
    #plt.ylim([0, 11])

    # Subplot 4: Plot total energy
    # A few patients (2) have erroneous values (~25000) in the energy matrix (mean ~2.0, max is aroung 5.5). 
    # Replace rows with any energy value greater than 6 with 5 (approx mean) in the second dimension from the plots
    mtl_energy = df['reg1_fdg'+type]
    ftl_energy = df['reg2_fdg'+type]
    mtl_energy = np.where(mtl_energy > 6, 5, mtl_energy)
    ftl_energy = np.where(ftl_energy > 6, 5, ftl_energy)
    
    # Calculate the total energy by adding MTL and Frontal energy 
    df['total_energy'] =  mtl_energy + ftl_energy
    plt.subplot(2, 2, 4)
    sns.lineplot(data=df, x="Years", y="total_energy", hue='RID', palette=palette, marker="o")
    sns.lineplot(data=df, x="Years", y="total_energy", marker="o", color="black", linewidth="2.5")
    plt.title(f"Mean Total Metabolic Cost:{np.round(df['total_energy'].mean(), 2)}")
    print(f"Mean Total Metabolic Cost:{np.round(df['total_energy'].mean(), 2)}")
    plt.legend([], [], frameon=False)
    #plt.ylim([3, 20])

    plt.tight_layout()
    # Save the figure to the specified filepath
    plt.savefig(filepath)

def plot_patient(df, filepath, method, score='MMSE'):
    """
    Generate individual patient plots for ADNI data.

    Parameters:
        - df (pandas.DataFrame): Input DataFrame containing ADNI data.
        - filepath (str): File path for saving the plot.
        - score (str, optional): Which cogsc score to use ('MMSE', 'ADAS11', 'ADAS13') 

    Returns:
        None
    """
    color_rl = "darkorange"
    color_groundtruth = "black"
    patient_fig_w, patient_fig_h = 5, 6
    legend_fontsize = 14

    # Subplot 1: Plot cognitive scores over time for each participant
    fig1 = plt.figure(figsize=(patient_fig_w, patient_fig_h))
    sns.lineplot(data=df, x="Years", y="cogsc_rl", color=color_rl, marker="o", label=method)
    sns.lineplot(data=df, x="Years", y=score + "_norm", color=color_groundtruth, marker="X", linestyle="--", label='Ground Truth')
    plt.legend(fontsize=legend_fontsize)
    #plt.title(f"Total Cognition")
    plt.ylim([0, 10])
    plt.ylabel("MMSE Score" if "MMSE" in score else "ADAS13 Score", fontsize=label_fontsize)
    plt.xticks(range(11))
    plt.tight_layout()
    plt.savefig(filepath + '_cognition.pdf', bbox_inches='tight')

    # Subplot 2: Plot HC size over time for each participant
    fig2 = plt.figure(figsize=(patient_fig_w, patient_fig_h))
    sns.lineplot(data=df, x="Years", y="reg1_mri_rl", color=color_rl, marker="o", label=method)
    sns.lineplot(data=df, x="Years", y="mri_HIPPO_norm", color=color_groundtruth, marker="X", linestyle="--", label='Ground Truth')
    #plt.legend(loc='lower left')
    #plt.title(f"Hippocampus Size")
    plt.ylim([0, 5])
    plt.ylabel("Hippocampus Size", fontsize=label_fontsize)
    plt.xticks(range(11))
    plt.tight_layout()
    #plt.legend().remove()
    ##plt.savefig(filepath + '_hc_size.pdf', bbox_inches='tight')

    # Subplot 3: Plot PFC size over time for each participant
    fig3 = plt.figure(figsize=(patient_fig_w, patient_fig_h))
    sns.lineplot(data=df, x="Years", y="reg2_mri_rl", color=color_rl, marker="o", label=method)
    sns.lineplot(data=df, x="Years", y="mri_FRONT_norm", color=color_groundtruth, marker="X", linestyle="--", label='Ground Truth')
    #plt.legend(loc='lower left')
    #plt.title(f"Prefrontal Cortex Size")
    plt.ylim([0, 5])
    plt.ylabel("Prefrontal Cortex Size", fontsize=label_fontsize)
    plt.xticks(range(11))
    plt.tight_layout()
    #plt.legend().remove()
    ##plt.savefig(filepath + '_pfc_size.pdf', bbox_inches='tight')

    plt.close()


def plot_all_methods(rl_methods_df, algos, plot_best, filepath, score_type):
    """
    Plots cognition score for all RL methods, supervised baselines and ground truth.

    Parameters:
    - rl_methods_df (pandas.DataFrame): A DataFrame containing simulated data for each RL method
    - algos (list): A list of the names of the RL algorithms.
    - plot_best (bool, optional): If True, only the best performing RL method will be plotted alongwith supervised baselines. 
                                If False, all RL methods will be plotted.
    - filepath (str, optional): The path where the plot will be saved.

    Returns:
        None
    """
        
    plt.figure(figsize=(8,6))

    # Values for MiniRNN and SVR estimated from Fig 5 of Saboo et al (2021)
    # Code for these methods at https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/predict_phenotypes/Nguyen2020_RNNAD
    sb_values = {
        "minimalRNN": {
            "x-values": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "mean": np.array([10, 9.996, 9.982, 9.97, 9.963, 9.956, 9.944, 9.937, 9.926, 9.919]),
            "std": np.array([0, 0.004, 0.007, 0.0085, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015])
        },
        "SVR": {
            "x-values": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "mean": np.array([9.637, 9.630, 9.604, 9.5852, 9.5481, 9.5259, 9.5148, 9.5778, 9.7111, 8.8]),
            "std": np.array([0.0408, 0.0408, 0.0408, 0.0408, 0.0408, 0.0408, 0.0408, 0.0408, 0.0408, 0.0408])
        }
    }

    # Define a colorblind-friendly palette, markers
    palette = sns.color_palette("colorblind", 8)
    markers = ['s', 'D', 'o', 'P', '^', '<', '>', 'p', 'v']

    # Create a dictionary to map each algorithm to a color
    color_dict = dict(zip(algos, palette))
    marker_dict = dict(zip(algos, markers))
    
    # Temporary fix to plot supervised baselines with RL-LSTMS and not best RL method
    only_LSTMs = False

    # Plot the best performing RL method with Supervised Baselines
    if plot_best:
        try:
            print("\tPlotting only TRPO with Supervised Baselines")

            # Plot miniRNN
            x = sb_values["minimalRNN"]["x-values"]
            y = sb_values["minimalRNN"]["mean"]
            std = sb_values["minimalRNN"]["std"]
            sns.lineplot(x=x, y=y, marker="s", legend="full", label="miniRNN")
            plt.fill_between(x, y-std, y+std, alpha=0.3)

            # Plot SVR
            x = sb_values["SVR"]["x-values"]
            y = sb_values["SVR"]["mean"]
            std = sb_values["SVR"]["std"]
            sns.lineplot(x=x, y=y, marker="o", legend="full", label="SVR")
            plt.fill_between(x, y-std, y+std, alpha=0.3)
        
            # When required to plot LSTMs with supervised baselines
            if only_LSTMs:
                sns.lineplot(data=rl_methods_df['TRPO-LSTM'], x="Years", y="cogsc_rl", marker="D", label='TRPO-LSTM')
                sns.lineplot(data=rl_methods_df['PPO-LSTM'], x="Years", y="cogsc_rl", marker="P", label='PPO-LSTM')
                #plt.yticks(np.arange(4.5, 10.1))    # Set y-axis ticks to 8.5, 9.0, 9.5, 10.0
                plt.ylim(4.1, 10.1)
            else:
                # plot TRPO
                sns.lineplot(data=rl_methods_df['TRPO'], x="Years", y="cogsc_rl", marker="D", label='TRPO')
                
                # plt.xticks(np.arange(0, 11, 2))        # Set x-axis ticks to 0, 2, 4, 6, 8, 10
                # plt.ylim(0, 10.1)
                plt.xticks(range(11))           
                plt.xlim(-0.5, 10.5)
                plt.yticks(np.arange(8.5, 10.1, 0.5))    # Set y-axis ticks to 8.5, 9.0, 9.5, 10.0
                plt.ylim(8.1, 10.1)
            
            # plot ground truth (we can use any df since all have the same ground truth values)
            sns.lineplot(data=rl_methods_df[algos[0]], x="Years", y="cogsc", color="black", 
                        markersize=8, marker="X", linestyle="--", label='Ground Truth')
    
        except Exception as e:
            print(f"\tBaselines plotted without Best RL method since it is hardcoded to be 'TRPO' "
                  f"which was not found in the list of algos. Here is the error: {e}")

    # Plot all RL method without Supervised Baselines
    else:
        print("Plotting all RL methods")
        order = ["TRPO", "PPO", "DDPG", "SAC"]
        ordered_algos = [algo for algo in order if algo in algos]
        for algo in ordered_algos:
            print(f"\tPlotting {algo}")
            sns.lineplot(data=rl_methods_df[algo], x="Years", y="cogsc_rl", color=color_dict[algo], 
                         marker=marker_dict[algo], label=f'{algo}')

        # plot ground truth (we can use any df since all have the same ground truth values)
        sns.lineplot(data=rl_methods_df[algo], x="Years", y="cogsc", color="black", 
                     markersize=8, marker="X", linestyle="--", label='Ground Truth')
        
        plt.xticks(range(11))
        plt.ylim(0, 10.1)
        plt.xlim(-0.5, 10.5)
        #plt.ylim(0, 10.1)
        #plt.xticks(np.arange(0, 11, 2))          # Set x-axis ticks to 0, 2, 4, 6, 8, 10
        #plt.yticks(np.arange(0.0, 10.1, 0.5))    # Set y-axis ticks to 8.5, 9.0, 9.5, 10.0

    plt.xlabel("Years", fontsize=label_fontsize)
    plt.ylabel(f'Cognition Score ({score_type})', fontsize=label_fontsize)
    plt.legend(loc="lower left", fontsize=18)
    #plt.grid()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def plot_graphic(filename='cognition'):
    """
    Plot a simple line graph with a legend and labels for use in infographic.
    """
    
    # dummy RL predictions
    RL_dummy = {
    "RL": {
        "x-values": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "mean": np.array([9.5, 9.1, 8.6, 8.3, 8.1, 7.8, 7.6, 7.4, 7.2, 7.0, 6.8]),
        "std": np.array([0.5, 0.45, 0.5, 0.4, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2])
        },
    }
    # dummy ground truth
    GT = [9.5, 9.1, 8.5, None, 8.2, 7.5, 7, None, 7.1, 6.5, 6]

    color_rl = "darkorange"
    color_groundtruth = "darkblue"
    fig_w, fig_h = 4, 4

    plt.figure(figsize=(fig_w, fig_h))
    x = RL_dummy["RL"]["x-values"]
    y = RL_dummy["RL"]["mean"]
    std = RL_dummy["RL"]["std"]

    sns.lineplot(x=x, y=y, color=color_rl, marker="o", label="RL")
    plt.fill_between(x, y-std, y+std, color=color_rl, alpha=0.3)
    sns.lineplot(x=x, y=GT, color=color_groundtruth, marker="o", linestyle="--", label='Ground Truth')
    
    plt.legend(loc='best', fontsize=12)
    plt.xlabel("Years", fontsize=16)
    plt.ylabel("Cognition", fontsize=16)
    plt.xticks(np.arange(0, 11, 2))        # Set x-axis ticks to 0, 2, 4, 6, 8, 10
    plt.tick_params(axis='x', labelsize=10)
    plt.tick_params(axis='y', labelsize=10) 
    plt.xlim(-0.5, 10.5)
    plt.ylim([4, 10])
    plt.tight_layout()
    plt.savefig(f'{filename}.png')