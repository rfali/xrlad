import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import numpy as np
import os

font = {'size'   : 14}
matplotlib.rc('font', **font)

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
    plt.ylabel(ylab)
    plt.xlabel("Time (years)")
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


def plot_trajectories(mtl_load, ftl_load, mtl_energy, ftl_energy, mtl_h, ftl_h, output_dir):
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
        print("This is per-patient data")
        is_patient = True
    else:
        print("This is train/valid/test/per-year data")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    verbose = False
    
    # Create a new figure with a specified size
    # fig = plt.figure(figsize=(8, 6)))

    # Subplot 1: Plot cognition using the error_plot function
    cognition = mtl_load + ftl_load
    plt.figure(figsize=(8, 6)) 
    error_plot(cognition[:, :], "Cognition $C(t)$", is_patient)
    stats_cog = f"Mean Total Cognition:{np.round(cognition.sum(axis=1).mean()/11, 2)}"
    plt.title(stats_cog)
    if verbose: print(stats_cog)
    plt.xticks(range(11))
    plt.ylim([0, 10])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, "cognition_plot.png")
    plt.savefig(save_path)
    plt.close()

    # Subplot 2: Plot total energy
    mtl_energy = np.where(mtl_energy > 6, 5, mtl_energy)
    ftl_energy = np.where(ftl_energy > 6, 5, ftl_energy)
    total_energy = mtl_energy + ftl_energy
    plt.figure(figsize=(8, 6))  # Create a new figure for the next subplot
    error_plot(total_energy, "Energy Cost $M(t)$", is_patient)
    stats_energy = f"Mean Energy Cost:{np.round(total_energy.sum(axis=1).mean()/11, 2)}"
    plt.title(stats_energy)
    if verbose: print(stats_energy)
    plt.xticks(range(11))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, "energy_cost_plot.png")
    plt.savefig(save_path)
    plt.close()

    # Subplot 3: Plot activity
    plt.figure(figsize=(8, 6))  # Create a new figure for the next subplot
    plt.plot(np.mean(mtl_energy, axis=0), label="$Y_{HC}(t)$")
    plt.plot(np.mean(ftl_energy, axis=0), label="$Y_{PFC}(t)$")
    stats_activity = f"Mean Brain Activity, HC:{np.round(np.mean(mtl_energy), 2)} PFC:{np.round(np.mean(ftl_energy), 2)}"
    plt.title(stats_activity)
    if verbose: print(stats_activity)
    plt.legend()
    plt.xlabel("Time (years)")
    plt.xticks(range(11))
    plt.ylabel("Brain Activity $Y_{v}(t)$")
    plt.ylim([0, 5])
    plt.tight_layout()
    save_path = os.path.join(output_dir, "region_activity_plot.png")
    plt.savefig(save_path)
    plt.close()

    # Subplot 4: Plot Information Processed
    plt.figure(figsize=(8, 6))  # Create a new figure for the next subplot
    plt.plot(np.mean(mtl_load, axis=0), label="$I_{HC}(t)$")
    plt.plot(np.mean(ftl_load, axis=0), label="$I_{PFC}(t)$")
    stats_info = f"Mean Information Processed, HC:{np.round(np.mean(mtl_load), 2)} PFC:{np.round(np.mean(ftl_load), 2)}"
    plt.title(stats_info)
    if verbose: print(stats_info)
    plt.legend()
    plt.xlabel("Time (years)")
    plt.xticks(range(11))
    plt.ylabel("Information $I_{v}(t)$")
    plt.ylim([0, 10])
    plt.tight_layout()
    save_path = os.path.join(output_dir, "region_information_processed_plot.png")
    plt.savefig(save_path)
    plt.close()

    # Subplot 5: Plot Brain Size
    plt.figure(figsize=(8, 6))  # Create a new figure for the next subplot
    plt.plot(np.mean(mtl_h, axis=0), label="$X_{HC}(t)$")
    plt.plot(np.mean(ftl_h, axis=0), label="$X_{PFC}(t)$")
    stats_brain_size = f"Mean Brain Size, HC:{np.round(np.mean(mtl_h), 2)} PFC:{np.round(np.mean(ftl_h), 2)}"
    plt.title(stats_brain_size)
    if verbose: print(stats_brain_size)
    plt.legend()
    plt.xlabel("Time (years)")
    plt.xticks(range(11))
    plt.ylabel("Brain Size $X_{v}(t)$")
    plt.ylim([0, 5])
    plt.tight_layout()
    save_path = os.path.join(output_dir, "region_size_plot.png")
    plt.savefig(save_path)
    plt.close()

def plot_cognition(df, filepath, score='MMSE'):
    """
    Plot predicted cognition scores against actual cognition scores.

    Parameters:
        df (pandas.DataFrame): DataFrame containing the columns 'cogsc' and 'cogsc_rl'.

    Returns:
        None
    """

    fig1 = plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x="Years", y="cogsc_rl", color="darkorange", marker="o", label='RL Prediction')
    sns.lineplot(data=df, x="Years", y=score + "_norm", color="dodgerblue", marker="o", label='Ground Truth')
    plt.legend(loc='lower left')
    plt.title('Actual vs Predicted Cognition Scores')
    plt.title(f"Mean GT:{np.round(df[score + '_norm'].mean(), 2)}, Mean RL:{np.round(df['cogsc_rl'].mean(), 2)}")
    plt.ylim([5, 10])
    plt.ylabel("MMSE Score" if "MMSE" in score else "ADAS13 Score")
    plt.xticks(range(11))
    plt.tight_layout()
    plt.savefig(filepath)


def plot_comparison(df, name, var_rl, var_gt, y_min, y_max, filepath):
    """
    Plot predicted cognition scores against actual cognition scores.

    Parameters:
        df (pandas.DataFrame): DataFrame containing the columns 'cogsc' and 'cogsc_rl'.

    Returns:
        None
    """

    fig1 = plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x="Years", y=var_rl, color="darkorange", marker="o", label='RL Prediction')
    sns.lineplot(data=df, x="Years", y=var_gt, color="dodgerblue", marker="o", label='Ground Truth')
    plt.legend(loc='lower left')
    plt.title('Actual vs Predicted: {name}')
    plt.title(f"Mean GT:{np.round(df[var_gt].mean(), 2)}, Mean RL:{np.round(df[var_rl].mean(), 2)}")
    plt.ylim([y_min, y_max])
    plt.ylabel(f"{name}")
    plt.xticks(range(11))
    plt.tight_layout()
    plt.savefig(filepath)


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
    fig = plt.figure(figsize=(16, 6))

    # Subplot 1: Plot cognitive scores over time for each participant
    plt.subplot(1, 2, 1)
    sns.lineplot(data=df, x="Years", y="cogsc", hue='RID', palette=palette, marker="o")
    plt.title(f"Mean Cognition:{np.round(df['cogsc'].mean(), 2)}")
    plt.legend([], [], frameon=False)
    plt.xticks(range(11))
    plt.ylim([0, 11])
    #plt.ylim([df['cogsc'].min() + 1, df['cogsc'].max()])
    plt.ylabel("Cognition Score (RL Predicted)") if type== '_rl' else plt.ylabel(f'Cognition Score ({score_type})')

    # Subplot 2: Plot the mean total cognition score over time for all participants
    plt.subplot(1, 2, 2)
    sns.lineplot(data=df, x="Years", y="cogsc", marker="o")
    plt.title(f"Mean Cognition:{np.round(df['cogsc'].mean(), 2)}")
    plt.legend([], [], frameon=False)
    plt.xticks(range(11))
    plt.ylim([0, 11])
    plt.ylabel("Cognition Score (RL Predicted)") if type== '_rl' else plt.ylabel(f'Cognition Score ({score_type})')

    plt.tight_layout()    
    # Save the plot to the specified file path
    plt.savefig(filepath)


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
    fig = plt.figure(figsize=(8, 6))
    
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

def plot_patient(df, filepath, score='MMSE'):
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
    color_groundtruth = "dodgerblue"
    
    # Subplot 1: Plot cognitive scores over time for each participant
    fig1 = plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x="Years", y="cogsc_rl", color=color_rl, marker="o", label='RL Prediction')
    sns.lineplot(data=df, x="Years", y=score + "_norm", color=color_groundtruth, marker="o", label='Ground Truth')
    plt.legend(loc='lower left')
    plt.title(f"Total Cognition")
    plt.ylim([5, 10])
    plt.ylabel("MMSE Score" if "MMSE" in score else "ADAS13 Score")
    plt.xticks(range(11))
    plt.tight_layout()
    plt.savefig(filepath + '_cognition.png')

    # Subplot 2: Plot PFC size over time for each participant
    fig2 = plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x="Years", y="reg1_mri_rl", color=color_rl, marker="o", label='RL Prediction')
    sns.lineplot(data=df, x="Years", y="mri_FRONT_norm", color=color_groundtruth, marker="o", label='Ground Truth')
    plt.legend(loc='lower left')
    plt.title(f"Prefrontal Cortex Size")
    plt.ylim([0, 5])
    plt.ylabel("Prefrontal Cortex Size")
    plt.xticks(range(11))
    plt.tight_layout()
    plt.savefig(filepath + '_pfc_size.png')

    # Subplot 3: Plot HC size over time for each participant
    fig3 = plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x="Years", y="reg2_mri_rl", color=color_rl, marker="o", label='RL Prediction')
    sns.lineplot(data=df, x="Years", y="mri_HIPPO_norm", color=color_groundtruth, marker="o", label='Ground Truth')
    plt.legend(loc='lower left')
    plt.title(f"Hippocampus Size")
    plt.ylim([0, 5])
    plt.ylabel("Hippocampus Size")
    plt.xticks(range(11))
    plt.tight_layout()
    plt.savefig(filepath + '_hc_size.png')