from brain_env import BrainEnv
import tensorflow as tf
import pandas as pd
import numpy as np
import garage
from garage.envs import normalize
from garage.experiment import Snapshotter

debug = False    # Flag for debugging, set to True to print obs, action, rew, shap_values for the first patient in the test split

class EvalPolicy:

    def __init__(self, T=11, snapshot_dir=None, log_dir=None, gamma=2.1, gamma_type='fixed', cog_init=None, 
                 adj=None, action_type='delta', action_limit=1.0, w_lambda=1.0, energy_model='inverse'):
        """
        Initialize the evaluation policy.

        Parameters:
        - T (int): Number of time steps.
        - snapshot_dir (str): Directory for storing snapshots.
        - log_dir (str): Directory for logging.
        - gamma (float): gamma parameter from the paper
        - gamma_type (str): Type of gamma ('fixed' or 'variable').
        - cog_init: Initial cognition value.
        - adj: Adjacency matrix for the environment.
        - action_type (str): Type of action ('delta' or other).
        - action_limit (float): Action limit.
        - w_lambda (float): Lambda weight.
        - energy_model (str): Energy model type ('inverse' or other).
        """
        self.T = T  
        self.gamma = gamma  
        self.snapshot_dir = snapshot_dir  
        self.log_dir = log_dir  
        self.cog_init = cog_init  
        self.adj = adj  
        self.gamma_type = gamma_type 
        self.action_type = action_type 
        self.action_limit = action_limit  
        self.w_lambda = w_lambda  
        self.energy_model = energy_model  

    def simulate(self, data=None, data_type='test', scale_state=True, normalize_state=False):
        """
        Run the trained policy on patient data and simulate change in cognition and other predictions (X_V, Y_V etc.).

        Parameters:
        - data: Initial values for X_V, D, alpha1, alpha2, beta, and cog.
        - data_type (str): Type of data split ('train', 'valid, or 'test').
        - scale_state (bool): Flag for scaling state.
        - normalize_state (bool): Flag for normalizing state.

        Returns:
        - state_log: Log of states used for SHAP analysis.
        - action_log: Log of actions used for SHAP analysis.
        - output: DataFrame containing the output (RID, Years, All RL predictions).
        """
                
        # Initialize gamma values based on data
        gamma_init = np.ones(len(data[0])) * self.gamma
        gamma_val = data[2][0]
        alpha2_init_new = gamma_val * data[1] / self.gamma
        scale_factor = 10.0 if scale_state else 1.0  # Scaling factor for state
        
        # Initialize variables to store simulation results
        snapshotter = Snapshotter()  # Snapshotter object for loading data
        tf.compat.v1.reset_default_graph()  # Reset TensorFlow default graph

        RIDs = data[-2]  # List of patient IDs
        n_sim = len(RIDs)  # Number of simulations which is equal to the number of patients
        self.mtl_load = np.zeros((n_sim, self.T))  # Memory load values for MTL
        self.ftl_load = np.zeros((n_sim, self.T))  # Memory load values for FTL

        self.mtl_energy = np.zeros((n_sim, self.T))  # Energy values for MTL
        self.ftl_energy = np.zeros((n_sim, self.T))  # Energy values for FTL
        self.mtl_h = np.zeros((n_sim, self.T))  # Health values for MTL
        self.ftl_h = np.zeros((n_sim, self.T))  # Health values for FTL

        self.mtl_d = np.zeros((n_sim, self.T))  # D values for MTL
        self.ftl_d = np.zeros((n_sim, self.T))  # D values for FTL

        out_data = []  # List to store output data
        self.cognition_vec_rl = []  # Cognition vector for RL
        self.reward_vec_rl = []  # Reward vector for RL

        action_log = []  # Log of actions
        state_log = []  # Log of states
        state_log_dict = {}
        num_patients = 0  # Number of patients
        total_steps = 0  # Total number of simulation steps

        # Flag for debugging
        if data_type == 'test' and debug: 
            print(f'\nDEBUGGING...')

        with tf.compat.v1.Session():  # Create a TensorFlow session
            trained_data = snapshotter.load(self.log_dir)  # Load trained data
            policy = trained_data['algo'].policy  # Get the policy

            for i, j in enumerate(RIDs):  # Iterate over patient IDs
                done = False

                # Create the BrainEnv for simulation
                env = BrainEnv(max_time_steps=self.T + 1, alpha1_init=data[0], alpha2_init=alpha2_init_new, 
                               beta_init=data[3], gamma_init=gamma_init, X_V_init=data[5], D_init=data[4], 
                               cog_init=data[-1], adj=self.adj, action_limit=self.action_limit, w_lambda=self.w_lambda, 
                               patient_idx=i, gamma_type=self.gamma_type, action_type=self.action_type, 
                               scale=scale_state, energy_model=self.energy_model)

                obs = env.reset()  # The initial observation
                policy.reset()
                steps, max_steps = 0, self.T  

                # Start the evaluation loop
                while steps < max_steps:  
                    # Get policy action (if ContinuousMLPPolicy, directly outputs mean of the actions. If GaussianMLPPolicy, get mean action.
                    if isinstance(policy, garage.tf.policies.ContinuousMLPPolicy):
                        action = policy.get_action(obs)[0]
                    else:
                        action = policy.get_action(obs)[1]['mean']

                    obs_old = obs.copy()  # Copy the current observation, just used here for debugging

                    # Take a step in the environment
                    obs, rew, done, info = env.step(action)  

                    # following code is just for debugging. Only print for 'test' split
                    if data_type == 'test' and debug: # and j == 2373 (we can also debug for a single patient like this, but this works only for a single experiment)
                        print(f'RID: {j}, step: {steps}, \n ' +
                            f'curr_obs: \t {obs_old}, \n ' +
                            f'action: \t {action}, \n ' +    
                            
                            f'calc_cog: \t {(obs_old[:2]*10.0 + action)/10.0}, \n ' +   # current cognition I_V is calculated in step() through [previous_cognition->upscale+action->downscale] and stored as next_state in obs[:2]    
                            f'curr_I: \t {((obs_old[:2]*10.0 + action)/10.0)*10.0}, \n ' +   # the calculated current cognition is returned as obs[0] * scale_factor and obs[1] * scale_factor a few lines below in out.append   
                            f'curr_X: \t {info["health"]}, \n ' +    # current size/health X_V is stored in info["health"], next/updated size is calculated in step(), / by 5 and stored as self.X_V in obs[3:5]
                            f'curr_D: \t {info["D"]}, \n ' +    # current amyloid D_V is stored in info["D"], next/updated amyloid is calculated in step() and stored as self.D in obs[4:6]
                            
                            f'next_obs: \t {obs}, \n ' +     # next_state is obs[:2] * scale_factor, obs[3:5] * scale_factor, obs[4:6] * scale_factor
                            f'reward: \t {np.round(rew, 6)}')
                        
                        # action_log.append(action) # we can also just collect this patient's state and action logs for use later with shap
                        # state_log.append(obs)     # however this is not tested for more than 1 experiment being run concurrently.
                        
                        # if steps == 1:            # in case we just want to see the first two steps
                        #     exit()
                            
                    # Store simulation data
                    self.mtl_energy[i, steps], self.ftl_energy[i, steps] = info['y']            # Y_V values, related to energy M
                    self.mtl_h[i, steps], self.ftl_h[i, steps] = info['health']                 # X_V values
                    self.mtl_d[i, steps], self.ftl_d[i, steps] = info['D']                      # D_V values
                    self.mtl_load[i, steps], self.ftl_load[i, steps] = obs[:2] * scale_factor   # I_t values (Cognition load C(t))

                    # Append a list of predicted values to the out_data list
                    out_data.append([j, steps, obs[0] * scale_factor, obs[1] * scale_factor, 
                                     info['y'][0], info['y'][1], info['health'][0], info['health'][1], 
                                     info['D'][0], info['D'][1], data[3][i], data[0][i], alpha2_init_new[i], gamma_init[i]])
                    
                    # Append the sum of the cognition scores to the cognition_vec_rl list
                    self.cognition_vec_rl.append(obs[0] + obs[1])
                    
                    # Append the reward to the reward_vec_rl list
                    self.reward_vec_rl.append(rew)

                    action_log.append(action)
                    state_log.append(obs)

                    if j not in state_log_dict:
                        state_log_dict[j] = []
                    state_log_dict[j].append(obs)

                    steps += 1
                    total_steps += 1

                num_patients += 1
                env.close()

                if debug and num_patients > 0: break         # Adjust for however many patients you want to print for debugging. Default is 1.   

        # Plot RL predicted values at the end of the simulation
        # try:
        #     print('\nPlotting RL trajectories at the end of Simulation...')
        #     plot_trajectories(self.mtl_load, self.ftl_load, self.mtl_energy, self.ftl_energy, self.mtl_h, self.ftl_h, f'{self.snapshot_dir}/plots/rl_trajectories_{data_type}')
        # except Exception as e:
        #     print(f"ERROR: Exception occurred while plotting plot_trajectories(): {e}")

        # Define column names for the output DataFrame
        columns = ['RID', 'Years', 'reg1_info', 'reg2_info', 'reg1_fdg', 'reg2_fdg', 'reg1_mri', 'reg2_mri', 'reg1_D', 'reg2_D', 'beta', 'alpha1', 'alpha2', 'gamma']
        new_columns = []
        
        # Create a pandas DataFrame from the out_data list with the specified column names
        self.output = pd.DataFrame(out_data, columns=columns)

        # Rename columns that correspond to RL variables (except RID and Years) with a "_rl" suffix
        for j, c in enumerate(columns):
            if j >= 2:
                new_columns.append(c + "_rl")
            else:
                new_columns.append(c)

        self.output.columns = new_columns

        # Add a new column to the DataFrame that contains the sum of the cognition scores for each patient
        self.output['cogsc_rl'] = self.output['reg1_info_rl'] + self.output['reg2_info_rl']
        
        # Print the total number of patients and simulation steps
        print('Total number of patients:', num_patients)  # Total number of patients
        print('Total number of steps:', total_steps)  # Total number of simulation steps

        # Return the state and action logs as numpy arrays, along with the output DataFrame
        return np.array(state_log), np.array(action_log), self.output, state_log_dict

 
    def compute(self, df, data_type='test', exp_type='adni', score='MMSE'):
        """
        Computes the output cognition values and frontal/mtl degradation and energy-related values, and stores them in an xlsx file.

        This method takes simulation results and evaluation data, calculates various metrics, and optionally
        plots and saves the results. It can be used for both ADNI and Synthetic datasets.

        Parameters:
        - df (DataFrame): The input data containing ground truth information.
        - data_type (str): The type of data being evaluated, such as 'train', 'valid', or 'test'.
        - exp_type (str): The type of experiment data, either 'adni' for ADNI data or 'synthetic' for synthetic data.
        - score (str): The cognitive score used for evaluation, e.g., 'MMSE', 'ADAS11', 'ADAS13'

        Returns:
        A tuple containing various evaluation metrics for cognition and reward, depending on the data type.

        Note:
        This method performs different calculations and visualizations based on the data type and experiment type.
        For ADNI data, it calculates metrics and plots RL predictions against ground truth.
        For synthetic data, it calculates mean absolute error and mean squared error for cognition and reward.

        """
        # plot_dir = f'{self.snapshot_dir}/plots/comparison/'
        # if not os.path.exists(plot_dir):
        #     os.makedirs(plot_dir, exist_ok=True)

        # For ADNI dataset
        if exp_type == 'adni':
            # Merge input data with simulation output
            # Resulting DataFrame will include all the rows from the right DataFrame (RL predictions) and only the matching rows from the left DataFrame (original dataset). 
            # This is basically to calculate the difference between the RL predicted cognition scores and the actual cognition scores
            df_join = pd.merge(df, self.output, how='right', left_on=['RID', 'Years'], right_on=['RID', 'Years'])

            # Set cognition score to be score_norm (MMSE_norm, ADAS11_norm, ADAS13_norm, etc.)
            df_join['cogsc'] = df_join[f'{score}_norm']

            # Set the file write mode based on the data type (train, valid or test)
            if data_type == 'train':
                # If the data type is 'train', set the mode to 'w' to overwrite the file
                mode = 'w'
            else:
                # If the data type is not 'train', set the mode to 'a' to append to the file
                mode = 'a'

            # Extract the name of the output file from the snapshot directory path
            outfile = self.snapshot_dir.split("results/")[1].replace("/", "_")

            # Calculate the difference between the cognition scores predicted by the RL model and the actual cognition scores
            df_join['cog_diff'] = df_join['cogsc_rl'] - df_join['cogsc']

            # ********** Saving results to file **********
            # Write the results (input data, RL predictions, cog_diff) to an Excel file {dataset}_{fold}_{method}_{score}_{seed}.xlsx} 
            print(f'Writing results to {self.snapshot_dir}/{outfile}.xlsx under sheet "{data_type}".')
            with pd.ExcelWriter(f'{self.snapshot_dir}/{outfile}.xlsx', engine="openpyxl", mode=mode) as writer:
                df_join.to_excel(writer, sheet_name=data_type, index=False)
            
            # ********** Common Data Points **********
            # Merge input data with simulation output again to generate common data points (input data and RL predictions)
            # Resulting DataFrame will include all the rows from the left DataFrame (original dataset) and only the matching rows from the right DataFrame (RL predictions).
            df_join = pd.merge(df, self.output, how='left', left_on=['RID', 'Years'], right_on=['RID', 'Years'])
            
            # set cognition score to be ground truth score (MMSE_norm, ADAS11_norm, ADAS13_norm)
            df_join['cogsc'] = df_join[f'{score}_norm']
            
            # set cognition score to be predicted cognition score (I = I_PFC + I_HC)
            df_join['cogsc_rl'] = df_join['reg1_info_rl'] + df_join['reg2_info_rl']

            # Calculate mean absolute error and mean squared error for cognition
            # C(t) = I_v1 + I_v2 - ground_truth (MMSE)
            cog_mae = np.abs(df_join['reg1_info_rl'] + df_join['reg2_info_rl'] - df_join[f'{score}_norm']).values.mean()
            cog_mse = np.square(df_join['reg1_info_rl'] + df_join['reg2_info_rl'] - df_join[f'{score}_norm']).values.mean()

            # Round the mean absolute error and mean squared error of the cognition scores to 3 decimal places
            cog_mae = np.round(cog_mae, 3)
            cog_mse = np.round(cog_mse, 3)

            # Define a list of categories to evaluate the cognition scores on
            categories = ['EMCI', 'CN', 'LMCI', 'SMC']

            # Initialize lists to store the mean absolute error and mean squared error for each category
            mae_cat = []
            mse_cat = []

            # Loop over each category and calculate the mean absolute error and mean squared error
            for cat in categories:
                # Calculate the absolute error for the current category
                # C(t) = I_v1 + I_v2 - ground_truth (MMSE)
                cat_mae = np.abs(
                    (df_join[df_join['DX_bl'] == cat]['reg1_info_rl'] + 
                     df_join[df_join['DX_bl'] == cat]['reg2_info_rl'] -
                     df_join[df_join['DX_bl'] == cat][f'{score}_norm']).values)
                
                # Calculate the squared error for the current category
                cat_mse = np.square(
                    (df_join[df_join['DX_bl'] == cat]['reg1_info_rl'] + 
                     df_join[df_join['DX_bl'] == cat]['reg2_info_rl'] -
                     df_join[df_join['DX_bl'] == cat][f'{score}_norm']).values)

                # Append the mean absolute error and mean squared error to their respective lists
                mae_cat.append(cat_mae.mean())
                mse_cat.append(cat_mse.mean())

            # Round the mean absolute error and mean squared error for each category to 3 decimal places
            mae_cat = [np.round(x, 3) for x in mae_cat]
            mse_cat = [np.round(x, 3) for x in mse_cat]

            # Round the mean absolute error of the cognition scores to 3 decimal places
            cog_mae = np.round(cog_mae, 3)

            # Round the mean squared error of the cognition scores to 3 decimal places
            cog_mse = np.round(cog_mse, 3)

            # Round the mean reward to 3 decimal places
            self.reward_vec_rl = np.round(np.mean(self.reward_vec_rl), 3)

            # Return the mean absolute error and mean squared error for each category, as well as the mean reward
            return cog_mae, mae_cat[0], mae_cat[1], mae_cat[2], mae_cat[3], \
                    cog_mse, mse_cat[0], mse_cat[1], mse_cat[2], mse_cat[3], \
                    0, self.reward_vec_rl, 0
        
        # For Synthetic dataset
        else:
            # Merge evaluation data with simulation output
            df_join = pd.merge(df, self.output, how='left', left_on=['RID', 'Years'], right_on=['RID', 'Years'])

            # Extract cognition and reward vectors
            cognition_vec = df_join['cogsc'].values

            # Set the file write mode based on the data type (train, valid or test)
            if data_type == 'train':
                # If the data type is 'train', set the mode to 'w' to overwrite the file
                mode = 'w'
            else:
                # If the data type is not 'train', set the mode to 'a' to append to the file
                mode = 'a'
            # Extract the name of the output file from the snapshot directory path
            outfile = self.snapshot_dir.split("/")[-1]

            # Calculate the difference between the cognition scores predicted by the RL model and the actual cognition scores
            df_join['cog_diff'] = df_join['cogsc_rl'] - df_join['cogsc']
            
            # Write the merged data to an Excel file
            with pd.ExcelWriter(f'{self.snapshot_dir}/{outfile}.xlsx', engine="openpyxl", mode=mode) as writer:
                df_join.to_excel(writer, sheet_name=data_type, index=False)

            # R(t) = −[λ|Ctask − C(t)| + M(t)] # eq 8 of Saboo et al. 2021
            # R(t) = − λ|C(t) - Ctask| - M(t)
            # R(t) = − |C(t) - Ctask|λ - M(t) where C(t)=I_v1 + I_v2, Ctask=10, M(t)=Y_V1 + Y_V2
            reward_vec = (-np.abs(df_join['reg1_info'] + df_join['reg2_info'] - 10) * self.w_lambda - (
                                    df_join['reg1_fdg'] + df_join['reg2_fdg'])).values

            # Process data for MTL and FTL
            # Group the data by patient ID (RID), select the first 7 rows for each group, and reset the index
            df_6 = df.groupby('RID').head(7).reset_index(drop=True)

            # Extract the information scores I_v(t) for MTL and FTL from df_6
            mtl_load = df_6['reg1_info'].values
            ftl_load = df_6['reg2_info'].values

            # Reshape the regional information scores for MTL and FTL into a 2D array and select all columns except the first column
            mtl_load = mtl_load.reshape(-1, 7)[:, 1:]
            ftl_load = ftl_load.reshape(-1, 7)[:, 1:]

            # Calculate the total cognition scores by adding the regional information scores for MTL and FTL
            cognition = mtl_load + ftl_load

            # Extract the regional FDG-PET activity Y_v(t) (energy) scores for MTL and FTL from df_6
            mtl_energy = df_6['reg1_fdg'].values
            ftl_energy = df_6['reg2_fdg'].values

            # Reshape the regional FDG-PET energy scores for MTL and FTL into a 2D array and select all columns except the first column
            mtl_energy = mtl_energy.reshape(-1, 7)[:, 1:]
            ftl_energy = ftl_energy.reshape(-1, 7)[:, 1:]

            # Calculate the total energy scores by adding the regional FDG-PET energy scores for MTL and FTL
            total_energy = mtl_energy + ftl_energy

            mtl_h = None
            ftl_h = None

            # Extract cognition and reward vectors for RL
            cognition_vec_rl = df_join['cogsc_rl'].values

            # R(t) = − |C(t) - Ctask|λ - M(t) where C(t)=I_v1 + I_v2, Ctask=10, M(t)=Y_V1 + Y_V2
            reward_vec_rl = (-np.abs(df_join['reg1_info_rl'] + df_join['reg2_info_rl'] - 10) * self.w_lambda - (
                                     df_join['reg1_fdg_rl'] + df_join['reg2_fdg_rl'])).values

            # Calculate mean absolute error and mean squared error for cognition (prediction - ground truth)
            cog_mae = np.abs(cognition_vec_rl - cognition_vec).mean()
            cog_mse = (np.square(cognition_vec_rl - cognition_vec)).mean()

            # Calculate the difference in reward vectors
            reward_diff = np.mean(reward_vec_rl - reward_vec)

            # Round the mean absolute error and mean squared error for the cognition scores to 3 decimal places
            cog_mae = np.round(cog_mae, 3)
            cog_mse = np.round(cog_mse, 3)

            # Round the mean reward difference, mean reward for the RL model, and mean reward for the baseline model to 3 decimal places
            reward_diff = np.round(reward_diff, 3)
            mean_reward_rl = np.round(np.mean(self.reward_vec_rl), 3)
            mean_reward_baseline = np.round(np.mean(reward_vec), 3)

            # Return the mean absolute error and mean squared error for the cognition scores, 
            # as well as the mean reward difference and mean reward for the RL and baseline models
            # Not calculating for the categories 'EMCI', 'CN', 'LMCI', 'SMC' (set to 0)
            return cog_mae, 0, 0, 0, 0, \
                    cog_mse, 0, 0, 0, 0, \
                    reward_diff, mean_reward_rl, mean_reward_baseline

        
        
        
            
        




