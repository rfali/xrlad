import numpy as np
import pandas as pd
import tensorflow as tf
from garage.experiment import Snapshotter
import shap
import matplotlib.pyplot as plt
import os, shutil
import garage
from matplotlib.colors import ListedColormap
import seaborn as sns
from eval import debug

plt.rcParams['figure.max_open_warning'] = 30
plt.rcParams['figure.dpi'] = 300 
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 16 
plt.rcParams['axes.labelsize'] = 20
custom_figsize = True
fig_w, fig_h = 4, 6
patient_fig_w, patient_fig_h = 6, 4

def generate_shap(log_dir, state_log, action_log, use_all_samples=False):
    """
    Generate SHAP values for a single experiment

    Args:
        log_dir (str): Directory containing trained model data.
        state_log (numpy.ndarray): Log of states or observations.
        action_log (numpy.ndarray): Log of actions taken.
        use_all_samples (bool, optional): Whether to use all samples for background data. Defaults to False.

    Returns:
        shape_values (numpy.ndarray): SHAP values for each input feature.
        state_log (numpy.ndarray): Log of states or observations.
        explainer_expected_value (numpy.ndarray): Expected value (mean) of the explainer for each action.
    """
    
    # Set True if need to print diagnostic information
    verbose = False
        
    #use_all_samples = #True, use all samples to build background, else use kmeans and generate `shap_samples_to_use` samples
    shap_samples_to_use = 100
    
    # Reset TensorFlow default graph
    tf.compat.v1.reset_default_graph()

    # Initialize the Snapshotter
    snapshotter = Snapshotter()

    # Create a TensorFlow session
    session = tf.compat.v1.Session()

    # Load trained model data
    with session:
        trained_data = snapshotter.load(log_dir)
        policy = trained_data['algo'].policy
        algo_name = trained_data["algo"].__class__.__name__

        if verbose:
            # Print info about state and action logs
            print(f'{algo_name} - state_log : shape:{state_log.shape} type:{type(state_log)} sample_0:{state_log[0]}')
            # print(f'{algo_name} - action_log: shape:{action_log.shape} type:{type(action_log)} sample_0:{action_log[0]}')

            # Get sample action from the policy
            action = policy.get_action(state_log[0])[0]                 # get sample action, equal to model.predict(obs)
            action_mean = policy.get_action(state_log[0])[1]['mean']    # get the mean of the action distribution. Refer to garage documentation e.g. gaussian_mlp_policy.get_action()
            
            # Print information about the sample action
            print(f'{algo_name} - action for sample 0: shape: {np.shape(action)} - action_values:{action}')
            print(f'{algo_name} - action_mean for sample 0: shape: {np.shape(action_mean)} - action_values:{action_mean}')
        
        # Define the RL model to be used for SHAP using the policy
        if isinstance(policy, garage.tf.policies.ContinuousMLPPolicy):
            rl_model = lambda x: policy.get_actions(x)[0]               # For ContinuousMLPPolicy used with DDPG, model output is the single best action, as done in eval.simluate()
        else:
            rl_model = lambda x: policy.get_actions(x)[1]['mean']       # For GaussianMLPPolicy used with TRPO, PPO and SAC, model output is the action distribution and we take the mean over actions.
        
        # Initialize a SHAP KernelExplainer
        explainer = shap.KernelExplainer(rl_model, state_log)

        # Use all samples to build shap_values for SHAP
        if use_all_samples:     
            print(f'Using all {state_log.shape[0]} samples to build shap\'s background data')
            shap_values = explainer.shap_values(state_log)  # use all 1133 samples to build background data
            if verbose: print('using all samples - shape of state_log', np.shape(state_log))
            if verbose: print('using all samples - shape of shap values', np.shape(shap_values))
        
        # Use limited samples to build shap_values. This is to counter "Using x background data samples could cause slower run times."
        else:                   
            print(f'Using limited samples ({shap_samples_to_use}) to build shap\'s background data')
            state_log = shap.sample(state_log, shap_samples_to_use)
            shap_values = explainer.shap_values(state_log) 
            if verbose: print('using limited samples - shape of state_log', np.shape(state_log))
            if verbose: print('using limited samples - shape of shap values', np.shape(shap_values))
        
        # Get the expected value of the explainer for each action (total actions = 2) and reshape it to add a dummy dimension.
        # This is used later for plots (decision, waterfall) which require a mean value to center the plot.
        explainer_expected_value = explainer.expected_value.reshape(1, 2)
        
        # Coupled with debugging in eval.simulate(), this helps to verify the actual state, actions and associated shap_values.
        if debug:
            print('\nDEBUGGING SHAP values...')
            np.set_printoptions(suppress=True)
            print('\nshap_values', type(shap_values), np.shape(shap_values)) 
            print('\nshap_values [0]\n', shap_values[0]) 
            print('\nshap_values [1]\n', shap_values[1]) 
            print('\nstate_log', type(state_log), np.shape(state_log), '\n', state_log)
            print('\nexplainer_exp_value', type(explainer_expected_value), np.shape(explainer_expected_value), '\n', explainer_expected_value)                             
            np.set_printoptions(suppress=False)

        return np.array(shap_values), state_log, explainer_expected_value
    
def plot_shap(algo, shap_values, state_log, explainer_expected_value, plot_dir, show_fig=False, is_patient=False):
        """
        Generate SHAP plots for model explanation for all samples for a given algorithm.

        Args:
            algo (str): Algorithm name. In case of patient SHAP, this is prefixed by patient record ID (RID-xxxx_algo)
            shap_values (numpy.ndarray): SHAP values for each input feature.
            state_log (numpy.ndarray): Log of states or observations.
            explainer_expected_value (numpy.ndarray): Expected value of the explainer for each action.
            plot_dir (str): Directory to contain SHAP plots.
            show_fig (bool, optional): Whether to display plots. Defaults to False.
            is_patient (bool, optional): Whether to generate plots for a single patient. The passed inputs will be for that patient only. Defaults to False.

        Returns:
            None
        """

        # Set True if need to print diagnostic information
        verbose = False
        
        print(f'\n*** Generating SHAP plots for {algo} and saving them to {plot_dir} ***\n')
        print('Received following data for plotting SHAP figures')
        print('shap_values', type(shap_values), np.shape(shap_values))                              # shap_values <class 'numpy.ndarray'> (2, N, 6)
        print('shap_values[0]', type(shap_values[0]), np.shape(shap_values[0]))                     # shap_values[0] <class 'numpy.ndarray'> (N, 6)
        
        # # Convert shap_values to a list for summary bar plot (otherwise it is unable to detect the correct number of features)
        # # using tolist() recursively converts all dimensions to lists, which is not what we want as it throws error in detecting correct number of features.
        # # In order to correctly convert shap_values to a list, we need to only convert the outer type to list and keep the inner types as numpy arrays
        # shap_values_list = list(shap_values) 
        # print('shap_values_list', type(shap_values_list), np.shape(shap_values_list))               # shap_values_list <class 'list'> (2, N, 6)
        # print('shap_values_list[0]', type(shap_values_list[0]), np.shape(shap_values_list[0]))      # shap_values_list[0] <class 'numpy.ndarray'> (N, 6)
        
        print('state_log', type(state_log), np.shape(state_log))
        print('explainer_exp_value', type(explainer_expected_value), np.shape(explainer_expected_value))

        # Define names for input features and outputs
        # input features : Information processing capacity of brain regions at time t-1, Size of brain regions at time t and amyloid deposition at time t
        feature_names = ['$I_{HC}(t-1)$','$I_{PFC}(t-1)$' ,'$X_{HC}(t)$','$X_{PFC}(t)$','$D_{HC}(t)$','$D_{PFC}(t)$']
        feature_names_strings = ['I_HC(t-1)','I_PFC(t-1)','X_HC','X_PFC','D_HC','D_PFC']
        # outputs/actions : change in Information processing capacity of brain regions
        output_names = ['$\Delta I_{HC}(t)$', '$\Delta I_{PFC}(t)$']
        
        # Create a SHAP Explanation object using the shap_values
        explanation = shap.Explanation(shap_values, explainer_expected_value, state_log, feature_names=feature_names)
            
        if verbose: print('\nshap_values', type(shap_values), np.shape(shap_values)) # (2, 10, 6) -> (actions, shap_samples_to_use, features)
        if verbose: print('explanation.values', type(explanation.values), np.shape(explanation.values))
        if verbose: print('explanation.base_values', type(explanation.base_values), np.shape(explanation.base_values))
        if verbose: print(f'explanation.values {explanation.values[0][0]}  == \nshap_values {shap_values[0][0]}') # shap values are the explanation.values
        if verbose: print(f'explanation.base_values {explanation.base_values}  == \nexplainer_expected_value {explainer_expected_value}') 

        if is_patient:
            # Reshape the samples (years*seeds e.g. 11x5=55 to years, seeds) and take a mean across the seeds dimension to get a single value for each year.
            num_seeds = len(np.unique(pd.read_csv('results/summary_adni.csv')["seed"]))
            shap_values = np.mean(shap_values.reshape(2, num_seeds, 11, 6), axis=1)
            state_log = np.mean(state_log.reshape(num_seeds, 11, 6), axis=0)
        
        # Generate and save SHAP plots
        shap.initjs()

        # Clear the figure
        plt.clf()

        ################################################ Global Explanations ################################################################
        #####################################################################################################################################

        # Following are explanations for all patient samples (shap_values[0] = ΔI_HC, shap_values[1] = ΔI_PFC)
        print(f'Generating Global Explanations for {algo}...')
        if os.path.exists(plot_dir): shutil.rmtree(plot_dir)
        plot_dir = f'{plot_dir}/global'
        os.makedirs(plot_dir, exist_ok=True)

        # Summary bar plot
        print('Generating plot: summary_bar')
        # Convert shap_values back to a list for summary bar plot, otherwise it is unable to detect the correct number of features.
        my_cmap = ListedColormap(sns.color_palette(["orange", "purple"]).as_hex())
        shap.summary_plot(list(shap_values), features=state_log, feature_names=feature_names, class_names=output_names, color=my_cmap, show=show_fig)
        plt.title(f'{algo}: Summary Plot', y=1.0)
        plt.xlabel("mean |SHAP value|")
        if custom_figsize: plt.gcf().set_size_inches(fig_w, fig_h)
        # plt.gcf().set_size_inches(6, 6)
        plt.savefig(f'{plot_dir}/{algo}_summary_bar.pdf', bbox_inches='tight')
        plt.clf()

        # # Summary bar plots for each action (ΔI_HC and ΔI_PFC)
        # print('Generating plot: summary_bar_HC')
        # shap.summary_plot(shap_values[0], features=state_log, plot_type="bar", feature_names=feature_names, show=show_fig)
        # plt.title(f'{algo}: Summary Plot: {output_names[0]}', y=1.0)
        # plt.xlabel("mean |SHAP value|")
        # if custom_figsize: plt.gcf().set_size_inches(fig_w, fig_h)
        # plt.savefig(f'{plot_dir}/{algo}_summary_bar_HC.pdf')
        # plt.clf()
        
        # print('Generating plot: summary_bar_PFC')
        # shap.summary_plot(shap_values[1], features=state_log, plot_type="bar", feature_names=feature_names, show=show_fig)
        # plt.title(f'{algo}: Summary Plot: {output_names[1]}', y=1.0)
        # plt.xlabel("mean |SHAP value|")
        # if custom_figsize: plt.gcf().set_size_inches(fig_w, fig_h)
        # plt.savefig(f'{plot_dir}/{algo}_summary_bar_PFC.pdf')
        # plt.clf()
        
        # Beeswarm plots for each action (ΔI_HC and ΔI_PFC)
        print('Generating plot: beeswarm_HC')
        shap.summary_plot(shap_values[0], features=state_log, feature_names=feature_names, show=show_fig)
        plt.title(f'{algo}: Beeswarm Plot: {output_names[0]}', y=1.0)
        plt.xlabel("SHAP value")
        if custom_figsize: plt.gcf().set_size_inches(fig_w, fig_h)
        # plt.gcf().set_size_inches(6, 4)
        plt.savefig(f'{plot_dir}/{algo}_beeswarm_HC.pdf', bbox_inches='tight')
        plt.clf()
       
        print('Generating plot: beeswarm_PFC')
        shap.summary_plot(shap_values[1], features=state_log, feature_names=feature_names, show=show_fig)
        plt.title(f'{algo}: Beeswarm Plot: {output_names[1]}', y=1.0)
        plt.xlabel("SHAP value")
        if custom_figsize: plt.gcf().set_size_inches(fig_w, fig_h)
        # plt.gcf().set_size_inches(6, 4)
        plt.savefig(f'{plot_dir}/{algo}_beeswarm_PFC.pdf', bbox_inches='tight')
        plt.clf()
        
        # {Paper) Decision Plots Only needed for patient
        if algo.startswith('RID'):
            # Decision plots for each action (ΔI_HC and ΔI_PFC)
            print('Generating plot: decision_plot_HC')
            shap.decision_plot(explainer_expected_value[0], shap_values[0], features=state_log, feature_names=feature_names, show=show_fig, ignore_warnings=True)
            plt.title(f'{algo}: Decision Plot: {output_names[0]}', y=1.1)
            #plt.xlabel("Custom X-Axis Label")
            plt.subplots_adjust(top=0.9, bottom=0.2)  # Adjust margins as needed
            if custom_figsize: plt.gcf().set_size_inches(fig_w, fig_h)
            plt.savefig(f'{plot_dir}/{algo}_decision_plot_HC.pdf', bbox_inches='tight')
            plt.clf()

            print('Generating plot: decision_plot_PFC')
            shap.decision_plot(explainer_expected_value[1], shap_values[1], features=state_log, feature_names=feature_names, show=show_fig, ignore_warnings=True)
            plt.title(f'{algo}: Decision Plot: {output_names[1]}', y=1.1)
            #plt.xlabel("Custom X-Axis Label")
            plt.subplots_adjust(top=0.9, bottom=0.2)  # Adjust margins as needed
            if custom_figsize: plt.gcf().set_size_inches(fig_w, fig_h)
            plt.savefig(f'{plot_dir}/{algo}_decision_plot_PFC.pdf', bbox_inches='tight')
            plt.clf()

            # Probability Decision plots for each action (ΔI_HC and ΔI_PFC)
            # The link='logit' argument converts the base values and SHAP values to probabilities.
            print('Generating plot: decision_prob_plot_HC')
            shap.decision_plot(explainer_expected_value[0], shap_values[0], features=state_log, feature_names=feature_names, 
                                    link="logit", show=show_fig, ignore_warnings=True)
            plt.title(f'{algo}: Decision Plot: {output_names[0]}', y=1.1)
            plt.xlabel("Model output value (Probability)")
            plt.subplots_adjust(top=0.9, bottom=0.2)  # Adjust margins as needed
            if custom_figsize: plt.gcf().set_size_inches(fig_w, fig_h)
            plt.savefig(f'{plot_dir}/{algo}_decision_prob_plot_HC.pdf', bbox_inches='tight')
            plt.clf()

            print('Generating plot: decision_prob_plot_PFC')
            shap.decision_plot(explainer_expected_value[1], shap_values[1], features=state_log, feature_names=feature_names, 
                                    link="logit", show=show_fig, ignore_warnings=True)
            plt.title(f'{algo}: Decision Plot: {output_names[1]}', y=1.1)
            plt.xlabel("Model output value (Probability)")
            plt.subplots_adjust(top=0.9, bottom=0.2)  # Adjust margins as needed
            if custom_figsize: plt.gcf().set_size_inches(fig_w, fig_h)
            plt.savefig(f'{plot_dir}/{algo}_decision_prob_plot_PFC.pdf', bbox_inches='tight')
            plt.clf()
        
        # (Paper specific) Dependence Plots not needed for patient
        if not algo.startswith('RID'):
            # Dependence plots for each feature on each action (ΔI_HC and ΔI_PFC) 
            # https://shap-lrjball.readthedocs.io/en/latest/example_notebooks/plots/dependence_plot.html
            interaction_idx = "auto"    # From "auto", None, int, or string.  
            # If "auto" then shap.common.approximate_interactions picks approximate strongest interaction feature. If None, then effects are shown without interactions.
            for i in range(len(feature_names)):
                print(f'Generating plot: dependence_plot_HC for feature:{feature_names_strings[i]}')
                shap.dependence_plot(ind=i, shap_values=shap_values[0], features=state_log, interaction_index=interaction_idx, 
                                    feature_names=feature_names, show=show_fig)
                plt.title(f'{algo}: Dependence Plot: {output_names[0]}', y=1.0)
                #plt.xlabel("Custom X-Axis Label")
                plt.subplots_adjust(left=0.2)  # Adjust margins as needed
                if custom_figsize: plt.gcf().set_size_inches(fig_w, fig_h)
                plt.savefig(f'{plot_dir}/{algo}_dependence_HC_{feature_names_strings[i]}.pdf', bbox_inches='tight')
                plt.clf()

                print(f'Generating plot: dependence_plot_PFC for feature:{feature_names_strings[i]}')
                shap.dependence_plot(ind=i, shap_values=shap_values[1], features=state_log, interaction_index=interaction_idx, 
                                    feature_names=feature_names, show=show_fig)
                plt.title(f'{algo}: Dependence Plot: {output_names[1]}', y=1.0)
                #plt.xlabel("Custom X-Axis Label")
                plt.subplots_adjust(left=0.2)  # Adjust margins as needed
                if custom_figsize: plt.gcf().set_size_inches(fig_w, fig_h)
                plt.savefig(f'{plot_dir}/{algo}_dependence_PFC_{feature_names_strings[i]}.pdf', bbox_inches='tight')
                plt.clf()

            # # Generate dependence plots with specific interaction index. 
            # f1, f2= 0, 2 #  I_HC(t-1) (index 0) with X_HC (index 2). Indices from feature_names_strings
            # print(f'Generating plot: dependence_plot_HC for feature:{feature_names_strings[f1]} interaction with {feature_names_strings[f2]} ')
            # shap.dependence_plot(ind=f1, shap_values=shap_values[0], features=state_log, interaction_index=f2, feature_names=feature_names, show=show_fig)
            # plt.title(f'{algo}: Dependence Plot: {output_names[0]}', y=1.0)
            # #plt.xlabel("Custom X-Axis Label")
            # plt.subplots_adjust(left=0.2)  # Adjust margins as needed
            # if custom_figsize: plt.gcf().set_size_inches(fig_w, fig_h)
            # plt.savefig(f'{plot_dir}/{algo}_dependence_HC_{feature_names_strings[f1]}_with_{feature_names_strings[f2]}.pdf', bbox_inches='tight')
            # plt.clf()    

        # (Paper) Stack Force Plots only needed for patient
        if algo.startswith('RID'):
            # Stacked Force plots for each action (ΔI_HC and ΔI_PFC)
            print('Generating plot: stacked_force_HC')
            force_plot = shap.force_plot(explainer_expected_value[0], shap_values[0], feature_names=feature_names_strings, show=False)
            shap.save_html(f'{plot_dir}/{algo}_stacked_force_HC.html', force_plot)
            
            print('Generating plot: stacked_force_PFC')
            force_plot = shap.force_plot(explainer_expected_value[1], shap_values[1], feature_names=feature_names_strings, show=False)
            shap.save_html(f'{plot_dir}/{algo}_stacked_force_PFC.html', force_plot)

        plt.close()
        
        ################################################ Local Explanations ################################################################
        ####################################################################################################################################

        # Following are explanations for a particular sample (shap_values[0][X] = ΔI_HC, shap_values[1][X] = ΔI_PFC) where X is the index of the sample.
        # If is_patient is True, then generate plots for a single patient for specific years (0,3,6,9). 
        # If false, then generate plots for a single sample (the first sample at idx 0) from all input samples.

        if is_patient:
            # Reshape the samples (years*seeds e.g. 11x5=55 to years, seeds) and take a mean across the seeds dimension to get a single value for each year.
            # shap_values = np.mean(shap_values.reshape(2, 5, 11, 6), axis=1)
            # state_log = np.mean(state_log.reshape(5, 11, 6), axis=0)
            indices = [0, 3, 6, 9]      # generate shap local plots for each of these years
            sample_or_year = 'year'
        else:
            indices = [0]               # index of the sample to be explained
            sample_or_year = 'sample'

        original_plot_dir = plot_dir

        # {Paper) temporarily generating local plots only for patients 
        if is_patient:
            
            # Loop over indices (years in case of patient, otherwise the provided idx from all input samples)
            for idx in indices:

                print(f'\nGenerating Local Explanations of {sample_or_year} {idx} for {algo}...')

                if is_patient:
                    plot_dir = original_plot_dir.replace("global", "local_year_"+str(idx))
                else:
                    plot_dir = original_plot_dir.replace("global", "local_idx_"+str(idx))
                os.makedirs(plot_dir, exist_ok=True)

                state_log[idx] = [round(x, 4) for x in state_log[idx]] # round off values to 4 decimal places. Done for Force plots.

                # Force plots for each action (ΔI_HC and ΔI_PFC)
                print('Generating plot: force_HC')
                shap.force_plot(explainer_expected_value[0], shap_values[0][idx], features=state_log[idx], feature_names=feature_names, matplotlib=True, show=show_fig)
                #plt.title(f'{algo}: Force Plot: {output_names[0]} for {sample_or_year} {idx}', y=1.2)
                #plt.xlabel("Custom X-Axis Label")
                plt.subplots_adjust(top=0.9, bottom=0.2) 
                plt.savefig(f'{plot_dir}/{algo}_force_HC_{idx}.pdf', format="pdf", dpi=600, bbox_inches='tight')
                plt.clf()
                
                print('Generating plot: force_PFC')
                shap.force_plot(explainer_expected_value[1], shap_values[1][idx], features=state_log[idx], feature_names=feature_names, matplotlib=True, show=show_fig)
                #plt.title(f'{algo}: Force Plot: {output_names[1]} for {sample_or_year} {idx}', y=1.2)
                #plt.xlabel("Custom X-Axis Label")
                plt.subplots_adjust(top=0.9, bottom=0.2)
                plt.savefig(f'{plot_dir}/{algo}_force_PFC_{idx}.pdf', format="pdf", dpi=600, bbox_inches='tight')
                plt.clf()

                # local HTML plots used to work but no longer (investigate later)
                # print('Generating plot: force_HC_html')
                # # Force plots for each action (ΔI_HC and ΔI_PFC) in HTML format
                # force_plot = shap.force_plot(explainer_expected_value[0], shap_values[0][idx], features=state_log[idx], feature_names=feature_names_strings, show=False)
                # shap.save_html(f'{plot_dir}/{algo}_force_HC_{idx}.html', force_plot)

                # print('Generating plot: force_PFC_html')
                # force_plot = shap.force_plot(explainer_expected_value[1], shap_values[1][idx], features=state_log[idx], feature_names=feature_names_strings, show=False)
                # shap.save_html(f'{plot_dir}/{algo}_force_PFC_{idx}.html', force_plot)

                # Decision plots for each action (ΔI_HC and ΔI_PFC)
                print('Generating plot: decision_plot_HC')
                shap.decision_plot(explainer_expected_value[0], shap_values[0][idx], features=state_log[idx], feature_names=feature_names, show=show_fig)
                plt.title(f'{algo}: Decision Plot: {output_names[0]} for {sample_or_year} {idx}', y=1.1)
                #plt.xlabel("Custom X-Axis Label")
                plt.subplots_adjust(top=0.9, bottom=0.2) 
                if custom_figsize: plt.gcf().set_size_inches(patient_fig_w, patient_fig_h)
                plt.savefig(f'{plot_dir}/{algo}_decision_plot_HC_{idx}.pdf', bbox_inches='tight')
                plt.clf()

                print('Generating plot: decision_plot_PFC')
                shap.decision_plot(explainer_expected_value[1], shap_values[1][idx], features=state_log[idx], feature_names=feature_names, show=show_fig)
                plt.title(f'{algo}: Decision Plot: {output_names[1]} for {sample_or_year} {idx}', y=1.1)
                #plt.xlabel("Custom X-Axis Label")
                plt.subplots_adjust(top=0.9, bottom=0.2) 
                if custom_figsize: plt.gcf().set_size_inches(patient_fig_w, patient_fig_h)
                plt.savefig(f'{plot_dir}/{algo}_decision_plot_PFC_{idx}.pdf', bbox_inches='tight')
                plt.clf()

                # (Paper): Probability Decision Plots not needed for paper
                # # Probability Decision plots for each action (ΔI_HC and ΔI_PFC)
                # print('Generating plot: decision_prob_plot_HC')
                # shap.decision_plot(explainer_expected_value[0], shap_values[0][idx], features=state_log[idx], 
                #                 link='logit', feature_names=feature_names, show=show_fig)
                # plt.title(f'{algo}: Decision Plot: {output_names[0]} for {sample_or_year} {idx}', y=1.1)
                # plt.xlabel("Model output value (Probability)")
                # plt.subplots_adjust(top=0.9, bottom=0.2) 
                # if custom_figsize: plt.gcf().set_size_inches(patient_fig_w, patient_fig_h)
                # plt.savefig(f'{plot_dir}/{algo}_decision_prob_plot_HC_{idx}.pdf', bbox_inches='tight')
                # plt.clf()

                # print('Generating plot: decision_prob_plot_PFC')
                # shap.decision_plot(explainer_expected_value[1], shap_values[1][idx], features=state_log[idx], 
                #                 link='logit', feature_names=feature_names, show=show_fig)
                # plt.title(f'{algo}: Decision Plot: {output_names[1]} for {sample_or_year} {idx}', y=1.1)
                # plt.xlabel("Model output value (Probability)")
                # plt.subplots_adjust(top=0.9, bottom=0.2) 
                # if custom_figsize: plt.gcf().set_size_inches(patient_fig_w, patient_fig_h)
                # plt.savefig(f'{plot_dir}/{algo}_decision_prob_plot_PFC_{idx}.pdf', bbox_inches='tight')
                # plt.clf()

                # (Paper): Waterfall plots (without mean values displayed) not needed
                # # Legacy Waterfall plots for each action (ΔI_HC and ΔI_PFC)
                # print('Generating plot: waterfall_plot_HC')
                # shap.plots._waterfall.waterfall_legacy(explainer_expected_value[0], shap_values[0][idx], feature_names=feature_names, show=show_fig)   
                # plt.title(f'{algo}: Waterfall Plot: {output_names[0]} for {sample_or_year} {idx}', y=1.1)
                # #plt.xlabel("Custom X-Axis Label")
                # plt.subplots_adjust(left= 0.2, right=0.8, top=0.8, bottom=0.2)  
                # if custom_figsize: plt.gcf().set_size_inches(patient_fig_w, patient_fig_h)
                # plt.savefig(f'{plot_dir}/{algo}_waterfall_plot_HC_{idx}.pdf')
                # plt.clf()
                
                # print('Generating plot: waterfall_plot_PFC')
                # shap.plots._waterfall.waterfall_legacy(explainer_expected_value[1], shap_values[1][idx], feature_names=feature_names, show=show_fig)
                # #shap.plots.waterfall(explainer_expected_value[1], shap_values[1][idx], feature_names=feature_names, show=show_fig)  
                # plt.title(f'{algo}: Waterfall Plot: {output_names[1]} for {sample_or_year} {idx}', y=1.1)
                # #plt.xlabel("Custom X-Axis Label")
                # plt.subplots_adjust(left= 0.2, right=0.8, top=0.8, bottom=0.2)  
                # if custom_figsize: plt.gcf().set_size_inches(patient_fig_w, patient_fig_h)
                # plt.savefig(f'{plot_dir}/{algo}_waterfall_plot_PFC_{idx}.pdf')
                # plt.clf()

                # Waterfall plots (with mean value displayed on y-axis) for action (ΔI_HC and ΔI_PFC)
                print('Generating plot: waterfall_plot_HC_with_mean')
                explanation_waterfall_0 = shap.Explanation(shap_values[0], explainer_expected_value[0], state_log, feature_names=feature_names)
                shap.plots.waterfall(explanation_waterfall_0[idx], show=show_fig)  
                plt.title(f'{algo}: Waterfall Plot: {output_names[0]} for {sample_or_year} {idx}', y=1.1)
                plt.subplots_adjust(left= 0.3, right=0.8, top=0.8, bottom=0.2)  
                if custom_figsize: plt.gcf().set_size_inches(patient_fig_w, patient_fig_h)
                plt.savefig(f'{plot_dir}/{algo}_waterfall_plot_HC_{idx}_with_mean.pdf')
                plt.clf()
                
                print('Generating plot: waterfall_plot_PFC_with_mean')
                explanation_waterfall_1 = shap.Explanation(shap_values[1], explainer_expected_value[1], state_log, feature_names=feature_names)
                shap.plots.waterfall(explanation_waterfall_1[idx], show=show_fig)
                plt.title(f'{algo}: Waterfall Plot: {output_names[1]} for {sample_or_year} {idx}', y=1.1)
                plt.subplots_adjust(left= 0.3, right=0.8, top=0.8, bottom=0.2)  
                if custom_figsize: plt.gcf().set_size_inches(patient_fig_w, patient_fig_h)
                plt.savefig(f'{plot_dir}/{algo}_waterfall_plot_PFC_{idx}_with_mean.pdf')
                plt.clf()

                # MultiOutput-Decision plot. Plots all outputs for a single observation (given by row_index).
                print('Generating plot: multioutput_decision_plot')
                shap.multioutput_decision_plot(list(explainer_expected_value), list(shap_values), row_index=idx, feature_names=feature_names, 
                                            legend_labels=output_names, legend_location="lower right", show=show_fig)
                plt.title(f'{algo}: Multi-Output Decision Plot for {sample_or_year} {idx}', y=1.1)
                if custom_figsize: plt.gcf().set_size_inches(patient_fig_w, patient_fig_h)
                plt.savefig(f'{plot_dir}/{algo}_multioutput_decision_plot_{idx}.pdf', bbox_inches='tight')
                plt.clf()

                plt.close()

        
