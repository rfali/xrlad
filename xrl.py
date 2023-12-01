import numpy as np
import tensorflow as tf
from garage.experiment import Snapshotter
import shap
import matplotlib.pyplot as plt
import os

def shap_explain(log_dir, output_dir, state_log, action_log, split_type, cmap_value='coolwarm', use_all_samples=False, show_fig=False):
    """
    Generate SHAP (SHapley Additive exPlanations) plots for model interpretation.

    Args:
        log_dir (str): Directory containing trained model data.
        output_dir (str): Directory to save plots.
        state_log (numpy.ndarray): Log of states or observations.
        action_log (numpy.ndarray): Log of actions taken.
        split_type (str): Type of data split (e.g., 'train', 'test').
        cmap_value (str, optional): Colormap for plotting. Defaults to 'coolwarm'.
        use_all_samples (bool, optional): Whether to use all samples for background data. Defaults to False.
        show_fig (bool, optional): Whether to display plots. Defaults to False.

    Returns:
        None
    """
        

    # Create a directory for saving SHAP plots
    plot_dir = f'{output_dir}/shap_plots/{split_type}'
    os.makedirs(plot_dir, exist_ok=True)

    #use_all_samples = #True, use all samples to build background, else use kmeans and generate `shap_samples_to_use` samples
    shap_samples_to_use = 20
    
    # Define feature names
    feature_names = ['$I_{PFC} (t-1)$','$I_{HC} (t-1)$' ,'$X_{PFC}$','$X_{HC}$','$D_{PFC}$','$D_{HC}$']

    # Reset TensorFlow default graph
    tf.compat.v1.reset_default_graph()

    # Initialize the Snapshotter
    snapshotter = Snapshotter()

    # Create a TensorFlow session
    session = tf.compat.v1.Session()

    verbose = False

    # Load trained model data
    with session:
        trained_data = snapshotter.load(log_dir)
        policy = trained_data['algo'].policy

        if verbose:
            # Print shapes of state and action logs
            print('shape: state_log', state_log.shape)
            print('shape: action_log', action_log.shape)

            # Get sample action from the policy
            action = policy.get_action(state_log[0])[0]                 # get sample action, equal to model.predict(obs)
            action_mean = policy.get_action(state_log[0])[1]['mean']    # garage also has a way to get the mean of all posstible actions, refer to garage documentation
            
            # Print information about the sample action
            print('state for sample 0: ', state_log[0])
            print('action for sample 0: ', action)
            print('action_mean: ', action_mean)
            print('action shape: ', np.shape(action))
        
        # Define the RL model to be used for SHAP using the policy
        rl_model = lambda x: policy.get_actions(x)[0]
        #rl_model = np.reshape(lambda X: policy.get_action(X)[1]['mean'], (1133,6))
        
        # Initialize a SHAP KernelExplainer
        explainer = shap.KernelExplainer(rl_model, state_log)

        # takes about 7 minutes to build shap_values for all 1133 samples
        if use_all_samples:     
            print(f'Using all {state_log.shape[0]} samples to build shap\'s background data')
            shap_values = explainer.shap_values(state_log)  # use all 1133 samples to build background data
            if verbose: print('slow - shape of shap values', np.shape(shap_values))
        
        # to counter "Using x background data samples could cause slower run times."
        else:                   
            print(f'Using limited {shap_samples_to_use} samples to build shap\'s background data')
            background = shap.sample(state_log, shap_samples_to_use)
            shap_values = explainer.shap_values(background) 

            # Update the state_log (this is necessary for beeswarm plot as it requries shape of shap_values and state_log to be the same)
            state_log = shap.sample(state_log, shap_samples_to_use)
            if verbose: print('fast - shape of shap values', np.shape(shap_values))
            if verbose: print('fast - shape of state_log', np.shape(state_log))
        
        # Create a SHAP Explanation object using the shap_values
        explanation = shap.Explanation(shap_values, explainer.expected_value, state_log, feature_names=feature_names)
        
        # Generate and save SHAP plots
        shap.initjs()
        if verbose:print('shape: explanation.values', np.shape(explanation.values))
        if verbose: print('shape: shap_values', np.shape(shap_values)) # (2, 10, 6) -> (actions, shap_samples_to_use, features)

        # Clear the figure
        plt.clf()
        print(f'\n*** Saving SHAP plots to {plot_dir} ***\n')

        ########### Global Explanations  ##########  
        # Following are explanations for all patient samples (shap_values[0] = ΔI_PFC, shap_values[1] = ΔI_HC)
        print('Generating Global Explanations..')
        plot_dir = f'{plot_dir}/global'
        os.makedirs(plot_dir, exist_ok=True)

        # Summary bar plot
        print('Generating plot: summary_bar')
        class_names = ['$\Delta I_{PFC}(t)$', '$\Delta I_{HC}(t)$']
        shap.summary_plot(shap_values, feature_names=feature_names, class_names=class_names, show=show_fig)
        plt.title('Summary Plot', y=0.95)
        #plt.xlabel("Custom X-Axis Label")
        plt.savefig(f'{plot_dir}/summary_bar.png')
        plt.clf()
        
        # Summary bar plots for each action (ΔI_PFC and ΔI_HC)
        print('Generating plot: summary_bar_PFC')
        shap.summary_plot(shap_values[0], features=state_log, plot_type="bar", feature_names=feature_names, show=show_fig)
        plt.title(f'Summary Plot: {class_names[0]}', y=0.95)
        #plt.xlabel("Custom X-Axis Label")
        plt.savefig(f'{plot_dir}/summary_bar_PFC.png')
        plt.clf()
        
        print('Generating plot: summary_bar_HC')
        shap.summary_plot(shap_values[1], features=state_log, plot_type="bar", feature_names=feature_names, show=show_fig)
        plt.title(f'Summary Plot: {class_names[1]}', y=0.95)
        #plt.xlabel("Custom X-Axis Label")
        plt.savefig(f'{plot_dir}/summary_bar_HC.png')
        plt.clf()

        # Beeswarm plots for each action (ΔI_PFC and ΔI_HC)
        print('Generating plot: beeswarm_PFC')
        shap.summary_plot(shap_values[0], features=state_log, feature_names=feature_names, show=show_fig)
        plt.title(f'Beeswarm Plot: {class_names[0]}', y=0.98)
        #plt.xlabel("Custom X-Axis Label")
        plt.savefig(f'{plot_dir}/beeswarm_PFC.png')
        plt.clf()
       
        print('Generating plot: beeswarm_HC')
        shap.summary_plot(shap_values[1], features=state_log, feature_names=feature_names, show=show_fig)
        plt.title(f'Beeswarm Plot: {class_names[1]}', y=0.98)
        #plt.xlabel("Custom X-Axis Label")
        plt.savefig(f'{plot_dir}/beeswarm_HC.png')
        plt.clf()

        # Decision plots for each action (ΔI_PFC and ΔI_HC)
        print('Generating plot: decision_plot_PFC')
        shap.decision_plot(explainer.expected_value[0], shap_values[0], features=state_log, feature_names=feature_names, show=show_fig)
        #plt.title(f'Decision Plot: {class_names[0]} for sample {idx}', y=0.95)
        #plt.xlabel("Custom X-Axis Label")
        plt.subplots_adjust(top=0.9, bottom=0.2)  # Adjust margins as needed
        plt.savefig(f'{plot_dir}/decision_plot_PFC.png')
        plt.clf()

        print('Generating plot: decision_plot_HC')
        #plt.title(f'Decision Plot: {class_names[1]} for sample {idx}', y=0.95)
        #plt.xlabel("Custom X-Axis Label")
        shap.decision_plot(explainer.expected_value[1], shap_values[1], features=state_log, feature_names=feature_names, show=show_fig)
        plt.subplots_adjust(top=0.9, bottom=0.2)  # Adjust margins as needed
        plt.savefig(f'{plot_dir}/decision_plot_HC.png')
        plt.clf()

        # Stacked Force plots for each action (ΔI_PFC and ΔI_HC)
        print('Generating plot: stacked_force_PFC')
        force_plot = shap.force_plot(explainer.expected_value[0], shap_values[0], feature_names=feature_names, show=False)
        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        with open(f'{plot_dir}/stacked_force_PFC.html', "w") as file:
            file.write(shap_html)
        plt.clf()
        
        print('Generating plot: stacked_force_HC')
        force_plot = shap.force_plot(explainer.expected_value[1], shap_values[1], feature_names=feature_names, show=False)
        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        with open(f'{plot_dir}/stacked_force_HC.html', "w") as file:
            file.write(shap_html)
        plt.clf()
        
        ########### Local Explanations  ##########
        # Following are explanations for a particular patient sample (shap_values[0][X] = ΔI_PFC, shap_values[1][X] = ΔI_HC)
        print('\nGenerating Local Explanations..')
        idx = 0 # index of the sample to be explained
        plot_dir = plot_dir.replace("global", "local_yr_"+str(idx))
        os.makedirs(plot_dir, exist_ok=True)

        # Force plots for each action (ΔI_PFC and ΔI_HC)
        print('Generating plot: force_PFC')
        shap.force_plot(explainer.expected_value[0], shap_values[0][idx], features=feature_names, matplotlib=True, show=show_fig)
        #plt.title(f'Force Plot: {class_names[0]} for sample {idx}', y=0.95)
        #plt.xlabel("Custom X-Axis Label")
        plt.subplots_adjust(top=0.9, bottom=0.2) 
        plt.savefig(f'{plot_dir}/force_PFC_{idx}.png')
        plt.clf()
        
        print('Generating plot: force_HC')
        shap.force_plot(explainer.expected_value[1], shap_values[1][idx], features=feature_names, matplotlib=True, show=show_fig)
        #plt.title(f'Force Plot: {class_names[1]} for sample {idx}', y=0.95)
        #plt.xlabel("Custom X-Axis Label")
        plt.subplots_adjust(top=0.9, bottom=0.2)
        plt.savefig(f'{plot_dir}/force_HC_{idx}.png')
        plt.clf()

        # Decision plots for each action (ΔI_PFC and ΔI_HC)
        print('Generating plot: decision_plot_PFC')
        shap.decision_plot(explainer.expected_value[0], shap_values[0][idx], features=state_log, feature_names=feature_names, show=show_fig)
        #plt.title(f'Decision Plot: {class_names[0]} for sample {idx}', y=0.95)
        #plt.xlabel("Custom X-Axis Label")
        plt.subplots_adjust(top=0.9, bottom=0.2) 
        plt.savefig(f'{plot_dir}/decision_plot_PFC_{idx}.png')
        plt.clf()

        print('Generating plot: decision_plot_HC')
        shap.decision_plot(explainer.expected_value[1], shap_values[1][idx], features=state_log, feature_names=feature_names, show=show_fig)
        #plt.title(f'Decision Plot: {class_names[1]} for sample {idx}', y=0.95)
        #plt.xlabel("Custom X-Axis Label")
        plt.subplots_adjust(top=0.9, bottom=0.2) 
        plt.savefig(f'{plot_dir}/decision_plot_HC_{idx}.png')
        plt.clf()

        # Waterfall plots for each action (ΔI_PFC and ΔI_HC)
        print('Generating plot: waterfall_plot_PFC')
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shap_values[0][idx], feature_names=feature_names, show=show_fig)   
        plt.title(f'Waterfall Plot: {class_names[0]} for sample {idx}', y=1.10)
        #plt.xlabel("Custom X-Axis Label")
        plt.subplots_adjust(left= 0.2, right=0.8, top=0.8, bottom=0.2)  
        plt.savefig(f'{plot_dir}/waterfall_plot_PFC_{idx}.png')
        plt.clf()
        
        print('Generating plot: waterfall_plot_HC')
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1], shap_values[1][idx], feature_names=feature_names, show=show_fig)
        #shap.plots.waterfall(explainer.expected_value[1], shap_values[1][idx], feature_names=feature_names, show=show_fig)  
        plt.title(f'Waterfall Plot: {class_names[1]} for sample {idx}', y=1.10)
        #plt.xlabel("Custom X-Axis Label")
        plt.subplots_adjust(left= 0.2, right=0.8, top=0.8, bottom=0.2)  
        plt.savefig(f'{plot_dir}/waterfall_plot_HC_{idx}.png')
        plt.clf()

        # Waterfall plots (with mean value displayed on y-axis) for action (ΔI_PFC and ΔI_HC)
        # The resulting waterfall plots are almost identical, so leaving it commented out for now
        '''
        explanation_waterfall_0 = shap.Explanation(shap_values[0], explainer.expected_value, state_log, feature_names=feature_names)
        explanation_waterfall_1 = shap.Explanation(shap_values[1], explainer.expected_value, state_log, feature_names=feature_names)
        print('Generating plot: waterfall_mean_plot_PFC')
        shap.plots.waterfall(explanation_waterfall_0[idx], show=show_fig)  
        plt.title(f'Waterfall Plot: {class_names[0]}', y=1.10)
        plt.subplots_adjust(left= 0.2, right=0.8, top=0.8, bottom=0.2)  
        plt.savefig(f'{plot_dir}/waterfall_mean_plot_PFC_{idx}.png')
        plt.clf()
        
        print('Generating plot: waterfall_mean_plot_HC')
        shap.plots.waterfall(explanation_waterfall_1[idx], show=show_fig)
        plt.title(f'Waterfall Plot: {class_names[1]} for sample {idx}', y=1.10)
        plt.subplots_adjust(left= 0.2, right=0.8, top=0.8, bottom=0.2)  
        plt.savefig(f'{plot_dir}/waterfall_mean_plot_HC_{idx}.png')
        plt.clf()
        '''

        plt.close()

        #shap.summary_plot(explanation)
        #shap.summary_plot(explanation[0])
        #shap.summary_plot(explanation[1])
        #shap.multioutput_decision_plot(explainer.expected_value[0], shap_values[0][0])
        #shap.plots.waterfall(explanation[0])
        #shap.plots.beeswarm(explanation)