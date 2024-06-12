from multiprocessing import Process, JoinableQueue, Manager
import sys, json, time, os
from glob import glob
from munch import Munch
import numpy as np
import pandas as pd

from train import main
from xrl import generate_shap, plot_shap
from utils.postprocess import post_process
from utils.process_rl_plots import process_rl_plots
from utils.find_fold import find_fold
from utils.patient_filter import filter_patient
    
# Command-line argument for agent configurations
agent_configs = sys.argv[1]

# Create a joinable queue for task distribution
q = JoinableQueue()

# Number of threads for parallel processing
NUM_THREADS = 60

# Delete the results summary file if it already exists
output_filename = f'results/summary_adni.csv'
if os.path.isfile(output_filename):
    os.remove(output_filename)
    print(f"Deleted old result file:{output_filename}")

# Start the timer
master_start_time = time.time()

# Create shared dictionaries for storing SHAP data across threads
manager = Manager()
shap_values_per_algo = manager.dict()
state_log_per_algo = manager.dict()
state_log_per_algo_per_RID = manager.dict()
explainer_ev_per_algo = manager.dict()
# Create a shared queue for storing the experiment configuration
exp_config = manager.Queue() 

# ************************ Running Experiments (Train or Eval) *****************************

# Function to run a single configuration
def run_single_config(queue):
    while True:
        # Get a configuration path from the queue
        conf_path = queue.get()
        params = json.load(open(conf_path))
        exp_start_time = time.time()
        try:
            # Call the main function with configuration parameters
            shap_values, state_log, explainer_expected_value, state_log_dict = main(Munch(params))

            # Get the algo from the configuration file
            algo = params["algo"]
            exp_config.put(params)      # Put the configuration into the shared queue
            
            # Create a new entry (empty numpy array of correct shape) in the shared dictionaries if it doesn't exist
            if algo not in shap_values_per_algo:
                shap_values_per_algo[algo] = np.empty((2, 0, 6))    # 2 actions, N samples, 6 features
                state_log_per_algo[algo] = np.empty((0, 6))         # N samples, 6 features
                state_log_per_algo_per_RID[algo] = manager.list()
                explainer_ev_per_algo[algo] = np.empty((0, 2))      # N samples, 2 actions

            # Append the SHAP data to the shared dictionaries. Note the axes (append across N samples).
            shap_values_per_algo[algo] = np.concatenate((shap_values_per_algo[algo], shap_values), axis=1)
            state_log_per_algo[algo] = np.concatenate((state_log_per_algo[algo], state_log), axis=0)
            state_log_per_algo_per_RID[algo].append(state_log_dict)
            explainer_ev_per_algo[algo] = np.concatenate((explainer_ev_per_algo[algo], explainer_expected_value), axis=0)
               
        except Exception as e:
            print("ERROR", e)
            raise e
        exp_end_time = time.time() 
        print(f"Single Train/Eval Execution time: {round((exp_end_time - exp_start_time)/60, 2)} minutes")
        queue.task_done()

# Start multiple worker processes for parallel execution
for i in range(NUM_THREADS):
    worker = Process(target=run_single_config, args=(q,))
    worker.daemon = True
    worker.start()

# Put configuration file paths into the queue for processing
for fname in glob(os.path.join(agent_configs, "*.json")):
    q.put(fname)

# Wait for all tasks in the queue to be completed
q.join()

# retreive the config from the queue
while not exp_config.empty():
    config = exp_config.get()
    
print(f"\n\nExperiment Configuration (Sample): {config}")

master_end_time = time.time() 
master_execution_time = round((master_end_time - master_start_time)/60, 2)
print(f"\nTotal Train/Eval Execution time: {master_execution_time} minutes\n")

# ********************************* Post Processing Results ***************************************

# Post-process the results. This loads the summary_adni.csv file, sorts results, calculates mean scores for each method across seeds and folds.
post_process(output_filename)

algos = np.unique(pd.read_csv("results/summary_adni.csv")["algo"])

# ************************** Plotting RL Predictions and Ground Truth *****************************
print(f'\n***Starting RL plotting for algos: {algos}***\n')
plot_rl_start_time = time.time()

# Read the individual result files, aggregate metrics for each RL method and generate RL prediction plots
combined_df = process_rl_plots(algos, config['score'])

# Print the execution time for the experiment and plotting
plot_rl_end_time = time.time()
rl_plotting_time = round((plot_rl_end_time - plot_rl_start_time)/60, 2)
print(f"Total RL Plotting time: {rl_plotting_time} minutes")
print(f"Total Experiment time till now: {round((rl_plotting_time + master_execution_time), 2)} minutes\n")

# ********************************** SHAP Plots ****************************************************
# Now that all the tasks are done, we can get the results back from the shared dictionaries.

# Convert the Manager dict back to a regular dict. Create a new dict for mean_ev.
shap_values_per_algo_final = dict(shap_values_per_algo)
state_log_per_algo_final = dict(state_log_per_algo)
state_log_per_algo_per_RID_final = dict(state_log_per_algo_per_RID)
for d in state_log_per_algo_per_RID_final:
    state_log_per_algo_per_RID_final[d] = list(state_log_per_algo_per_RID_final[d])
explainer_ev_per_algo_final = dict(explainer_ev_per_algo)
mean_ev_per_algo_final = dict()

# algos = list(shap_values_per_algo_final.keys())

aggregated_state_log_per_algo_per_RID_final = {}
shap_algo_plotting_time, shap_patient_plotting_time = 0, 0

# if shap_enable_flag is True, then plot SHAP values (Paper: only plot SHAP for MMSE scores)
if config['shap_enable'] and config['score']=="MMSE":
    print(f'\n***Starting SHAP plotting***\n')

    # **************************** Plotting SHAP for each Algo ***************************************
    # Loop through algos and create SHAP plots for each algo
    shap_start_time = time.time()

    for algo in algos:
        if algo in ["TRPO-LSTM", "PPO-LSTM"]:
            print(f"\nWhile processing {algo}: SHAP is currently not supported for LSTM variants.")
            continue
        
        aggregated_state_log_per_algo_per_RID_final[algo] = {}
        for d in state_log_per_algo_per_RID_final[algo]:
            for id, values_list in d.items():
                values_array = np.array(values_list)
                if id not in aggregated_state_log_per_algo_per_RID_final[algo]:
                    aggregated_state_log_per_algo_per_RID_final[algo][id] = np.empty((0, 6))
                aggregated_state_log_per_algo_per_RID_final[algo][id] = np.concatenate((aggregated_state_log_per_algo_per_RID_final[algo][id], values_array), axis=0)

        print(f'\n***Preparing data for SHAP plotting for {algo}***\n')
        print(f'{algo}: Shap Values Shape:', np.shape(shap_values_per_algo_final[algo]))
        print(f'{algo}: State Log Shape:', np.shape(state_log_per_algo_final[algo]))
        print(f'{algo}: Explainer EV Shape:', np.shape(explainer_ev_per_algo_final[algo]))

        # Transpose the Explainer Expected Values since each experiment has 2 values, one for each action. 
        # We do this because we want to average the values for each action across all experiments. So (N, 2) -> (2, N). 
        # For example, if there are 3 experiments, the expected values are transormed from [[1,2], [3,4], [5,6]] -> [[1,3,5], [2,4,6]] 
        explainer_ev_per_algo_final[algo] = explainer_ev_per_algo_final[algo].T
        print(f'{algo}: Explainer EV Shape after transpose: ', np.shape(explainer_ev_per_algo_final[algo]))
        
        # Now we take mean across experiments for each action's expected values. So [[1,3,5], [2,4,6]] -> [9/3, 12/3] -> [3, 4]
        mean_ev_per_algo_final[algo] = np.mean(explainer_ev_per_algo_final[algo], axis=1)
        print(f'{algo}: mean_ev_per_algo_final: {mean_ev_per_algo_final[algo]}')

        # Finally, we pass the SHAP values, state log, and expected values (mean across experiments) to the SHAP plotting function
        plot_shap(algo, shap_values_per_algo_final[algo], state_log_per_algo_final[algo], mean_ev_per_algo_final[algo], plot_dir=f'plots_shap/{algo}')

    shap_end_time = time.time()
    shap_algo_plotting_time = round((shap_end_time - shap_start_time)/60, 2)
    print(f"SHAP Algo Plotting time: {shap_algo_plotting_time} minutes")

    # **************************** Plotting SHAP for Patients ***************************************
    # Plotting SHAP for selected patients
    shap_patient_start_time = time.time()

    for algo in algos:
        if algo in ["TRPO-LSTM", "PPO-LSTM"]:
            print(f"\nWhile processing {algo}: SHAP is currently not supported for LSTM variants.")
            continue

        seeds = np.unique(pd.read_csv('results/summary_adni.csv')["seed"])

        def per_patient_shap_evaluation(filter_feature, num_patients, num_seeds=5, years_of_data=6):

            sorted_rids = filter_patient(combined_df, algo, filter_feature, num_patients, num_seeds, years_of_data)
            for rid in sorted_rids:
                shap_values_final = np.empty((2, 0, 6))
                explainer_ev_final = np.empty((0, 2))

                for i, seed in enumerate(seeds):
                    fold = find_fold(rid)
                    print(f"Generating shap values for {algo}, fold {fold}, seed {seed} ")
                    shap_values, _, explainer_ev = generate_shap(f'progress/{algo}/adni_fold{fold}/seed_{seed}', 
                                                                    state_log=aggregated_state_log_per_algo_per_RID_final[algo][rid][i*11:(i*11)+11],   # separate state_log (e.g. 55) into 11 samples per seed (55->11,11,11,11,11)
                                                                    action_log=None, 
                                                                    use_all_samples=True)
                    # print(shap_values.shape)
                    # print(explainer_ev.shape)
                    shap_values_final = np.concatenate((shap_values_final, shap_values), axis=1)
                    explainer_ev_final = np.concatenate((explainer_ev_final, explainer_ev), axis=0)

                print("Shap Values Shape:", shap_values_final.shape)
                print("State Log Shape:", aggregated_state_log_per_algo_per_RID_final[algo][rid].shape)
                print("Explainer EV Shape:", explainer_ev_final.shape)

                # Transpose the Explainer Expected Values since each experiment has 2 values, one for each action. 
                explainer_ev_final = explainer_ev_final.T
                print("Explainer EV Shape after transpose:", explainer_ev_final.shape)

                # Now we take mean across experiments for each action's expected values. So [[1,3,5], [2,4,6]] -> [9/3, 12/3] -> [3, 4]
                mean_ev = np.mean(explainer_ev_final, axis=1)
                print("Mean of Explainer EV", mean_ev.shape, mean_ev)
                
                # Call the SHAP plotting function for the selected patient(s)
                # Temporary for generating plots only for this RID
                if rid == 4294:
                    plot_shap(f'RID-{rid}_{algo}', shap_values_final, aggregated_state_log_per_algo_per_RID_final[algo][rid], mean_ev, f'plots_shap/{algo}/patient_RID_{rid}', is_patient=True)
        
        # enable per-patient shap only for specific algos (e.g. TRPO)
        if algo == "TRPO":
            # Plot SHAP for selected patients using different filters
            #per_patient_shap_evaluation(f'{config["score"]}_norm', num_patients=1, num_seeds=len(seeds))     # select the top num_patients with the most decline in MMSE score
            #per_patient_shap_evaluation('mae', num_patients=1, num_seeds=len(seeds))               # select the top num_patients with the least MAE (best prediction)
            per_patient_shap_evaluation('range', num_patients=3, num_seeds=len(seeds))             # select the top num_patients with the MMSE decline in a certain range

    shap_patient_end_time = time.time()
    shap_patient_plotting_time = round((shap_patient_end_time - shap_patient_start_time)/60, 2)
    print(f"SHAP Patient Plotting time: {shap_patient_plotting_time} minutes")

print(f"\nTotal Train/Eval Execution time: {master_execution_time} minutes")
print(f"Total RL Plotting time: {rl_plotting_time} minutes")
if config['shap_enable']:
    print(f"SHAP Algo Plotting time: {shap_algo_plotting_time} minutes")
    print(f"SHAP Patient Plotting time: {shap_patient_plotting_time} minutes")
    print(f"Total Experiment time: {round((master_execution_time + rl_plotting_time + shap_algo_plotting_time + shap_patient_plotting_time), 2)} minutes")
else:
    print(f"Total Experiment time: {round((master_execution_time + rl_plotting_time), 2)} minutes")