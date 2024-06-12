import os, time
import json
from datetime import datetime
import numpy as np
import pandas as pd

import tensorflow as tf
from garage.envs import GarageEnv
from garage.envs import normalize
from garage.experiment.deterministic import set_seed
from garage import wrap_experiment
from garage.experiment import LocalTFRunner
from garage.experiment import Snapshotter
from garage.sampler.utils import rollout

from garage.np.baselines import LinearFeatureBaseline
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.algos import TRPO, DDPG, PPO
from garage.tf.policies import CategoricalMLPPolicy,ContinuousMLPPolicy, GaussianMLPPolicy, GaussianLSTMPolicy
from garage.tf.q_functions import ContinuousMLPQFunction
from garage.tf.optimizers import FirstOrderOptimizer, ConjugateGradientOptimizer, FiniteDifferenceHvp
from garage.torch.algos import SAC
from garage.torch.policies import TanhGaussianMLPPolicy, GaussianMLPPolicy as TorchGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction as TorchContinuousMLPQFunction

from garage.np.exploration_policies import AddOrnsteinUhlenbeckNoise
from garage.replay_buffer import PathBuffer
from garage.sampler import RaySampler, LocalSampler, WorkerFactory

from brain_env import BrainEnv
from eval import EvalPolicy
from xrl import generate_shap
from utils.plot import plot_patient

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def load_data_syn(filename):
    """
    Load Synthetic data from an Excel file.

    Args:
        filename (str): The path to the Excel file containing the synthetic data. The Excel file should contain three sheets named 'train', 'valid', and 'test'.

    Returns:
        tuple: A tuple containing three DataFrames for the train, validation, and test datasets.

    """
    df = {}
    for i in ['train', 'valid', 'test']:
        df[i] = pd.read_excel(filename, sheet_name=i)
    return df['train'], df['valid'], df['test']


def load_adni_data(args, filename):
    """
    Load ADNI data from an Excel file and join it with additional Parameters data from a separte file (with same name but with "_parameters" in the filename).

    Args:
        args: Config parameters specific to an experiment (in json format).
        filename (str): The path to the Excel file containing the data.

    Returns:
        tuple: A tuple containing three DataFrames for the train, validation, and test datasets.

    """
    df = {}
    for i in ['train', 'valid', 'test']:
        # Read data from the main Excel file
        df_temp = pd.read_excel(filename, sheet_name=i)

        # Read additional parameters data from a different Excel file
        df_params = pd.read_excel(filename.replace(".xls", "_parameters.xls"),
                                  sheet_name=f'{args.score}_norm_PTGENDER_APOEPOS')

        # Join the main data with the parameters data using a common column 'RID'
        df[i] = df_temp.set_index('RID').join(df_params.set_index('RID'), on='RID', how='inner', rsuffix='_right')

        # Reset the index to make 'RID' a regular column
        df[i].reset_index(level=0, inplace=True)

    return df['train'], df['valid'], df['test']


def extract_params(args, df):
    """
    Extract initial values of parameters for Synthetic dataset.

    Args:
        args: Config parameters specific to an experiment (in json format).
        df (DataFrame): The DataFrame containing the data.

    Returns:
        A 3-tuple containing initial values (list), brain activity values (numpy.ndarray), and time post onset of amyloid deposition (numpy.ndarray).
        
        Initial values (list): A list containing the following extracted and processed data:
            alpha1_init (numpy.ndarray): Initial values for alpha1.
            alpha2_init (numpy.ndarray): Initial values for alpha2.
            gamma_init (numpy.ndarray): Initial values for gamma.
            beta_init (numpy.ndarray): Initial values for beta.
            D1_init (numpy.ndarray): Initial values for D1.
            X_V_init (numpy.ndarray): Initial values for X_V.
            end_times (numpy.ndarray): The last year of patient data for each subject.
            RIDs (numpy.ndarray): Unique patient IDs (RIDs).
            info_init (numpy.ndarray): Initial values for Information Processing I(t).
        FDG_init (numpy.ndarray): Initial values for Brain activity Y_V(t).
        tpo_array (numpy.ndarray): Values representing time (tpo_array).
    """
    # Extract unique subject IDs (RIDs) from the DataFrame
    RIDs = np.unique(df['RID'].values)

    # Calculate the negative difference between the 'reg1_av45'/'reg2_av45' column for each participant
    # and their next year (grouped by 'RID'). av_45 is the amyloid PET SUVR (D)
    df['D1_1'] = -df.groupby('RID')['reg1_av45'].diff(-1)
    df['D1_2'] = -df.groupby('RID')['reg2_av45'].diff(-1)

    # Group the DataFrame by 'RID' and take the first entry for each group
    df_first = df.groupby('RID', as_index=False).first()

    # Filter df_first to include only RIDs in the specified range
    df_first = df_first[df_first.RID.isin(RIDs[:])]

    # Extract various initialization values from df_first
    alpha1_init = df_first['alpha1'].values
    alpha2_init = df_first['alpha2'].values
    beta_init = df_first['beta'].values
    gamma_init = df_first['gamma'].values
    D1_init = df_first[['D1_1', 'D1_2']].values

    X_V_init = df_first[['reg1_mri', 'reg2_mri']].values
    FDG_init = df_first[['reg1_fdg', 'reg2_fdg']].values

    # Create an array for info_init based on cog_init or 'baseline'
    info_init = np.array([cog_init] * len(RIDs)) if args.cog_init != 'baseline' \
           else np.array([df_first[f'{args.score}_norm'].values / 2, df_first[f'{args.score}_norm'].values] / 2).T

    # Extract end_times and tpo_array from the DataFrame
    end_times = df.groupby('RID', as_index=False).last()['Years'].values
    # get tpo:time post onset of amyloid deposition
    tpo_array = df_first['tpo'].values  

    # Return a tuple of extracted values and arrays
    return [alpha1_init, alpha2_init, gamma_init, beta_init, D1_init, X_V_init, end_times, RIDs, info_init], FDG_init, tpo_array


def extract_data(args, df, H, gamma):
    """
    Extract and process data for the ADNI dataset.

    Args:
        args: Config parameters specific to an experiment (in json format).
        df (DataFrame): DataFrame containing the data.
        H (array-like): Laplacian of the adjacency matrix of the graph
        gamma (float): Value of gamma.

    Returns:
        list: A list containing the following extracted and processed data:
        - alpha1_init (array-like): Initial values for alpha1.
        - alpha2_init (array-like): Initial values for alpha2.
        - gamma_init (array-like): Initial values for gamma.
        - beta_init (array-like): Initial values for beta.
        - D0_init (array-like): Initial values for D0.
        - X_V_init (array-like): Initial values for X_V.
        - end_times (array-like): The last year of patient data for each subject.
        - RIDs (array-like): Unique RID values.
        - info_init (array-like): Initial values for info.

    """
    # Extract unique RID values from the 'RID' column.
    RIDs = np.unique(df['RID'].values)

    # Group data by 'RID' and take the first row of each group.
    df_first = df.groupby('RID', as_index=False).first()

    # Extract beta_init values from the 'beta_estm' column.
    beta_init = df_first['beta_estm'].values

    # Extract X_V_init values from the 'mri_FRONT_norm' and 'mri_HIPPO_norm' columns.
    X_V_init = df_first[['mri_FRONT_norm', 'mri_HIPPO_norm']].values

    # Extract end_times by finding the last 'Years' value for each unique RID.
    end_times = df.groupby('RID', as_index=False).last()['Years'].values

    # Initialize gamma_init with a constant value 'gamma' for each RID. 'gamma' is specified in the config file and fixed to 1.0.
    gamma_init = np.ones(len(RIDs)) * gamma

    # Extract alpha1_init values from the 'alpha1_estm' column.
    alpha1_init = df_first['alpha1_estm'].values

    # Calculate alpha2_init values based on 'alpha2_gamma_estm' and 'gamma'.
    alpha2_init = df_first['alpha2_gamma_estm'].values / gamma

    # Extract tpo_array values from the 'tpo_estm' column.
    tpo_array = df_first['tpo_estm'].values

    # Extract D1_init values from the 'FRONTAL_SUVR' and 'HIPPOCAMPAL_SUVR' columns.
    D1_init = df_first[['FRONTAL_SUVR', 'HIPPOCAMPAL_SUVR']].values

    # Calculate info_init based on args.cog_init.
    info_init = np.array([cog_init] * len(RIDs)) if args.cog_init != 'baseline' else np.array([df_first[f'{args.score}_norm'].values, 0 * df_first[f'{args.score}_norm'].values / 2]).T 
    
    # The model requires D(t) for the simulation although only φ(0) is available from baseline data.
    #  D(1) is computed from φ(0) separately for each individual. See eq 13, Sec C.4 of Saboo et al. 2019.
    
    # Calculate v_j and U from H matrix.
    v_j, U = np.linalg.eig(H)

    # Initialize D0_init with a copy of X_V_init.
    D0_init = X_V_init.copy()
    
    # Compute D0_init values based on beta values and other calculations.
    for i, beta in enumerate(beta_init):
        tpo = tpo_array[i]

        # Calculate diag_array using v_j, beta, and tpo.
        diag_array = np.diag([v_j[0] * np.exp(-v_j[0] * beta * tpo) / (1 - np.exp(-v_j[0] * beta * tpo)), 1 / (beta * tpo)])   # eq 13 of Saboo et al. 2019.

        # Extract phi_tpo values from D1_init for the current RID.
        phi_tpo = D1_init[i]

        # Compute 'mult' using U, diag_array, and U.T (transpose of U).
        mult = U.dot(diag_array).dot(U.T)
        mult = U.dot(diag_array).dot(U.T)

        # Update D0_init values based on the calculated values.
        D0_init[i, :] = beta * (mult).dot(phi_tpo.T)   

    # Return the extracted and processed data as a list.
    return [alpha1_init, alpha2_init, gamma_init, beta_init, D0_init, X_V_init, end_times, RIDs, info_init]



def calculate_D(D1_init, beta_init, alpha1_init, alpha2_init, H, fdg, tpo):
    """
    Calculate and update D_init values for Synthetic data.
    Similar to eq 13, Sec C.4 of Saboo et al. 2019.

    Args:
        D1_init (numpy.ndarray): Initial values for D1.
        beta_init (numpy.ndarray): Initial values for beta.
        alpha1_init (numpy.ndarray): Initial values for alpha1.
        alpha2_init (numpy.ndarray): Initial values for alpha2.
        H (numpy.ndarray): Laplacian of the adjacency matrix of the graph
        fdg (numpy.ndarray): Brain activity values.
        tpo (numpy.ndarray): Time post onset of amyloid deposition

    Returns:
        numpy.ndarray: Updated D_init values.

    """
    # Create a copy of D1_init to store the updated values.
    D_init = D1_init.copy()
    
    # Loop over each row in D1_init.
    for i in range(D1_init.shape[0]):        
        # Calculate D_old using matrix operations.
        D_old = np.linalg.inv(np.exp(-beta_init[i] * H)).dot(D1_init[i, :])
        
        # Update the corresponding row in D_init with D_old.
        D_init[i, :] = D_old

    return D_init


def main(args):
    """
    Main function for running the experiment.

    Args:
        args: Config parameters specific to an experiment (in json format).

    Returns:
        None
    """

    # Global variables
    global adj, cog_init, log_dir  

    # Extract arguments
    gamma = args.gamma
    max_time_steps = args.trainsteps
    epochs = args.epochs
    batch_size=args.batch_size
    gamma_type = args.gammatype
    action_type = args.actiontype
    scale_state = args.scale
    score = args.score
    seed = args.seed

    start_time = time.time()

    name = args.filename.split(".xls")[0]  # Extracting filename without extension

    # Create a 2x2 adjacency matrix
    adj = np.array([[0,1],[1,0]])  
    # Create a diagonal matrix based on adjacency matrix
    H = np.diag(np.sum(adj, axis=1)) - adj  

    # Setting the initial cognition values based on arguments
    if args.cog_init == 'full':
        cog_init = np.array([args.cog_mtl,10.0 - args.cog_mtl])
    else:
        cog_init = [None, None]

    # Save experiment configuration to a dict and then to a JSON file
    exp_config = {
        "name": name, "algo": args.algo, "max_time_steps": max_time_steps, "action_type": action_type,
        "gamma_type": gamma_type, "gamma": gamma, "epochs": epochs, "batch_size": batch_size, "action_limit": args.action_limit,
        "cog_type": args.cog_type, "cog_init": args.cog_init, "discount": args.discount, "w_lambda": args.w_lambda,"trainsteps": args.trainsteps,
        "energy_model": args.energy_model,"score": args.score,"network": args.network, "seed": seed, "normalize": args.normalize,
        "shap_enable": args.shap_enable, "shap_use_all_samples": args.shap_use_all_samples, "shap_show_fig": args.shap_show_fig,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Set up the log_dir where training logs of the RL method will be saved
    log_dir = f'progress/{args.algo}/{name}/seed_{seed}'
    
    # Set up the results dir where RL predicted patient data will be saved
    output_dir = log_dir.replace("progress", "results")

    # Modifying log directory based on configuration
    if args.cog_type == 'variable':
        log_dir = f'results/{name}_{args.algo}_{max_time_steps}_{10.0}_{action_type}_{gamma_type}_{gamma}_{epochs}_{batch_size}_{args.action_limit}_{args.cog_type}_{args.cog_init}_{args.discount}_{args.w_lambda}_{args.trainsteps}_{args.energy_model}_{args.score}_{args.network}'

    # If in evaluation mode, update cog_init
    if args.eval:
        cog_init = np.array([args.cog_mtl, 10.0 - args.cog_mtl])    # [I_HC, I_PFC i.e. 10 - I_HC]

    # Create output directory if it doesn't exist
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Write experiment configuration to a JSON file
    with open(output_dir+'/exp_config.json', 'w') as f:
        json.dump(exp_config, f, indent=4)

    # Load data based on data type
    if args.datatype == 'synthetic':
        filename = f'dataset/synthetic/{args.filename}'
        train_df, valid_df, test_df = load_data_syn(filename=filename)
        global train_data
        train_data, train_fdg, train_tpo = extract_params(args, train_df)
        valid_data, valid_fdg, valid_tpo = extract_params(args, valid_df)
        test_data, test_fdg, test_tpo = extract_params(args, test_df)
        train_data[4] = calculate_D(train_data[4].copy(), train_data[3], train_data[0], train_data[1], H, train_fdg, train_tpo)
        test_data[4] = calculate_D(test_data[4].copy(), test_data[3], test_data[0], test_data[1], H, test_fdg, test_tpo)
        valid_data[4] = calculate_D(valid_data[4].copy(), valid_data[3], valid_data[0], valid_data[1], H, valid_fdg, valid_tpo)
    elif args.datatype == 'adni':
        filename = f'dataset/adni/{args.filename}'
        train_df, valid_df, test_df = load_adni_data(args, filename=filename)
        train_data = extract_data(args, train_df, H, gamma)
        valid_data = extract_data(args, valid_df, H, gamma)
        test_data = extract_data(args, test_df, H, gamma)


    # Define a function to train a policy
    @wrap_experiment(log_dir=log_dir, archive_launch_repo=False, use_existing_dir=True)
    def train_policy(ctxt=None, seed=1, n_epochs=50, batch_size=1000, action_type='delta', 
                     gamma_type='variable', gamma=2.1, max_time_steps=10, algo_name='TRPO'):
        """
        Train policy with Brain environment.

        Parameters:
        -----------
        ctxt : object, optional
            Context for the experiment.
        seed : int, optional
            Random seed for reproducibility.
        n_epochs : int, optional
            Number of training epochs.
        batch_size : int, optional
            Batch size for training.
        action_type : str, optional
            Type of action space, e.g., 'delta'.
        gamma_type : str, optional
            Type of gamma values, e.g., 'variable'.
        gamma : float, optional
            Value of gamma.
        max_time_steps : int, optional
            Maximum time steps in an episode.
        algo_name : str, optional
            Name of the RL algorithm to use (e.g., 'TRPO').

        Returns:
        --------
        policy : object
            Trained policy for the Brain environment.
        """

        # Set random seed
        set_seed(seed)
        
        # Start training using the train_split (train_data)
        # Create a LocalTFRunner instance for training the model
        with LocalTFRunner(snapshot_config=ctxt) as trainer:
            # Get the gamma value from training data
            gamma_val = train_data[2]
            
            # Determine gamma_init and alpha2_new_init based on gamma_type
            if gamma_type == 'variable':
                gamma_init = train_data[2]
                alpha2_new_init = train_data[1]
            else:    
                gamma_init = np.ones(len(train_data[0])) * gamma
                alpha2_new_init = gamma_val * train_data[1] / gamma

            # Create the environment for reinforcement learning
            if args.normalize:
                env = normalize(GarageEnv(
                    BrainEnv(
                        max_time_steps=max_time_steps + 1,
                        alpha1_init=train_data[0],
                        alpha2_init=alpha2_new_init,
                        beta_init=train_data[3],
                        gamma_init=gamma_init,
                        X_V_init=train_data[5],
                        D_init=train_data[4],
                        cog_type=args.cog_type,
                        cog_init=train_data[-1],
                        adj=adj,
                        action_limit=args.action_limit,
                        w_lambda=args.w_lambda,
                        gamma_type=gamma_type,
                        action_type=action_type,
                        scale=False,
                        energy_model=args.energy_model
                    )
                ), normalize_obs=True)
            else:
                env = GarageEnv(
                    BrainEnv(
                        max_time_steps=max_time_steps + 1,
                        alpha1_init=train_data[0],
                        alpha2_init=alpha2_new_init,
                        beta_init=train_data[3],
                        gamma_init=gamma_init,
                        X_V_init=train_data[5],
                        D_init=train_data[4],
                        cog_type=args.cog_type,
                        cog_init=train_data[-1],
                        adj=adj,
                        action_limit=args.action_limit,
                        w_lambda=args.w_lambda,
                        gamma_type=gamma_type,
                        action_type=action_type,
                        scale=args.scale,
                        energy_model=args.energy_model
                    )
                )
                        
            # Define the baseline for the environment
            baseline = LinearFeatureBaseline(env_spec=env.spec)

            print('Using RL Method: ', algo_name)

            if algo_name == 'TRPO':
                # Create a TRPO (Trust Region Policy Optimization) algorithm instance.
                policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=[args.network, args.network],
                                            max_std=None, adaptive_std=False, std_share_network=False, output_nonlinearity=None)
                algo = TRPO(
                    env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    discount=args.discount,
                    gae_lambda=0.97,
                    lr_clip_range=0.2,
                    policy_ent_coeff=0.0,
                    max_path_length=max_time_steps
                )
            
            elif algo_name == 'TRPO-LSTM':
                # Create a TRPO (Trust Region Policy Optimization) algorithm with LSTM instance.

                policy = GaussianLSTMPolicy(
                    env_spec=env.spec,
                    hidden_dim=32,
                    hidden_nonlinearity=tf.nn.tanh,
                    output_nonlinearity=None,
                )

                baseline = GaussianMLPBaseline(
                    env_spec=env.spec,
                    regressor_args=dict(
                        hidden_sizes=[args.network, args.network],
                        use_trust_region=False,
                        optimizer=FirstOrderOptimizer,
                        optimizer_args=dict(
                            batch_size=10,
                            max_epochs=4,
                            ),
                        ),
                    )
                algo = TRPO(
                    env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    discount=args.discount,
                    gae_lambda=0.97,
                    lr_clip_range=0.2,
                    policy_ent_coeff=0.0,
                    max_path_length=max_time_steps,
                    max_kl_step=0.01,
                    optimizer=ConjugateGradientOptimizer,
                    optimizer_args=dict(
                        hvp_approach=FiniteDifferenceHvp(
                        base_eps=1e-5)
                        ),
                    )

            elif algo_name == 'PPO':
                # Create a PPO (Proximal Policy Optimization) algorithm instance.
                policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=[args.network, args.network],
                            max_std=None, adaptive_std=False, std_share_network=False, output_nonlinearity=None)
                algo = PPO(
                    env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    max_path_length=max_time_steps,
                    discount=args.discount,
                    gae_lambda=0.95,
                    lr_clip_range=0.2,
                    optimizer_args=dict(
                        batch_size=10,
                        max_epochs=4,
                    ),
                    stop_entropy_gradient=True,
                    entropy_method='max',
                    policy_ent_coeff=0.02,
                    center_adv=False,
                )

            elif algo_name == 'PPO-LSTM':
                # Create a PPO (Proximal Policy Optimization) algorithm with LSTM instance.
                policy = GaussianLSTMPolicy(
                    env_spec=env.spec,
                    hidden_dim=32,
                    hidden_nonlinearity=tf.nn.tanh,
                    output_nonlinearity=None,
                )

                baseline = GaussianMLPBaseline(
                    env_spec=env.spec,
                    regressor_args=dict(
                        hidden_sizes=[args.network, args.network],
                        use_trust_region=False,
                        optimizer=FirstOrderOptimizer,
                        optimizer_args=dict(
                            batch_size=10,
                            max_epochs=4,
                        ),
                    ),
                )
                algo = PPO(
                    env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    max_path_length=max_time_steps,
                    discount=args.discount,
                    gae_lambda=0.95,
                    lr_clip_range=0.2,
                    optimizer_args=dict(
                        batch_size=10,
                        max_epochs=4,
                    ),
                    stop_entropy_gradient=True,
                    entropy_method='max',
                    policy_ent_coeff=0.02,
                    center_adv=False,
                )

            
            elif algo_name == 'DDPG':
                # Create a DDPG (Deep Deterministic Policy Gradient) algorithm instance.
                policy = ContinuousMLPPolicy(env_spec=env.spec, hidden_sizes=[args.network, args.network])
                exploration_policy = AddOrnsteinUhlenbeckNoise(env.spec, policy, sigma=0.2)     # Ornstein-Uhlenbeck Noise is a better choice for continuous action spaces (than e.g. Epsilon Greedy)
                qf = ContinuousMLPQFunction(env_spec=env.spec, hidden_sizes=[args.network, args.network])
                replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))
                algo = DDPG(
                    env_spec=env.spec,
                    policy=policy,
                    qf=qf,
                    replay_buffer=replay_buffer,
                    exploration_policy=exploration_policy,
                    max_path_length=max_time_steps,
                    discount=args.discount,
                    steps_per_epoch=1,)
            
            elif algo_name == 'SAC':
                # Create a SAC (Soft Actor Critic) algorithm instance.
                policy = TanhGaussianMLPPolicy(env_spec=env.spec, hidden_sizes=[args.network, args.network])
                qf1 = TorchContinuousMLPQFunction(env_spec=env.spec, hidden_sizes=[args.network, args.network])
                qf2 = TorchContinuousMLPQFunction(env_spec=env.spec, hidden_sizes=[args.network, args.network])
                replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))
                algo = SAC(
                    env_spec=env.spec,
                    policy=policy,
                    qf1=qf1,
                    qf2=qf2,
                    replay_buffer=replay_buffer,
                    gradient_steps_per_itr=1000,
                    max_path_length=max_time_steps,
                    discount=args.discount,
                    min_buffer_size=int(0),
                    steps_per_epoch=1,)
                algo.sampler_cls = LocalSampler
            
            # Set up the trainer with the selected algorithm and environment.
            trainer.setup(algo, env)

            # Train the policy using the specified number of epochs and batch size.
            trainer.train(n_epochs=n_epochs, batch_size=batch_size)

            # Return the trained policy.
            return policy
    
    #   clear the default graph and start with a fresh graph.
    tf.compat.v1.reset_default_graph()

    # Training the policy if not in evaluation mode
    if not args.eval:
        print('Starting training...')
        # Train the policy and store it in the 'policy' variable
        policy = train_policy(seed=args.seed, n_epochs=epochs, batch_size=batch_size, action_type=action_type, 
                              gamma_type=gamma_type, gamma=gamma, max_time_steps=max_time_steps, algo_name=args.algo)
        print('Training complete.')

    print('\n***Loading Trained Agent for Evaluation***')
    # Initialize evaluator to assess policy performance
    evaluator = EvalPolicy(T=11, snapshot_dir=output_dir, log_dir=log_dir, gamma=gamma, gamma_type=gamma_type, 
                           cog_init=cog_init, adj=adj, action_type=action_type, action_limit=args.action_limit, 
                           w_lambda=args.w_lambda, energy_model=args.energy_model)

    # Evaluate on the Training set
    print('\n*****Starting evaluation on Train set*****')
    # Simulate the policy on the training data. Log states and actions
    state_log_train, action_log_train, output_train, state_log_train_dict = evaluator.simulate(train_data, 'train', scale_state, args.normalize)
    
    # Evaluate and compute metrics for the training set
    train_mae, train_mae_emci, train_mae_cn, train_mae_lmci, train_mae_smc, \
    train_mse, train_mse_emci, train_mse_cn, train_mse_lmci, train_mse_smc, \
    train_reward_gain, train_reward_rl, train_reward = evaluator.compute(train_df, 'train', args.datatype, score)
    
    # # Generate SHAP (Shapley additive explanations) plots for the training set
    # if args.shap_enable:
    #     print('Generating SHAP values for train split ...')   
    #     shap_values, state_log, explainer_exp_value = generate_shap(log_dir, state_log_train, action_log_train, 
    #                                                                 use_all_samples=args.shap_use_all_samples)

    # Evaluate on the Validation set
    print('\n*****Starting evaluation on Validation set*****')
    # Simulate the policy on the validation data. Log states and actions
    state_log_valid, action_log_valid, output_valid, state_log_valid_dict  = evaluator.simulate(valid_data, 'valid', scale_state, args.normalize)
    
    # Evaluate and compute metrics for the validation set
    valid_mae, valid_mae_emci, valid_mae_cn, valid_mae_lmci, valid_mae_smc, \
    valid_mse, valid_mse_emci, valid_mse_cn, valid_mse_lmci, valid_mse_smc, \
    valid_reward_gain, valid_reward_rl, valid_reward = evaluator.compute(valid_df, 'valid', args.datatype, score)
    
    # Generate SHAP plots for the validation set
    # if args.shap_enable:   
    #     print('Generating SHAP values for validation split ...')   
    #     shap_values, state_log, explainer_exp_value = generate_shap(log_dir, state_log_valid, action_log_valid, 
    #                                                                 use_all_samples=args.shap_use_all_samples)

    # Evaluate on the Test set
    print('\n*****Starting evaluation on Test set*****')
    # Simulate the policy on the test data. Log states and actions
    state_log_test, action_log_test, output_test, state_log_test_dict = evaluator.simulate(test_data, 'test', scale_state, args.normalize)
    
    # Evaluate and compute metrics for the test set
    test_mae, test_mae_emci, test_mae_cn, test_mae_lmci, test_mae_smc, \
    test_mse, test_mse_emci, test_mse_cn, test_mse_lmci, test_mse_smc, \
    test_reward_gain, test_reward_rl, test_reward = evaluator.compute(test_df, 'test', args.datatype, score)

    # create empty numpy arrays for the shap data
    shap_values = np.empty((2, 0, 6))       # 2 actions, N samples, 6 features
    state_log = np.empty((0, 6))            # N samples, 6 features
    explainer_exp_value = np.empty((0, 2))  # N samples, 2 actions


    # Generate SHAP values for the test set
    if args.shap_enable and args.algo not in ['TRPO-LSTM', 'PPO-LSTM']:     # SHAP is currently not supported for LSTM variants.
        print('Generating SHAP values for test split...')   
        shap_values, state_log, explainer_exp_value = generate_shap(log_dir, state_log_test, action_log_test, 
                                                                    use_all_samples=args.shap_use_all_samples)
    
    # Print results
    print(f'\n****Results for algo:{args.algo}-{name}-seed:{seed}-epochs:{epochs}****')
    print('train mae', train_mae, 'validation mae', valid_mae, 'test mae', test_mae)

    end_time = time.time()
    exp_time = round((end_time - start_time)/60, 2)

    # Create a results dataframe to store the results
    results_df = pd.DataFrame({ 'algo':args.algo, 'name':name, 'seed': seed, 'epochs':epochs, 'batch_size':batch_size,
                                'network':args.network, 'score':args.score, 'time_taken (min)':exp_time,
                                
                                'gamma':gamma,'gamma_type':gamma_type, 'w_lambda':args.w_lambda,  
                                'discount':args.discount, 'max_time_steps':args.trainsteps, 'action_lim':args.action_limit,  
                                'cog_mtl':args.cog_mtl, 'cog_init':args.cog_init, 'cog_type':args.cog_type,
                                'energy_model':args.energy_model, 'category':'APOE', 

                                'train_mae':train_mae, 'valid_mae':valid_mae, 'test_mae':test_mae,
                                'train_mse':train_mse, 'valid_mse':valid_mse, 'test_mse':test_mse,
                                
                                'train_mae_emci':train_mae_emci, 'valid_mae_emci':valid_mae_emci, 'test_mae_emci':test_mae_emci,
                                'train_mae_cn':train_mae_cn, 'valid_mae_cn':valid_mae_cn, 'test_mae_cn':test_mae_cn,
                                'train_mae_lmci':train_mae_lmci, 'valid_mae_lmci':valid_mae_lmci, 'test_mae_lmci':test_mae_lmci,
                                'train_mae_smc':train_mae_smc, 'valid_mae_smc':valid_mae_smc, 'test_mae_smc':test_mae_smc,

                                'train_mse_emci':train_mse_emci, 'valid_mse_emci':valid_mse_emci, 'test_mse_emci':test_mse_emci,
                                'train_mse_cn':train_mse_cn, 'valid_mse_cn':valid_mse_cn, 'test_mse_cn':test_mse_cn,
                                'train_mse_lmci':train_mse_lmci, 'valid_mse_lmci':valid_mse_lmci, 'test_mse_lmci':test_mse_lmci,
                                'train_mse_smc':train_mse_smc, 'valid_mse_smc':valid_mse_smc, 'test_mse_smc':test_mse_smc,

                                'train_reward_rl':train_reward_rl, 'valid_reward_rl':valid_reward_rl, 'test_reward_rl':test_reward_rl,
                                'train_reward':train_reward, 'valid_reward':valid_reward, 'test_reward':test_reward
                            }, index=[0])
    
    # # Create a condensed results dataframe to store the results
    # results_df = pd.DataFrame({ 'algo':args.algo, 'name':name, 'seed': seed, 'epochs':epochs, 'batch_size':batch_size, 
    #                             'train_mae':train_mae, 'test_mae':test_mae,
    #                             'train_mse':train_mse, 'test_mse':test_mse,
    #                             'train_reward_rl':train_reward_rl, 'test_reward_rl':test_reward_rl,
    #                         }, index=[0])
    
    # Save the results dataframe to an output file (CSV) which has results for each experiment appended to it
    output_filename = f'results/summary_{args.datatype}.csv'
    if os.path.isfile(output_filename):
        # If the output file already exists, append the results to it
        results_df.to_csv(output_filename, mode='a', header=False, index=False)
    else:
        # If the output file does not exist, create it and write the results with headers    
        results_df.to_csv(output_filename, mode='w', header=True, index=False)

    # Return the shape values, state log, and explainer expected value for shap plotting
    return shap_values, state_log, explainer_exp_value, state_log_test_dict