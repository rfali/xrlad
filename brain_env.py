import gym
import numpy as np
import random
from garage.envs.step import Step

class BrainEnv(gym.Env):
    """
    A custom OpenAI Gym environment modeling brain health and amyloid deposition.
    References: Saboo et al. 2021 (https://arxiv.org/abs/2106.16187).
    """ 

    def __init__(self, network_size=2, num_edges=1, max_time_steps=7, alpha1_init=None, alpha2_init=None, beta_init=None, gamma_init=None, \
                 X_V_init=None, D_init=None, cog_type='fixed', cog_init=None, adj=None, action_limit=1.0, w_lambda=1.0, patient_idx=-1, \
                 gamma_type='fixed', action_type='delta', scale=False, use_gamma=False, energy_model='inverse'):
        """
        Initialize the BrainEnv environment.

        Args:
        - network_size: Number of nodes/brain regions in the network (default=2 i.e. Pre-frontal Cortex PFC and Hippocampus HC).
        - num_edges: Number of edges in the network (default=1).
        - max_time_steps: Maximum number of time steps in an episode (default=7).
        - alpha1_init: Initial value for alpha1 (default=None).
        - alpha2_init: Initial value for alpha2 (default=None).
        - beta_init: Initial value for beta (default=None).
        - gamma_init: Initial value for gamma (default=None).
        - X_V_init: Initial values for brain region's size (default=None).
        - D_init: Initial values for instantaneous amyloid accumulation (default=None).
        - cog_type: Type of cognitive data, 'fixed' or 'variable' (default='fixed').
        - cog_init: Initial values for cognitive measurements (default=None).
        - adj: Adjacency matrix of the graph (default=None).
        - action_limit: Limit for the action values (default=1.0).
        - w_lambda: Weight applied to the mismatch term in the reward function (default=1.0).
        - patient_idx: Index of the patient (default=-1).
        - gamma_type: Type of gamma parameter, 'fixed' or 'variable' (default='fixed').
        - action_type: Type of action, 'delta' or 'absolute' (default='delta').
        - scale: Whether to normalize the environment (default=False).
        - use_gamma: Whether to use gamma (default=False).
        - energy_model: Type of energy model, 'inverse' or 'inverse-squared' (default='inverse').

        """
        self.env_name = 'brain_env'
        self.patient_idx = patient_idx
        
        self.gamma_type = gamma_type
        self.action_type = action_type
        self.cog_type = cog_type
        self.use_gamma_ = use_gamma
        self.energy_model = energy_model
        self.normalize_ = scale
        
        self.cog_init_ = cog_init
        self.beta_init_ = beta_init
        self.alpha1_init_ = alpha1_init 
        self.alpha2_init_ = alpha2_init 
        self.gamma_init_ = gamma_init 
        self.X_V_init_ = X_V_init 
        self.D_init_ = D_init
        
        self.lambda_ = w_lambda
        self.C_task = 10    # a hypothetical variable which represents the cognitive demand on the brain. Can be thought of as the maximum score on a cognitive test.
        self.normalize_factor = 1/self.C_task if self.normalize_ else 1.0
        self.state_limits = np.array([[0,10],[0,5],[0,2]])  # limits for the state variables (cog C, size X, amyloid D)
        self.action_limit = action_limit    # constrain the continuous action space in the range [-2,2]
        self.reward_bound = 2000.0          # to stabilize learning, reward is constrained in the range [-2000, 2000]
        
        self.adj_ = adj     # adjacency matrix , set to np.array([[0,1],[1,0]]) 
        self.H = np.diag(np.sum(self.adj_, axis=1)) - self.adj_     # H is the Laplacian of the adjacency matrix of the graph
        
        self.patient_count = D_init.shape[0]
        self.max_time_steps = max_time_steps
        
        if use_gamma:
            self.observation_space = gym.spaces.Box(low=np.array([0,0,0,0,0,0,0,0]), 
                                                    high=np.array([self.C_task,self.C_task,5,5,2,2,self.C_task*5,1]), 
                                                    shape=(8, ), 
                                                    dtype=np.float64)
        else:
            self.observation_space = gym.spaces.Box(low=np.array([0.0,0.0,0,0,0,0]), 
                                                    high=np.array([self.C_task, self.C_task,5,5,2,2]),  # I_V1, I_V2, X_V1, X_V2, D_V1, D_V2
                                                    shape=(6, ), 
                                                    dtype=np.float64)
            
        self.action_space = gym.spaces.Box(low=-action_limit, 
                                           high=action_limit, 
                                           shape=(2, ), 
                                           dtype=np.float64)
        self.reset()


    def calc_node_activity(self, Iv, Xv):
        """
        Calculate the brain activity at frontal temporal lobe (ftl/PFC) and medial temporal lobe (mtl/HC) nodes.

        Args:
        - Iv: Information processed by the nodes (e.g., cognition scores)
        - Xv: Size or health of the brain regions

        Returns:
        - Yv: Brain activity of a node/region, which also translates into the node's energy (M) . 
        """

        if self.energy_model == 'inverse':
            return self.gamma_v * Iv / Xv                   #  eq 4 of Saboo et al. 2021
        elif self.energy_model == 'inverse-squared':
            return self.gamma_v * Iv / (Xv**2)              #  eq 12 of Saboo et al. 2021
        else:
            raise NotImplementedError("relationship between Iv, Xv, and Yv not defined")
    
    
    def calc_reward(self, state, Y_V, C_first=None):    
        """
        Calculate the reward for the reinforcement learning agent.

        Args:
            state (numpy.ndarray): Only the first two elements of the state vector are used, which are Iv_1 and Iv_2 (cognitive measurements for PFC and HC)
            Y_V (numpy.ndarray): The brain activity of regions.
            C_first (float, optional): The initial cumulative value (None by default).

        Returns:
            float: The calculated reward value.
        """
        # Calculate the cumulative state value (total cognition value Iv_1+Iv_2 at this time step)
        Ct = state.sum()                # eq 3 of Saboo et al. 2021
        
        # Calculate the cumulative energy cost (total brain activity Yv_1+Yv_2 at this time step)
        # The sum of activity of all brain regions equals the total energy cost of the brain
        Mt = np.sum(Y_V)                # eq 5 of Saboo et al. 2021
        
        # Determine the task cumulative value, either from C_first or the predefined C_task
        C_task = self.C_task if C_first is None else C_first    # C_task is the maximum score on a cognitive test, set to 10

        # To constrain the agent from assigning C(t) > 10 to an individual, a penalty factor is 
        # incorporated in the reward function, based on the mismatch between Ctask and C(t). 
        # 100^[max(Ct − C_task, 0)] See Section C.4 of Saboo et al. 2021 for details
        power_factor = np.clip(-C_task + Ct, a_min=0, a_max=None)
        factor = 100**power_factor

        # Reward is the trade-off between the competing criteria of 
        # (i) reducing mismatch between Ctask and C(t) and 
        # (ii) reducing the cost M(t) of supporting cognition
        # R(t) = −[λ|Ctask − C(t)| × 100^[max(Ct − C_task, 0)] + M(t)]
        reward = -(np.abs(C_task - Ct) * factor * self.lambda_ + Mt)    # eq 8 of Saboo et al. 2021 

        # Constrain the reward to be within the specified range [-2000, 2000]
        if np.isnan(reward):
            reward = -self.reward_bound
        elif reward > self.reward_bound:
            reward = self.reward_bound
        elif reward < -self.reward_bound:
            reward = -self.reward_bound

        return reward    


    def get_new_structure(self, D_old, Y_V, X_V):
        """
        Update brain size and amyloid deposition based on previous info.

        Args:
            D_old (numpy.ndarray): The previous instantaneous amyloid deposition.
            Y_V (numpy.ndarray): Brain region activity.
            X_V (numpy.ndarray): Brain region sizes.

        Returns:
            X_V_new (numpy.ndarray): Updated brain region sizes.
            D_new (numpy.ndarray): Updated instantaneous amyloid deposition.
        """
        # Calculate the evolution of amyloid deposition over time
        D_new = D_old - self.beta * self.H @ D_old      # eq 1 of Saboo et al. 2021 (H is the Laplacian of the adjacency matrix of the graph).
        
        # Calculate the new health of brain regions
        if self.energy_model == 'inverse-squared':
            delta_X = -self.alpha_1 * D_new - self.alpha_2 * Y_V / X_V  # related to eq 12 and eq 6 of Saboo et al. 2021. See Section C.3 of the paper for details. 
        else:
            delta_X = -self.alpha_1 * D_new - self.alpha_2 * Y_V        #  eq 6 of Saboo et al. 2021
        
        X_V_new = X_V + delta_X
        return X_V_new, D_new
    

    def reset(self, state_type='healthy', randomize=True):
        """
        Reset the environment.

        Args:
            state_type (str, optional): The type of state to initialize ('healthy' by default).
            randomize (bool, optional): Whether to randomize patient selection (True by default).

        Returns:
            numpy.ndarray: The initial state of the environment.
        """

        # Initialize the time step
        self.t = 0
        # Get the current patient index
        patient_idx = self.patient_idx
        # Sample gamma_e from a normal distribution with mean 3 and standard deviation 1
        # gamma_e is not used anywhere in the code.
        self.gamma_e = np.random.normal(3, 1)   

        # Check if the patient index is set to -1, indicating random selection
        if patient_idx == -1:
            # Randomly select a patient index from the available patient count
            patient_idx = np.random.randint(self.patient_count)

        # Set environment parameters based on the selected patient
        self.beta = self.beta_init_[patient_idx]
        self.alpha_1 = self.alpha1_init_[patient_idx]
        self.alpha_2 = self.alpha2_init_[patient_idx]
        prod = self.alpha2_init_[patient_idx] * self.gamma_init_[patient_idx]

        # Check if gamma_type is 'variable'
        if self.gamma_type == 'variable':
            # Randomly choose a value for gamma_v from a predefined set
            self.gamma_v = np.random.choice([0.5, 1.0, 1.5, 2.1, 2.5, 3.0])
            # Update alpha_2 based on the chosen gamma_v
            self.alpha_2 = prod / self.gamma_v
        else:
            # Use the predefined gamma value for the patient
            self.gamma_v = self.gamma_init_[patient_idx]

        # Set initial values for brain variables (X_V, D) based on the selected patient
        self.X_V = self.X_V_init_[patient_idx]
        self.D = self.D_init_[patient_idx]

        # Set the initial cognitive measurements (cog) based on the selected patient
        cog = self.cog_init_[patient_idx]
        if self.cog_type == 'variable':
            # Initialize cognitive measurements with a variable approach
            base = cog.sum() // 2
            mtl_init = base + base * random.random()
            cog = np.array([mtl_init, base * 2 - mtl_init])
        self.cog = cog

        # Initialize the state of the patient based on environment settings
        if self.use_gamma_:
            # Construct the state vector with gamma adjustment, shape = (8, )
            self.state = np.array([self.cog[0] * self.normalize_factor, 
                                   self.cog[1] * self.normalize_factor, 
                                   self.X_V[0], 
                                   self.X_V[1], 
                                   self.D[0], 
                                   self.D[1], 
                                   1.0 / self.X_V[0] * self.gamma_v, 
                                   0.0])
        else:
            # Construct the state vector without gamma adjustment, shape = (6, )
            self.state = np.array([self.cog[0] * self.normalize_factor, 
                                   self.cog[1] * self.normalize_factor, 
                                   self.X_V[0] / 5.0, 
                                   self.X_V[1] / 5.0, 
                                   self.D[0], 
                                   self.D[1]])

        # Initialize the reward to None
        self.reward = None
        # Return the initial state
        return self.state

 
    def step(self, action):
        """
        Take a step in the environment based on the given action.

        Args:
            action (numpy.ndarray): The action taken by the agent. It has 2 elements i.e. change in cognition score for each brain region (delta_Iv_1 and delta_Iv_2)

        Returns:
            Step: An object representing state, reward, done and info [Y_V, X_V (health/size), D_old]
        """
        # Increment the time step
        self.t += 1
        # Create a copy of the current state for modification
        next_state = self.state.copy()
        
        # Normalize the first two elements of the state (Iv_1 and Iv_2) if necessary
        if self.normalize_:
            next_state[:2] = next_state[:2] * self.C_task
        
        # Clip the action to fit within the action space bounds
        a = action.copy()
        a = np.clip(a, self.action_space.low, self.action_space.high)
        
        # Update the state based on the action type
        if self.action_type == 'delta':
            if self.t == 1:
                next_state[:2] += a
            else:
                next_state[:2] += a
        else:
            next_state[:2] = a
        
        # Clip the resulting state to fit within the observation space bounds
        next_state = np.clip(next_state, a_min=self.observation_space.low, a_max=self.observation_space.high)
        
        # Calculate the energy consumption at frontal and mtl nodes (Y_V)
        Y_V = self.calc_node_activity(next_state[:2], self.X_V)
        # Clip values to a maximum of 10 (some Y_V/energy values were very large)
        # Not being used right now to avoid deviating from published results
        # Y_V = np.clip(Y_V, a_min=None, a_max=10)  
        
        # Create a copy of the brain health (size) variable
        health = self.X_V.copy()
        
        # Calculate the reward using the calc_reward function
        reward = self.calc_reward(next_state[:2], Y_V)
        
        # Create copies of D and X_V
        D_old = self.D.copy()
        X_V_old = self.X_V.copy()
        
        # Update brain size X_V and amyloid deposition D based on previous info
        self.X_V, self.D = self.get_new_structure(D_old.copy(), Y_V.copy(), X_V_old.copy())
        
        # Clip X_V to ensure it's not less than 0 
        # Caution: 0.0001 resulted in exhorbitantly large Y_V values (Y_V = gamma * I_V/X_V) for some cases (< 5)
        self.X_V = np.clip(self.X_V, a_min=0.0001, a_max=None)
        
        # Update the reward attribute
        self.reward = reward
        
        # Update the state based on whether gamma adjustment is used
        if self.use_gamma_:
            self.state = np.array([next_state[0] * self.normalize_factor, 
                                   next_state[1] * self.normalize_factor, 
                                   self.X_V[0], 
                                   self.X_V[1], 
                                   self.D[0], 
                                   self.D[1], 
                                   Y_V.sum() * self.normalize_factor, 
                                   self.t / 11.])
        else:
            self.state = np.array([next_state[0] * self.normalize_factor, 
                                   next_state[1] * self.normalize_factor, 
                                   self.X_V[0] / 5.0, 
                                   self.X_V[1] / 5.0, 
                                   self.D[0], 
                                   self.D[1]])
        
        # Check if the episode is done
        done = self.is_done()
        
        # Return the step information
        return Step(observation=self.state, reward=reward, done=done, y=Y_V, health=health, D=D_old)


    def observe(self):
        """
        Get the current state of the environment.

        Returns:
            numpy.ndarray: The current state of the environment.
        """
        return self.state


    def is_done(self):
        """
        Check if the episode is complete.

        Returns:
            bool: True if the episode is complete (the current time step is greater than or equal to the maximum allowed time steps), False otherwise.
        """
        return True if self.t >= self.max_time_steps else False
    
    
    def render(self, mode='human'):
        """
        Render the current state and reward of the environment.

        Args:
            mode (str, optional): The rendering mode. Default is 'human' for human-readable output.
        """
        print(self.state, self.reward)

    