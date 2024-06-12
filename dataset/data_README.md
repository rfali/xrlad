## How to read the Experiment Configs and Input/Output CSV Files

### Input Dataset

**1. Input Variables (Ground-truth)**

The ADNI patient dataset (`dataset/adni/adni_fold{i}.xls`) has the following columns.

| Column Name                         | Description                                                                         |
| ----------------------------------- | ----------------------------------------------------------------------------------- |
| RID                                 | Patient ID                                                                          |
| VISCODE                             | Baseline (bl) or month of measurement (mXX)                                         |
| Years                               | Year of clinical measurement                                                        |
| DX_bl/ DX_bl_num                    | Diagnosis at baseline (year 0) - Type of cognitive impairment (EMCI, CN, LMCI, SMC) |
| CurAGE                              | Patient's age                                                                       |
| PTGENDER/ PTGENDER_num              | Gender (Male/Female)                                                                |
| PTEDUCAT                            | Years of education                                                                  |
| APOEPOS                             | Presence of  APOE ε4 gene                                                           |
| MMSE_norm, ADAS11_norm, ADAS13_norm | Normalized MMSE, ADAS11, ADAS13 scores                                              |
| mri_FRONT_norm, mri_HIPPO_norm      | $X(t)$ - Normalized Frontal/Hippocampal region size                                 |
| FRONTAL_SUVR, HIPPOCAMPAL_SUVR      | $D(t)$ - Instantaneous amyloid accumulation in Frontal/Hippocampal regions from florbetapir-PET scans  |
| cogsc                               | $C(t)$ - Cognition score (MMSE/ADAS13 was used in these experiments)                |


**2. Estimated parameters for differential equations**

The differential equations' parameters that were estimated based on demographics of ADNI patient dataset (`dataset/adni/adni_fold{i}_parameters.xls`) has the following columns.

| Column Name       | Description                                            |
| ----------------- | ------------------------------------------------------ |
| beta_estm         | $\beta$ parameter for amyloid propagation            |
| tpo_estm          | Actual pathology time-period at baseline (CurAGE - 50) |
| alpha1_estm       | $\alpha_1$ for brain degeneration                    |
| alpha2_gamma_estm | $\alpha_2 \gamma$ for computing activity Y(t)        |



### Experiment Configurations
Each experiment's config is saved under `configs\train_configs` or `configs\eval_config` folder.


| Column Name       | Description                                                                                        |
|-------------------|----------------------------------------------------------------------------------------------------|
| name              | Experiment name                                                                                                               |
| seed              | Random seed used in the experiment or data generation.                                                                        |
| gamma             | The gamma parameter used in modeling the relationship between Y(t), X(t) and I(t)                                             |
| gamma_type        | Type of gamma parameter, which can be 'variable' or 'fixed'.                                                                  |
| epochs            | Number of training epochs or iterations in an experiment.                                                                     |
| batch_size        | Size of data batches used in training.                                                                                        |
| cog_mtl           | $I_{HC}(0)$ Initial cognition score (baseline year 0) for Hippocampus (HC) region. $I_{PFC}(0) = 10.0 - I_{HC}(0)$            |
| discount          | Discount factor applied to rewards in RL.                                                                                     |
| max_time_steps    | Maximum number of time steps (years in this case) n a training episode.                                                       |
| w_lambda          | Trade-off between the mismatch (C_task - C(t)) and the energy cost M(t) in the reward function (see Eq 8 of the original paper)|
| action_lim        | Limit or constraint applied to action values. Set to 2.0 , so $\Delta I(t)$ = [-2, 2]                                         |
| cog_init          | Initial value or setting for cognitive measurements. Set to `full` (a value of 10.0)                                          |
| cog_type          | Type of cognitive data, e.g., 'variable' or 'fixed'.                                                                          |
| energy_model      | Type or name of the energy model used in the experiment. `inverse` or `inverse_squared`                                       | 
| score             | cognition score to use (MMSE, ADAS11, ADAS13).                                                                                | 
| network           | MLP network hidden layer size. defauts to 2-layer MLP with hidden_size = 32, so [32,32].                                      |          
| algo              | Name or type of the machine learning or RL algorithm.                                                                         |
| category          | Fixed to 'APOE' which is the APOE ε4 gene.                                                                                    |


### Results

**1. Variables computed using estimated DE parameters and information allocation by RL model**

The results for each experiment run are saved in `results/{algo}/{fold}/{seed}/{experiment_name}.xlsx` and contains the following RL model's predictions for each timestep (in addition to the Input Variables (Ground Truth)):

**reg1**: medial temporal lobe (**mtl/HC**) i.e. Hippocampus or hippocampal region

**reg2**: frontal temporal lobe (**ftl/PFC**) i.e. Pre-Frontal Cortex (PFC) or frontal region

| Column Name                             | Description                                                                           |
| --------------------------------------- | ------------------------------------------------------------------------------------- |
| reg1_info_rl                            | $I_{v1} (t)$ = Information processed by hippocampal region                              |
| reg2_info_rl                            | $I_{v1} (t)$ = Information processed by frontal region                          |
| reg1_fdg_rl                             | $Y_{v1} (t)$ = Hippocampal activity (fgd:fluorodeoxyglucose). Interchangeably used for energy consumption $M=\sum Y$  |
| reg2_fdg_rl                             | $Y_{v2} (t)$ = Frontal activity (fgd:fluorodeoxyglucose). Interchangeably used for energy consumption $M=\sum Y$ |
| reg1_mri_rl                             | $X_{v1} (t)$ = Hippocampal region size                                                  |
| reg2_mri_rl                             | $X_{v2} (t)$ = Frontal region size                                              |
| reg1_D_rl                               | $D_{v1} (t)$ = Hippocampal instantaneous amyloid accumulation                           |
| reg2_D_rl                               | $D_{v2} (t)$ = Frontal instantaneous amyloid accumulation                       |
| beta_rl, alpha1_rl, alpha2_rl, gamma_rl | Parameters used by RL model for the DE-based simulator                              |
| cogsc_rl                                | $C(t) = \sum I_v (t) $ Cognition score computed by RL (reg1_info_rl + reg2_info_rl) |
| cogsc                                   | $C(t)$ Cognition score (MMSE in our case)   |
| cog_diff                                | Difference between cogsc_rl and cogsc  |

**2. RL Reward $\Delta I(t)$ and Errors (MAE and MSE) for the Experiment**

Each experiment's errors between RL predictions and ground truth values (Mean Absolute Error and Mean Square Error) are saved in `results/summary_{dataset}.csv` with the following data.                                                                          |

| Column Name       | Description                                                                                        |
|-------------------|----------------------------------------------------------------------------------------------------|
| train_mae, valid_mae, test_mae                    | Mean Absolute Error (`MAE`) on the train, validation and test split.      |
| train_mse, valid_mse, test_mse                    | Mean Squared Error (`MSE`) on the train, validation and test split.       |
| train_mae_emci, valid_mae_emci, test_mae_emci     | `MAE` for Early Mild Cognitive Impairment (EMCI) category for the 3 splits|
| train_mae_cn, valid_mae_cn, test_mae_cn           | `MAE` for Early Cognitive Normal (CN) category for the 3 splits           |
| train_mae_lmci, valid_mae_lmci, test_mae_lmci     | `MAE` for Late Mild Cognitive Impairment (LMCI) category for the 3 splits |
| train_mae_smc, valid_mae_smc, test_mae_smc        | `MAE` for Signficant Memory Concern (SMC) category for the 3 splits       |
| train_mse_emci, valid_mse_emci, test_mse_emci     | `MSE` for Early Mild Cognitive Impairment (EMCI) category for the 3 splits|
| train_mse_cn, valid_mse_cn, test_mse_cn           | `MSE` for Early Cognitive Normal (CN) category for the 3 splits           |
| train_mse_lmci, valid_mse_lmci, test_mse_lmci     | `MSE` for Late Mild Cognitive Impairment (LMCI) category for the 3 splits |
| train_mse_smc, valid_mse_smc, test_mse_smc        | `MSE` for Signficant Memory Concern (SMC) category for the 3 splits       |
| train_reward_rl, valid_reward_rl, test_reward_rl  | RL-based reward for the 3 splits                                        |
| train_reward, valid_reward,test_reward            | Reward metric for the 3 splits                                          |
