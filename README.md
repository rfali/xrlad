## Explainable Reinforcement Learning for Alzheimer’s Disease Progression Prediction

Code associated with the paper ["Explainable Reinforcement Learning for Alzheimer’s Disease Progression Prediction"](https://openreview.net/pdf?id=joaWGug1CU) presented at the NeurIPS 2023's XAIA (XAI in Action: Past, Present, and Future Applications) workshop. If you use this code in your own work, please cite our paper:
```
@inproceedings{ali2023explainable,
  title={Explainable Reinforcement Learning for Alzheimer’s Disease Progression Prediction},
  author={Ali, Raja Farrukh and Farooq, Ayesha and Adeniji, Emmanuel and Woods, John and Sun, Vinny and Hsu, William},
  booktitle={XAI in Action Workshop NeurIPS 2023},
  year={2023}
}
```
### Setup

Clone the repo. The conda yml file is provided in the setup folder (tested on Ubuntu 22.04). Use miniconda/anaconda to create a conda env by:

```
conda env create -f setup/xrlad.yml
```

### Generate Configs

An explanation of all available config variables is given towards the end. The base config file is `brain.json`. Specify the train configs using the `train.config.py` file in which multiple values for a hyperparameter can be specified (e.g. `"algo": ["TRPO"]` to `"algo": ["TRPO", "PPO"]`). The same is true for `eval.config.py`. All the possible configs will be generated as `.json` files in their respective folders.

```
python configs/gen_train_configs.py
python configs/gen_eval_configs.py
```

### Train the model

The folder containing the configs files (0.json, 1.json, ...) will be input to the run_agent.py, which launches the `trainer_tf.py`. Edit the `NUM_THREADS` variable in `run_agents.py` according to your local machine.

```
python run_agents.py configs/train_configs
```

Two subfolders will be created for each experiment under `progress` and `results`.

1. The `progress` folder stores training progress using tensorboard events, the RL method's training parameters under `progress.csv` and the trained RL model as `params.pkl`.
2. The `results` folder stores the results of the experiment in a spreadsheet, with each patient's predicted parameters (see below) and cognition scores for each year. Within each experiment folder, the experiment's config will be saved as `exp_config.json` and some plots that are generated as part of the `evaluation`.
3. Finally, there will be `summary.csv` in the `results/_summary` folder which will have a summary of all experiments, with each experiment's configuration and the MAE and MSE values across all subjects in a train/val/test fold saved in a row.

We carried out 5-fold cross validation, with each fold undergoing 5 experiments (each with a differen seed.) Due to size contraints, only the trained models (`progress`) and `results` of the best performing fold's best performing seed are being provided here. Each experiment takes about 30 minutes to run on a 32-core, single GPU machine, hence the experimentation is easily re-doable.

### Evaluate a trained model

Launches the `trainer_tf.py`. with `config[eval]=True` as set in `eval_configs`, training will be skipped, the (already) trained model will be reloaded, and evaluation performed by calling `eval.py`.

```python
python run_agents.py configs/eval_configs
```

### How to read the Input and Output CSV Files

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
| APOEPOS                             | Presence of  APOE ε4 gene                                                          |
| MMSE_norm, ADAS11_norm, ADAS13_norm | Normalized MMSE, ADAS11, ADAS13 scores                                              |
| mri_FRONT_norm, mri_HIPPO_norm      | $X(t)$ - Normalized Frontal/Hippocampal region size                                   |
| FRONTAL_SUVR, HIPPOCAMPAL_SUVR      | $D(t)$ - Instantaneous amyloid accumulation in Frontal/Hippocampal regions from florbetapir-PET scans  |
| cogsc                               | $C(t)$ - Cognition score (MMSE was used in these experiments)                                           |


**2. Estimated parameters for differential equations**

The differential equations' parameters that were estimated based on demographics of ADNI patient dataset (`dataset/adni/adni_fold{i}_parameters.xls`) has the following columns.

| Column Name       | Description                                            |
| ----------------- | ------------------------------------------------------ |
| beta_estm         | $\beta$ parameter for amyloid propagation            |
| tpo_estm          | Actual pathology time-period at baseline (CurAGE - 50) |
| alpha1_estm       | $\alpha_1$ for brain degeneration                    |
| alpha2_gamma_estm | $\alpha_2 \gamma$ for computing activity Y(t)        |

### Results

**1. Variables computed using estimated DE parameters and information allocation by RL model**

The results for each experiment run are saved in `results/{experiment_name}/{experiment_name}.xlsx` and contains the following RL model's predictions for each timestep (in addition to the Input Variables (Ground Truth)):

| Column Name                             | Description                                                                           |
| --------------------------------------- | ------------------------------------------------------------------------------------- |
| reg1_info_rl                            | $I_{v1} (t)$ = Information processed by frontal region                              |
| reg2_info_rl                            | $I_{v1} (t)$ = Information processed by hippocampal region                          |
| reg1_fdg_rl                             | $Y_{v1} (t)$ = Frontal activity (fluorodeoxyglucose). Interchangeably used for energy consumption $M=\sum Y$  |
| reg2_fdg_rl                             | $Y_{v2} (t)$ = Hippocampal activity (fluorodeoxyglucose). Interchangeably used for energy consumption $M=\sum Y$                |
| reg1_mri_rl                             | $X_{v1} (t)$ = Frontal region size                                                  |
| reg2_mri_rl                             | $X_{v2} (t)$ = Hippocampal region size                                              |
| reg1_D_rl                               | $D_{v1} (t)$ = Frontal instantaneous amyloid accumulation                           |
| reg2_D_rl                               | $D_{v2} (t)$ = Hippocampal instantaneous amyloid accumulation                       |
| beta_rl, alpha1_rl, alpha2_rl, gamma_rl | Parameters used by RL model for the DE-based simulator                              |
| cogsc_rl                                | $C(t) = \sum I_v (t)$ Cognition score computed by RL (reg1_info_rl + reg2_info_rl)  |
| cogsc                                   | $C(t)$ Actual cognition score (MMSE in our experiments)                             |
| cog_diff                                | Difference between cogsc_rl and cogsc                                               |

**2. Experiment Configurations and Errors (MAE and MSE) for the Experiment**

Each experiment's config and the errors between RL predictions and ground truth values (Mean Absolute Error and Mean Square Error) are saved in `results/_summary/summary.csv` with the following data. The mean results for each fold are saved in `results/_summary/final_results.csv`.

#### Experiment Configurations

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
| w_lambda          | Trade-off between the mismatch ($C_{task}$ - C(t)) and the energy cost M(t) in the reward function (see Eq 1 of the paper)        |
| action_lim        | Limit or constraint applied to action values. Set to 2.0 , hence $\Delta I(t)$ = [-2, 2]                                      |
| cog_init          | Initial value or setting for cognitive measurements. Set to `full` (a value of 10.0)                                          |
| cog_type          | Type of cognitive data, e.g., 'variable' or 'fixed'.                                                                          |
| energy_model      | Type or name of the energy model used in the experiment. `inverse` or `inverse_squared`                                       | 
| score             | cognition score to use (MMSE, ADAS11, ADAS13).                                                                                | 
| network           | MLP network hidden layer size. defauts to 2-layer MLP with hidden_size = 32, so [32,32].                                      |          
| algo              | Name or type of the machine learning or RL algorithm.                                                                         |
| category          | Fixed to 'APOE' which is the APOE ε4 gene.                                                                                    |

#### RL Reward $\Delta I(t)$ and Errors (MAE and MSE) for the Experiment
| Column Name       | Description                                                                                        |
|-------------------|----------------------------------------------------------------------------------------------------|
| train_mae         | Mean Absolute Error (MAE) on the training data.                                                    |
| valid_mae         | MAE on the validation data.                                                                         |
| test_mae          | MAE on the test data.                                                                               |
| train_mse         | Mean Squared Error (MSE) on the training data.                                                      |
| valid_mse         | MSE on the validation data.                                                                         |
| test_mse          | MSE on the test data.                                                                               |
| train_mae_emci    | MAE for a specific category ('EMCI') on the training data.                                           |
| valid_mae_emci    | MAE for 'EMCI' category on the validation data.                                                       |
| test_mae_emci     | MAE for 'EMCI' category on the test data.                                                             |
| train_mae_cn      | MAE for 'CN' category on the training data.                                                           |
| valid_mae_cn      | MAE for 'CN' category on the validation data.                                                           |
| test_mae_cn       | MAE for 'CN' category on the test data.                                                                 |
| train_mae_lmci    | MAE for 'LMCI' category on the training data.                                                           |
| valid_mae_lmci    | MAE for 'LMCI' category on the validation data.                                                           |
| test_mae_lmci     | MAE for 'LMCI' category on the test data.                                                               |
| train_mae_smc     | MAE for 'SMC' category on the training data.                                                             |
| valid_mae_smc     | MAE for 'SMC' category on the validation data.                                                             |
| test_mae_smc      | MAE for 'SMC' category on the test data.                                                                 |
| train_mse_emci    | MSE for 'EMCI' category on the training data.                                                           |
| valid_mse_emci    | MSE for 'EMCI' category on the validation data.                                                           |
| test_mse_emci     | MSE for 'EMCI' category on the test data.                                                               |
| train_mse_cn      | MSE for 'CN' category on the training data.                                                             |
| valid_mse_cn      | MSE for 'CN' category on the validation data.                                                             |
| test_mse_cn       | MSE for 'CN' category on the test data.                                                                 |
| train_mse_lmci    | MSE for 'LMCI' category on the training data.                                                           |
| valid_mse_lmci    | MSE for 'LMCI' category on the validation data.                                                           |
| test_mse_lmci     | MSE for 'LMCI' category on the test data.                                                               |
| train_mse_smc     | MSE for 'SMC' category on the training data.                                                             |
| valid_mse_smc     | MSE for 'SMC' category on the validation data.                                                             |
| test_mse_smc      | MSE for 'SMC' category on the test data.                                                                 |
| train_reward_rl   | RL-based reward on the training data.                                                                  |
| valid_reward_rl   | RL-based reward on the validation data.                                                                  |
| test_reward_rl    | RL-based reward on the test data.                                                                      |
| train_reward      | Reward metric on the training data.                                                                     |
| valid_reward      | Reward metric on the validation data.                                                                     |
| test_reward       | Reward metric on the test data.                                                                         |


### Acknowledgement

This code is based on [this](https://github.com/anic46/ADProgModel) open-sourced implementation of AD progression using RL, but has been refined and improved, along with the addition of explainable RL (XRL) components (i.e. [SHAP library](https://github.com/shap/shap)).
