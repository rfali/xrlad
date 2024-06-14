# Unifying Interpretability and Explainability for Alzheimer’s Disease Progression Prediction

Code associated with the paper "Unifying Interpretability and Explainability for Alzheimer’s Disease Progression Prediction". This work has been presented at the [NeurIPS 2023's XAIA](https://xai-in-action.github.io/) (XAI in Action: Past, Present, and Future Applications) workshop [[PDF](https://openreview.net/pdf?id=joaWGug1CU)] and [AAAI 2024's XAI4DRL](https://xai4drl.github.io/) (eXplainable AI approaches for Deep Reinforcement Learning) workshop [[PDF](https://openreview.net/pdf?id=OuhYChnUeP)]. If you use this code in your own work, we request that you kindly cite our paper:
```
@article{ali2024unifying,
  title={Unifying Interpretability and Explainability for Alzheimer’s Disease Progression Prediction},
  author={Ali, Raja Farrukh and Milani, Stephanie and Woods, John and Adeniji, Emmanuel and Farooq, Ayesha and Mansel, Clayton and Burns, Jeffrey and Hsu, William},
  journal={arXiv preprint arXiv:2406.07777}
  year={2024},
  url={https://arxiv.org/abs/2406.07777}
}
```

### Setup

Clone the repo. The conda yml file is in the setup folder. Use miniconda/anaconda to create a conda env by:

```
conda env create -f setup/adenv.yml
```

### Configs, Dataset and Results Description

Information related to dataset variables used as input to the model, config variables passed to an experiment, and the result/output files is available under [dataset/data_README.md](dataset/data_README.md).

### Experiment Setup

#### Generate Configs

The base config file is `brain.json`. Specify the train configs using the `train.config.py` file in which multiple values for a hyperparameter can be specified (e.g. `"algo": ["TRPO"]` to `"algo": ["TRPO", "PPO"]`). The same is true for `eval.config.py`. All the possible configs will be generated as `.json` files in their respective folders.

```
python configs/gen_train_configs.py
python configs/gen_eval_configs.py
```

#### Train the model

The folder containing the configs files (0.json, 1.json, ...) will be input to the run_agent.py, which launches the `train_tf.py`. Edit the `NUM_THREADS` variable in `run_agents.py` according to your computational setup. After training is complete, evaluation will be carried out, followed by RL and SHAP plotting.

```
python run_agents.py configs/train_configs
```

#### Evaluate a trained model

One can also evaluate a trained model separately. This launches the `train.py`. with `config[eval]=True` as set in `eval_configs`, training will be skipped, the (already) trained model will be loaded, and evaluation performed by calling `eval.py`. This will be followed by plotting.

```python
python run_agents.py configs/eval_configs
```

### Directory Structure

The entry point for the code is `run_agents.py`.
```
.
├── configs             # Configs and the code to generate train/eval configs
├── dataset             # The filtered ADNI dataset divided into 5-folds for k-fold CV
├── plots_rl            # RL predictions vs Ground Truth plots are saved here
├── plots_shap          # SHAP plots are saved here for each algo
├── progress            # TF/PyTorch models and associated training files
├── results             # The predicted RL variables (cognition, activity, size etc.) per-patient per-year
├── results_summary     # A summary of results by folds and algos
├── setup               # Conda yaml and requirements.txt files
├── utils               # Tools and utilities 
├── brain_env.py        # Custom Gym environment simulating domain knowledge via Differential Equations
├── eval.py             # Run evaluation on trained agent
├── run_agents.py       # Entry point for running agent training, evaluation and plotting
├── train.py            # Train agents
└── xrl.py              # SHAP calculation and plotting code.
```
Different subfolders will be created during training and evaluation, and each algorithm's output will be under its own subfolder:

1. `progress` folder stores training progress using tensorboard events, console output in `debug.log`, the RL method's training parameters under `progress.csv`, and the trained RL model as `params.pkl`.
2. `results` folder stores the results of the experiment in a spreadsheet, with each patient's predicted parameters including cognition scores, brain region size, amyloid, energetic cost/activity for each year (see [dataset/data_README.md](dataset/data_README.md) for a breakdown). The experiment's config is saved as `exp_config.json`. The `summary_adni.csv` in the `results` folder stores a summary of all experiments in this batch, with each experiment's configuration and the MAE and MSE values across all subjects in a train/val/test split saved in a row.
3. `plots_rl` folder saves all plots comparing RL predictions to Ground Truth variables. `common` contains only those data points whose ground truth values were available and `all` stands for all data points (full trajectory). `baselines` compare best RL method (currently TRPO) against the supervised baselines MiniRNN and SVR. `rl` compares all RL methods among each other alongside Ground Truth cognition scores. There will be 5 subfolders for each RL algorithm. 
    * `all` contains cognition plots using all samples (RL predictions for all patients for all years), 
    * `common` will include predictions for only those samples found in the original dataset (common data points), 
    * `comparison` will have RL vs Ground Truth plots for 5 variables found in the dataset (Cognition, HC/PFC Size, HC/PFC Amyloid). 
    * `per_patient` will contain the cognition and HC/PFC size plots for the patient in test split with the max cognition decline. 
    * `rl_trajectories` folder will have RL predictions for cognition, energy cost, activity (HC+PFC), information (HC+PFC) and size (HC+PFC).
4. `plots_shap` will contain the SHAP plots at the global level, local level (first sample by default), and selected patient plots (suffix with RID), each under its own folder.
5. `results_summary` will contain 3 spreadsheets under a timestamped folder; a sorted version of `summary_adni.csv`, aggregate results by fold, and aggregate results by method.

### Acknowledgement

The dataset used in this research belongs to [ADNI](https://adni.loni.usc.edu/data-samples/access-data/), and permission is required to use this data for research purposes. Code is based on [this](https://github.com/anic46/ADProgModel) open-sourced implementation of AD progression using RL, but has been extensively refactored and improved, along with integration of XRL components (SHAP). 
