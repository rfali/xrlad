import os
import json
import shutil
from gen_train_configs import dict_product

with open("configs/brain.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "filename":["adni_fold0.xls", "adni_fold1.xls", "adni_fold2.xls", "adni_fold3.xls", "adni_fold4.xls"],
    #"filename":["synthetic_fold0.xls", "synthetic_fold1.xls", "synthetic_fold2.xls", "synthetic_fold3.xls", "synthetic_fold4.xls"],
    "datatype":["adni"],
    "algo": ["TRPO", "PPO", "DDPG", "SAC", "TRPO-LSTM", "PPO-LSTM"],
    "w_lambda":[1.0],
    "gamma":[1.0],
    "cog_init":["full"],
    "cog_type":["fixed"],
    "cog_mtl":[7.0],
    "epochs":[1000],
    "batch_size":[1000],
    "eval": [True],
    "score": ["MMSE"],
    "scale":[True],
    "discount":[1.00],
    "network":[32],
    "energy_model":["inverse"],
    "normalize":[False],
    "seed":[613, 1727, 2594, 3954, 4910],
    "shap_enable": [True],
    "shap_use_all_samples": [False],
    "shap_show_fig": [False],
}

all_configs = [{**BASE_CONFIG, **p} for p in dict_product(PARAMS)]
if os.path.isdir("configs/eval_configs"):
    shutil.rmtree("configs/eval_configs")

os.makedirs("configs/eval_configs/")

for i, config in enumerate(all_configs):
    with open(f"configs/eval_configs/{i}.json", "w") as f:
        json.dump(config, f, indent=4, sort_keys=True)