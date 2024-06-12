import os
import json
import shutil
import itertools

def dict_product(d):
    '''
    Implementing itertools.product for dictionaries.
    E.g. {"a": [1,4],  "b": [2,3]} -> [{"a":1, "b":2}, {"a":1,"b":3} ..]
    Inputs:
    - d, a dictionary {key: [list of possible values]}
    Returns;
    - A list of dictionaries with every possible configuration
    '''
    keys = d.keys()
    vals = d.values()
    prod_values = list(itertools.product(*vals))
    all_dicts = map(lambda x: dict(zip(keys, x)), prod_values)
    return all_dicts

with open("configs/brain.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "filename":["adni_fold0.xls", "adni_fold1.xls", "adni_fold2.xls", "adni_fold3.xls", "adni_fold4.xls"],
    #"filename":["synthetic_fold0.xls", "synthetic_fold1.xls", "synthetic_fold2.xls", "synthetic_fold3.xls", "synthetic_fold4.xls"],
    "datatype":["adni"],
    #"datatype":["synthetic"],
    "algo": ["TRPO", "PPO", "DDPG", "SAC", "TRPO-LSTM", "PPO-LSTM"],
    "w_lambda":[1.0],
    "gamma":[1.0],
    "cog_init":["full"],
    "cog_type":["fixed"],
    "cog_mtl":[7.0],
    "epochs":[1000],
    "batch_size":[1000],
    "eval": [False],
    "score": ["MMSE"],
    "scale":[True],
    "discount":[1.00],
    "network":[32],
    "energy_model":["inverse"],
    "normalize":[False],
    "action_limit":[2.0],
    "trainsteps":[11],
    "seed":[613, 1727, 2594, 3954, 4910],
    "shap_enable": [True],
    "shap_use_all_samples": [False],
    "shap_show_fig": [False],
}

all_configs = [{**BASE_CONFIG, **p} for p in dict_product(PARAMS)]
if os.path.isdir("configs/train_configs"):
    shutil.rmtree("configs/train_configs")
    
os.makedirs("configs/train_configs/")

for i, config in enumerate(all_configs):
    with open(f"configs/train_configs/{i}.json", "w") as f:
        json.dump(config, f, indent=4, sort_keys=True)
