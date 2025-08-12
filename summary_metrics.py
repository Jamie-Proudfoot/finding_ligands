import os
import glob
import math

import numpy as np
import pandas as pd
from tqdm import tqdm

pred_target = "pKi" #"pEC50" for LIT-PCBA
data_suffix = "data_3d_delta_pKi" #"data_full" for LIT-PCBA
w = 1   # batch size
m = 10  # initial pool size
M = 10_000 # budget, e.g. 10_000: arbitrarily high for target termination
EF_target = 9.0 #4.0 for LIT-PCBA
dir = "results"
suffix = "" # e.g "10%" or "100" for fixed budgets
p = None # proportion of budget (M) to consider (None => 100%)

def random_analytic(D,v):
    """
    Analytic form of random sampling without replacement
    derived from the negative hypergeometic distribution
    D :: Data (list or 1D array of numerics)
    N :: Total finite population size
    v :: Hit target value
    H :: Number of 'hits'
    returns :: Expected number of random samples
    required to reach at least one 'hit'
    """
    N = len(D)
    quantile = (D < v).mean()
    H = int(round((1 - quantile) * N))
    return (N + 1) / (H + 1)

for file in tqdm(glob.glob(os.path.join(dir,"*ID.csv"))):
    filename = os.path.basename(os.path.normpath(file))
    job = filename[:-7]
    name = filename.split("_")[0]
    data_df = pd.read_csv(os.path.join("data",f"{name}_{data_suffix}.csv"))
    id_df = pd.read_csv(file)

    if p: M = int(len(data_df) * p/100)

    target_data = data_df[pred_target].values
    mol_target = np.max(target_data)

    steps_to_maximum = []
    recall = []
    enrichment_factor = []
    max_val = []
    for i in id_df.columns:
        train_idx = id_df[i].values
        train_idx = list(train_idx[~np.isnan(train_idx)].astype(int))[:M]
        Ytrain = data_df[pred_target].iloc[train_idx].values

        # Collect max value found
        mval = np.max(Ytrain)
        max_val.append(mval)

        # Collect number of steps needed to reach the maximum
        # Catch case where max(Ytrain) < target
        # check if run has found the maximum
        is_greater = Ytrain.flatten() >= mol_target
        if not np.any(is_greater):
            steps = len(Ytrain)*2 # arbitrary
            found = 0
        else:
            steps = np.argmax(is_greater)+1
            found = 1
        # Round up to batch size
        steps = math.ceil(steps/w)* w
        # Round up to initial pool if smaller (optional)
        # if steps < m: steps = m
        steps_to_maximum.append(steps)
        recall.append(found)

        # Collect enrichment factor
        rate = np.sum(target_data > EF_target) / len(target_data)
        sampled_rate = np.sum(Ytrain > EF_target) / len(Ytrain)
        ef = sampled_rate / rate
        enrichment_factor.append(ef)

    mean_steps = np.mean(steps_to_maximum)
    std_steps = round(np.std(steps_to_maximum),4)
    mean_recall = np.mean(recall)
    mean_EF = np.mean(enrichment_factor)
    std_EF = round(np.std(enrichment_factor),4)
    mean_max = np.mean(max_val)
    std_max = round(np.std(max_val),4)

    # Collect results as csv
    results = {
        "mean_steps":mean_steps,
        "std_steps":std_steps,
        "mean_EF":mean_EF,
        "std_EF":std_EF,
        "mean_max":mean_max,
        "std_max":std_max,
        "recall":mean_recall,
    }
    results_df = pd.DataFrame.from_dict(results,orient="index").transpose()
    if suffix: name = f"{job}_{suffix}_SM.csv"
    else: name = f"{job}_SM.csv"
    results_df.to_csv(os.path.join(dir,name),index=False)
