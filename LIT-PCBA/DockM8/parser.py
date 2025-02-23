
#%%

import os
import random
import time

import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import pearsonr, gaussian_kde
from scipy.spatial.distance import jaccard, pdist, cdist

from rdkit import Chem
from rdkit.Chem import Draw

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tqdm import tqdm

seed=42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)

#%%

target = "ESR1ago"
measures = {"ESR1ago":"Ratio-Fit_LogAC50-Replicate_1",
            "ESR1ant":"Fit_LogAC50-Replicate_1",
            "PPARG":"Ratio Potency (uM)",
            "TP53":"Ratio-Fit_LogAC50-Replicate_1",}
measure = measures[target]

with open(os.path.join(target,f"{target}_actives.smi")) as f:
    actives = np.array([l.strip("\n").split() for l in f.readlines()])
with open(os.path.join(target,f"{target}_inactives.smi")) as f:
    inactives = np.array([l.strip("\n").split() for l in f.readlines()])

#%%

docking_df = pd.read_csv(os.path.join(target,f"{target}_scores.csv"))
activity_df = pd.read_csv(os.path.join(target,f"{target}_datatable.csv"))

#%%

docking_df.head()

#%%

activity_df.head()

#%%

def get_sid(id,actives,inactives):
    if "INACT" in id: sid = int(id.split("_")[0][5:])
    else: sid = int(id.split("_")[0][3:])
    return float(sid)

def get_smi(id,actives,inactives):
    if "INACT" in id: 
        sid = id.split("_")[0][5:]
        i = np.argwhere(inactives[:,1]==sid)[0]
        smi = inactives[:,0][i][0]
    else: 
        sid = id.split("_")[0][3:]
        i = np.argwhere(actives[:,1]==sid)[0]
        smi = actives[:,0][i][0]
    return smi

#%%

docking_df["PUBCHEM_SID"] = docking_df["Pose ID"].apply(get_sid, args=(actives,inactives))
print(docking_df["PUBCHEM_SID"])

#%%

docking_df["smiles"] = docking_df["Pose ID"].apply(get_smi, args=(actives,inactives))
docking_df.head()

#%%

activity_df = activity_df[activity_df["PUBCHEM_ACTIVITY_OUTCOME"].isin(["Active","Inactive","Inconclusive"])]
if measure == "Fit_LogAC50":
    activity_df["pEC50"] = activity_df[measure].apply(lambda x: -float(x))
elif measure == "EC50": # measured in uM
    activity_df["pEC50"] = activity_df[measure].apply(lambda x: 6-np.log10(float(x)))
elif measure == "Ratio Potency (uM)":
    activity_df["pEC50"] = activity_df[measure].apply(lambda x: 6-np.log10(float(x))) 
elif measure == "Potency":
    activity_df["pEC50"] = activity_df[measure].apply(lambda x: 6-np.log10(float(x))) 
elif measure == "Potency-Replicate_1":
    measures = [col for col in activity_df.columns if "Potency-Replicate_" in col]
    # activity_df["pEC50"] = np.nanmean(6-np.log10(activity_df[measures].values.astype(float)),axis=1)
    AC50_replicates = 6-np.log10(activity_df[measures].values.astype(float))
    mean_pEC50 = [np.nanmean(replicates) if sum(~np.isnan(replicates)) > 1 else np.nan for replicates in AC50_replicates]
    activity_df["pEC50"] = mean_pEC50
elif measure == "Fit_LogAC50-Replicate_1":
    measures = [col for col in activity_df.columns if "Fit_LogAC50-Replicate_" in col]
    # activity_df["pEC50"] = np.nanmean(-activity_df[measures].values.astype(float),axis=1)
    AC50_replicates = -activity_df[measures].values.astype(float)
    mean_pEC50 = [np.nanmean(replicates) if sum(~np.isnan(replicates)) > 1 else np.nan for replicates in AC50_replicates]
    activity_df["pEC50"] = mean_pEC50
elif measure == "Ratio-Fit_LogAC50-Replicate_1":
    measures = [col for col in activity_df.columns if "Ratio-Fit_LogAC50-Replicate_" in col]
    # activity_df["pEC50"] = np.nanmean(-activity_df[measures].values.astype(float),axis=1)
    AC50_replicates = -activity_df[measures].values.astype(float)
    mean_pEC50 = [np.nanmean(replicates) if sum(~np.isnan(replicates)) > 1 else np.nan for replicates in AC50_replicates]
    activity_df["pEC50"] = mean_pEC50
elif measure == "W460-Fit_LogAC50-Replicate_1":
    measures = [col for col in activity_df.columns if "W460-Fit_LogAC50-Replicate_" in col]
    # activity_df["pEC50"] = np.nanmean(-activity_df[measures].values.astype(float),axis=1)
    AC50_replicates = -activity_df[measures].values.astype(float)
    mean_pEC50 = [np.nanmean(replicates) if sum(~np.isnan(replicates)) > 1 else np.nan for replicates in AC50_replicates]
    activity_df["pEC50"] = mean_pEC50
elif measure == "W530-Fit_LogAC50-Replicate_1":
    measures = [col for col in activity_df.columns if "W530-Fit_LogAC50-Replicate_" in col]
    # activity_df["pEC50"] = np.nanmean(-activity_df[measures].values.astype(float),axis=1)
    AC50_replicates = -activity_df[measures].values.astype(float)
    mean_pEC50 = [np.nanmean(replicates) if sum(~np.isnan(replicates)) > 1 else np.nan for replicates in AC50_replicates]
    activity_df["pEC50"] = mean_pEC50
elif measure == "Fit_LogAC50":
    measures = [col for col in activity_df.columns if "Fit_LogAC50" in col]
    # activity_df["pEC50"] = np.nanmean(-activity_df[measures].values.astype(float),axis=1)
    AC50_replicates = -activity_df[measures].values.astype(float)
    mean_pEC50 = [np.nanmean(replicates) if sum(~np.isnan(replicates)) > 1 else np.nan for replicates in AC50_replicates]
    activity_df["pEC50"] = mean_pEC50
activity_df["pEC50"] = activity_df["pEC50"].fillna(4.0)
activity_df.head()

#%%

new_df = pd.merge(activity_df[["PUBCHEM_SID","PUBCHEM_CID","PUBCHEM_EXT_DATASOURCE_SMILES",
                               "PUBCHEM_ACTIVITY_OUTCOME",measure,"pEC50","PUBCHEM_ACTIVITY_SCORE",]],
                  docking_df, on="PUBCHEM_SID")
new_df.rename(columns={"PUBCHEM_ACTIVITY_SCORE":"activity"},inplace=True)
new_df["active"] = new_df["PUBCHEM_ACTIVITY_OUTCOME"].apply(lambda x: int("Inactive" not in x))

#%%

print(f"Size before dropping duplicate molecules: {new_df.shape[0]}")
# Drop duplicates by taking the mean of reported values 
_, idx = np.unique(new_df["smiles"].values, return_index=True)
molecules = new_df["smiles"].values[np.sort(idx)]
for molecule in tqdm(molecules[:]):
    mol_df = new_df[new_df["smiles"] == molecule]
    new_df.loc[new_df["smiles"]==molecule, "pEC50"] = np.mean(mol_df["pEC50"].values)
new_df.drop_duplicates("smiles", keep="first", inplace=True)
new_df.reset_index(drop=True, inplace=True)
print(f"Size after dropping duplicate molecules: {new_df.shape[0]}")

#%%

new_df.head()

#%%

new_df.to_csv(os.path.join(target,f"{target}_data.csv"),index=False)

#%%