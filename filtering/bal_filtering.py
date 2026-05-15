
#%%

import os, sys
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import time
import requests
from rcsbapi.data import DataQuery as Query
from rcsbapi.search import Sort
from unipressed import IdMappingClient
from rdkit import Chem


#%%

LinF9_IDs = pd.read_csv("LinF9_train.csv")["pdb"].values.tolist()
DeltaLinF9XGB_IDs = pd.read_csv("DeltaLinF9XGB_train.csv")["pdb"].values.tolist()

#%%

targets = ["D2R","USP7","Mpro","TYK2"]

#%%

# Mode = "any": any ligands in the LinF9 / DeltaLinF9XGB training set regardless of target
# Mode = "target": any ligands for the given target in the LinF9 / DeltaLinF9XGB training set

LinF9_df = pd.read_csv("LinF9_train_labelled.csv")
DeltaLinF9XGB_df = pd.read_csv("DeltaLinF9XGB_train_labelled.csv")
LinF9_SMILES_unique = np.unique([v for x in LinF9_df["SMILES"].values.astype(str) for v in x.split() if v])
DeltaLinF9XGB_SMILES_unique = np.unique([v for x in DeltaLinF9XGB_df["SMILES"].values.astype(str) for v in x.split() if v])
training_SMILES = set(LinF9_df["SMILES"].values) | set(DeltaLinF9XGB_df["SMILES"].values)

mode = "target" # "any"

for target in targets:
    data_csv = os.path.join(".",f"{target}_final.csv")
    df = pd.read_csv(data_csv)
    smiles = [Chem.CanonSmiles(s) for s in df["SMILES"].values] # rdkit canonical smiles
    LinF9_target_SMILES = [v for x in LinF9_df[LinF9_df["GeneID"]==target]["SMILES"].values for v in x.split()]
    DeltaLinF9XGB_target_SMILES = [v for x in DeltaLinF9XGB_df[DeltaLinF9XGB_df["GeneID"]==target]["SMILES"].values for v in x.split()]
    training_target_SMILES = LinF9_target_SMILES + DeltaLinF9XGB_target_SMILES
    if mode == "target": filter_SMILES = training_target_SMILES
    elif mode == "any": filter_SMILES  = training_SMILES 
    else: filter_SMILES = training_SMILES
    overlapping = []
    for smi in smiles:
        if smi in filter_SMILES: overlapping.append(smi)
    print(target, len(overlapping), len(overlapping)/len(df) * 100)
    print(overlapping)
    if not os.path.exists("data_filtered"): os.mkdir("data_filtered")
    if len(overlapping) > 0:
        df = df[~df["SMILES"].isin(filter_SMILES)]
        df.to_csv(os.path.join("data_filtered",f"{target}_filtered_{mode}.csv"),index=False)
    print()

#%%