
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

seed=42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)

#%%

target = "ESR1ago"

#%%

def converter(target,file_name):
    sppl = Chem.SDMolSupplier(file_name)
    outname = file_name.replace(".sdf", ".smi")
    out_file = open(os.path.join(target,outname), "w")
    smiles = []
    for mol in sppl:
        if mol is not None:# some compounds cannot be loaded.
            smi = Chem.MolToSmiles(mol)
            if smi not in smiles:
                name = mol.GetProp("_Name")
                out_file.write(f"{smi} {name}\n")
                smiles.append(smi)
    out_file.close()
converter(target,os.path.join(target,f"{target}_inactives.sdf"))

#%%
