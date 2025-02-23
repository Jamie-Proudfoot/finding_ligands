#%%
import sys
import os
import math
import traceback

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
tqdm.pandas()

import time
import logging

import warnings
warnings.filterwarnings("ignore")

from natsort import natsorted

# Load local version of delta_LinF9_XGB (modified from original)
path_to_XGB = os.path.realpath(os.path.join(".","delta_LinF9_XGB","script"))
sys.path.append(os.path.abspath(path_to_XGB))
from runXGB_all import run_XGB # importing from local folder
# WARNING: above must be run using the "DXGB" environment
# envs/DXGB.yml

#%%

target_name = "EGFR"
fpLen = 2048

#%%

os.chdir(f"{target_name}")
cwd = os.getcwd()
path_to_ligands = os.path.join(cwd,"conformers")
path_to_protein = os.path.join(cwd,f"{target_name}.pdb")
data_file = f"{target_name}_data_3d_pKi.csv"
l = 2048 # 2048-bit Morgan FP
data = pd.read_csv(data_file)
data = data.iloc[:]

#%%

vinaF = [f"vinaF_{i}" for i in range(49)]
betaScore = ["betaScore"]
ligCover = ["ligCover"]
sasaF = [f"sasaF_{i}" for i in range(30)]
ligF = [f"ligF_{i}" for i in range(10)]
watF = [f"watF_{i}" for i in range(3)]
LinF9 = ["LinF9"]
XGB = ["XGB"]
descList = vinaF + betaScore + ligCover + sasaF + ligF + watF + LinF9 + XGB

#%%

temp = []
t0_0 = time.time()
pro = path_to_protein
os.chdir(path_to_ligands)
for molec in data["molecule_chembl_id"].values:
    lig = f"{molec}.sdf"
    try:
        delta = {}
        print(f"Re-scoring docking for {lig}")
        results  = run_XGB(pro, lig).values()
        flat_results = [x for xs in results for x in xs]
        delta.update(zip(descList,flat_results))
    except Exception as e:
        delta = {desc: None for desc in descList}
        print(f"Re-scoring failed for {lig}")
        logging.error(e)
    temp.append(delta)
t1_0 = time.time()
print(f"Total time taken (s): {t1_0-t0_0}")

temp = pd.DataFrame(temp)
data.reset_index(drop=True, inplace=True)
output = pd.concat((data, pd.DataFrame(temp)), axis=1)
print(f"Shape before adding delta descriptors: {data.shape}")
print(f"Shape before filtering NaN values: {output.shape}")
output.dropna(axis=0, how="any", inplace=True)
print(f"Shape after filtering NaN values: {output.shape}")
output.reset_index(drop=True, inplace=True)
print(f"Shape after adding delta descriptors: {output.shape}")
os.chdir(cwd)
output.to_csv(f"{target_name}-{l}_data_3d_delta_pKi.csv",index=False)
