
#%%

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import pearsonr
from tqdm import tqdm

#%%

target = "DRD2"

if target in ["DRD2","USP7","Mpro"]: suffix = "data_3d_delta"
elif target in ["TYK2"]: suffix = "data"
if target in ["DRD2","TYK2"]: y = "pKi"
elif target in ["Mpro","USP7"]: y = "pIC50"
if target == "TYK2": x = "DockingScore"
else: x = "XGB"

file = os.path.join(target,f"{target}_{suffix}.csv")
df = pd.read_csv(file)

#%%

def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'ρ = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)

g = sns.pairplot(df[list([x])+[y]])
g.map_lower(corrfunc)
plt.show()

#%%

def get_2D_descriptors(mol):
    """
    Calculate the full set of 2D RDKit descriptors for a molecule
    missingVal is used if the descriptor cannot be calculated
    """
    res = {}
    for nm, fn in Descriptors._descList:
        try: val = fn(mol)
        except:
            traceback.print_exc()
            val = None
        val = fn(mol)
        res[nm] = val
    return res

def get_MACCS(mol):
    """
    Calculate MACCS keys (166-bit)
    """
    maccs = {f"maccs_{i}": xi for i, xi in enumerate(MACCSkeys.GenMACCSKeys(mol))}
    return maccs

def get_Morgan2(mol):
    """
    Calculate radius-2 Morgan fingerprints
    """
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
    morgan2 = {f"morgan2_{i}": xi for i, xi in enumerate(fpgen.GetFingerprint(mol))}
    return morgan2

def get_Morgan3(mol):
    """
    Calculate radius-3 Morgan fingerprints
    """
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=2048)
    morgan3 = {f"morgan3_{i}": xi for i, xi in enumerate(fpgen.GetFingerprint(mol))}
    return morgan3

#%%

output_csv = os.path.join(target,f"{target}_data_all.csv")
print(f"Shape before adding descriptors: {df.shape}")
PandasTools.AddMoleculeColumnToFrame(df, 'smiles')
print(f"Computing 2D RDKit descriptors ...")
rdkit2d = pd.DataFrame([get_2D_descriptors(m) for m in tqdm(df["ROMol"])])
df = df.join(rdkit2d)
df.dropna(axis=0, how="any", inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Shape after adding 2D descriptors: {df.shape}")
# print(f"Computing MACCS keys ...")
# maccs = pd.DataFrame([get_MACCS(m) for m in tqdm(df["ROMol"])])
# df = df.join(maccs)
print(f"Computing radius-3 Morgan fingerprints ...")
morgan3 = pd.DataFrame([get_Morgan3(m) for m in tqdm(df["ROMol"])])
df = df.join(morgan3)
print(f"Shape after adding FP descriptors: {df.shape}")
print(f"Saving data to {output_csv}")
df = df.drop("ROMol", axis=1)
df.dropna(axis=0, how="any", inplace=True)
df.reset_index(drop=True, inplace=True)
df.to_csv(output_csv,index=False)

#%%
