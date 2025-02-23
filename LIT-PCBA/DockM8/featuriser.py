#%%

import os
import math
import time
import re
import random
import logging
import traceback
from pathlib import Path
from zipfile import ZipFile
from tempfile import TemporaryDirectory

from IPython.display import display, SVG

import numpy as np
import pandas as pd

from chembl_webresource_client.new_client import new_client
from tqdm.auto import tqdm

from rdkit import Chem
from rdkit import rdBase
from rdkit import RDLogger 
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors, Draw, PandasTools, MACCSkeys, AllChem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdmolops import AddHs
#from rdkit.Chem.Descriptors3D import CalcMolDescriptors3D
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.rdMolDescriptors import CalcPBF, CalcPMI1, CalcPMI2, CalcPMI3, CalcNPR1, CalcNPR2
from rdkit.Chem.rdMolDescriptors import CalcRadiusOfGyration, CalcInertialShapeFactor
from rdkit.Chem.rdMolDescriptors import CalcEccentricity, CalcAsphericity, CalcSpherocityIndex
from rdkit.Chem.rdMolDescriptors import CalcAUTOCORR3D, CalcRDF, CalcMORSE, CalcWHIM, CalcGETAWAY

rdBase.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.*')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tqdm import tqdm

seed=42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)

#%%

descList = ['PMI1','PMI2','PMI3','NPR1', 'NPR2',
            'RadiusOfGyration','InertialShapeFactor',
            'Eccentricity','Asphericity','SpherocityIndex','PBF']
AUTOCORR3D_desclist = [f"AUTOCORR3D_{i}" for i in range(80)]
RDF_desclist = [f"RDF_{i}" for i in range(210)]
MORSE_desclist = [f"MORSE_{i}" for i in range(224)]
GETAWAY_desclist = [f"GETAWAY_{i}" for i in range(273)]
WHIM_desclist = [f"WHIM_{i}" for i in range(114)]
descriptors = sum([
    descList,
    AUTOCORR3D_desclist,
    RDF_desclist,
    MORSE_desclist,
    GETAWAY_desclist,
    WHIM_desclist],[])

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

def get_Morgan3(mol):
    """
    Calculate radius-3 Morgan fingerprints
    """
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=2048)
    morgan3 = {f"morgan3_{i}": xi for i, xi in enumerate(fpgen.GetFingerprint(mol))}
    return morgan3

def get_3D_descriptors(mol,keys=descriptors):
    values = []
    try:
        values.append(CalcPMI1(mol))
        values.append(CalcPMI2(mol))
        values.append(CalcPMI3(mol))
        values.append(CalcNPR1(mol))
        values.append(CalcNPR2(mol))
        values.append(CalcRadiusOfGyration(mol))
        values.append(CalcInertialShapeFactor(mol))
        values.append(CalcEccentricity(mol))
        values.append(CalcAsphericity(mol))
        values.append(CalcSpherocityIndex(mol))
        values.append(CalcPBF(mol))
        values += CalcAUTOCORR3D(mol)
        values += CalcRDF(mol)
        values += CalcMORSE(mol)
        values += CalcWHIM(mol)
        values += CalcGETAWAY(mol)
    except: values += [None]*len(keys)
    return dict(zip(keys, values))


#%%

target = "ESR1ago"

with open(os.path.join(target,f"{target}_actives.smi")) as f:
    actives = np.array([l.strip("\n").split() for l in f.readlines()])
with open(os.path.join(target,f"{target}_inactives.smi")) as f:
    inactives = np.array([l.strip("\n").split() for l in f.readlines()])

#%%

df = pd.read_csv(os.path.join(target,f"{target}_data.csv"))
df.head()
print(len(df))

#%%

print("Loading docked poses ...")
confs = Chem.SDMolSupplier(f"{target}_poses.sdf")
ids = df["PUBCHEM_SID"].values
found, mols = [], []
for conf in tqdm(confs):
    id = float(re.search(r'\d+', conf.GetProp("_Name")).group(0))
    if id not in found and id in ids:
        found.append(id)
        mol = AddHs(conf,addCoords=True)
        mols.append(mol)
order = {v:i for i,v in enumerate(ids)}
mols3d = [mol for name, mol in sorted(zip(found,mols), key=lambda x: order[x[0]])]

#%%

print(f"Computing 2D RDKit descriptors ...")
mols2d = [Chem.MolFromSmiles(smi) for smi in df["smiles"].values]
rdkit2d = pd.DataFrame([get_2D_descriptors(m) for m in tqdm(mols2d)])
rdkit2d.fillna(0, inplace=True)
df = df.join(rdkit2d)
print(f"Computing MACCS keys ...")
maccs = pd.DataFrame([get_MACCS(m) for m in tqdm(mols2d)])
maccs.fillna(0, inplace=True)
df = df.join(maccs)
print(f"Computing radius-3 Morgan fingerprints ...")
morgan3 = pd.DataFrame([get_Morgan3(m) for m in tqdm(mols2d)])
morgan3.fillna(0, inplace=True)
df = df.join(morgan3)
print(f"Computing 3D RDKit descriptors ...")
rdkit3d = pd.DataFrame([get_3D_descriptors(m) for m in tqdm(mols3d)])
rdkit3d.fillna(0, inplace=True)
df = df.join(rdkit3d)

#%%

df.head()

#%%

df.to_csv(os.path.join(target,f"{target}_data_full.csv"),index=False)

#%%