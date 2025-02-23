#%%
import os
import math
import traceback
import sys

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
tqdm.pandas()

#from dockstring import load_target # (unmodified dockstring)

# Load local version of dockstring (modified from original)
path_to_dockstring = os.path.realpath(os.path.join(".","dockstring"))
sys.path.append(os.path.abspath(path_to_dockstring))
from dockstring import load_target # importing from local folder

from natsort import natsorted

import time
import logging

from rdkit import Chem
from rdkit.Chem.rdmolops import AddHs
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.rdMolDescriptors import CalcPBF, CalcPMI1, CalcPMI2, CalcPMI3, CalcNPR1, CalcNPR2
from rdkit.Chem.rdMolDescriptors import CalcRadiusOfGyration, CalcInertialShapeFactor
from rdkit.Chem.rdMolDescriptors import CalcEccentricity, CalcAsphericity, CalcSpherocityIndex
from rdkit.Chem.rdMolDescriptors import CalcAUTOCORR3D, CalcRDF, CalcMORSE, CalcWHIM, CalcGETAWAY

import warnings

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

#%%

def ConfToMol(mol, conf_id):
    conf = mol.GetConformer(conf_id)
    new_mol = Chem.Mol(mol)
    new_mol.RemoveAllConformers()
    new_mol.AddConformer(Chem.Conformer(conf))
    return new_mol

def get_features(mol,keys=descriptors):
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


def dock_ligand(target,smi,CPUs):
    mol = Chem.MolFromSmiles(smi)
    # Generate each stereoisomer for chiral centres
    chiral_centers = Chem.FindMolChiralCenters(mol)
    n_centers = len(chiral_centers)
    isosmiles = []
    # Do not enumerate if there are too many centers
    if n_centers <= 6:
        opts = StereoEnumerationOptions(tryEmbedding=True)
        isomers = tuple(EnumerateStereoisomers(mol,options=opts))
        isosmiles = [Chem.MolToSmiles(x,isomericSmiles=True) for x in isomers]
    if len(isosmiles) == 0: isosmiles += [smi]
    isoligands = []
    isoscores = []
    # Dock each stereoisomer
    for isosmi in isosmiles:
        try:
            score, aux = target.dock(isosmi, num_cpus=CPUs)
            ligand = aux["ligand"]
            scores = aux["affinities"]
            if score >= 0: raise ValueError("Best binding energy is >= 0")
        except Exception as e:
            ligand = None
            scores = [10]
            err = e
        isoligands.append(ligand)
        isoscores.append(scores)
    # Choose best stereoisomer from docking score
    best = np.argmin([scores[0] for scores in isoscores])
    ligand = isoligands[best]
    scores = isoscores[best]
    # Catch case where all isomers fail
    if scores == [10]: raise err
    Chem.SanitizeMol(ligand)
    ligand = AddHs(ligand, addCoords=True)
    return ligand, scores

#%%

target_name = "EGFR"
confdir = os.path.join(target_name, "conformers")
if not os.path.exists(confdir): os.makedirs(confdir)
os.chdir(target_name)
data_file = os.path.join(f"{target_name}_data_pKi.csv")
dockstring_target = target_name
CPUs = 12 # number of CPUs to use
data = pd.read_csv(data_file)
data = data.iloc[:5]

#%%

print(f"DataFrame shape: {data.shape}")
print(f"Computing RDKit 3D features and docking scores ...")
target = load_target(dockstring_target)

#%%

output = []
n = 100
c = int((len(data)-1)/100)+1
splits = [data[i:i+c] for i in range(0,len(data),c)]
chunks = [df for df in splits if not df.empty]
t0_0 = time.time()
j = 1
for i, df in enumerate(chunks):
    print(f"Chunk number: {i}")
    t0 = time.time()
    temp = []
    for smi, id in zip(df["smiles"].values,df["molecule_chembl_id"].values):
        try:
            ligand, scores = dock_ligand(target,smi,CPUs)
            features = get_features(ConfToMol(ligand,0))
            features["score"] = scores[0]
            if np.isnan(list(features.values())).any():
                raise ValueError("NaN values in 3D features")
                #for key, value in features.items():
                    #if value != value: features[key] = 0
            writer = Chem.SDWriter(os.path.join("conformers",f"{id}.sdf"))
            for conf, score in zip(ligand.GetConformers(), scores):
                ligand.SetProp("score", str(score))
                ligand.SetProp("index", str(j))
                writer.write(ligand, confId=conf.GetId())
            j += 1
        except Exception as e:
            features = {desc: None for desc in descriptors+["score"]}
            print(f"Docking failed for {smi}")
            logging.error(e)
        temp.append(features)
    df.reset_index(drop=True, inplace=True)
    df = df.join(pd.DataFrame(temp))
    t1 = time.time()
    output.append(df)
    print(f"Time taken (s): {t1-t0}")
t1_0 = time.time()
print(f"Total time taken (s): {t1_0-t0_0}")
output = pd.concat(output)
print(output)
print(f"Shape before filtering NaN values: {output.shape}")
output.dropna(axis=0, how="any", inplace=True)
print(f"Shape after filtering NaN values: {output.shape}")
output.reset_index(drop=True, inplace=True)
output.to_csv(os.path.join(f"{target_name}_data_3d_pKi.csv"),index=False)

