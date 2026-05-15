
#%%

import numpy as np
import pandas as pd

from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem.rdmolops import AddHs
from rdkit.Chem.rdMolDescriptors import CalcPBF, CalcPMI1, CalcPMI2, CalcPMI3, CalcNPR1, CalcNPR2
from rdkit.Chem.rdMolDescriptors import CalcRadiusOfGyration, CalcInertialShapeFactor
from rdkit.Chem.rdMolDescriptors import CalcEccentricity, CalcAsphericity, CalcSpherocityIndex


#%%


descList = ['PMI1','PMI2','PMI3','NPR1', 'NPR2',
            'RadiusOfGyration','InertialShapeFactor',
            'Eccentricity','Asphericity','SpherocityIndex','PBF']
AUTOCORR3D_desclist = [f"AUTOCORR3D_{i}" for i in range(80)]
RDF_desclist = [f"RDF_{i}" for i in range(210)]
MORSE_desclist = [f"MORSE_{i}" for i in range(224)]
GETAWAY_desclist = [f"GETAWAY_{i}" for i in range(273)]
WHIM_desclist = [f"WHIM_{i}" for i in range(114)]
descriptors = descList

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
    except: values += [None]*len(keys)
    return dict(zip(keys, values))


#%%

df = pd.read_csv("TYK2_data_pKi.csv")

#%%

mols = Chem.SDMolSupplier("10k_most_similar_tyk2_charged.sdf", removeHs=False)
features = []
for i, mol in enumerate(tqdm(mols)): 
    if i == 0: continue #skip reference ligand
    feat3d = get_features(mol)
    features.append(feat3d)

#%%

print(f"Shape before adding 3D features: {df.shape}")
df.reset_index(drop=True, inplace=True)
df = df.join(pd.DataFrame(features))
print(f"Shape after adding 3D features: {df.shape}")


#%%

df.to_csv("TYK2_data_3d_pKi.csv")

#%%