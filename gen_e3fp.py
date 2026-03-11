#%%

import os 
import tarfile

from rdkit import Chem

from python_utilities.parallel import Parallelizer
from e3fp.pipeline import fprints_from_mol
from e3fp.conformer.util import smiles_to_dict

import pandas as pd

#%%

target = "EGFR"
bits = 2048 # FP bit length
cores = 10 # cpu cores

#%%

# Load all SDF files into RDKit mols
conf_tar = os.path.join(target,"conformers.tar.gz")
tar = tarfile.open(conf_tar, "r:gz")

# generator of (mol, ID), top-ranked pose only
# updated to just load mols directly from tar.gz archive now
# (no need to unzip the tar file currently :>)
mols = {}
for file in tar.getmembers():
    if file.name.endswith("sdf"):
        name = os.path.basename(file.name).split(".")[0]
        f = tar.extractfile(file)
        if not f: continue 
        # ForwardSDMolsupplier is not iterable like SDMolSupplier so we use next(x, None)
        mol = next(Chem.ForwardSDMolSupplier(f, removeHs=False), None)
        assert mol != None
        mols[name] = mol
mols_iter = ((mol, name) for name, mol in mols.items())

#%%

# Generate all E3FP fingerprints

def fp_wrapper(mol, name, fprint_params=None):
    fps = fprints_from_mol(mol, fprint_params=fprint_params)
    return name, fps

# 2048-bit E3FP fingerprints
fprint_params = {'bits': bits, 'radius_multiplier': 1.5, 'rdkit_invariants': True}
kwargs = {"fprint_params": fprint_params} # parallel implementation
parallelizer = Parallelizer(parallel_mode="processes", num_proc=cores)
fprints_list = parallelizer.run(fp_wrapper, mols_iter, kwargs=kwargs)

#%%

# Convert fingerprints to dataframe

def fp_to_dict(fp, nbits=2048):
    bits = fp.to_vector(sparse=False)
    return {f"e3fp_{i}": int(bits[i]) for i in range(nbits)}

fingerprint_dict = {}
for name, fps in zip(mols.keys(), fprints_list):
    fp = fps[0][1][0]
    fingerprint_dict[name] = fp_to_dict(fp, bits)

df_fp = pd.DataFrame.from_dict(fingerprint_dict, orient="index")
df_fp = df_fp.rename_axis("molecule_chembl_id").reset_index()
# df_fp.head()

#%%

# Replace 2D fingerprints with 3D fingerprints
# "paste" into final dataframe using Pandas merge on IDs

# Load data csv file
path_to_data = os.path.join("data",f"{target}-2048_data_3d_delta_pKi.csv.tar.gz")
with tarfile.open(path_to_data, "r:gz") as tar:
    # assume only one CSV in the archive
    csv_member = [m for m in tar.getmembers() if m.name.endswith(".csv")][0]
    f = tar.extractfile(csv_member)
    data = pd.read_csv(f)
# data.head()

# Get indices to cut
c1 = data.columns.get_loc("morgan3_0")
c2 = data.columns.get_loc("morgan3_2047")
# Cut out old fps
df1 = data[data.columns[:c1]]
df2 = data[data.columns[c2+1:]]
# Replace w/ new ones
df1 = df1.merge(df_fp, on="molecule_chembl_id", how="inner")
df_final = pd.concat([df1,df2],axis=1)

#%%

# Re-cluster data using new E3FP fingerprints
from cluster import cluster_pca, cluster_tsne
fp_column = [[f"e3fp_{i}" for i in range(bits)]]
df_final = cluster_pca(df_final,fp_column)
df_final = cluster_tsne(df_final,fp_column)
# Save to csv
df_final.to_csv(os.path.join("data",f"{target}-e3fp_data_3d_delta_pKi.csv"),index=False)

#%%