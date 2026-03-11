#%%

import os 
import tarfile

import prolif as plf
import MDAnalysis as mda
from MDAnalysis.topology.guessers import guess_types
from rdkit import Chem

import numpy as np
import pandas as pd

#%%

target = "EGFR"
residues = "all"
if type(residues) == list: residues = list(map(str.upper, residues))
cores = 10 # cpu cores
cut = False # replace columns (True) or append to dataframe (False)

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


#%%

# Load protein with RDKit
protein_file = os.path.join(target,f"{target}.pdb")
u = mda.Universe(protein_file)
rdkit_prot = Chem.MolFromPDBFile(protein_file, removeHs=False)
chainIDs = [a.GetPDBResidueInfo().GetChainId() for a in rdkit_prot.GetAtoms()]
if len(np.unique(chainIDs)) == 1 and np.unique(chainIDs)[0] in ["", " "]: 
    # we set chainID to 'A' if singular and missing (will be white space " " if missing from PDB)
    for atom in rdkit_prot.GetAtoms(): atom.GetPDBResidueInfo().SetChainId('A')
protein_mol = plf.Molecule.from_rdkit(rdkit_prot)

#%%

interactions = plf.Fingerprint.list_available() # use all possible interactions
# print(interactions)
# update interaction thresholds to (generous) GLIDE-like settings
hbd_params = {"distance": 3.5, "DHA_angle": (100,180)}
hba_params = {"distance": 3.5, "DHA_angle": (100,180)}
custom_vdw = {"vdwradii": {"Fe": 2.15}} # Fe is missing from default params :,<
fp = plf.Fingerprint(interactions=interactions, parameters={"HBDonor": hbd_params, "HBAcceptor": hba_params, "VdWContact": custom_vdw}, count=True)
ligand_mols = [plf.Molecule.from_rdkit(mol) for mol in mols.values()]
# parallelism is implemented natively in prolif package :>
fp.run_from_iterable(ligand_mols, protein_mol, residues=residues, n_jobs=cores)
df_fp = fp.to_dataframe(drop_empty=False)
df_fp.columns = ["_".join(c[1:]) for c in df_fp.columns.to_flat_index()]
df_fp = df_fp.loc[:, (df_fp != 0).any(axis=0)] # drop columns where all values are 0
# n.b. fingerprint length will therefore depend on the protein and the set of ligands
df_fp["molecule_chembl_id"] = list(mols.keys())
df_fp = df_fp.set_index("molecule_chembl_id")
fp_column = df_fp.columns.to_list()
df_fp.head()

#%%

# %matplotlib ipympl

# View 2d plot of interactions
indx=100
view = fp.plot_lignetwork(ligand_mols[indx], kind="frame", frame=indx, display_all=True)
view

#%%

# View 3D plot of interactions
view = fp.plot_3d(ligand_mols[indx], protein_mol, frame=indx, display_all=False)
view

#%%

fig = fp.plot_barcode(xlabel="Ligand",figsize=(8,24),residues_tick_location="bottom")
fig.tick_params(axis='y', which='major', labelsize=8, labelrotation=0)
fig

#%%

# Replace/merge 2D fingerprints with 3D fingerprints
# "paste" into final dataframe using Pandas merge on IDs

# Load data csv file (also no need to unzip this tar gz compressed file :>)
path_to_data = os.path.join("data",f"{target}-2048_data_3d_delta_pKi.csv.tar.gz")
with tarfile.open(path_to_data, "r:gz") as tar:
    # assume only one CSV in the archive
    csv_member = [m for m in tar.getmembers() if m.name.endswith(".csv")][0]
    f = tar.extractfile(csv_member)
    data = pd.read_csv(f)
# data.head()

if cut:
    # Get indices to cut
    c1 = data.columns.get_loc("morgan3_0")
    c2 = data.columns.get_loc("morgan3_2047")
    # Cut out old fps
    df1 = data[data.columns[:c1]]
    df2 = data[data.columns[c2+1:]]
    # Replace w/ new ones
    df1 = df1.merge(df_fp, on="molecule_chembl_id", how="inner")
    df_final = pd.concat([df1,df2],axis=1)
else:
    df_final = data.merge(df_fp, on="molecule_chembl_id", how="inner")

#%%

# Re-cluster data using new E3FP fingerprints
from cluster import cluster_pca, cluster_tsne
fp_column = [fp_column]
df_final = cluster_pca(df_final,fp_column,label="plif-cpca")
df_final = cluster_tsne(df_final,fp_column,label="plif-ctsne")
# Save to csv
df_final.to_csv(os.path.join("data",f"{target}-plif_data_3d_delta_pKi.csv"),index=False)

#%%