#!/usr/bin/env python3

#%%

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import random
import os
import time

from IPython.display import display

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

from rdkit.DataStructs import ExplicitBitVect
from rdkit.DataManip.Metric import GetTanimotoDistMat
from rdkit.Chem import AllChem, Draw, rdFMCS, PandasTools
from rdkit.Chem import MolFromSmiles, MolToSmiles, MolFromSmarts, MolToSmarts
from rdkit.ML.Cluster import Butina
from rdkit.Chem.MolStandardize.rdMolStandardize import Cleanup
from rdkit import RDLogger

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering, KMeans

from tqdm import tqdm

SEED=42
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

#%%

morgan3_fp = [f"morgan3_{i}" for i in range(2048)] #2048
rdkit3d = ['PMI1','PMI2','PMI3','NPR1','NPR2',
            'RadiusOfGyration','InertialShapeFactor',
            'Eccentricity','Asphericity','SpherocityIndex','PBF']
delta = ["score","betaScore","ligCover","LinF9"]
docking = ["XGB"]

#%%

def np_to_bv(fv):
	"""
	Convert Numpy binary array to RDKit ExplicitBitVect
	"""
	bv = ExplicitBitVect(len(fv))
	bv.SetBitsFromList(np.where(fv)[0].tolist())
	return bv

def get_mcs(mols):
	"""
	Find the MCS (maximum common substructure) of a set of molecules
	"""
	mcs = rdFMCS.FindMCS(mols,timeout=60)
	return mcs.smartsString

def get_Morgan3(mol):
    """
    Calculate radius-3 Morgan fingerprints (2048-bit)
    """
    fpgen = AllChem.GetMorganGenerator(radius=3,fpSize=2048)
    morgan3 = {f"morgan3_{i}": xi for i, xi in enumerate(fpgen.GetFingerprint(mol))}
    return morgan3

def cluster(data,cutoff,fp,get_fp,label,smiles):
	"""
	Cluster molecules by fingerprint Tanimoto distance with the Butina algorithm
	"""
	PandasTools.AddMoleculeColumnToFrame(data,smiles)
	mols = data["ROMol"].to_numpy()
	data = data.drop("ROMol", axis=1)
	print(f"Collecting fingerprints ...")
	fp_arr = data[fp].to_numpy()
	fp_list = [np_to_bv(fp) for fp in fp_arr]
	print(f"Calculating Tanimoto distances ...")
	distance_matrix = GetTanimotoDistMat(fp_list)
	n = len(fp_list)
	print(f"Clustering molecules ...")
	clusters = list(Butina.ClusterData(distance_matrix, n, cutoff, isDistData=True))
	random.shuffle(clusters)
	# clusters = sorted(clusters, key=len, reverse=True)
	print(f"Creating cluster data ...")
	cluster_data = []
	cluster_vals = np.empty(n)
	for cluster_idx, cluster in enumerate(tqdm(clusters)):
		cluster_dict = {}
		cluster = list(cluster)
		cluster_dict["size"] = len(cluster)
		cluster_dict["best_idx"] = cluster[np.argmax(data.loc[cluster][label])]
		cluster_dict["best_target"] = data[label].values[cluster_dict["best_idx"]][0]
		cluster_dict["centroid"] = cluster[0]
		cluster_dict["centroid_target"] = data[label].values[cluster[0]][0]
		centroid_fp = data[fp].iloc[cluster_dict["centroid"]]
		centroid_fp.keys().values[:] = [f"centroid_fp_{i}" for i in range(len(fp))]
		cluster_dict |= centroid_fp.to_dict()
		if len(cluster) > 1: cluster_dict["mcs"] = get_mcs(list(mols[cluster]))
		else: cluster_dict["mcs"] = MolToSmarts(mols[cluster[0]])
		try: mcs_mol = Cleanup(MolFromSmarts(cluster_dict["mcs"]))
		except: mcs_mol = mols[cluster_dict["centroid"]]
		mcs_fp = pd.Series(get_fp(mcs_mol))
		mcs_fp.keys().values[:] = [f"mcs_fp_{i}" for i in range(len(fp))]
		cluster_dict |= mcs_fp.to_dict()
		cluster_data.append(cluster_dict)
		for idx in cluster: cluster_vals[idx] = cluster_idx
	cluster_data = pd.DataFrame(cluster_data)
	data["cluster"] = cluster_vals
	print("Clustering complete.")
	return data, cluster_data

def cluster_pca(data,desc,n=2):
	"""
	Cluster molecules by fingerprint PCA distance with the K-Means algorithm
	"""
	print(f"Calculating PCA features ...")
	features = data[sum(desc,[])].values
	pca = PCA(n_components=n)
	pca_crds = pca.fit_transform(features)
	print(f"PCA variance explained: {round(np.sum(pca.explained_variance_ratio_),2)}")
	print(f"Clustering molecules ...")
	for k in tqdm([5,10,20,30,50,100]):
		clustering = KMeans(n_clusters=k,n_init="auto").fit(pca_crds)
		cluster_vals = clustering.labels_
		data[f"cpca_{k}"] = cluster_vals
	print("Clustering complete.")
	return data

def cluster_tsne(data,desc,n=2):
	"""
	Cluster molecules by fingerprint tSNE distance with the K-Means algorithm
	"""
	print(f"Calculating PCA features ...")
	features = data[sum(desc,[])].values
	pca = PCA(n_components=50)
	crds = pca.fit_transform(features)
	print(f"PCA variance explained: {round(np.sum(pca.explained_variance_ratio_),2)}")
	print(f"Calculating tSNE features ...")
	tsne = TSNE(n_components=n)
	tsne_crds = tsne.fit_transform(crds)
	print(f"Clustering molecules ...")
	for k in tqdm([5,10,20,30,50,100]):
		clustering = KMeans(n_clusters=k,n_init="auto").fit(tsne_crds)
		cluster_vals = clustering.labels_
		data[f"ctsne_{k}"] = cluster_vals
	print("Clustering complete.")
	return data

#%%

if __name__ == "__main__":
	target = "ESR1ago"
	l = 2048
	mol_data = pd.read_csv(os.path.join(target,f"{target}_data_full.csv"))
	mol_desc = [morgan3_fp]
	mol_data = cluster_pca(mol_data,mol_desc)
	mol_data = cluster_tsne(mol_data,mol_desc)
	mol_data.to_csv(os.path.join(target,f"{target}_data_full.csv"),index=False)

#%%