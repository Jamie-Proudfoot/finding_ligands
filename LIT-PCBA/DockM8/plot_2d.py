#%%

import os
import random
import time

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display, SVG
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from tqdm import tqdm
import seaborn as sns

seed=42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)

plt.rcParams['figure.dpi'] = 300

#%%

morgan3_fp = [f"morgan3_{i}" for i in range(2048)] # 1024
rdkit3d = ['PMI1','PMI2','PMI3','NPR1','NPR2',
            'RadiusOfGyration','InertialShapeFactor',
            'Eccentricity','Asphericity','SpherocityIndex','PBF']
docking = ["CNN-Affinity"]

#%%

target = "ESR1ago"
highlevel = "pEC50"
lowlevel = "CNN-Affinity"
mol_data = pd.read_csv(os.path.join("..","..","data",f"{target}_data_full.csv"))
# scaler = StandardScaler()
# mol_data[rdkit3d] = scaler.fit_transform(mol_data[rdkit3d].values)
# mol_data[delta] = scaler.fit_transform(mol_data[delta].values)
# mol_data[docking] = scaler.fit_transform(mol_data[docking].values)
features = mol_data[morgan3_fp]

pca = PCA(n_components=2)
pca_crds = pca.fit_transform(features)
activity_values = mol_data[highlevel].values
print(round(np.sum(pca.explained_variance_ratio_)*100,2))
plt.style.use("default")

#%%

from matplotlib.colors import LinearSegmentedColormap
np.random.seed(0)
fig, axis = plt.subplots(2, 2)
# Clusters
cluster = "cpca_10"
clusters = mol_data[cluster].values.astype(int).tolist()
n_clusters = len(np.unique(clusters))
colormap = plt.cm.rainbow(np.linspace(0,1,n_clusters))
cmap = LinearSegmentedColormap.from_list('custom', colormap, N=len(colormap))
try: mpl.colormaps.register(cmap=cmap)
except: pass
a0 = axis[0,0].scatter(pca_crds[:,0],pca_crds[:,1],s=4,c=colormap[clusters],alpha=0.8)
axis[0,0].set_xlabel("PC1",fontsize=15)
axis[0,0].set_ylabel("PC2",fontsize=15)
axis[0,0].set_title(f"Molecule PCA Plot (coloured by cluster)",fontsize=15)
# plt.colorbar(a0,ax=axis[0,0])
bounds = [i for i in range(0,n_clusters+1)]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
scalarmap = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
# plt.colorbar(colormap, ax=axis[0,0], label="Cluster")
cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),ax=axis[0,0],ticks=bounds)
# Diverse pool
diverse_pool = []
cluster = "cpca_10"
for i in range(10):
    cluster_idx = mol_data.index[mol_data[cluster]==i]
    if len(cluster_idx) > 0: diverse_pool.append(np.random.choice(cluster_idx))
docking_values = mol_data[lowlevel].values
pEC50_values = mol_data["pEC50"].values
idmax = np.argmax(pEC50_values)
# actives = np.argwhere(mol_data["active"].values==1)
# b0 = axis[0,1].scatter(pca_crds[:,0][actives],pca_crds[:,1][actives],s=20,c="cyan",zorder=3,label="active")
a1 = axis[0,1].scatter(pca_crds[:,0][idmax],pca_crds[:,1][idmax],s=20,c="red",zorder=3,label="top-1")
a2 = axis[0,1].scatter(pca_crds[:,0][diverse_pool],pca_crds[:,1][diverse_pool],s=60,c="lime",zorder=2,label="diverse initial pool",marker="x")
a3 = axis[0,1].scatter(pca_crds[:,0],pca_crds[:,1],s=4,c=pEC50_values,cmap="plasma",alpha=0.8,vmin=4,vmax=6,zorder=1)
plt.colorbar(a3,ax=axis[0,1])
axis[0,1].legend(fontsize=12)
axis[0,1].set_xlabel("PC1", fontsize=15)
axis[0,1].set_ylabel("PC2",fontsize=15)
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)
axis[0,1].set_title("Molecule PCA Plot (coloured by $pEC5_{50}$)",fontsize=15)
# Docking
docking_values = mol_data[lowlevel].values
top10 = np.argwhere(docking_values >= np.percentile(docking_values,90))
idmax = np.argmax(pEC50_values)
valmax = round(pEC50_values[idmax],2)
b1 = axis[1,0].scatter(pca_crds[:,0][idmax],pca_crds[:,1][idmax],s=20,c="red",label="top-1",zorder=2)
b2 = axis[1,0].scatter(pca_crds[:,0][top10],pca_crds[:,1][top10],s=10,c="orange",alpha=1,label="top-10% docking",zorder=3)
b3 = axis[1,0].scatter(pca_crds[:,0],pca_crds[:,1],s=4,c=docking_values,cmap="cividis",alpha=0.8,vmin=5,vmax=9)
plt.colorbar(b3,ax=axis[1,0])
axis[1,0].legend(fontsize=12)
axis[1,0].set_xlabel("PC1",fontsize=15)
axis[1,0].set_ylabel("PC2",fontsize=15)
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)
axis[1,0].set_title("Molecule PCA Plot (coloured by docking score)", fontsize=15)
# Top-10 docking
docking_values = mol_data[lowlevel].values
pEC50_values = mol_data["pEC50"].values
docking_pool = np.random.choice(np.where(docking_values >= np.percentile(docking_values,90))[0],10,replace=False).tolist()
idmax = np.argmax(pEC50_values)
# actives = np.argwhere(mol_data["active"].values==1)
# c0 = axis[1,1].scatter(pca_crds[:,0][actives],pca_crds[:,1][actives],s=20,c="cyan",zorder=3,label="active")
c1 = axis[1,1].scatter(pca_crds[:,0][idmax],pca_crds[:,1][idmax],s=20,c="red",zorder=3,label="top-1")
c2 = axis[1,1].scatter(pca_crds[:,0][docking_pool],pca_crds[:,1][docking_pool],s=60,c="lime",zorder=2,label="docking initial pool",marker="x")
c3 = axis[1,1].scatter(pca_crds[:,0],pca_crds[:,1],s=4,c=pEC50_values,cmap="plasma",alpha=0.8,vmin=4,vmax=6,zorder=1)
plt.colorbar(c3,ax=axis[1,1])
axis[1,1].legend(fontsize=12)
axis[1,1].set_xlabel("PC1",fontsize=15)
axis[1,1].set_ylabel("PC2",fontsize=15)
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)
axis[1,1].set_title("Molecule PCA Plot (coloured by $pEC_{50}$)",fontsize=15)
fig.subplots_adjust(hspace=0.3)
fig.subplots_adjust(wspace=0.15)
fig.set_size_inches(15, 16)
plt.show()
# fig.savefig(os.path.join(target,f"{target}_PCA.png"))

#%%