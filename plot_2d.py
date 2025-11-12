#%%
import os
import random
import time

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display, SVG
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.interpolate import griddata
# import umap
from tqdm import tqdm
import seaborn as sns

seed=42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)

mpl.rcParams['figure.dpi'] = 300

#%%

target = "EGFR"

CHEMBL = ["EGFR","JAK2","LCK","NOS1","MAOB","ACHE","PARP1","PDE5A","PTGS2","ESR1","NR3C1","AR","F10","ADRB2"]
LITPCBA = ["ESR1ago","ESR1ant","PPARG","TP53"]

if target in CHEMBL: mode = "delta"
elif target in LITPCBA: mode = "litpcba"
else: raise ValueError("Target not available.")
l = 2048

if mode == "delta": 
    highlevel = "pKi"
    lowlevel = "XGB" # delta_LinF9_XGB docking
    mol_data = pd.read_csv(f"data/{target}-{l}_data_3d_{mode}_pKi.csv")
    vmin, vmax = 5, 9
    ylabel = "p$K_i$"
elif mode == "litpcba": 
    highlevel = "pEC50"
    lowlevel = "CNN-Affinity" # GNINA docking
    mol_data = pd.read_csv(f"data/{target}_data_full.csv")
    vmin, vmax = 4, 5
    ylabel = "p$EC_{50}$"

morgan3_fp = [f"morgan3_{i}" for i in range(l)] # 2048-bit Morgan FP
rdkit3d = ['PMI1','PMI2','PMI3','NPR1','NPR2',
            'RadiusOfGyration','InertialShapeFactor',
            'Eccentricity','Asphericity','SpherocityIndex','PBF']
docking = [lowlevel]
features = mol_data[morgan3_fp]

pca = PCA(n_components=100)
pca_crds = pca.fit_transform(features)
y_values = mol_data[highlevel].values
print(f"Percentage variance explained by this PCA: {round(np.sum(pca.explained_variance_ratio_)*100,2)}")
# plt.style.use("default")

#%%

# Main figure 
np.random.seed(0)
fig, axis = plt.subplots(2, 2)
# fig.set_figheight(20)
# fig.set_figwidth(15)
# Clusters
cluster = "cpca_10"
clusters = mol_data[cluster].values.astype(int).tolist()
n_clusters = len(np.unique(clusters))
colormap = plt.cm.rainbow(np.linspace(0,1,n_clusters))
cmap = LinearSegmentedColormap.from_list('custom', colormap, N=len(colormap))
try: mpl.colormaps.register(cmap=cmap)
except: pass
a0 = axis[0,0].scatter(pca_crds[:,0],pca_crds[:,1],s=10,c=colormap[clusters],alpha=0.8)
axis[0,0].set_xlabel("PC1",fontsize=10)
axis[0,0].set_ylabel("PC2",fontsize=10)
axis[0,0].set_title(f"Molecular clusters",fontsize=11)
axis[0,0].xaxis.set_tick_params(labelsize=8)
axis[0,0].yaxis.set_tick_params(labelsize=8)
# plt.colorbar(a0,ax=axis[0,0])
bounds = np.array([i for i in range(0,n_clusters+1)])
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
scalarmap = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
# plt.colorbar(colormap, ax=axis[0,0], label="Cluster")
cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),ax=axis[0,0],ticks=bounds[:-1]+1)
cbar.ax.tick_params(labelsize=8)
# Diverse pool
diverse_pool = []
cluster = "cpca_10"
for i in range(10):
    cluster_idx = mol_data.index[mol_data[cluster]==i]
    if len(cluster_idx) > 0: diverse_pool.append(np.random.choice(cluster_idx))
docking_values = mol_data[lowlevel].values
y_values = mol_data[highlevel].values
idmax = np.argmax(y_values)
a1 = axis[0,1].scatter(pca_crds[:,0][idmax],pca_crds[:,1][idmax],s=25,c="red",zorder=3,label="top-1")
a2 = axis[0,1].scatter(pca_crds[:,0][diverse_pool],pca_crds[:,1][diverse_pool],s=40,c="lime",zorder=2,label="initial pool",marker="x",linewidths=1.5)
a3 = axis[0,1].scatter(pca_crds[:,0],pca_crds[:,1],s=8,c=y_values,cmap="plasma",alpha=0.8,vmin=vmin,vmax=vmax,zorder=1)
cbar = plt.colorbar(a3,ax=axis[0,1])
cbar.ax.tick_params(labelsize=8)
cbar.set_label(ylabel, fontsize=10)
axis[0,1].legend(fontsize=8, frameon=True, handletextpad=0.025)
# axis[0,1].set_xlabel("PC1", fontsize=15)
# axis[0,1].set_ylabel("PC2",fontsize=15)
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)
axis[0,1].set_title("Diversity-based initialisation",fontsize=11)
axis[0,1].xaxis.set_tick_params(labelsize=8)
axis[0,1].yaxis.set_tick_params(labelsize=8)
# Docking
docking_values = mol_data[lowlevel].values
top10 = np.argwhere(docking_values >= np.percentile(docking_values,90))
idmax = np.argmax(y_values)
valmax = round(y_values[idmax],2)
b1 = axis[1,0].scatter(pca_crds[:,0][idmax],pca_crds[:,1][idmax],s=25,c="red",label="top-1",zorder=3)
b2 = axis[1,0].scatter(pca_crds[:,0][top10],pca_crds[:,1][top10],s=8,c="orange",alpha=0.8,label="top-10% docking",zorder=2)
b3 = axis[1,0].scatter(pca_crds[:,0],pca_crds[:,1],s=8,c=docking_values,cmap="cividis",alpha=0.8,vmin=4,vmax=9)
cbar = plt.colorbar(b3,ax=axis[1,0])
cbar.ax.tick_params(labelsize=8)
axis[1,0].legend(fontsize=8, frameon=True, handletextpad=0.025)
# axis[1,0].set_xlabel("PC1",fontsize=15)
# axis[1,0].set_ylabel("PC2",fontsize=15)
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)
axis[1,0].set_title("Docking scores", fontsize=11)
axis[1,0].xaxis.set_tick_params(labelsize=8)
axis[1,0].yaxis.set_tick_params(labelsize=8)
# Top-10 docking
docking_values = mol_data[lowlevel].values
y_values = mol_data[highlevel].values
docking_pool = np.random.choice(np.where(docking_values >= np.percentile(docking_values,90))[0],10,replace=False).tolist()
idmax = np.argmax(y_values)
c1 = axis[1,1].scatter(pca_crds[:,0][idmax],pca_crds[:,1][idmax],s=25,c="red",zorder=3,label="top-1")
c2 = axis[1,1].scatter(pca_crds[:,0][docking_pool],pca_crds[:,1][docking_pool],s=40,c="lime",zorder=2,label="initial pool",marker="x",linewidths=1.5)
c3 = axis[1,1].scatter(pca_crds[:,0],pca_crds[:,1],s=8,c=y_values,cmap="plasma",alpha=0.8,vmin=vmin,vmax=vmax,zorder=1)
cbar = plt.colorbar(c3,ax=axis[1,1])
cbar.ax.tick_params(labelsize=8)
cbar.set_label(ylabel, fontsize=10)
axis[1,1].legend(fontsize=8, frameon=True, handletextpad=0.025)
# axis[1,1].set_xlabel("PC1",fontsize=15)
# axis[1,1].set_ylabel("PC2",fontsize=15)
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)
axis[1,1].set_title("Docking-based initialisation",fontsize=11)
axis[1,1].xaxis.set_tick_params(labelsize=8)
axis[1,1].yaxis.set_tick_params(labelsize=8)
fig.subplots_adjust(hspace=0.3)
fig.subplots_adjust(wspace=0.15)
fig.set_size_inches(8, 8)
plt.show()
# fig.savefig(os.path.join("..","images",f"{target.lower()}_pca.png"),bbox_inches='tight')

#%%

# Molecule PCA Plot (coloured by activity)
idmax = np.argmax(y_values)
valmax = round(y_values[idmax],2)
plt.scatter(pca_crds[:,0][idmax],pca_crds[:,1][idmax],s=20,c="red",zorder=2)
plt.scatter(pca_crds[:,0],pca_crds[:,1],s=10,c=y_values,cmap="plasma",alpha=0.8,vmin=vmin,vmax=vmax)
plt.colorbar()
plt.xlabel("PC1",fontsize=15)
plt.ylabel("PC2",fontsize=15)
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)
plt.title(f"Molecule PCA Plot (coloured by {ylabel})",fontsize=15)

#%%

# Molecule PCA Plot (coloured by cluster)
cluster = "cpca_10"
clusters = mol_data[cluster].values.astype(int).tolist()
n_clusters = len(np.unique(clusters))
colormap = plt.cm.rainbow(np.linspace(0,1,n_clusters))
plt.figure(figsize=(5,5)) 
plt.scatter(pca_crds[:,0],pca_crds[:,1],s=10,c=colormap[clusters],alpha=0.8)
plt.xlabel("PC1",fontsize=15)
plt.ylabel("PC2",fontsize=15)
plt.title(f"Molecule PCA Plot (coloured by cluster)",fontsize=15)
# plt.savefig(f"{target}_cluster.svg")

#%%

# Molecule PCA Plot (including diverse initial pool)
diverse_pool = []
cluster = "cpca_10"
for i in range(10):
    cluster_idx = mol_data.index[mol_data[cluster]==i]
    if len(cluster_idx) > 0: diverse_pool.append(np.random.choice(cluster_idx))
docking_values = mol_data[lowlevel].values
y_values = mol_data[highlevel].values
idmax = np.argmax(y_values)
plt.figure(figsize=(5,5)) 
plt.scatter(pca_crds[:,0][idmax],pca_crds[:,1][idmax],s=20,c="red",zorder=3,label="top-1")
plt.scatter(pca_crds[:,0][diverse_pool],pca_crds[:,1][diverse_pool],s=40,c="lime",zorder=2,label="diverse",marker="x")
plt.scatter(pca_crds[:,0],pca_crds[:,1],s=10,c=y_values,cmap="plasma",alpha=0.8,vmin=vmin,vmax=vmax,zorder=1)
plt.colorbar()
plt.legend(fontsize=12)
plt.xlabel("PC1", fontsize=15)
plt.ylabel("PC2",fontsize=15)
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)
plt.title(f"Molecule PCA Plot (coloured by {ylabel})",fontsize=15)


#%%

# Molecule PCA Plot (coloured by docking)
docking_values = mol_data[lowlevel].values
top10 = np.argwhere(docking_values >= np.percentile(docking_values,90))
idmax = np.argmax(y_values)
valmax = round(y_values[idmax],2)
plt.figure(figsize=(5,5)) 
plt.scatter(pca_crds[:,0][idmax],pca_crds[:,1][idmax],s=20,c="red",label="top-1",zorder=2)
# plt.scatter(pca_crds[:,0][top10],pca_crds[:,1][top10],s=2,c="orange",alpha=1,label="top-10% docking",zorder=3)
plt.scatter(pca_crds[:,0],pca_crds[:,1],s=8,c=docking_values,cmap="cividis",alpha=0.8,vmin=4,vmax=9)
plt.colorbar()
# plt.legend(fontsize=12)
plt.xlabel("PC1",fontsize=15)
plt.ylabel("PC2",fontsize=15)
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)
plt.title("Molecule PCA Plot (coloured by docking)", fontsize=15)

#%%

# Molecule PCA Plot (including top-10 initial pool)
docking_values = mol_data[lowlevel].values
y_values = mol_data[highlevel].values
docking_pool = np.argsort(-docking_values)[:10].tolist()
idmax = np.argmax(y_values)
plt.figure(figsize=(5,5))
plt.scatter(pca_crds[:,0][idmax],pca_crds[:,1][idmax],s=20,c="red",zorder=3,label="top-1")
plt.scatter(pca_crds[:,0][docking_pool],pca_crds[:,1][docking_pool],s=40,c="lime",zorder=2,label="top-10 docking",marker="x")
plt.scatter(pca_crds[:,0],pca_crds[:,1],s=10,c=y_values,cmap="plasma",alpha=1.0,vmin=vmin,vmax=vmax,zorder=1)
plt.colorbar()
# plt.legend(fontsize=12)
plt.xlabel("PC1",fontsize=15)
plt.ylabel("PC2",fontsize=15)
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)
plt.title(f"Molecule PCA Plot (coloured by {ylabel})",fontsize=15)


#%%

# Molecule PCA Plot coloured by cutoffs
idmax = np.argmax(y_values)
valmax = round(y_values[idmax],2)
idx_10 = np.argwhere(y_values >= 10)
idx_9 = np.argwhere((y_values >= 9) & (y_values < 10))
idx_8 = np.argwhere((y_values >= 8) & (y_values < 9))
idx_7 = np.argwhere((y_values >= 7) & (y_values < 8))
plt.figure(figsize=(4.5,5))
plt.scatter(pca_crds[:,0],pca_crds[:,1],s=5,c="b",alpha=0.8)
plt.scatter(pca_crds[:,0][idx_7],pca_crds[:,1][idx_7],s=10,c="yellow",alpha=1,label=f"{ylabel} $\geq$ 7")
plt.scatter(pca_crds[:,0][idx_8],pca_crds[:,1][idx_8],s=20,c="coral",alpha=1,label=f"{ylabel} $\geq$ 8")
plt.scatter(pca_crds[:,0][idx_9],pca_crds[:,1][idx_9],s=30,c="red",alpha=1,label=f"{ylabel} $\geq$ 9")
plt.scatter(pca_crds[:,0][idx_10],pca_crds[:,1][idx_10],s=40,c="purple",alpha=1,label=f"{ylabel} $\geq$ 10")
plt.scatter(pca_crds[:,0][idmax],pca_crds[:,1][idmax],s=50,c="cyan",alpha=1,label=f"{ylabel} = {valmax}")
plt.legend(fontsize=12)
plt.xlabel("PC1",fontsize=15)
plt.ylabel("PC2",fontsize=15)
plt.title("Molecule PCA Plot ",fontsize=15)

# %%
