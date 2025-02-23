#%%
import os
import random
import time

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from IPython.display import display, SVG
#from mayavi import mlab # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import seaborn as sns

seed=42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)

mpl.rcParams['figure.dpi'] = 300

#%%

target = "EGFR"
l = 2048
mode = "delta"
mol_data = pd.read_csv(os.path.join("data",f"{target}-{l}_data_3d_{mode}_pKi.csv"))

#%%

morgan3_fp = [f"morgan3_{i}" for i in range(l)] # 2048-bit Morgan FP
rdkit3d = ['PMI1','PMI2','PMI3','NPR1','NPR2',
            'RadiusOfGyration','InertialShapeFactor',
            'Eccentricity','Asphericity','SpherocityIndex','PBF']
delta = ["score","betaScore","ligCover","LinF9"]
docking = ["XGB"]

features = mol_data[morgan3_fp]

pca = PCA(n_components=2)
pca_crds = pca.fit_transform(features)
pKi_values = mol_data["pKi"].values
print(f"Percentage variance explained by this PCA: {round(np.sum(pca.explained_variance_ratio_)*100,2)}")
plt.style.use("dark_background")

XGB_values = mol_data["XGB"].values
pKi_values = mol_data["pKi"].values
diverse_pool = []
cluster = "cpca_10"
for i in range(10):
    cluster_idx = mol_data.index[mol_data[cluster]==i]
    if len(cluster_idx) > 0: diverse_pool.append(np.random.choice(cluster_idx))
docking_pool = np.argsort(-XGB_values)[:10].tolist()
idmax = np.argmax(pKi_values)

#%%

# 3D PCA plot (raw data)
# %matplotlib widget
plt.clf()
# 3D Plot
X = pca_crds[:,0]
Y = pca_crds[:,1]
# X, Y = np.meshgrid(X, Y)
Z = pKi_values
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(X, Y, Z, linewidth=0, antialiased=False, cmap='terrain',
                        vmin=5, vmax=9)
ax.set_xlabel("PC_1",fontsize=7,labelpad=-2,loc="right")
ax.set_ylabel("PC_2",fontsize=7,labelpad=2,loc="top")
fig.colorbar(surf, label="pKi")
ax.set_title('3D PCA Plot (triangular surface)')
ax.set_box_aspect(None, zoom=1.0)
ax.azim = -100
ax.elev = 45
ax.set_box_aspect(aspect=None, zoom=1.2)
ax.grid(False)
ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
plt.show()

#%%

# 3D PCA plot (nearest neighbour interpolation)
# %matplotlib widget
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = pca_crds[:,0]
Y = pca_crds[:,1]
Z = pKi_values
resolution = 300 # Increase resolution if necessary
xi = np.linspace(X.min(),X.max(),resolution)
yi = np.linspace(Y.min(),Y.max(),resolution)
zi = griddata((X, Y), Z, (xi[None,:], yi[:,None]), method='nearest') # linear
xig, yig = np.meshgrid(xi, yi)
surf = ax.plot_surface(xig, yig, zi, cmap='terrain', linewidth=0,
                       rstride=1, cstride=1, vmin=5, vmax=9)
ax.set_xlabel("PC_1",fontsize=7,labelpad=-2,loc="right")
ax.set_ylabel("PC_2",fontsize=7,labelpad=2,loc="top")
fig.colorbar(surf, label="pKi")
ax.set_title('3D PCA Plot (interpolated: nearest neighbour)')
ax.set_box_aspect(None, zoom=1.0)
ax.azim = -100
ax.elev = 45
ax.set_box_aspect(aspect=None, zoom=1.2)
ax.grid(False)
ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
plt.show()

#%%

# 3D PCA plot (linear interpolation)
# %matplotlib widget
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = pca_crds[:,0]
Y = pca_crds[:,1]
Z = pKi_values
resolution = 300 # Increase resolution if necessary
xi = np.linspace(X.min(),X.max(),resolution)
yi = np.linspace(Y.min(),Y.max(),resolution)
zi = griddata((X, Y), Z, (xi[None,:], yi[:,None]), method='linear') 
xig, yig = np.meshgrid(xi, yi)
surf = ax.plot_surface(xig, yig, zi, cmap='terrain', linewidth=0,
                       rstride=1, cstride=1, vmin=5, vmax=9)
ax.xaxis.set_tick_params(pad=2)
ax.yaxis.set_tick_params(pad=2)
ax.set_xlabel("PC_1",fontsize=7,labelpad=-2,loc="right")
ax.set_ylabel("PC_2",fontsize=7,labelpad=2,loc="top")
fig.colorbar(surf, label="pKi")
ax.set_title('3D PCA Plot (interpolated: linear)')
ax.set_box_aspect(None, zoom=1.0)
ax.azim = -100
ax.elev = 45
ax.set_box_aspect(aspect=None, zoom=1.2)
ax.grid(False)
ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
# ax.scatter(pca_crds[:,0][idmax],pca_crds[:,1][idmax],pKi_values[idmax]+0.5,s=50,c="red",label="top-1 pKi")
plt.tight_layout()
plt.figure(figsize=(10, 5), dpi=300)
plt.show()

#%%
