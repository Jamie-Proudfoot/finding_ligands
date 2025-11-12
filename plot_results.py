#%%
import os
import numpy as np
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.patches as mpatches
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, gaussian_kde

mpl.rcParams['figure.dpi'] = 600

#%%

target = "EGFR"

CHEMBL = ["EGFR","JAK2","LCK","MAOB","NOS1","ACHE","PARP1","PTGS2","PDE5A","ESR1","NR3C1","AR","ADRB2","F10"]
LITPCBA = ["ESR1ago","ESR1ant","PPARG","TP53"]
if target in CHEMBL: mode = "delta"
elif target in LITPCBA: mode = "litpcba"
config = "BRR_greedy"

if mode == "delta":
    Nreps = 25
    l = "-2048"
    lowlevel = "XGB"
    results = "results"
    y = "pKi"
    hit = 9.0
    eofs = ["random_10","tanimoto_morgan3_10","morgan3_rdkit2d_10","morgan3_rdkit2d_10_cpca",f"morgan3_rdkit2d_rdkit3d_delta_docking_10_top_{lowlevel}_P90"]
    configs = ["baseline","baseline",config,config,config]
    names = ["random", "similarity", "BO (B1)", "BO (B2)", "BO + docking (D)"]
    pal = {"random": "r", "similarity": "brown", "BO (B1)": "orange","BO (B2)": "b", "D": "g"}
    ticks = 50
    ylabel="\\text{p}K_i"
elif mode == "litpcba":
    Nreps = 25
    l = ""
    lowlevel = "CNN-Affinity"
    results = "results_litpcba"
    y = "pEC50"
    hit = 7.0
    eofs = ["random_10","tanimoto_morgan3_10","morgan3_rdkit2d_10","morgan3_rdkit2d_10_cpca",f"morgan3_rdkit2d_rdkit3d_docking_10_top_{lowlevel}_P90"]
    configs = ["baseline","baseline",config,config,config]
    names = ["random", "similarity", "BO (B1)", "BO (B2)", "morgan3_rdkit2d_rdkit3d_docking // docking"]
    pal = {"random": "r", "similarity": "brown", "BO (B1)": "orange","BO (B2)": "b", "morgan3_rdkit2d_rdkit3d_docking // docking": "g"}
    ticks = 200
    ylabel="\\text{p}EC_{50}"

#%%

def random_analytic(D,v):
    """
    Analytic form of random sampling without replacement
    derived from the negative hypergeometic distribution
    D :: Data (list or 1D array of numerics)
    N :: Total finite population size
    v :: Hit target value
    H :: Number of 'hits'
    returns :: Expected number of random samples
    required to reach at least one 'hit'
    """
    N = len(D)
    quantile = (D < v).sum() / N
    H = int(round((1 - quantile) * N))
    return (N + 1) / (H + 1)

def random_analytic_std(D,v):
    """
    Analytic form of random sampling without replacement
    derived from the negative hypergeometic distribution
    D :: Data (list or 1D array of numerics)
    N :: Total finite population sizes
    v :: Hit target value
    H :: Number of 'hits'
    returns :: Standard deviation of the number of random samples
    required to reach at least one 'hit'
    """
    N = len(D)
    quantile = (D < v).sum() / N
    H = int(round((1 - quantile) * N))
    return np.sqrt(((N - H)*(N + 1)*H) / ((H + 1)**2 * (H + 2)))

#%%

# Mean steps to target plots

dfs = []
for eof,config in zip(eofs,configs):
    folder = os.path.join(results,config)
    if config != "baseline": file = f"{target}{l}_{config.split('_')[0]}_{eof}.csv"
    else: file = f"{target}{l}_{eof}.csv"
    df = pd.read_csv(os.path.join(folder,file))
    dfs.append(df)
    if mode == "litpcba": data = pd.read_csv(os.path.join("data",f"{target}_data_full.csv"))
    else: data = pd.read_csv(os.path.join("data",f"{target}{l}_data_3d_{mode}_pKi.csv"))

#%%

Y = data[y].values
targets = dfs[0]["targets"].values
random = dfs[0]["mean_queries"].values
random_std = dfs[0]["std_queries"].values
nodocking = dfs[-2]["mean_queries"].values
nodocking_std = dfs[-2]["std_queries"].values
docking = dfs[-1]["mean_queries"].values
docking_std = dfs[-1]["std_queries"].values

#%%

plt.plot(random,targets,c="y",marker=".",label="random")
plt.fill_betweenx(y=targets,x1=random-random_std,x2=random+random_std,color="y",alpha=0.2)
plt.plot(nodocking,targets,c="m",marker=".",label="BO (B2)")
plt.fill_betweenx(y=targets,x1=nodocking-nodocking_std,x2=nodocking+nodocking_std,color="m",alpha=0.1)
plt.plot(docking,targets,c="c",marker=".",label="BO + docking (D)")
plt.fill_betweenx(y=targets,x1=docking-docking_std,x2=docking+docking_std,color="c",alpha=0.1)
plt.xticks(np.arange(0,max(random[-1],nodocking[-1])+1.5*ticks,ticks),rotation=65)
plt.yticks(np.arange(min(targets)-0.2,max(targets)+0.2,0.2)+0.2)
plt.xlim(0,max(random[-1],nodocking[-1])+1.5*ticks)
plt.ylim(bottom=min(targets)-0.1,top=max(targets)+0.2)
plt.ylabel("$"+ylabel+"$",fontsize=14)
plt.xlabel("Mean steps",fontsize=14)
plt.legend(loc="lower right",fontsize=14)
plt.grid(True)
plt.title(target, fontsize=16)
plt.tight_layout()
plt.show()

#%%

# Mean best activity plots

best_means = []
best_medians = []
best_stds = []
best_q1s = []
best_q3s = []
for eof,config in zip(eofs,configs):
    folder = os.path.join(results,config)
    if config != "baseline": file = f"{target}{l}_{config.split('_')[0]}_{eof}_ID.csv"
    else: file = f"{target}{l}_{eof}_ID.csv"
    df = pd.read_csv(os.path.join(folder,file))
    all_ids = [df[column].values for column in df.columns]
    all_ids = [ids[~np.isnan(ids)].astype(int).tolist() for ids in all_ids]
    all_Ys = [[Y[i] for i in ids] for ids in all_ids]
    steps = [len(ids) for ids in all_ids]
    max_len = np.max([len(id) for id in all_ids])
    all_best = np.array([[np.max(Y[ids[:i]]) for i in range(1,len(ids)+1)]+[np.max(Y)]*(max_len-len(ids)) for ids in all_ids])
    best_mean = np.mean(all_best,axis=0)
    best_med = np.median(all_best,axis=0)
    best_std = np.std(all_best,axis=0)
    best_q1 = np.quantile(all_best,0.25,axis=0)
    best_q3 = np.quantile(all_best,0.75,axis=0)
    best_means.append(best_mean)
    best_medians.append(best_med)
    best_stds.append(best_std)
    best_q1s.append(best_q1)
    best_q3s.append(best_q3)

#%%

best_random = best_means[0]
best_random_std = best_stds[0]
best_random_q1 = best_q1s[0]
best_random_q3 = best_q3s[0]
best_nodocking = best_means[-2]
best_nodocking_std = best_stds[-2]
best_nodocking_q1 = best_q1s[-2]
best_nodocking_q3 = best_q3s[-2]
best_docking = best_means[-1]
best_docking_std = best_stds[-1]
best_docking_q1 = best_q1s[-1]
best_docking_q3 = best_q3s[-1]

#%%

minval = min(Y)
maxval = max(Y)
# plt.figure(figsize=(4,4.5)) 
plt.plot(np.arange(1,best_random.shape[-1]+1,1),best_random,c="y",label="random")
plt.fill_between(x=np.arange(1,best_random.shape[-1]+1,1),y1=np.clip(best_random-best_random_std,minval,maxval),y2=np.clip(best_random+best_random_std,minval,maxval),color="y",alpha=0.2)
plt.plot(np.arange(1,best_nodocking.shape[-1]+1,1),best_nodocking,c="m",label="BO (B2)")
plt.fill_between(x=np.arange(1,best_nodocking.shape[-1]+1,1),y1=np.clip(best_nodocking-best_nodocking_std,minval,maxval),y2=np.clip(best_nodocking+best_nodocking_std,minval,maxval),color="m",alpha=0.1)
plt.plot(np.arange(1,best_docking.shape[-1]+1,1),best_docking,c="c",label="BO + docking (D)")
plt.fill_between(x=np.arange(1,best_docking.shape[-1]+1,1),y1=np.clip(best_docking-best_docking_std,minval,maxval),y2=np.clip(best_docking+best_docking_std,minval,maxval),color="c",alpha=0.1)
plt.ylabel(f"Mean of best ${ylabel}$", fontsize=14)
if mode == "litpcba": yscale = 2 
else: yscale = 4
plt.xlim(-10,max(200,len(Y)/yscale))
plt.ylim(np.mean(Y),np.max(Y)+0.2)
plt.xlabel("Compounds Sampled", fontsize=14)
plt.legend(loc="lower right", fontsize=14)
plt.grid(True)
plt.title(target, fontsize=16)
plt.tight_layout()
plt.show()

#%%

# TOC-style diagram -- mean of best activity
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
plt.figure(figsize=(5.4,6)) 
minval = min(Y)
maxval = max(Y)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.plot(np.arange(1,best_nodocking.shape[-1]+1,1),best_nodocking,c="m",label=names[-2],marker="o",ms="1.5")
plt.plot(np.arange(1,best_docking.shape[-1]+1,1),best_docking,c="c",label=names[-1],marker="o",ms="1.5")
if mode == "litpcba": yscale = 2 
else: yscale = 4
plt.xlim(-10,best_nodocking.shape[-1]+30)
plt.ylim(np.mean(Y)+1.2,np.max(Y)+0.1)
plt.xlabel("Compounds sampled", fontsize=16)
plt.ylabel(f"${ylabel}$", fontsize=16)
c_patch = mpatches.Patch(color='c', label='ML + Docking')
m_patch = mpatches.Patch(color='m', label='ML')
plt.xticks(rotation=45)
if len(best_docking) < len(best_nodocking): plt.fill_between(np.arange(1,best_nodocking.shape[-1]+1,1),best_nodocking,np.concatenate((best_docking,[np.max(Y)]*(len(best_nodocking)-len(best_docking)))),color="limegreen",alpha=0.7)
else: plt.fill_between(np.arange(1,best_nodocking.shape[-1]+1,1),best_nodocking,best_docking[:len(best_nodocking)],color="limegreen",alpha=0.7)
plt.legend(handles=[m_patch,c_patch], fontsize=16, loc="lower right")
plt.tight_layout()
plt.show()


#%%

# TOC-style diagram -- steps-to-target
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
plt.figure(figsize=(4,5)) 
plt.plot(nodocking,targets,c="m",marker=".",label="ML")
plt.plot(docking,targets,c="c",marker=".",label="ML + docking")
plt.xticks(np.arange(0,nodocking[-1]+1.0*ticks,ticks),rotation=65)
plt.yticks(np.arange(min(targets)-0.2,max(targets)+0.2,0.2)+0.2)
plt.xlim(0,nodocking[-1]+1.0*ticks)
plt.ylim(bottom=min(targets),top=max(targets)+0.2)
plt.ylabel("$"+ylabel+"$",fontsize=14)
plt.xlabel("Mean steps",fontsize=14)
plt.legend(loc="lower right",fontsize=14)
plt.fill_betweenx(targets, nodocking, color="aquamarine",alpha=0.7)
plt.fill_betweenx(targets, docking, color="white")
# plt.fill_between(np.arange(1,nodocking.shape[-1]+1,1),nodocking,np.concatenate((docking,[np.max(Y)]*(len(nodocking)-len(docking)))),color="limegreen",alpha=0.7)
plt.tight_layout()
plt.show()


#%%