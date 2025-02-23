#%%
import os
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr 

mpl.rcParams['figure.dpi'] = 600

#%%

target = "EGFR"
mode = "delta"
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
    names = ["random", "similarity search", "morgan3_rdkit2d // random", "morgan3_rdkit2d // diverse", "morgan3_rdkit2d_rdkit3d_delta_docking // docking"]
    pal = {"random": "r", "similarity search": "brown", "morgan3_rdkit2d // random": "orange","morgan3_rdkit2d // diverse": "b", "morgan3_rdkit2d_rdkit3d_delta_docking // docking": "g"}
    ticks = 50
    ylabel="$pK_i$"
elif mode == "litpcba":
    Nreps = 25
    l = ""
    lowlevel = "CNN-Affinity"
    results = "results_litpcba"
    y = "pEC50"
    hit = 7.0
    eofs = ["random_10","tanimoto_morgan3_10","morgan3_rdkit2d_10","morgan3_rdkit2d_10_cpca",f"morgan3_rdkit2d_rdkit3d_docking_10_top_{lowlevel}_P90"]
    configs = ["baseline","baseline",config,config,config]
    names = ["random", "similarity search", "morgan3_rdkit2d // random", "morgan3_rdkit2d // diverse", "morgan3_rdkit2d_rdkit3d_docking // docking"]
    pal = {"random": "r", "similarity search": "brown", "morgan3_rdkit2d // random": "orange","morgan3_rdkit2d // diverse": "b", "morgan3_rdkit2d_rdkit3d_docking // docking": "g"}
    ticks = 200
    ylabel="$pEC_{50}$"

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

# try: plt.style.use("seaborn-v0_8")
# except: plt.style.use("seaborn")
Y = data[y].values

#%%

targets = dfs[0]["targets"].values
random = dfs[0]["mean_queries"].values
random_std = dfs[0]["std_queries"].values
nodocking = dfs[-2]["mean_queries"].values
nodocking_std = dfs[-2]["std_queries"].values
docking = dfs[-1]["mean_queries"].values
docking_std = dfs[-1]["std_queries"].values
# similarity_random = dfs[1]["mean_queries"].values

#%%

# plt.style.use("default")
# plt.figure(figsize=(5,8)) 
plt.plot(random,targets,c="y",marker=".",label="random")
plt.fill_betweenx(y=targets,x1=random-random_std,x2=random+random_std,color="y",alpha=0.2)
# plt.plot(similarity_random,targets,c="coral",marker=".",label="similarity // random")
# plt.plot(similarity_diverse,targets,c="peru",marker=".",label="similarity // diverse")
# plt.plot(similarity_docking,targets,c="orange",marker=".",label="similarity // docking")
plt.plot(nodocking,targets,c="m",marker=".",label=names[-2])
plt.fill_betweenx(y=targets,x1=nodocking-nodocking_std,x2=nodocking+nodocking_std,color="m",alpha=0.1)
plt.plot(docking,targets,c="c",marker=".",label=names[-1])
plt.fill_betweenx(y=targets,x1=docking-docking_std,x2=docking+docking_std,color="c",alpha=0.1)
# plt.xticks(np.arange(0,max((max(random),max(similarity_random),max(similarity_diverse)))+4*ticks,ticks),rotation=45)
plt.xticks(np.arange(0,max(random[-1],nodocking[-1])+1.5*ticks,ticks),rotation=65)
plt.yticks(np.arange(min(targets)-0.2,max(targets)+0.2,0.2)+0.2)
plt.xlim(0,max(random[-1],nodocking[-1])+1.5*ticks)
plt.ylim(bottom=min(targets)-0.1,top=max(targets)+0.2)
plt.ylabel(ylabel)
plt.xlabel("Mean steps")
plt.legend(loc="lower right")
plt.grid(True)
plt.title(f"{target} {' '.join(config.split('_'))}")
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
    df = os.path.join(folder,file)
    if mode == "litpcba": data = pd.read_csv(f"{target}_data_full.csv")
    else: data = pd.read_csv(f"{target}{l}_data_3d_{mode}_pKi.csv")
    # avg_tanimoto = TS(data.index,data)
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

# best_random = best_medians[0]
best_random = best_means[0]
best_random_std = best_stds[0]
best_random_q1 = best_q1s[0]
best_random_q3 = best_q3s[0]
# best_nodocking = best_medians[-2]
best_nodocking = best_means[-2]
best_nodocking_std = best_stds[-2]
best_nodocking_q1 = best_q1s[-2]
best_nodocking_q3 = best_q3s[-2]
# best_docking = best_medians[-1]
best_docking = best_means[-1]
best_docking_std = best_stds[-1]
best_docking_q1 = best_q1s[-1]
best_docking_q3 = best_q3s[-1]

#%%

# plt.style.use("default")
# plt.figure(figsize=(5,8)) 
minval = min(Y)
maxval = max(Y)
plt.plot(np.arange(1,best_random.shape[-1]+1,1),best_random,c="y",label=names[0])
plt.fill_between(x=np.arange(1,best_random.shape[-1]+1,1),y1=np.clip(best_random-best_random_std,minval,maxval),y2=np.clip(best_random+best_random_std,minval,maxval),color="y",alpha=0.2)
# plt.fill_between(x=np.arange(1,best_random.shape[-1]+1,1),y1=(best_random_q1),y2=(best_random_q3),color="y",alpha=0.2)
plt.plot(np.arange(1,best_nodocking.shape[-1]+1,1),best_nodocking,c="m",label=names[-2])
plt.fill_between(x=np.arange(1,best_nodocking.shape[-1]+1,1),y1=np.clip(best_nodocking-best_nodocking_std,minval,maxval),y2=np.clip(best_nodocking+best_nodocking_std,minval,maxval),color="m",alpha=0.1)
# plt.fill_between(x=np.arange(1,best_nodocking.shape[-1]+1,1),y1=(best_nodocking_q1),y2=(best_nodocking_q3),color="m",alpha=0.1)
plt.plot(np.arange(1,best_docking.shape[-1]+1,1),best_docking,c="c",label=names[-1])
plt.fill_between(x=np.arange(1,best_docking.shape[-1]+1,1),y1=np.clip(best_docking-best_docking_std,minval,maxval),y2=np.clip(best_docking+best_docking_std,minval,maxval),color="c",alpha=0.1)
# plt.fill_between(x=np.arange(1,best_docking.shape[-1]+1,1),y1=(best_docking_q1),y2=(best_docking_q3),color="c",alpha=0.1)
plt.ylabel(f"Mean of best {ylabel}")
if mode == "litpcba": yscale = 2 
else: yscale = 4
plt.xlim(-10,max(200,len(Y)/yscale))
plt.ylim(np.mean(Y),np.max(Y)+0.5)
plt.xlabel("Steps")
plt.legend(loc="lower right")
plt.grid(True)
plt.title(f"{target} {' '.join(config.split('_'))}")
plt.tight_layout()
plt.show()

#%%

# TOC-style diagram
# plt.style.use("default")
# plt.figure(figsize=(5,8)) 
plt.figure(figsize=(4.2,4))
minval = min(Y)
maxval = max(Y)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.plot(np.arange(1,best_random.shape[-1]+1,1),best_random,c="y",label=names[0],marker="o",ms="1.5")
# plt.fill_between(x=np.arange(1,best_random.shape[-1]+1,1),y1=np.clip(best_random-best_random_std,minval,maxval),y2=np.clip(best_random+best_random_std,minval,maxval),color="y",alpha=0.2)
# plt.fill_between(x=np.arange(1,best_random.shape[-1]+1,1),y1=(best_random_q1),y2=(best_random_q3),color="y",alpha=0.2)
plt.plot(np.arange(1,best_nodocking.shape[-1]+1,1),best_nodocking,c="m",label=names[-2],marker="o",ms="1.5")
# plt.fill_between(x=np.arange(1,best_nodocking.shape[-1]+1,1),y1=np.clip(best_nodocking-best_nodocking_std,minval,maxval),y2=np.clip(best_nodocking+best_nodocking_std,minval,maxval),color="m",alpha=0.1)
# plt.fill_between(x=np.arange(1,best_nodocking.shape[-1]+1,1),y1=(best_nodocking_q1),y2=(best_nodocking_q3),color="m",alpha=0.1)
plt.plot(np.arange(1,best_docking.shape[-1]+1,1),best_docking,c="c",label=names[-1],marker="o",ms="1.5")
plt.fill_between(x=np.arange(1,best_docking.shape[-1]+1,1),y1=np.clip(best_docking-best_docking_std,minval,maxval),y2=np.clip(best_docking+best_docking_std,minval,maxval),color="c",alpha=0.1)
# plt.fill_between(x=np.arange(1,best_docking.shape[-1]+1,1),y1=(best_docking_q1),y2=(best_docking_q3),color="c",alpha=0.1)
# plt.ylabel(f"Mean of best {ylabel}")
if mode == "litpcba": yscale = 2 
else: yscale = 4
plt.xlim(-10,400)
plt.ylim(9,np.max(Y)+0.2)
# plt.xlabel("Steps")
# plt.legend(loc="lower right")
plt.grid(True)
# plt.title(f"{target} {' '.join(config.split('_'))}")
plt.tight_layout()
plt.show()
