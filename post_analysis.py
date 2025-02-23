#%%

import os
import random
import time

import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import pearsonr, gaussian_kde
from scipy.spatial.distance import jaccard, pdist, cdist

from rdkit import Chem
from rdkit.Chem import Draw

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl

import seaborn as sns

from scipy.stats import tukey_hsd, f_oneway, friedmanchisquare, ttest_ind, kstest
from statsmodels.stats.anova import AnovaRM
from statsmodels.sandbox.stats.multicomp import multipletests

seed=42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)

sns.set_theme()
# plt.style.use("seaborn-v0_8")
plt.rcParams.update({'font.size': 18})
mpl.rcParams['figure.dpi'] = 600

from tqdm import tqdm
from rdkit import DataStructs
from rdkit.DataManip.Metric.rdMetricMatrixCalc import GetTanimotoSimMat

#%%

def np_to_bv(fp):
    bitvector = DataStructs.ExplicitBitVect(len(fp))
    for i,v in enumerate(fp): 
        if v: bitvector.SetBit(i)
    return bitvector

def EF(ids,df,y="pKi",hit=9.0):
    rand_hits = len(ids)/len(df) * np.sum(df[y]>=hit)
    hits = np.sum(df[y].iloc[ids]>=hit)
    enrichment = hits/rand_hits
    return enrichment

def TS(ids,df,i=None,l=2048):
    fps = [f"morgan3_{i}" for i in range(l)]
    if not i: i = len(ids)
    mat = GetTanimotoSimMat([np_to_bv(fp) for fp in df[fps].iloc[ids[:i]].values])
    return np.mean(mat)

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
    names = ["random", "similarity search", "morgan3_rdkit2d\n(random)", "morgan3_rdkit2d\n(diverse)", "morgan3_rdkit2d_rdkit3d_delta_docking\n(docking)"]
    pal = {"random": "r", "similarity search": "brown", "morgan3_rdkit2d\n(random)": "orange","morgan3_rdkit2d\n(diverse)": "b", "morgan3_rdkit2d_rdkit3d_delta_docking\n(docking)": "g"}
    ticks = 50
    ylabel="$pK_i$"
elif mode == "litpcba":
    Nreps = 25
    l = ""
    lowlevel = "CNN-Affinity"
    results = "results_litpcba"
    y = "pEC50"
    hit = 4.0 + 1e-3 # > 4.0 instead of >= 4.0
    eofs = ["random_10","tanimoto_morgan3_10","morgan3_rdkit2d_10","morgan3_rdkit2d_10_cpca",f"morgan3_rdkit2d_rdkit3d_docking_10_top_{lowlevel}_P90"]
    configs = ["baseline","baseline",config,config,config]
    names = ["random", "similarity search", "morgan3_rdkit2d\n(random)", "morgan3_rdkit2d\n(diverse)", "morgan3_rdkit2d_rdkit3d_docking\n(docking)"]
    pal = {"random": "r", "similarity search": "brown", "morgan3_rdkit2d\n(random)": "orange","morgan3_rdkit2d\n(diverse)": "b", "morgan3_rdkit2d_rdkit3d_docking\n(docking)": "g"}
    ticks = 200
    ylabel="$pEC_{50}$"

#%%

if mode == "litpcba": data = pd.read_csv(os.path.join("data",f"{target}_data_full.csv"))
else: data = pd.read_csv(os.path.join("data",f"{target}{l}_data_3d_{mode}_pKi.csv"))
avg_tanimoto = TS(data.index,data)

#%%

# Data collection

y_EF = []
y_EF_err = []
y_TS_start = []
y_TS_start_err = []
y_TS_end = []
y_TS_end_err = []
all_steps = []
all_EFs = []
for eof,config in zip(eofs,configs):
    folder = os.path.join(results,config)
    if config != "baseline": file = f"{target}{l}_{config.split('_')[0]}_{eof}_ID.csv"
    else: file = f"{target}{l}_{eof}_ID.csv"
    df = pd.read_csv(os.path.join(folder,file))
    all_ids = [df[column].values for column in df.columns]
    all_ids = [ids[~np.isnan(ids)].tolist() for ids in all_ids]
    steps = [len(ids) for ids in all_ids]
    all_steps += steps
    EFs = [EF(ids,data,y=y,hit=hit) for ids in all_ids]
    all_EFs += EFs
    mean_EF = np.round(np.mean(EFs),3)
    std_EF = np.round(np.std(EFs),3)
    print(f"EF: {mean_EF} +/- {std_EF}")
    if "random" not in eof:
        TS_start = [TS(ids,data,i=10) for ids in tqdm(all_ids)]
        mean_TS_start = np.round(np.mean(TS_start),3)
        std_TS_start = np.round(np.std(TS_start),3)
        TS_end = [TS(ids,data) for ids in tqdm(all_ids)]
        mean_TS_end = np.round(np.mean(TS_end),3)
        std_TS_end = np.round(np.std(TS_end),3)
        y_EF.append(mean_EF)
        y_EF_err.append(std_EF)
        y_TS_start.append(mean_TS_start)
        y_TS_start_err.append(std_TS_start)
        y_TS_end.append(mean_TS_end)
        y_TS_end_err.append(std_TS_end)
        print(f"TS (initial pool): {mean_TS_start} +/- {std_TS_start}")
        print(f"TS (full): {mean_TS_end} +/- {std_TS_end}")

#%%

# Box plots

legend=True
all_names = []
for name in names: all_names += [name]*Nreps
datadict = {"config": all_names, "steps_to_maximum": all_steps, "EF": all_EFs}
# df = pd.DataFrame(datadict)
df = pd.DataFrame({k:pd.Series(v) for k,v in datadict.items()})

means  = [np.mean(df["steps_to_maximum"][df["config"]==name]) for name in names]
print(means)
stds = [np.std(df["steps_to_maximum"][df["config"]==name]) for name in names]
print(stds)
ax = sns.boxplot(x="config", y="steps_to_maximum", data=df, palette=pal, hue="config", legend=legend, linewidth=1.2)
if not legend:
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', labelrotation = 40)
else:
    ax.legend_.set_title(None)
    plt.legend(fancybox=True, framealpha=0.4)
    # plt.setp(ax.get_legend().get_texts(), fontsize='8')
    # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set(xlabel=None,xticklabels=[])
    ax.tick_params(bottom=False)
plt.tight_layout()
plt.title(f"{target}")
plt.show()

means  = [np.mean(df["EF"][df["config"]==name]) for name in names]
print(means)
stds = [np.std(df["EF"][df["config"]==name]) for name in names]
print(stds)
ax = sns.boxplot(x="config", y="EF", data=df, palette=pal, hue="config", legend=legend, linewidth=1.2)
if not legend:
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', labelrotation = 40)
else:
    ax.legend_.set_title(None)
    # plt.legend(fancybox=True, framealpha=0.4)
    # plt.setp(ax.get_legend().get_texts(), fontsize='8')
    # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set(xlabel=None,xticklabels=[])
    ax.tick_params(bottom=False)
plt.tight_layout()
plt.title(f"{target}")
plt.show()

#%%

# Statistical tests

# steps_to_maximum
# df["run"] = [i for i in range(Nreps)]*len(names)
y = "steps_to_maximum"
grouped = [df[y][df["config"]==name].values for name in names]
grouped = [arr[~np.isnan(arr)] for arr in grouped]
print(names)
# Tukey HSD
tukey = tukey_hsd(*grouped[-2:])
print(tukey)
# ANOVA
anova = f_oneway(*grouped[-2:])
print(anova)
# ANOVA repeated measurements
# anovaRM = AnovaRM(data=df[df["config"].isin(names[-2:])], depvar='steps_to_maximum',  subject='run', within=['config']).fit() 
# print(anovaRM)
# Two-sample one-way Welch's t-test
welch = ttest_ind(grouped[-1],grouped[-2],equal_var=False,alternative='less')
print(welch)
# Kolmogorov-Smirnov test
ks = kstest(grouped[-1],grouped[-2])
print(ks)

print()
# Enrichment factor
# df["run"] = [i for i in range(Nreps)]*len(names)
y = "EF"
grouped = [df[y][df["config"]==name].values for name in names]
grouped = [arr[~np.isnan(arr)] for arr in grouped]
print(names)
# Tukey HSD
tukey = tukey_hsd(*grouped[-2:])
print(tukey)
# ANOVA
anova = f_oneway(*grouped[-2:])
print(anova)
# ANOVA repeated measurements
# anovaRM = AnovaRM(data=df[df["config"].isin(names[-2:])], depvar='steps_to_maximum',  subject='run', within=['config']).fit() 
# print(anovaRM)
# Two-sample one-way Welch's t-test
welch = ttest_ind(grouped[-1],grouped[-2],equal_var=False,alternative='greater')
print(welch)
# Kolmogorov-Smirnov test
ks = kstest(grouped[-1],grouped[-2])
print(ks)

#%%

# Similarity plots

plt.rcParams.update({'font.size': 18})
y = [i for ij in zip(y_TS_start, y_TS_end) for i in ij]
yerr = [i for ij in zip(y_TS_start_err, y_TS_end_err) for i in ij]
width = 10
height = 8
fig, ax = plt.subplots(figsize=(width, height))
colors = ['lightcoral','firebrick','sandybrown','saddlebrown','salmon','tomato','lightblue','darkblue','lightgreen','darkgreen',]
labels = [
    "random (initial)",
    "random (end)",
    "similarity search (initial)",
    "similarity search (end)",
    "morgan3_rdkit2d // random (initial)",
    "morgan3_rdkit2d // random (end)",
    "morgan3_rdkit2d // diverse (initial)",
    "morgan3_rdkit2d // diverse (end)",
    "morgan3_rdkit2d_rdkit3d_delta_docking // docking (initial)",
    "morgan3_rdkit2d_rdkit3d_delta_docking // docking (end)",
]
y = y
yerr = yerr
colors = colors[2:]
labels = labels[2:]
x = [i for i in range(len(y))]
lab = [mpatches.Patch(color=c,label=l) for c,l in zip(colors,labels)]
plt.hlines(y=avg_tanimoto, xmin=-0.5, xmax=len(x), colors='red', linestyles='--', lw=2, label='dataset average')
plt.bar(x, y, color=colors, label=labels)
plt.title(f'{target} Tanimoto similarity')
plt.legend(handles=lab, prop={'size': 14})
plt.ylabel('Mean Tanimoto similarity')
plt.xticks(x)
plt.ylim(0,1)
plt.yticks(0.1*np.arange(0, 11, 1)) 
plt.errorbar(x, y, yerr, fmt='.', color='Black', elinewidth=2, capthick=10, errorevery=1, alpha=0.5, ms=4, capsize=2)
ax.set(xlabel=None,xticklabels=[])
ax.tick_params(bottom=False)
