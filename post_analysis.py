#%%

import os
import random
import time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

from scipy.stats import tukey_hsd, f_oneway, friedmanchisquare, ttest_ind, kstest, wilcoxon
from statsmodels.stats.anova import AnovaRM
from statsmodels.sandbox.stats.multicomp import multipletests

from tqdm import tqdm
from rdkit import DataStructs
from rdkit.DataManip.Metric.rdMetricMatrixCalc import GetTanimotoSimMat

seed=42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
np.set_printoptions(legacy='1.25')

sns.set_theme()
# plt.style.use("seaborn-v0_8")
plt.rcParams.update({'font.size': 18})
mpl.rcParams['figure.dpi'] = 300

#%%

def np_to_bv(fp):
    bitvector = DataStructs.ExplicitBitVect(len(fp))
    for i,v in enumerate(fp): 
        if v: bitvector.SetBit(i)
    return bitvector

def EF(ids,df,y="pKi",hit=9.0,percentile=None):
    if percentile: hit = np.percentile(df[y],percentile)
    rand_hits = len(ids)/len(df) * np.sum(df[y]>=hit)
    hits = np.sum(df[y].iloc[ids]>=hit)
    enrichment = hits/rand_hits
    return enrichment

def TS(ids,df,i=None,l=2048):
    fps = [f"morgan3_{i}" for i in range(l)]
    if not i: i = len(ids)
    mat = GetTanimotoSimMat([np_to_bv(fp) for fp in df[fps].iloc[ids[:i]].values])
    return np.mean(mat)

def round_sf(value, sig_figs):
    if value == 0: return 0
    else: return round(value, sig_figs - int(np.floor(np.log10(abs(value))))-1)

#%%

target = "EGFR"

CHEMBL = ["EGFR","JAK2","LCK","MAOB","NOS1","ACHE","PARP1","PTGS2","PDE5A","ESR1","NR3C1","AR","ADRB2","F10"]
LITPCBA = ["ESR1ago","ESR1ant","PPARG","TP53"]
BAL = ["DRD2","USP7","Mpro","TYK2"]
if target in CHEMBL: mode = "delta"
elif target in LITPCBA: mode = "litpcba"
elif target in BAL: mode = "bal"
config = "BRR_greedy"

if mode == "delta":
    Nreps = 25
    l = "-2048"
    lowlevel = "XGB"
    results = "results"
    y = "pKi"
    hit = 9.0
    percentile = 99
    eofs = ["random_10","tanimoto_morgan3_10","morgan3_rdkit2d_10","morgan3_rdkit2d_10_cpca",f"morgan3_rdkit2d_rdkit3d_delta_docking_10_top_{lowlevel}_P90"]
    configs = ["baseline","baseline",config,config,config]
    names = ["random", "similarity", "BO (B1)", "BO (B2)", "BO + docking (D)"]
    pal = {"random": "r", "similarity": "brown", "BO (B1)": "orange", "BO (B2)": "b", "BO + docking (D)": "g"}
    ticks = 50
    ylabel="\\text{p}K_i"
    data = pd.read_csv(os.path.join("data",f"{target}{l}_data_3d_{mode}_{y}.csv"))
    mval = np.max(data[y])
elif mode == "litpcba":
    Nreps = 25
    l = ""
    lowlevel = "CNN-Affinity"
    results = "results_litpcba"
    y = "pEC50"
    hit = 4.0 + 1e-3
    percentile = None
    eofs = ["random_10","tanimoto_morgan3_10","morgan3_rdkit2d_10","morgan3_rdkit2d_10_cpca",f"morgan3_rdkit2d_rdkit3d_docking_10_top_{lowlevel}_P90"]
    configs = ["baseline","baseline",config,config,config]
    names = ["random", "similarity", "BO (B1)", "BO (B2)", "BO + docking (D)"]
    pal = {"random": "r", "similarity": "brown", "BO (B1)": "orange","BO (B2)": "b", "BO + docking (D)": "g"}
    ticks = 200
    ylabel="\\text{p}EC_{50}"
    data = pd.read_csv(os.path.join("data",f"{target}_data_full.csv"))
    mval = np.max(data[y])
elif mode == "bal":
    Nreps = 25
    l = ""
    lowlevel = "XGB"
    if target in  ["USP7","Mpro"]: y, ylabel = "pIC50", "\\text{pIC}_{50}"
    else: y, ylabel = "pKi", "\\text{p}K_i"
    results = "results_bal"
    hit = 9.0
    percentile = 99
    eofs = ["random_10","tanimoto_morgan3_10","morgan3_rdkit2d_10","morgan3_rdkit2d_10_cpca",f"morgan3_rdkit2d_rdkit3d_delta_docking_10_top_{lowlevel}_P90"]
    configs = ["baseline","baseline",config,config,config]
    names = ["random", "similarity", "BO (B1)", "BO (B2)", "BO + docking (D)"]
    pal = {"random": "r", "similarity": "brown", "BO (B1)": "orange","BO (B2)": "b", "BO + docking (D)": "g"}
    ticks = 100
    data = pd.read_csv(os.path.join("data",f"{target}_data_3d_delta_{y}.csv"))
    mval = np.max(data[y])

#%%

# Data collection

# NEW: hit = P99 (99th percentile of activities) i.e. EF_1%
if percentile: hit = np.percentile(data[y].values,percentile)

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
    # steps = [len(ids) for ids in all_ids]
    steps = []
    for ids in all_ids:
        Ytrain = data[y].iloc[ids].values
        is_greater = Ytrain.flatten() >= mval
        if not np.any(is_greater): stps, found = len(Ytrain), 0
        else: stps, found = np.argmax(is_greater)+1, 1
        steps.append(stps)
    all_steps += steps
    EFs = [EF(ids,data,y=y,hit=hit) for ids in all_ids]
    all_EFs += EFs
    mean_EF = np.round(np.mean(EFs),3)
    std_EF = np.round(np.std(EFs),3)
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
print([round_sf(m,3) for m in means])
stds = [np.std(df["steps_to_maximum"][df["config"]==name]) for name in names]
# print([round_sf(s,3) for s in stds])
CIs = [1.96*s/np.sqrt(Nreps) for s in stds]
print([round_sf(c,3) for c in CIs])
pchange = 100*((means[-1]-means[-2])/means[-2])
print(round_sf(pchange,3))
std_pchange = 100*np.sqrt((stds[-1]*means[-2])**2 + (stds[-2]*means[-1])**2)/(means[-2]**2)
# print(round_sf(std_pchange,3))
CI_pchange = 1.96*std_pchange/np.sqrt(Nreps)
print(round_sf(CI_pchange,3))

ax = sns.boxplot(x="config", y="steps_to_maximum", data=df, palette=pal, hue="config", legend=legend, linewidth=1.2)
plt.ylabel("Steps to maximum")
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
print([round_sf(m,3) for m in means])
stds = [np.std(df["EF"][df["config"]==name]) for name in names]
# print([round_sf(s,3) for s in stds])
CIs = [1.96*s/np.sqrt(Nreps) for s in stds]
print([round_sf(c,3) for c in CIs])
pchange = 100*((means[-1]-means[-2])/means[-2])
print(round_sf(pchange,3))
std_pchange = 100*np.sqrt((stds[-1]*means[-2])**2 + (stds[-2]*means[-1])**2)/(means[-2]**2)
# print(round_sf(std_pchange,3))
CI_pchange = 1.96*std_pchange/np.sqrt(Nreps)
print(round_sf(CI_pchange,3))

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
anova_df = df[df["config"].isin(names[-2:])].reset_index(drop=True)
anova_df["run"] = [i for i in range(Nreps)]+[i for i in range(Nreps)]
anova_df["config"] = [0]*Nreps+[1]*Nreps

# steps_to_maximum
y = "steps_to_maximum"
grouped = [df[y][df["config"]==name].values for name in names]
grouped = [arr[~np.isnan(arr)] for arr in grouped]
print(names)
# Tukey HSD
tukey = tukey_hsd(*grouped[-2:])
print(tukey)
# ANOVA repeated measurements
anovaRM = AnovaRM(data=anova_df, depvar='steps_to_maximum', subject='run', within=['config']).fit() 
print(anovaRM)
# Two-sample one-way Welch's t-test
welch = ttest_ind(grouped[-1],grouped[-2],equal_var=False,alternative='less')
print(welch)
# Paired Wilcoxon signed rank test
wx = wilcoxon(grouped[-1],grouped[-2],alternative='less')
print(wx)

print()
# Enrichment factor
y = "EF"
grouped = [df[y][df["config"]==name].values for name in names]
grouped = [arr[~np.isnan(arr)] for arr in grouped]
print(names)
# Tukey HSD
tukey = tukey_hsd(*grouped[-2:])
print(tukey)
# ANOVA repeated measurements
anovaRM = AnovaRM(data=anova_df, depvar='EF', subject='run', within=['config']).fit() 
print(anovaRM)
# Two-sample one-way Welch's t-test
welch = ttest_ind(grouped[-1],grouped[-2],equal_var=False,alternative='greater')
print(welch)
# Paired Wilcoxon signed rank test
wx = wilcoxon(grouped[-1],grouped[-2],alternative='greater')
print(wx)

#%%

# Similarity plots

avg_tanimoto = TS(data.index,data)
plt.rcParams.update({'font.size': 18})
y = [i for ij in zip(y_TS_start, y_TS_end) for i in ij]
yerr = [i for ij in zip(y_TS_start_err, y_TS_end_err) for i in ij]
width = 10
height = 8
fig, ax = plt.subplots(figsize=(width, height))
colors = ['red','darkred','lightcoral','firebrick','gold','goldenrod','cornflowerblue','darkblue','limegreen','darkgreen',]
labels = [
    "random (initial)",
    "random (end)",
    "similarity (initial)",
    "similarity (end)",
    "B1 (initial)",
    "B1 (end)",
    "B2 (initial)",
    "B2 (end)",
    "D (initial)",
    "D (end)",
]
y = y
yerr = yerr
colors = colors[2:]
labels = labels[2:]
x = [i for i in range(len(y))]
lab = [mpatches.Patch(color=c,label=l) for c,l in zip(colors,labels)]
plt.hlines(y=avg_tanimoto, xmin=-0.5, xmax=len(x), colors='red', linestyles='--', lw=2, label='dataset average')
plt.bar(x, y, color=colors, label=labels)
plt.title(f'{target} Tanimoto similarity', fontsize=20)
plt.legend(handles=lab, prop={'size': 16})
plt.ylabel('Mean Tanimoto similarity', fontsize=18)
plt.xticks(x)
plt.ylim(0,1)
plt.yticks(0.1*np.arange(0, 11, 1), fontsize=16) 
plt.errorbar(x, y, yerr, fmt='.', color='Black', elinewidth=2, capthick=10, errorevery=1, alpha=0.5, ms=4, capsize=2)
ax.set(xlabel=None,xticklabels=[])
ax.tick_params(bottom=False)

#%%


# # Box plots (only B2 and D)

# df = df[df["config"].isin(['BO (B2)', 'BO + docking (D)'])]

# ax = sns.boxplot(x="config", y="steps_to_maximum", data=df, palette=pal, hue="config", linewidth=1.2, boxprops=dict(alpha=.3))
# sns.stripplot(data=df, x="config", y="steps_to_maximum", palette=pal, hue="config", dodge=True, ax=ax)
# plt.show()

# ax = sns.boxplot(x="config", y="EF", data=df, palette=pal, hue="config", linewidth=1.2, boxprops=dict(alpha=.3))
# sns.stripplot(data=df, x="config", y="EF", palette=pal, hue="config", dodge=True, ax=ax)
# plt.show()

#%%