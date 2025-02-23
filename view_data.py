#%%
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pandas as pd
import seaborn as sns
import numpy as np
from IPython.display import display
from scipy.stats import pearsonr
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from rdkit import Chem
from rdkit.Chem.Descriptors import _descList
# from rdkit.Chem.Descriptors3D import descList
descList = ['PMI1','PMI2','PMI3','NPR1','NPR2',
            'RadiusOfGyration','InertialShapeFactor',
            'Eccentricity','Asphericity','SpherocityIndex','PBF']
from rdkit.Chem.rdMolDescriptors import CalcAUTOCORR3D, CalcRDF, CalcMORSE, CalcWHIM, CalcGETAWAY
# from mordred import Calculator, descriptors
# from mordred._base.result import Result
from tqdm import tqdm

params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)
mpl.rcParams['figure.dpi'] = 300

#%%

# making dataframe  
target = "EGFR"
l = 2048
df = pd.read_csv(os.path.join("data",f"{target}-{l}_data_3d_delta_pKi.csv"))  
# output the dataframe 
# display(df)

#%%

rdkit2d_features = [desc[0] for desc in set(_descList)]
maccs_features = [f"maccs_{i}" for i in range(167)]
morgan3_features = [f"morgan3_{i}" for i in range(2048)]
rdkit3d_features = descList
autocorr3d_features = [f"AUTOCORR3D_{i}" for i in range(80)]
rdf_features = [f"RDF_{i}" for i in range(210)]
morse_features = [f"MORSE_{i}" for i in range(224)]
getaway_features = [f"GETAWAY_{i}" for i in range(273)]
whim_features = [f"WHIM_{i}" for i in range(114)]
vina_features = ["score"]+[f"vinaF_{i}" for i in range(49)]
sasa_features = [f"sasaF_{i}" for i in range(30)]
lig_features = [f"ligF_{i}" for i in range(10)]
wat_features = [f"watF_{i}" for i in range(3)]
delta_features = ["score","betaScore","ligCover","LinF9","XGB"]
good_rdkit3d = ["PMI1", "PMI2", "RadiusOfGyration", "PBF"]
good_rdkit2d = ["NumAromaticHeterocycles", "NumAromaticRings", "AvgIpc", "NumAromaticCarbocycles", 
    "Chi0v", "MolLogP", "MolMR", "NumValenceElectrons", "MolWt", "BertzCT", "MinEStateIndex", "MaxAbsPartialCharge", 
    "RingCount", "FractionCSP3", "HallKierAlpha", "NumHeteroatoms", "TPSA", "NumAliphaticHeterocycles", 
    "NumHAcceptors", "NumHDonors", "NumSaturatedRings", "LabuteASA", "fr_benzene", "BCUT2D_CHGLO", "BalabanJ", 
    "Chi1", "fr_C_O", "fr_pyridine", "MaxAbsPartialCharge", "MinAbsPartialCharge", "Chi2v", "fr_aniline", "NumRotatableBonds", 
    "fr_ArN", "qed", "fr_bicyclic", "SPS", "fr_NH0", "NOCount", "fr_halogen", "MinPartialCharge", "fr_phenol", "BCUT2D_LOGPLOW", 
    "Kappa1", "Kappa2", "BCUT2D_CHGHI", "FpDensityMorgan3", "fr_Nhpyrrole", "SlogP_VSA1"]

fsets = {
    "rdkit2d": rdkit2d_features,
    "maccs": maccs_features,
    "rdkit3d": rdkit3d_features,
    "autocorr3d": autocorr3d_features,
    "rdf": rdf_features,
    "morse": morse_features,
    "getaway": getaway_features,
    "whim": whim_features,
    "vina": vina_features,
    "sasa": sasa_features,
    "lig": lig_features,
    "delta": delta_features,
    "good_rdkit2d": good_rdkit2d,
    "good_rdkit3d": good_rdkit3d,
}
#%%

outdir = os.path.join(target,"features")
if not os.path.exists(outdir): os.mkdir(outdir)

# Plot correlation heatmaps of different variables
for (name, varlist) in tqdm(fsets.items()):
    Y = "pKi"
    N = len(varlist)
    for n in range(int(N/20)+1):
        m = n*20
        variables = varlist[m:m+20]+[Y]
        if len(variables) == 1: continue
        df_ = df[variables]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        data = df_.corr()
        cax = ax.matshow(data, cmap='bwr', vmin=-1, vmax=1, interpolation='nearest')
        bands = np.linspace(0, 1, 17, endpoint=True)
        fig.colorbar(cax, ticks=[-1,0,1], shrink=0.8)
        ax.set_xticks(range(len(variables)))
        ax.set_yticks(range(len(variables)))
        ax.set_xticklabels(variables,rotation=90)
        ax.set_yticklabels(variables)
        for (i, j), z in np.ndenumerate(data):
            ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', size=5)
        plt.savefig(os.path.join(outdir,f"{name}_corr_{n}.png"),dpi=300,bbox_inches='tight')
        plt.close()

#%%

# Test docking scores are for correct compound
sns.pairplot(df[['HeavyAtomMolWt','ligF_0']],diag_kind='kde')

#%%

# DOCKSTRING Vina score
x = -df["score"].values
y = df["pKi"].values
xtop = (x[x > np.percentile(x,90)])
ytop = (y[x > np.percentile(x,90)])
R = round(pearsonr(x,y).statistic**2,3)
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
plt.scatter(x,y,c=z,cmap="viridis",alpha=0.8,s=20)
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),c="black")
plt.scatter(xtop,ytop,c="orange",s=20,label="top 10%")
plt.xlim(0,15)
# plt.xlim(0,150)
plt.ylim(0,15)
plt.xlabel("Docking score", fontsize=15)
plt.ylabel("$pK_i$", fontsize=15)
plt.title(f"{target} ($R^2$: {R})", fontsize=15)
plt.legend(fontsize=12)

#%%

# delta_LinF9_XGB Vina score
x = -df["vinaF_0"].values
y = df["pKi"].values
xtop = (x[x > np.percentile(x,90)])
ytop = (y[x > np.percentile(x,90)])
R = round(pearsonr(x,y).statistic**2,3)
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
plt.scatter(x,y,c=z,cmap="viridis",alpha=0.8,s=20)
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),c="black")
plt.scatter(xtop,ytop,c="orange",s=20,label="top 10%")
plt.xlim(0,15)
# plt.xlim(0,150)
plt.ylim(0,15)
plt.xlabel("Docking score", fontsize=15)
plt.ylabel("$pK_i$", fontsize=15)
plt.title(f"{target} ($R^2$: {R})", fontsize=15)
plt.legend(fontsize=12)

#%%

# LinF9 score
x = df["LinF9"].values
y = df["pKi"].values
xtop = (x[x > np.percentile(x,90)])
ytop = (y[x > np.percentile(x,90)])
R = round(pearsonr(x,y).statistic**2,3)
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
plt.scatter(x,y,c=z,cmap="viridis",alpha=0.8,s=20)
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),c="black")
plt.scatter(xtop,ytop,c="orange",s=20,label="top 10%")
plt.xlim(0,15)
# plt.xlim(0,150)
plt.ylim(0,15)
plt.xlabel("Docking score", fontsize=15)
plt.ylabel("$pK_i$", fontsize=15)
plt.title(f"{target} ($R^2$: {R})", fontsize=15)
plt.legend(fontsize=12)

#%%

# delta_LinF9_XGB score
# x = -df["score"].values
x = df["XGB"].values
y = df["pKi"].values
xtop = (x[x > np.percentile(x,90)])
ytop = (y[x > np.percentile(x,90)])
R = round(pearsonr(x,y).statistic**2,3)
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
plt.scatter(x,y,c=z,cmap="viridis",alpha=0.8,s=20)
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),c="black")
plt.scatter(xtop,ytop,c="orange",s=20,label="top 10%")
plt.xlim(0,15)
# plt.xlim(0,150)
plt.ylim(0,15)
plt.xlabel("Docking score", fontsize=15)
plt.ylabel("$pK_i$", fontsize=15)
plt.title(f"{target} ($R^2$: {R})", fontsize=15)
plt.legend(fontsize=12)

#%%

# delta_LinF9_XGB score (actives only)
# x = -df["score"].values
x = df["XGB"][df["active"]==1].values
y = df["pKi"][df["active"]==1].values
xtop = (x[x > np.percentile(x,90)])
ytop = (y[x > np.percentile(x,90)])
R = round(pearsonr(x,y).statistic**2,3)
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
plt.scatter(x,y,c=z,cmap="viridis",alpha=0.8,s=20)
# plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),c="black")
# plt.scatter(xtop,ytop,c="orange",s=20,label="top 10%")
plt.xlim(0,15)
# plt.xlim(0,150)
plt.ylim(0,15)
plt.xlabel("Docking score", fontsize=15)
plt.ylabel("$pK_i$", fontsize=15)
plt.title(f"{target} ($R^2$: {R})", fontsize=15)
plt.legend(fontsize=12)
