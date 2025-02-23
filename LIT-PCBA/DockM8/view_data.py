#%%

import os
import random
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pandas as pd
import seaborn as sns
import numpy as np
from IPython.display import display
from scipy.stats import pearsonr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
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

seed=42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)

mpl.rcParams['figure.dpi'] = 600

#%%

# making dataframe  
protein = "ESR1ago"
target = "pEC50"
df = pd.read_csv(os.path.join("..","..","data",f"{protein}_data_full.csv"))  
# output the dataframe 
# display(df)

#%%

rdkit2d_features = [desc[0] for desc in set(_descList)]
rdkit2d_features = [x for x in rdkit2d_features if x not in ['SPS','AvgIpc']]
maccs_features = [f"maccs_{i}" for i in range(167)]
morgan3_features = [f"morgan3_{i}" for i in range(2048)]
rdkit3d_features = descList
autocorr3d_features = [f"AUTOCORR3D_{i}" for i in range(80)]
rdf_features = [f"RDF_{i}" for i in range(210)]
morse_features = [f"MORSE_{i}" for i in range(224)]
getaway_features = [f"GETAWAY_{i}" for i in range(273)]
whim_features = [f"WHIM_{i}" for i in range(114)]
# mordred3d_features = list(Calculator(descriptors, ignore_3D=False)(
#     Chem.MolFromSmiles("c1ccccc1")).asdict().keys())[-213:]
vina_features = ["score"]+[f"vinaF_{i}" for i in range(49)]
sasa_features = [f"sasaF_{i}" for i in range(30)]
lig_features = [f"ligF_{i}" for i in range(10)]
wat_features = [f"watF_{i}" for i in range(3)]
good_rdkit3d = ["PMI1", "PMI2", "RadiusOfGyration", "PBF"]
good_rdkit2d = ['MaxEStateIndex','MinEStateIndex','qed','MolWt','MaxPartialCharge','MinPartialCharge',
    'FpDensityMorgan3','BCUT2D_CHGHI','BCUT2D_CHGLO','BCUT2D_LOGPHI','BCUT2D_LOGPLOW','BalabanJ',
    'BertzCT','Chi0n','Chi0v','Chi1n','Chi1v','Chi2n','Chi2v','Chi3n','Chi3v','Chi4n','Chi4v',
    'HallKierAlpha','Ipc','Kappa1','Kappa2','Kappa3','LabuteASA',
    'SlogP_VSA1','SlogP_VSA10','SlogP_VSA11','SlogP_VSA12','SlogP_VSA2','SlogP_VSA3','SlogP_VSA4',
    'SlogP_VSA5','SlogP_VSA6','SlogP_VSA7','SlogP_VSA8','SlogP_VSA9',
    'TPSA','FractionCSP3','NHOHCount','NOCount','NumAliphaticCarbocycles','NumAliphaticHeterocycles',
    'NumAliphaticRings','NumAromaticCarbocycles','NumAromaticHeterocycles','NumAromaticRings',
    'NumHAcceptors','NumHDonors','NumHeteroatoms','NumRotatableBonds','NumSaturatedCarbocycles',
    'NumSaturatedHeterocycles','NumSaturatedRings','RingCount','MolLogP','MolMR',
    'fr_benzene','fr_phenol','fr_aniline','fr_ArN','fr_pyridine','fr_Nhpyrrole','fr_bicyclic',
    'fr_NH0','fr_C_O','fr_halogen']

fsets = {
    "rdkit2d": rdkit2d_features,
    "maccs": maccs_features,
    "rdkit3d": rdkit3d_features,
    "autocorr3d": autocorr3d_features,
    "rdf": rdf_features,
    "morse": morse_features,
    "getaway": getaway_features,
    "good_rdkit2d": good_rdkit2d,
    "good_rdkit3d": good_rdkit3d+["CNN-Affinity"],
}
#%%

for (name, varlist) in tqdm(fsets.items()):
    N = len(varlist)
    for n in range(int(N/20)+1):
        m = n*20
        variables = varlist[m:m+20]+[target]
        if len(variables) == 1: continue
        df_ = df[variables][df["active"]==1]
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
        if not os.path.exists(os.path.join(protein,"features")): os.makedirs(os.path.join(protein,"features"))
        plt.savefig(os.path.join(protein,"features",f"{name}_corr_{n}.png"),dpi=300,bbox_inches='tight')
        plt.close()

#%%

# GNINA CNN-Affinity docking score
x = df["CNN-Affinity"].values
y = df["pEC50"].values
xtop = x[x >= np.percentile(x,90)]
ytop = y[x >= np.percentile(x,90)]
xact = x[y > 4]
yact = y[y > 4]
R = round(pearsonr(xact,yact).statistic**2,3)
Rall = round(pearsonr(x,y).statistic**2,3)
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
plt.ylabel("$pEC_{50}$", fontsize=15)
plt.title(f"{protein}, $R^2$ (actives): {R}, $R^2$ (all): {Rall}", fontsize=15)
plt.legend(fontsize=12)

#%%

a = 0
measure = "activity"
scores = ["RFScoreVS","CHEMPLP","CNN-Affinity","ConvexPLR","AD4","CNN-Score","KORP-PL","RTMScore","Vinardo","LinF9","GNINA-Affinity"]
colours = cm.rainbow(np.linspace(0, 1, len(scores)))
for c,score in zip(colours,scores):
    plt.scatter(df[score],df[measure],c=c)
    x = df[score][df["activity"]>0]
    y = df[measure][df["activity"]>0]
    R = round(pearsonr(x,y).statistic,2)
    if R > 0: top10 = np.where(df[score].values>=np.percentile(df[score].values,90))[0]
    else: top10 = np.where(df[score].values<=np.percentile(df[score].values,10))[0]
    plt.scatter(df[score].loc[top10],df[measure].loc[top10],c="black")
    plt.title(f"{score} (R: {R})")
    plt.show()
    # plt.scatter(x,y,c=c)
    # R = round(pearsonr(x,y).statistic,2)
    # plt.title(f"{score} (R: {R})")
    # plt.show()

#%%