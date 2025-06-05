#%%

import os
import random
import time
import inspect

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ARDRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR, LinearSVR
from xgboost import XGBRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, ExpSineSquared, Matern, WhiteKernel
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import LearningCurveDisplay
from sklearn.metrics import mean_absolute_error

from scipy.stats import pearsonr, mode, rankdata, percentileofscore, zscore, gaussian_kde
from scipy.spatial.distance import cdist

from rdkit import Chem
from rdkit import RDLogger
from rdkit.ML.Cluster import Butina
from rdkit.DataStructs import ExplicitBitVect
from rdkit.DataManip.Metric import GetTanimotoDistMat
from rdkit.Chem import AllChem, Draw, rdFMCS
from rdkit.Chem import MolFromSmiles, MolToSmiles, MolFromSmarts, MolToSmarts
from rdkit.Chem.MolStandardize.rdMolStandardize import Cleanup
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Descriptors import _descList

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS']='ignore'

seed=42
random.seed(seed)
os.environ["PYTHONHASHSEED"]=str(seed)
np.random.seed(seed)
rng = np.random.RandomState(seed)

#%%

def fit_model(model,Xtrain,Ytrain,parameter_ranges):
    """
    Fit SK-Learn model by 5-fold cross validation hyperparameter tuning
    """
    grid_search = GridSearchCV(
        model,
        parameter_ranges,
        scoring="neg_mean_absolute_error",
        cv=5,
        refit=True,
        n_jobs=5, #-1
        verbose=0
    )
    grid_search.fit(Xtrain, Ytrain)
    return grid_search.best_estimator_

def inference(model,X,sd=False):
    """
    Return Ypred, Upred if Upred is avalilable, else return Ypred, 0
    Ypred :: model predictions on X
    Upred :: model uncertainty (standard deviation) on X
    """
    if sd and "return_std" in inspect.getfullargspec(model.predict)[0]:
        Ypred, Upred =  model.predict(X,return_std=True)
    else: Ypred, Upred = model.predict(X), np.zeros_like(X)
    Ypred = Ypred.reshape(-1,1)
    Upred = Upred.reshape(-1,1)
    return Ypred, Upred

def fit_predict(model,parameter_ranges,Xtrain,Ytrain,Xtest,sd=False):
    """
    Fit model to data (with Ytrain scaled to mean 0 and variance 1)
    Run inferences of trained model on test data Xtest
    Return Ypred, Upred if Upred is avalilable, else return Ypred, 0
    Ypred :: model predictions on Xtest
    Upred :: model uncertainty (standard deviation) on Xtest
    """
    Ytrain = StandardScaler().fit_transform(Ytrain)
    trained_model = fit_model(model,Xtrain,Ytrain,parameter_ranges)
    if sd and "return_std" in inspect.getfullargspec(model.predict)[0]:
        Ypred, Upred =  trained_model.predict(Xtest,return_std=True)
    else: Ypred, Upred = trained_model.predict(Xtest), np.zeros_like(Xtest)
    Ypred = Ypred.reshape(-1,1)
    Upred = Upred.reshape(-1,1)
    return trained_model, Ypred, Upred

def maximise(Ypred,Upred,Ytrain,test_idx):
    """
    Predicted mean acquisiton function
    a(x) = mu(X)
    return idx that maximises a(x)
    """
    a = Ypred
    idx = np.argmax(a)
    best_idx = test_idx[idx]
    return idx

def UCB(Ypred,Upred,Ytrain,test_idx,l=1):
    """
    Upper confidence bound acquisition function
    a(x) = mu + lambda * sigma(x)
    return idx that maximises a(x)
    """
    a = Ypred + l * Upred
    idx = np.argmax(a)
    best_idx = test_idx[idx]
    return idx

def PI(Ypred,Upred,Ytrain,test_idx):
    """
    Predicted Improvement acquisition function
    a(x) = max[f(x) - f(x*)]
    return idx that maximises a(x)
    """
    Ytrain = StandardScaler().fit_transform(Ytrain.reshape(-1,1))
    Ybest = np.max(Ytrain)
    a = sp.stats.norm.cdf((Ypred - Ybest) / Upred)
    idx = np.argmax(a)
    best_idx = test_idx[idx]
    return idx

def EI(Ypred,Upred,Ytrain,test_idx):
    """
    Expected Improvement acquisition function
    a(x) = <max[f(x) - f(x*)]>
    return idx that maximises a(x)
    """
    Ytrain = StandardScaler().fit_transform(Ytrain.reshape(-1,1))
    Ybest = np.max(Ytrain)
    a = (Ypred - Ybest) * sp.stats.norm.cdf((Ypred - Ybest) / Upred) + \
       Upred * sp.stats.norm.pdf((Ypred - Ybest) / Upred)
    idx = np.argmax(a)
    best_idx = test_idx[idx]
    return idx

def random_sampling(Ytrain,Xtrain,Xtest,test_idx):
    """
    Random acquisition function
    return idx selected by uniform random sampling
    """
    return np.random.choice(len(test_idx))

def tanimoto(Ytrain,Xtrain,Xtest,test_idx):
    """
    Similarity based acquisition function
    return idx of nearest neighbour to Xbest
    using Tanimoto (Jaccard) similarity (must use with binary features only)
    """
    Xbest = Xtrain[np.argmax(Ytrain)]
    distances = cdist([Xbest],Xtest,"jaccard")
    idx = np.argmin(distances)
    return idx

def cycle_limit(cycle,cycles,Ytrain,target):
    """
    Stop optimization after hitting a cycle limit
    """
    if cycle >= cycles: return False
    else: return True

def target_limit(cycle,cycles,Ytrain,target):
    """
    Stop optimization after reaching a target threshold
    """
    if np.max(Ytrain) >= target: return False
    else: return True

def preprocess(data,descriptors,train_idx,test_idx,scaler=StandardScaler(),pca=False):
    """
    Preprocessing training and testing data without data leakage
    data :: dataframe containing training and testing features
    descriptors :: list of features corresponding to columns in data
    train_idx :: dataframe row indices for training examples
    test_idx :: dataframe row indices for testing examples
    """
    Xtrain = data[descriptors].loc[train_idx]
    Xtest = data[descriptors].loc[test_idx]
    zero_columns = Xtrain.columns[(Xtrain == 0).all()].values.tolist()
    Xtrain.drop(zero_columns,axis=1,inplace=True)
    Xtest.drop(zero_columns,axis=1,inplace=True)
    descriptors = Xtrain.columns
    fp = [desc for desc in descriptors if data[desc].isin([0,1]).all()]
    non_fp = [desc for desc in descriptors if desc not in fp]
    corr_matrix = Xtrain[non_fp].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    Xtrain.drop(to_drop,axis=1,inplace=True)
    Xtest.drop(to_drop,axis=1,inplace=True)
    descriptors = Xtrain.columns
    for desc in non_fp:
        s = scaler
        Xtrain[desc] = s.fit_transform(Xtrain[desc].values.reshape(-1,1))
        Xtest[desc] = s.transform(Xtest[desc].values.reshape(-1,1))
    if pca:
        pca = PCA(n_components=0.95)
        PCAtrain = pd.DataFrame(pca.fit_transform(Xtrain[non_fp]))
        PCAtest = pd.DataFrame(pca.transform(Xtest[non_fp]))
        PCAtrain[fp] = Xtrain[fp]
        PCAtest[fp] = Xtest[fp]
        Xtrain = PCAtrain
        Xtest = PCAtest
    return Xtrain.values.tolist(), Xtest.values.tolist()


def optimization(data,descriptors,label,
                    oracle_fn,acquisition_fn,termination_fn,
                    n,N,target,
                    model_class,parameter_ranges,
                    initial="random",lowlevel="",
                    predict=True,sd=False):
    """
    Main optimization loop
    data :: Pandas Dataframe object
    descriptors :: List of strings specifying descriptors (X value) column name
    label :: List of a string specifying label (Y value) column name
    oracle_fn :: Function for parsing the label (Y) for a given data point (idx)
    acquisition_fn :: Function for selecting a given data point (idx) from a pool
    termination_fn :: Function for determining when to stop optimization cycle
    n :: Number of training data points in initial pool
    N :: Maximum number of optimization loops
    target :: Label value target threshold
    model_class :: Machine Learning model
    parameter_ranges :: Hyperparameter ranges for tuning
    initial :: Method for choosing initial pool
    lowlevel :: Feature acting as a low-level predictor of the target
    """

    # Initial sample
    test_idx = list(data.index.values)
    size = len(test_idx)
    if size <= n:
        # Do not attempt AL for datasets smaller than n
        sample_size = size
        cycles = 0
        train_idx = test_idx
        active = False
    elif size > n:
        # Random initial sample for first AL cycle
        sample_size = n
        cycles = min(N,size-n)
        if initial=="random":
            # Initial pool of size n selected at random
            rand_idx = np.random.choice(size,n,replace=False)
            train_idx = np.array(test_idx)[rand_idx].tolist()
        elif initial=="top":
            # Initial pool of size n selected by low-level feature
            vals = mol_data[lowlevel].values
            # train_idx = np.argsort(vals)[::-1][:n].tolist() # (top-n)
            P = np.where(vals >= np.percentile(vals,90))[0] # (P90)
            train_idx = np.random.choice(P,n,replace=False).tolist() # (P90)
        elif initial=="cpca":
            # Initial pool of size n selected at random from n-means pca-2 clusters
            cluster = f"cpca_{n}"
            train_idx = []
            for i in range(n):
                cluster_idx = data.index[data[cluster]==i]
                if len(cluster_idx) > 0: train_idx.append(np.random.choice(cluster_idx))
        elif initial=="ctsne":
            # Initial pool of size n selected at random from n-means tsne-95 clusters
            cluster = f"ctsne_{n}"
            train_idx = []
            for i in range(n):
                cluster_idx = data.index[data[cluster]==i]
                if len(cluster_idx) > 0: train_idx.append(np.random.choice(cluster_idx))
        elif initial=="dockdiv":
            # Initial pool of size n selected at random from n-means pca-2 clusters of top 50%
            cluster = f"dcpca_{n}"
            train_idx = []
            for i in range(n):
                cluster_idx = data.index[data[cluster]==i]
                if len(cluster_idx) > 0: train_idx.append(np.random.choice(cluster_idx))
        elif initial=="divdock":
            # Initial pool of size n selected at random from top 50% of n-means pca-2 clusters
            cluster = f"cpca_{n}"
            train_idx = []
            for i in range(n):
                cluster_idx = data.index[data[cluster]==i]
                if len(cluster_idx) > 0:
                    vals = mol_data[lowlevel].iloc[cluster_idx].values
                    P = cluster_idx[np.where(vals >= np.percentile(vals,90))[0]] # 50
                    train_idx.append(np.random.choice(P))
        test_idx = np.delete(test_idx,train_idx).tolist()
        Xtrain = data[descriptors].loc[train_idx].values.tolist()
        Xtest = data[descriptors].loc[test_idx].values.tolist()
        # Xtrain, Xtest = preprocess(data,descriptors,train_idx,test_idx)
        active = True

    # Query oracle for training data
    Ytrain = [oracle_fn(idx) for idx in train_idx]


    # Begin optimization loop
    cycle = 0
    while active:
        cycle += 1

        # Select data point
        # trained_model = fit_model(model,Xtrain,Ytrain,parameter_ranges)
        # Ypred, Upred = inference(trained_model,Xtest,sd)
        if predict:
            model = model_class
            trained_model, Ypred, Upred = fit_predict(model,parameter_ranges,Xtrain,Ytrain,Xtest,sd)
            idx = acquisition_fn(Ypred,Upred,Ytrain,test_idx)
        elif not predict:
            if acquisition_fn in [random_sampling, tanimoto]:
                new_fn = acquisition_fn
            else: new_fn = random_sampling
            # Random or similarity-based search
            idx = new_fn(Ytrain,Xtrain,Xtest,test_idx)

        train_idx.append(test_idx.pop(idx))
        Xtrain.append(Xtest.pop(idx))

        # Query oracle for selected data point
        Ytrain_new = oracle_fn(train_idx[-1])
        Ytrain.append(Ytrain_new)

        # Test for termination conditions
        active = termination_fn(cycle,cycles,Ytrain,target)
        if len(test_idx) == 0: active = False

    # oracle_calls = sample_size + cycle
    if predict: idx = acquisition_fn(Ytrain,np.zeros_like(Ytrain),Ytrain,train_idx)
    elif not predict: idx = acquisition_fn(Ytrain,Xtrain,Xtrain,train_idx)
    best_idx = train_idx[idx]
    best_Y = Ytrain[idx]

    return train_idx, Ytrain, best_idx, best_Y

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

#%%

# Model constants
models = [
	LinearRegression(),
	Ridge(random_state=rng),
        BayesianRidge(),
        LinearSVR(random_state=rng),
        SVR(),
	RandomForestRegressor(random_state=rng),
	KernelRidge(),
	GaussianProcessRegressor(random_state=rng),
	XGBRegressor(random_state=rng)
]
model_names = [
    "LR","RR","BRR","lSVR","SVR","RFR","KRR","GPR","XGB"
]
RBF_kernel = RBF()
RQ_kernel = RationalQuadratic()
Matern_kernel = Matern()
ESS_kernel = ExpSineSquared()
White_kernel = WhiteKernel()
parameter_ranges = [
    {"fit_intercept": [True, False]},
    {"alpha": [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100]},
    {"alpha_1":[1e-6],
     "alpha_2":[1e-6],
     "lambda_1":[1e-6],
     "lambda_2":[1e-6]
    },
    {"C": [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100]},
    {"C": [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100],
     "kernel": ["linear","poly","rbf","sigmoid"],
     "degree": [1,2,3],
     "epsilon": [1.0,0.1,0.001,0.0001],
    },
    {"max_features": ["sqrt","log2"],
     "max_depth": [5, 50, 100]
    },
    {"kernel": ["poly"],
     "alpha": [1e-3,1e-2,1e-1,1],
     "gamma": [1e-3,1e-2,1e-1,1],
     "degree": [2, 3, 4]
    },
    {"kernel": [
	1.0 * RBF_kernel,
        1.0 * RQ_kernel,
        1.0 * Matern_kernel,
	1.0 * RQ_kernel + 1.0 * Matern_kernel,
        ],
    },
    {"n_estimators": [50, 100, 200],
     "max_depth": [5, 10, 50, 100],
     "eta": [0.1, 0.01, 0.001],
    },
]


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


# Descriptor constants
descriptors = {
    #"rdkit2d":[desc[0] for desc in set(_descList)],
    "rdkit2d":good_rdkit2d,
    "maccs":[f"maccs_{i}" for i in range(167)],
    "morgan3":[f"morgan3_{i}" for i in range(2048)], #2048
    #"rdkit3d":['PMI1','PMI2','PMI3','NPR1', 'NPR2', 'RadiusOfGyration',
    #    'InertialShapeFactor','Eccentricity','Asphericity','SpherocityIndex','PBF'],
    "rdkit3d": good_rdkit3d,
    "autocorr3d":[f"AUTOCORR3D_{i}" for i in range(80)],
    "rdf":[f"RDF_{i}" for i in range(210)],
    "getaway":[f"GETAWAY_{i}" for i in range(273)],
    "whim":[f"WHIM_{i}" for i in range(114)],
    "vina":[f"vinaF_{i}" for i in range(49)],
    "sasa":[f"sasaF_{i}" for i in range(30)],
    "lig":[f"ligF_{i}" for i in range(10)],
    "wat":[f"watF_{i}" for i in range(3)],
    "delta":["score","betaScore","ligCover","LinF9"],
    "docking":["XGB"],
    "noised04":["noised_04"],
    "noised08":["noised_08"],
    "noised12":["noised_12"],
    "noised16":["noised_16"],
    "noised24":["noised_24"],
    "noised32":["noised_32"],
    "noise01": ["noise_01"]
}

#%%

configs =[
    {"dataset":"EGFR-2048","label":"pKi","model":"BRR","acquisition":"random","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"EGFR-2048","label":"pKi","model":"BRR","acquisition":"tanimoto","descriptors":["morgan3",],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"EGFR-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"EGFR-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"cpca","lowlevel":"XGB"},
    {"dataset":"EGFR-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"EGFR-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3","docking",],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"EGFR-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3","rdkit3d","delta","docking"],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"LCK-2048","label":"pKi","model":"BRR","acquisition":"random","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"LCK-2048","label":"pKi","model":"BRR","acquisition":"tanimoto","descriptors":["morgan3",],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"LCK-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3",],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"LCK-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"cpca","lowlevel":"XGB"},
    {"dataset":"LCK-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"LCK-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3","docking",],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"LCK-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3","rdkit3d","delta","docking"],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"JAK2-2048","label":"pKi","model":"BRR","acquisition":"random","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"JAK2-2048","label":"pKi","model":"BRR","acquisition":"tanimoto","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"JAK2-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"JAK2-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"cpca","lowlevel":"XGB"},
    {"dataset":"JAK2-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"JAK2-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3","docking",],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"JAK2-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3","rdkit3d","delta","docking"],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"MAOB-2048","label":"pKi","model":"BRR","acquisition":"random","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"MAOB-2048","label":"pKi","model":"BRR","acquisition":"tanimoto","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"MAOB-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"MAOB-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"cpca","lowlevel":"XGB"},
    {"dataset":"MAOB-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"MAOB-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3","docking",],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"MAOB-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3","rdkit3d","delta","docking"],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"NOS1-2048","label":"pKi","model":"BRR","acquisition":"random","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"NOS1-2048","label":"pKi","model":"BRR","acquisition":"tanimoto","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"NOS1-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"NOS1-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"cpca","lowlevel":"XGB"},
    {"dataset":"NOS1-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"NOS1-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3","docking",],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"NOS1-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3","rdkit3d","delta","docking"],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"PARP1-2048","label":"pKi","model":"BRR","acquisition":"random","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"PARP1-2048","label":"pKi","model":"BRR","acquisition":"tanimoto","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"PARP1-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"PARP1-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"cpca","lowlevel":"XGB"},
    {"dataset":"PARP1-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"PARP1-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3","docking",],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"PARP1-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3","rdkit3d","delta","docking"],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"ACHE-2048","label":"pKi","model":"BRR","acquisition":"random","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"ACHE-2048","label":"pKi","model":"BRR","acquisition":"tanimoto","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"ACHE-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"ACHE-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"cpca","lowlevel":"XGB"},
    {"dataset":"ACHE-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"ACHE-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3","docking",],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"ACHE-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3","rdkit3d","delta","docking"],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"PDE5A-2048","label":"pKi","model":"BRR","acquisition":"random","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"PDE5A-2048","label":"pKi","model":"BRR","acquisition":"tanimoto","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"PDE5A-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"PDE5A-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"cpca","lowlevel":"XGB"},
    {"dataset":"PDE5A-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"PDE5A-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3","docking",],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"PDE5A-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3","rdkit3d","delta","docking"],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"PTGS2-2048","label":"pKi","model":"BRR","acquisition":"random","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"PTGS2-2048","label":"pKi","model":"BRR","acquisition":"tanimoto","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"PTGS2-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"PTGS2-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"cpca","lowlevel":"XGB"},
    {"dataset":"PTGS2-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"PTGS2-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3","docking",],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"PTGS2-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3","rdkit3d","delta","docking"],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"ESR1-2048","label":"pKi","model":"BRR","acquisition":"random","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"ESR1-2048","label":"pKi","model":"BRR","acquisition":"tanimoto","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"ESR1-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"ESR1-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"cpca","lowlevel":"XGB"},
    {"dataset":"ESR1-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"ESR1-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3","docking",],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"ESR1-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3","rdkit3d","delta","docking"],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"NR3C1-2048","label":"pKi","model":"BRR","acquisition":"random","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"NR3C1-2048","label":"pKi","model":"BRR","acquisition":"tanimoto","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"NR3C1-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"NR3C1-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"cpca","lowlevel":"XGB"},
    {"dataset":"NR3C1-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"NR3C1-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3","docking",],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"NR3C1-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3","rdkit3d","delta","docking"],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"AR-2048","label":"pKi","model":"BRR","acquisition":"random","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"AR-2048","label":"pKi","model":"BRR","acquisition":"tanimoto","descriptors":["morgan3",],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"AR-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"AR-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"cpca","lowlevel":"XGB"},
    {"dataset":"AR-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"AR-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3","docking",],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"AR-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3","rdkit3d","delta","docking"],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"F10-2048","label":"pKi","model":"BRR","acquisition":"random","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"F10-2048","label":"pKi","model":"BRR","acquisition":"tanimoto","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"F10-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"F10-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"cpca","lowlevel":"XGB"},
    {"dataset":"F10-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"F10-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3","docking",],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"F10-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3","rdkit3d","delta","docking"],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"ADRB2-2048","label":"pKi","model":"BRR","acquisition":"random","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"ADRB2-2048","label":"pKi","model":"BRR","acquisition":"tanimoto","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"ADRB2-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"random","lowlevel":"XGB"},
    {"dataset":"ADRB2-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"cpca","lowlevel":"XGB"},
    {"dataset":"ADRB2-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3"],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"ADRB2-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3","docking",],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
    {"dataset":"ADRB2-2048","label":"pKi","model":"BRR","acquisition":"maximise","descriptors":["morgan3","rdkit3d","delta","docking"],"m":10,"M":10000,"initial":"top","lowlevel":"XGB"},
]

#%%

for config in configs:

    # Determine data, model, descriptors and optimization configuration
    Nrep = 25
    datafile = f"{config['dataset']}_data_3d_delta_pKi.csv"
    mol_data = pd.read_csv(os.path.join("data",datafile))
    mol_label = [config["label"]]
    label_values = mol_data[mol_label].values
    m = config["m"]
    M = config["M"]
    if "mol_target" in config.keys(): mol_target = config["mol_target"]
    else: mol_target = np.max(label_values)
    def mol_oracle(mol_idx,data=mol_data,label=mol_label):
        return data[label].iloc[mol_idx].values.tolist()
    if "acquisition" in config.keys():
        if config["acquisition"] == "maximise":
            acquisition = ""
            mol_acquisition = maximise
            sd = False
            predict = True
        elif config["acquisition"] == "UCB":
            acquisition = "UCB"
            mol_acquisition = UCB
            sd = True
            predict = True
        elif config["acquisition"] == "PI":
            acquisition = "PI"
            mol_acquisition = PI
            sd = True
            predict = True
        elif config["acquisition"] == "EI":
            acquisition = "EI"
            mol_acquisition = EI
            sd = True
            predict = True
        elif config["acquisition"] == "random":
            acquisition = "random"
            mol_acquisition = random_sampling
            sd = False
            predict = False
        elif config["acquisition"] == "tanimoto":
            acquisition = "tanimoto"
            mol_acquisition = tanimoto
            sd = False
            predict = False
    else:
        acquisition = ""
        mol_acquisition = maximise
        sd = False
        predict = True
    if predict:
        mol_model = models[model_names.index(config["model"])]
        mol_ranges = parameter_ranges[model_names.index(config["model"])]
    elif not predict:
        config["model"] = ""
        mol_model = None
        mol_ranges = None
    if "initial" in config.keys(): initial = config["initial"]
    else: initial="random"
    if "lowlevel" in config.keys(): lowlevel = config["lowlevel"]
    else: lowlevel=""
    mol_termination = target_limit
    if "scaling" in config.keys(): scaling = config["scaling"]
    else: scaling = True
    if "pca" in config.keys(): pca = config["pca"]
    else: pca = False
    mol_desc = sum([descriptors[d] for d in config["descriptors"]], [])
    job = ""
    job += config["dataset"]
    if config["model"] != "": job += "_"+config["model"]
    if acquisition != "": job += "_"+acquisition
    if acquisition != "random":
        for d in config["descriptors"]: job += "_"+d
    if pca: job += "_pca"
    job += "_"+str(config["m"])
    if initial == "top" and lowlevel: job += "_top_"+lowlevel+"_P90"
    elif initial == "cpca": job += "_cpca"
    elif initial == "ctsne": job += "_ctsne"
    elif initial == "dockdiv": job += "_dockdiv_"+lowlevel+"_P90"
    elif initial == "divdock": job += "_divdock_"+lowlevel+"_P90"
    print(f"Running {job} ...")

    # Preprocess descriptors by removing zero-variance columns and highly correlated features
    # also scale descriptors to zero mean and unit standard deviation, and optionally reduce dimensionality
    zero_columns = np.where(mol_data[mol_desc].std() == 0)[0]
    mol_desc = [desc for desc in mol_desc if desc not in zero_columns]
    fp = [desc for desc in mol_desc if mol_data[desc].isin([0,1]).all()]
    non_fp = [desc for desc in mol_desc if desc not in fp]
    corr_matrix = mol_data[non_fp].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    mol_desc = [desc for desc in mol_desc if desc not in to_drop]
    if scaling:
        for desc in non_fp:
            mol_data[desc] = StandardScaler().fit_transform(mol_data[desc].values.reshape(-1,1))
    if pca:
        pca_data = pd.DataFrame(PCA(n_components=0.95).fit_transform(mol_data[non_fp]))
        pca_desc = [f"PCA_{i}" for i in range(pca_data.shape[1])]
        pca_data.columns = pca_desc
        pca_data[mol_label+fp] = mol_data.copy()[mol_label+fp]
        mol_data = pca_data
        mol_desc = pca_desc+fp
    print(f"Feature length: {len(mol_desc)}")
    # Begin optimization over Nrep (default: 25) random seeds
    # determine number of queries required to reach a given checkpoint target
    # always "full pass" so trajectories run until global maxima is located
    targets = [7+0.1*i for i in range(int((mol_target-7)/0.1)+1)]+[mol_target]
    #print(targets)
    queries = []
    ids = []
    for i in tqdm(range(Nrep)):
        np.random.seed(i)
        try:
            train_idx, Ytrain, best_pred_idx, best_Ypred = optimization(
                mol_data,mol_desc,mol_label,
                mol_oracle,mol_acquisition,mol_termination,
                m,M,mol_target,
                mol_model,mol_ranges,
                initial,lowlevel,
                predict,sd)
            checkpoints = [np.argmax(np.array(Ytrain).flatten() >= target)+1 for target in targets]
            queries.append(checkpoints)
            ids.append(train_idx)
        except Exception as e: print(e) # in some cases, errors may arise from np.linalg

    # Collect results as csv
    outdir = "results"
    if not os.path.exists(outdir): os.mkdir(outdir)
    queries = np.array(queries)
    id_df = pd.DataFrame(ids).transpose()
    id_df.columns = [f"run_{i}" for i in range(len(ids))]
    id_df.to_csv(os.path.join(outdir,f"{job}_ID.csv"),index=False)
    target_data = mol_data[mol_label].values
    target_data = mol_data[mol_label].values
    mean_queries = np.mean(queries,axis=0)
    std_queries = np.std(queries,axis=0)
    max_queries = np.max(queries,axis=0)
    min_queries = np.min(queries,axis=0)
    rand_queries = [random_analytic(target_data,target) for target in targets]

    results = {
        "targets":targets,
        "mean_queries":mean_queries,
        "std_queries":std_queries,
        "max_queries":max_queries,
        "min_queries":min_queries,
        "rand_queries":rand_queries,
    }
    results_df = pd.DataFrame.from_dict(results,orient="index").transpose()
    results_df.to_csv(os.path.join(outdir,f"{job}.csv"),index=False)
