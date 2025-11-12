
#!/usr/bin/env python3

#%%

import os
import random
import time
import inspect
import math

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
    else: Ypred, Upred = model.predict(X), np.zeros(len(X))
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
    else: Ypred, Upred = trained_model.predict(Xtest), np.zeros(len(Xtest))
    Ypred = Ypred.reshape(-1,1)
    Upred = Upred.reshape(-1,1)
    return trained_model, Ypred, Upred

def fit_predict_rebalance(data,train_idx,model,parameter_ranges,Xtrain,Ytrain,Xtest,sd=False,balanced=0.1):
    """
    Rebalance data to reduce the effect of 'rare events' in unbalanced data
    Fit model to data (with Ytrain scaled to mean 0 and variance 1)
    Run inferences of trained model on test data Xtest
    Return Ypred, Upred if Upred is avalilable, else return Ypred, 0
    Ypred :: model predictions on Xtest
    Upred :: model uncertainty (standard deviation) on Xtest
    """
    common = []
    uncommon = []
    for i,id in enumerate(train_idx):
        # In this context, active compounds are uncommon events
        if data["active"].iloc[id] == 1: uncommon.append(i)
        else: common.append(i)
    balance = 1.0
    if len(uncommon) != 0 and len(common) != 0: 
        balance = len(uncommon) / len(common)
    sw_train = np.ones(len(Ytrain))
    if balance < balanced: # training data is 'imbalanced'
        n_common = int(len(uncommon) / balanced) # downsample
        selected = np.random.choice(common,n_common,replace=False)
        idx = np.concatenate([selected,uncommon])
        Ytrain = [Ytrain[i] for i in idx]
        Xtrain = [Xtrain[i] for i in idx]
        upweight = 1/balance
        sw_train[selected] = upweight # upweight loss
        sw_train = sw_train[idx]
    Ytrain = StandardScaler().fit_transform(Ytrain)
    grid_search = GridSearchCV(
        model,
        parameter_ranges,
        scoring="neg_mean_absolute_error",
        cv=5,
        refit=True,
        n_jobs=5, #-1
        verbose=0
    )
    grid_search.fit(Xtrain, Ytrain, sample_weight=sw_train)
    trained_model = grid_search.best_estimator_
    if sd and "return_std" in inspect.getfullargspec(model.predict)[0]:
        Ypred, Upred =  trained_model.predict(Xtest,return_std=True)
    else: Ypred, Upred = trained_model.predict(Xtest), np.zeros(len(Xtest))
    Ypred = Ypred.reshape(-1,1)
    Upred = Upred.reshape(-1,1)
    return trained_model, Ypred, Upred

def maximise(Ypred,Upred,Ytrain,test_idx,batch_size=1):
    """
    Predicted mean acquisiton function
    a(x) = mu(X)
    return idx that maximises a(x)
    """
    a = Ypred
    if batch_size == 1: idx = [np.random.choice(np.where(a == np.max(a))[0])]
    elif len(np.where(a == np.max(a))[0]) >= batch_size: # multiple maxima
        idx = np.random.choice(np.where(a==np.max(a))[0],batch_size,replace=False)
    else: idx = (-a.flatten()).argsort(kind="stable")[:batch_size]
    return idx

def UCB(Ypred,Upred,Ytrain,test_idx,l=1,batch_size=1):
    """
    Upper confidence bound acquisition function
    a(x) = mu + lambda * sigma(x)
    return idx that maximises a(x)
    """
    a = Ypred + l * Upred
    if batch_size == 1: idx = [np.random.choice(np.where(a == np.max(a))[0])]
    elif len(np.where(a == np.max(a))[0]) >= batch_size: # multiple maxima
        idx = np.random.choice(np.where(a==np.max(a))[0],batch_size,replace=False)
    else: idx = (-a.flatten()).argsort(kind="stable")[:batch_size]
    return idx

def PI(Ypred,Upred,Ytrain,test_idx,batch_size=1):
    """
    Predicted Improvement acquisition function
    a(x) = p[f(x) > f(x*)]
    return idx that maximises a(x)
    """
    Ytrain = StandardScaler().fit_transform(Ytrain.reshape(-1,1))
    Ybest = np.max(Ytrain)
    a = sp.stats.norm.cdf((Ypred - Ybest) / Upred)
    if batch_size == 1: idx = [np.random.choice(np.where(a == np.max(a))[0])]
    elif len(np.where(a == np.max(a))[0]) >= batch_size: # multiple maxima
        idx = np.random.choice(np.where(a==np.max(a))[0],batch_size,replace=False)
    else: idx = (-a.flatten()).argsort(kind="stable")[:batch_size]
    return idx

def EI(Ypred,Upred,Ytrain,test_idx,batch_size=1):
    """
    Expected Improvement acquisition function
    a(x) = E[f(x) > f(x*)]
    return idx that maximises a(x)
    """
    Ytrain = StandardScaler().fit_transform(Ytrain.reshape(-1,1))
    Ybest = np.max(Ytrain)
    a = (Ypred - Ybest) * sp.stats.norm.cdf((Ypred - Ybest) / Upred) + \
       Upred * sp.stats.norm.pdf((Ypred - Ybest) / Upred)
    if batch_size == 1: idx = [np.random.choice(np.where(a == np.max(a))[0])]
    elif len(np.where(a == np.max(a))[0]) >= batch_size: # multiple maxima
        idx = np.random.choice(np.where(a==np.max(a))[0],batch_size,replace=False)
    else: idx = (-a.flatten()).argsort(kind="stable")[:batch_size]
    return idx

def random_sampling(Ytrain,Xtrain,Xtest,test_idx,batch_size=1):
    """
    Random acquisition function
    return idx selected by uniform random sampling
    """
    if batch_size > len(test_idx): return np.arange(len(test_idx))
    elif batch_size == 1: return [np.random.choice(len(test_idx))]
    return np.random.choice(len(test_idx),size=batch_size,replace=False)

def tanimoto(Ytrain,Xtrain,Xtest,test_idx,batch_size=1):
    """
    Similarity based acquisition function
    return idx of nearest neighbour to Xbest
    using Tanimoto (Jaccard) similarity (must use with binary features only)
    """
    Xbest = Xtrain[np.argmax(Ytrain)]
    distances = cdist([Xbest],Xtest,"jaccard")
    idx = (distances.flatten()).argsort(kind="stable")[:batch_size]
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
    Preprocessing training and testing data
    data :: dataframe containing training and testing features
    descriptors :: list of features corresponding to columns in data
    train_idx :: dataframe row indices for training examples
    test_idx :: dataframe row indices for testing examples
    """
    X = data[descriptors]
    zero_columns = X.columns[(X == 0).all()].values.tolist()
    X.drop(zero_columns,axis=1,inplace=True)
    descriptors = X.columns
    fp = [desc for desc in descriptors if data[desc].isin([0,1]).all()]
    non_fp = [desc for desc in descriptors if desc not in fp]
    corr_matrix = X[non_fp].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    X.drop(to_drop,axis=1,inplace=True)
    descriptors = X.columns
    for desc in non_fp:
        s = scaler
        X[desc] = s.fit_transform(X[desc].values.reshape(-1,1))
    if pca:
        pca = PCA(n_components=0.95)
        PC = pd.DataFrame(pca.fit_transform(X[non_fp]))
        PC[fp] = X[fp]
        X = PC
    Xtrain = X.loc[train_idx]
    Xtest = X.loc[test_idx]
    return Xtrain.values.tolist(), Xtest.values.tolist()


def optimization(data,descriptors,label,
                    oracle_fn,acquisition_fn,termination_fn,
                    n,N,target,
                    model_class,parameter_ranges,
                    initial="random",lowlevel="",
                    predict=True,sd=False,
                    batch_size=1):
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
    predict :: Whether to perform ML inference or not
    sd :: Whether to return prediction uncertainties or not
    batch_size :: Acquisition function batch size
    """

    # Initial sample
    test_idx = list(data.index.values)
    size = len(test_idx)
    if size <= n:
        # Do not attempt for datasets smaller than n
        sample_size = size
        cycles = 0
        train_idx = test_idx
        active = False
    elif size > n:
        # Random initial sample for first cycle
        sample_size = n
        cycles = min(N,size-n)
        if initial=="random":
            # Initial pool of size n selected at random
            rand_idx = np.random.choice(size,n,replace=False)
            train_idx = np.array(test_idx)[rand_idx].tolist()
        elif initial=="top":
            # Initial pool of size n selected by low-level feature (random from top-10%)
            vals = mol_data[lowlevel].values
            P = np.where(vals >= np.percentile(vals,90))[0] # (P90)
            train_idx = np.random.choice(P,n,replace=False).tolist() # (P90)
        elif initial=="top-n":
            # Initial pool of size n selected by low-level feature (absolute top-n)
            vals = mol_data[lowlevel].values
            train_idx = (-vals).argsort(kind="stable")[:n].tolist()
        elif initial=="cpca":
            # Initial pool of size n selected at random from n-means pca-2 clusters
            cluster = f"cpca_{n}"
            train_idx = []
            for i in range(n):
                cluster_idx = data.index[data[cluster]==i]
                if len(cluster_idx) > 0: train_idx.append(np.random.choice(cluster_idx))
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

        # No informative labels => random sampling
        if np.std(Ytrain) == 0: predict = False
        elif acquisition_fn not in [random_sampling, tanimoto]:  predict = True

        # Select data point(s)
        if predict:
            model = model_class
            try:
                trained_model, Ypred, Upred = fit_predict_rebalance(data,train_idx,model,parameter_ranges,Xtrain,Ytrain,Xtest,sd)
                idx = acquisition_fn(Ypred,Upred,Ytrain,test_idx,batch_size=batch_size)
            except Exception as e:
                print(e)
                predict = False
        if not predict:
            if acquisition_fn in [random_sampling, tanimoto]:
                new_fn = acquisition_fn
            else: new_fn = random_sampling
            # Random or similarity-based search
            idx = new_fn(Ytrain,Xtrain,Xtest,test_idx,batch_size=batch_size)

        train_idx += [test_idx[i] for i in idx]
        test_idx = [test_idx[i] for i in range(len(test_idx)) if i not in idx]
       	Xtrain += [Xtest[i] for i in idx]
        Xtest = [Xtest[i] for i in range(len(Xtest)) if i not in idx]

        # Query oracle for selected data point(s)
        Ytrain_new = oracle_fn(train_idx[-batch_size:])
        Ytrain += Ytrain_new

        # Test for termination conditions
        active = termination_fn(cycle,cycles,Ytrain,target)
        if len(test_idx) == 0: active = False

    return train_idx, Ytrain

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
    quantile = (D < v).mean()
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
	XGBRegressor(random_state=rng, seed=seed)
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
    {"subsample": [0.1, 0.5, 1.0],
     "max_depth": [2, 4, 6],
     "eta": [0.5, 0.1],
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
        #'InertialShapeFactor','Eccentricity','Asphericity','SpherocityIndex','PBF'],
    "rdkit3d":good_rdkit3d,
    "autocorr3d":[f"AUTOCORR3D_{i}" for i in range(80)],
    "rdf":[f"RDF_{i}" for i in range(210)],
    "getaway":[f"GETAWAY_{i}" for i in range(273)],
    "docking":["CNN-Affinity"],
    "noised04":["noised_04"],
    "noised08":["noised_08"],
    "noised12":["noised_12"],
    "noised16":["noised_16"],
    "noised24":["noised_24"],
    "noised32":["noised_32"],
    "noise01": ["noise_01"]
}

#%%

# Parallel experimentation
# multi-well plate
# batch-size = w (may choose initial-size = w)
w = 1

configs =[
    {"dataset":"ESR1ago","label":"pEC50","model":"BRR","acquisition":"random","batch":w,"descriptors":["morgan3",],"m":10,"M":10000,"initial":"random","lowlevel":"CNN-Affinity","termination":"target"},
    {"dataset":"ESR1ago","label":"pEC50","model":"BRR","acquisition":"tanimoto","batch":w,"descriptors":["morgan3",],"m":10,"M":10000,"initial":"random","lowlevel":"CNN-Affinity","termination":"target"},
    {"dataset":"ESR1ago","label":"pEC50","model":"BRR","acquisition":"maximise","batch":w,"descriptors":["morgan3","rdkit2d",],"m":10,"M":10000,"initial":"random","lowlevel":"CNN-Affinity","termination":"target"},
    {"dataset":"ESR1ago","label":"pEC50","model":"BRR","acquisition":"maximise","batch":w,"descriptors":["morgan3","rdkit2d",],"m":10,"M":10000,"initial":"cpca","lowlevel":"CNN-Affinity","termination":"target"},
    #{"dataset":"ESR1ago","label":"pEC50","model":"BRR","acquisition":"maximise","batch":w,"descriptors":["morgan3","rdkit2d",],"m":10,"M":10000,"initial":"top","lowlevel":"CNN-Affinity","termination":"target"},
    #{"dataset":"ESR1ago","label":"pEC50","model":"BRR","acquisition":"maximise","batch":w,"descriptors":["morgan3","rdkit2d","docking"],"m":10,"M":10000,"initial":"top","lowlevel":"CNN-Affinity","termination":"target"},
    #{"dataset":"ESR1ago","label":"pEC50","model":"BRR","acquisition":"maximise","batch":w,"descriptors":["morgan3","rdkit2d","rdkit3d","docking"],"m":10,"M":10000,"initial":"cpca","lowlevel":"CNN-Affinity","termination":"target"},
    {"dataset":"ESR1ago","label":"pEC50","model":"BRR","acquisition":"maximise","batch":w,"descriptors":["morgan3","rdkit2d","rdkit3d","docking"],"m":10,"M":10000,"initial":"top","lowlevel":"CNN-Affinity","termination":"target"},
    {"dataset":"ESR1ant","label":"pEC50","model":"BRR","acquisition":"random","batch":w,"descriptors":["morgan3",],"m":10,"M":10000,"initial":"random","lowlevel":"CNN-Affinity","termination":"target"},
    {"dataset":"ESR1ant","label":"pEC50","model":"BRR","acquisition":"tanimoto","batch":w,"descriptors":["morgan3",],"m":10,"M":10000,"initial":"random","lowlevel":"CNN-Affinity","termination":"target"},
    {"dataset":"ESR1ant","label":"pEC50","model":"BRR","acquisition":"maximise","batch":w,"descriptors":["morgan3","rdkit2d",],"m":10,"M":10000,"initial":"random","lowlevel":"CNN-Affinity","termination":"target"},
    {"dataset":"ESR1ant","label":"pEC50","model":"BRR","acquisition":"maximise","batch":w,"descriptors":["morgan3","rdkit2d",],"m":10,"M":10000,"initial":"cpca","lowlevel":"CNN-Affinity","termination":"target"},
    #{"dataset":"ESR1ant","label":"pEC50","model":"BRR","acquisition":"maximise","batch":w,"descriptors":["morgan3","rdkit2d",],"m":10,"M":10000,"initial":"top","lowlevel":"CNN-Affinity","termination":"target"},
    #{"dataset":"ESR1ant","label":"pEC50","model":"BRR","acquisition":"maximise","batch":w,"descriptors":["morgan3","rdkit2d","docking"],"m":10,"M":10000,"initial":"top","lowlevel":"CNN-Affinity","termination":"target"},
    #{"dataset":"ESR1ant","label":"pEC50","model":"BRR","acquisition":"maximise","batch":w,"descriptors":["morgan3","rdkit2d","rdkit3d","docking"],"m":10,"M":10000,"initial":"cpca","lowlevel":"CNN-Affinity","termination":"target"},
    {"dataset":"ESR1ant","label":"pEC50","model":"BRR","acquisition":"maximise","batch":w,"descriptors":["morgan3","rdkit2d","rdkit3d","docking"],"m":10,"M":10000,"initial":"top","lowlevel":"CNN-Affinity","termination":"target"},
    {"dataset":"PPARG","label":"pEC50","model":"BRR","acquisition":"random","batch":w,"descriptors":["morgan3",],"m":10,"M":821,"initial":"random","lowlevel":"CNN-Affinity","termination":"target"},
    {"dataset":"PPARG","label":"pEC50","model":"BRR","acquisition":"tanimoto","batch":w,"descriptors":["morgan3",],"m":10,"M":821,"initial":"random","lowlevel":"CNN-Affinity","termination":"target"},
    {"dataset":"PPARG","label":"pEC50","model":"BRR","acquisition":"maximise","batch":w,"descriptors":["morgan3","rdkit2d",],"m":10,"M":821,"initial":"random","lowlevel":"CNN-Affinity","termination":"target"},
    {"dataset":"PPARG","label":"pEC50","model":"BRR","acquisition":"maximise","batch":w,"descriptors":["morgan3","rdkit2d",],"m":10,"M":821,"initial":"cpca","lowlevel":"CNN-Affinity","termination":"target"},
    #{"dataset":"PPARG","label":"pEC50","model":"BRR","acquisition":"maximise","batch":w,"descriptors":["morgan3","rdkit2d",],"m":10,"M":821,"initial":"top","lowlevel":"CNN-Affinity","termination":"target"},
    #{"dataset":"PPARG","label":"pEC50","model":"BRR","acquisition":"maximise","batch":w,"descriptors":["morgan3","rdkit2d","docking"],"m":10,"M":821,"initial":"top","lowlevel":"CNN-Affinity","termination":"target"},
    #{"dataset":"PPARG","label":"pEC50","model":"BRR","acquisition":"maximise","batch":w,"descriptors":["morgan3","rdkit2d","rdkit3d","docking"],"m":10,"M":821,"initial":"cpca","lowlevel":"CNN-Affinity","termination":"target"},
    {"dataset":"PPARG","label":"pEC50","model":"BRR","acquisition":"maximise","batch":w,"descriptors":["morgan3","rdkit2d","rdkit3d","docking"],"m":10,"M":821,"initial":"top","lowlevel":"CNN-Affinity","termination":"target"},
    {"dataset":"TP53","label":"pEC50","model":"BRR","acquisition":"random","batch":w,"descriptors":["morgan3",],"m":10,"M":10000,"initial":"random","lowlevel":"CNN-Affinity","termination":"target"},
    {"dataset":"TP53","label":"pEC50","model":"BRR","acquisition":"tanimoto","batch":w,"descriptors":["morgan3",],"m":10,"M":10000,"initial":"random","lowlevel":"CNN-Affinity","termination":"target"},
    {"dataset":"TP53","label":"pEC50","model":"BRR","acquisition":"maximise","batch":w,"descriptors":["morgan3","rdkit2d",],"m":10,"M":10000,"initial":"random","lowlevel":"CNN-Affinity","termination":"target"},
    {"dataset":"TP53","label":"pEC50","model":"BRR","acquisition":"maximise","batch":w,"descriptors":["morgan3","rdkit2d",],"m":10,"M":10000,"initial":"cpca","lowlevel":"CNN-Affinity","termination":"target"},
    #{"dataset":"TP53","label":"pEC50","model":"BRR","acquisition":"maximise","batch":w,"descriptors":["morgan3","rdkit2d",],"m":10,"M":10000,"initial":"top","lowlevel":"CNN-Affinity","termination":"target"},
    #{"dataset":"TP53","label":"pEC50","model":"BRR","acquisition":"maximise","batch":w,"descriptors":["morgan3","rdkit2d","docking"],"m":10,"M":10000,"initial":"top","lowlevel":"CNN-Affinity","termination":"target"},
    #{"dataset":"TP53","label":"pEC50","model":"BRR","acquisition":"maximise","batch":w,"descriptors":["morgan3","rdkit2d","rdkit3d","docking"],"m":10,"M":10000,"initial":"cpca","lowlevel":"CNN-Affinity","termination":"target"},
    {"dataset":"TP53","label":"pEC50","model":"BRR","acquisition":"maximise","batch":w,"descriptors":["morgan3","rdkit2d","rdkit3d","docking"],"m":10,"M":10000,"initial":"top","lowlevel":"CNN-Affinity","termination":"target"},
]


for config in configs:

    # Determine data, model, descriptors and optimization configuration
    Nrep = 25 # number of repeats (random seeds)
    datafile = f"{config['dataset']}_data_full.csv"
    mol_data = pd.read_csv(os.path.join("data",datafile))
    mol_label = [config["label"]]
    label_values = mol_data[mol_label].values
    m = config["m"]
    M = config["M"]

    # Choice of optimization target
    if "target" in config.keys(): mol_target = config["target"]
    else: mol_target = np.max(label_values)
    def mol_oracle(mol_idx,data=mol_data,label=mol_label):
        return data[label].iloc[mol_idx].values.tolist()

    # Acquisition config
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
    # Batch size
    if "batch" in config.keys(): batch_size = config["batch"]
    else: batch_size = 1

    # Initialization config
    if "initial" in config.keys(): initial = config["initial"]
    else: initial="random"
    if "lowlevel" in config.keys(): lowlevel = config["lowlevel"]
    else: lowlevel=""

    # Termination config
    if "termination" in config.keys(): termination = config["termination"]
    else: termination = "target"
    termination_functions =  {"cycle":cycle_limit, "target":target_limit}
    mol_termination = termination_functions[termination]

    # Pre-processing config
    if "scaling" in config.keys(): scaling = config["scaling"]
    else: scaling = True
    if "pca" in config.keys(): pca = config["pca"]
    else: pca = False
    mol_desc = sum([descriptors[d] for d in config["descriptors"]], [])

    # Save config name
    job = ""
    job += config["dataset"]
    if config["model"] != "": job += "_"+config["model"]
    if acquisition != "": job += "_"+acquisition
    if acquisition != "random":
        for d in config["descriptors"]: job += "_"+d
    if pca: job += "_pca"
    job += "_"+str(config["m"])
    if initial == "top" and lowlevel: job += "_top_"+lowlevel+"_P90"
    elif initial == "top-n" and lowlevel: job += f"_top-{m}"+"_"+lowlevel
    elif initial == "cpca": job += "_cpca"
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
    targets = [4+0.1*i for i in range(int((mol_target-4)/0.1)+1)]+[mol_target]
    queries = []
    ids = []
    for i in tqdm(range(Nrep)):
        np.random.seed(i)
        train_idx, Ytrain = optimization(
            mol_data,mol_desc,mol_label,
            mol_oracle,mol_acquisition,mol_termination,
            m,M,mol_target,
            mol_model,mol_ranges,
            initial,lowlevel,
            predict,sd,
            batch_size)

        # Collect number of steps needed to reach a given target value
        # Catch case where max(Ytrain) < target
        checkpoints = []
        for target in targets:
            is_greater = np.array(Ytrain).flatten() >= target
            if not np.any(is_greater): checkpoints.append(len(Ytrain)*2)
            else: checkpoints.append(np.argmax(is_greater)+1)
        checkpoints = [math.ceil(n/w)*w for n in checkpoints] # round up to nearest multiple of batch

        queries.append(checkpoints)
        ids.append(train_idx)

    # Collect results as csv
    outdir = "results_litpcba" #f"litpcba_batch_{w}" #f"litpcba_batch_{w}_budget_{int(M*w+m)}"
    if not os.path.exists(outdir): os.mkdir(outdir)
    queries = np.array(queries)
    id_df = pd.DataFrame(ids).transpose()
    id_df.columns = [f"run_{i}" for i in range(len(ids))]
    id_df.to_csv(os.path.join(outdir,f"{job}_ID.csv"),index=False)
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
#%%
