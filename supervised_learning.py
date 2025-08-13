#%%

import os
import random
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import time
import itertools
import inspect

from IPython.display import display

import warnings
warnings.filterwarnings('ignore') 

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

from scipy.stats import pearsonr, spearmanr, kendalltau

from rdkit import Chem
from rdkit.Chem.Descriptors import _descList

from tqdm.auto import tqdm

seed=42
random.seed(seed)
os.environ["PYTHONHASHSEED"]=str(seed)
np.random.seed(seed)
rng = np.random.RandomState(seed)

#%%

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
descriptors = {
    # "rdkit2d":[desc[0] for desc in set(_descList)],
	"rdkit2d":good_rdkit2d,
    "maccs":[f"maccs_{i}" for i in range(166)],
    "morgan3":[f"morgan3_{i}" for i in range(2048)],
    # "rdkit3d":['PMI1','PMI2','PMI3','NPR1', 'NPR2', 'RadiusOfGyration',
    #     'InertialShapeFactor','Eccentricity','Asphericity','SpherocityIndex','PBF'],
	"rdkit3d":good_rdkit3d,
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

def fit_model(model, parameter_ranges, Xtrain, Ytrain):
	grid_search = GridSearchCV(
		model, 
		parameter_ranges, 
		scoring="neg_mean_absolute_error", 
		cv=5, 
		refit=True, 
		n_jobs=2,
		verbose=1
	)
	grid_search.fit(Xtrain, Ytrain)
	print(grid_search.best_params_)
	return grid_search.best_estimator_

def fit_predict(model,parameter_ranges,Xtrain,Ytrain,Xtest,sd=False):
    Ytrain = StandardScaler().fit_transform(Ytrain)
    trained_model = fit_model(model,Xtrain,Ytrain,parameter_ranges)
    if sd and "return_std" in inspect.getfullargspec(model.predict)[0]:
        Ypred, Upred =  trained_model.predict(Xtest,return_std=True)
    else: Ypred, Upred = trained_model.predict(Xtest), np.zeros_like(Xtest)
    Ypred = Ypred #.reshape(-1,1)
    Upred = Upred #.reshape(-1,1)
    return trained_model, Ypred, Upred

def make_plot(model, Ypred, Y):
	plt.scatter(Ypred, Y, c="blue", s=2, alpha=0.5)
	plt.xlabel("Predicted")
	plt.ylabel("True")
	plt.title(f"{str(model).split('(')[0]}() test results")
	# plt.axis("equal")
	plt.show()

def evaluate_model(model, parameter_ranges, Xtrain, Ytrain, Xtest, Ytest, plot=False):
	train_scaler = StandardScaler().fit(Ytrain.reshape(-1,1))
	train_scale = train_scaler.scale_
	Ytrain = train_scaler.transform(Ytrain.reshape(-1,1)).ravel() #
	test_scaler = StandardScaler().fit(Ytest.reshape(-1,1))
	test_scale = test_scaler.scale_
	Ytest = test_scaler.transform(Ytest.reshape(-1,1)).ravel() #
	trained_model = fit_model(model, parameter_ranges, Xtrain, Ytrain)
	Ytrain_pred = trained_model.predict(Xtrain)
	Ytest_pred = trained_model.predict(Xtest)
	if plot:
		make_plot(model, Ytest_pred, Ytest)
		LearningCurveDisplay.from_estimator(trained_model, Xtrain, Ytrain, scoring="neg_mean_absolute_error")
		plt.show()
	Ytrain = Ytrain.flatten()
	Ytrain_pred = Ytrain_pred.flatten()
	Ytest = Ytest.flatten()
	Ytest_pred = Ytest_pred.flatten()
	train_error = mean_absolute_error(Ytrain_pred, Ytrain) * train_scale
	test_error = mean_absolute_error(Ytest_pred, Ytest) * test_scale
	train_r = pearsonr(Ytrain_pred, Ytrain).statistic
	test_r = pearsonr(Ytest_pred, Ytest).statistic
	train_sr = spearmanr(Ytrain_pred, Ytrain).statistic
	test_sr = spearmanr(Ytest_pred, Ytest).statistic
	train_kr = kendalltau(Ytrain_pred, Ytrain).statistic
	test_kr = kendalltau(Ytest_pred, Ytest).statistic
	results = (
		train_error, test_error, train_r, test_r, train_sr, test_sr, train_kr, test_kr,
	)
	print(len(results))
	return results

def generate_datasets(data, rstate, train_size=10,
					  features=["morgan3"], descriptors=descriptors,
					  scaling=True, pca=False,
					  target="pKi",
					  initial="random", lowlevel="XGB",):
	mol_desc = sum([descriptors[d] for d in features], [])
	target = target
	print("Removing zero-valued features...")
	zero_columns = data[mol_desc].columns[(data[mol_desc] == 0).all()].values.tolist()
	print(zero_columns)
	mol_desc = [desc for desc in mol_desc if desc not in zero_columns]
	print("Removing highly correlated features...")
	fp = [desc for desc in mol_desc if data[desc].isin([0,1]).all()]
	non_fp = [desc for desc in mol_desc if desc not in fp]
	corr_matrix = data[non_fp].corr().abs()
	upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
	to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
	print(to_drop)
	mol_desc = [desc for desc in mol_desc if desc not in to_drop]
	if scaling:
		print("Scaling non-fingerprint features...")
		for desc in non_fp: data[desc] = StandardScaler().fit_transform(data[desc].values.reshape(-1,1))
	if pca:
		print("PCA decomposition on non-fignerprint features...")
		pca = PCA(n_components=0.95)
		PCAset = pd.DataFrame(pca.fit_transform(data[non_fp]))
		pca_desc = [f"PC_{i}" for i in range(len(PCAset.columns))]
		PCAset.columns = pca_desc
		PCAset[fp+[target]] = data[fp+[target]]
		mol_desc = pca_desc + fp
		data = PCAset
	X = data[mol_desc].to_numpy(dtype=float)
	Y = data[target].to_numpy(dtype=float)
	Xtrain, Ytrain, Xtest, Ytest = initialisation(
		data,X,Y,train_size,initial,lowlevel,rstate)
	return Xtrain, Ytrain, Xtest, Ytest

def initialisation(data, X, Y, n, initial, lowlevel, rstate):
	np.random.seed(rstate)
	size = len(data.index)
	test_idx = data.index
	if initial=="random":
		# Initial pool of size n selected at random
		rand_idx = np.random.choice(size,n,replace=False)
		train_idx = np.array(test_idx)[rand_idx].tolist()
	elif initial=="top":
		# Initial pool of size n selected by low-level feature
		vals = data[lowlevel].values
		# train_idx = np.argsort(vals)[::-1][:n].tolist() # (top-n)
		P = np.where(data[lowlevel].values >= np.percentile(vals,90))[0] #(P90)
		train_idx = np.random.choice(P,n,replace=False).tolist() #(P90)
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
	test_idx = np.delete(test_idx,train_idx)
	Xtrain = X[train_idx]
	Ytrain = Y[train_idx]
	Xtest = X[test_idx]
	Ytest = Y[test_idx]
	return Xtrain, Ytrain, Xtest, Ytest

def test_models(models, parameter_ranges, data, 
				features=["morgan3"], target="pKi",
				n_repeats=5, train_size=10, 
				initial="random", lowlevel="XGB",
				plot=False):
	train_scores = np.zeros(len(models))
	test_scores = np.zeros(len(models))
	train_square_errors = np.zeros(len(models))
	test_square_errors = np.zeros(len(models))
	train_pearsonr = np.zeros(len(models))
	test_pearsonr = np.zeros(len(models))
	train_spearmanr = np.zeros(len(models))
	test_spearmanr = np.zeros(len(models))
	train_kendalltau = np.zeros(len(models))
	test_kendalltau = np.zeros(len(models))
	for r in tqdm(range(n_repeats)):
		print(f"Repeat number: {r}")
		Xtrain, Ytrain, Xtest, Ytest = generate_datasets(
			data, rstate=r, 
			features=features, target=target,
			initial=initial, lowlevel=lowlevel,
			train_size=train_size)
		for i, (model, parameter_range) in enumerate(zip(models, parameter_ranges)):
			print(f"Model: {model_names[i]}")
			t0 = time.time()
			(train_error, test_error,
			train_r, test_r,
			train_sr, test_sr, 
			train_kr, test_kr,) = evaluate_model(
				model, parameter_range, 
				Xtrain, Ytrain, Xtest, Ytest, 
				plot=plot)
			t1 = time.time()
			print(f"Time taken: {round(t1-t0,3)}\n")
			train_scores[i] += train_error
			test_scores[i] += test_error
			train_square_errors[i] += train_error**2
			test_square_errors[i] += test_error**2
			train_pearsonr[i] += train_r
			test_pearsonr[i] += test_r
			train_spearmanr[i] += train_sr
			test_spearmanr[i] += test_sr
			train_kendalltau[i] += train_kr
			test_kendalltau[i] += test_kr
	avg_train_mae = train_scores / n_repeats
	avg_test_mae = test_scores / n_repeats
	avg_train_maes = train_square_errors / n_repeats
	avg_test_maes = test_square_errors / n_repeats
	avg_train_sd = np.sqrt(np.abs(avg_train_maes - avg_train_mae**2))
	avg_test_sd = np.sqrt(np.abs(avg_test_maes - avg_test_mae**2))
	avg_train_pearsonr = train_pearsonr / n_repeats
	avg_test_pearsonr = test_pearsonr / n_repeats
	avg_train_spearmanr = train_spearmanr / n_repeats
	avg_test_spearmanr = test_spearmanr / n_repeats
	avg_train_kendalltau = train_kendalltau / n_repeats
	avg_test_kendalltau = test_kendalltau / n_repeats
	output = (
		avg_train_mae,
		avg_test_mae,
		avg_train_sd,
		avg_test_sd,
		avg_train_pearsonr,
		avg_test_pearsonr,
		avg_train_spearmanr,
		avg_test_spearmanr,
		avg_train_kendalltau,
		avg_test_kendalltau,
	)
	return output

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
    {"n_estimators": [50, 100, 200],
     "max_depth": [5, 10, 50, 100]
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

#%%

# Evaluate data on different combinations of features and models

if __name__ == "__main__":
	targets = [
		"EGFR","JAK2","LCK","MAOB","NOS1","PARP1","ACHE",
		"PDE5A","PTGS2","ESR1","AR","NR3C1","F10","ADRB2"]
	for target in targets:
		print()
		print(target)
		dataset = f"{target}-2048_data_3d_delta_pKi.csv"
		data = pd.read_csv(os.path.join("data",dataset))
		lowlevel = "XGB"
		train_size = 10
		list_initials = [
			"random",
			"cpca",
			"top"
		]
		list_features = [
			["morgan3"],
			["morgan3","rdkit2d"],
			["morgan3","docking"],
			["morgan3","rdkit2d","docking"],
			["morgan3","rdkit3d","delta","docking"],
			["morgan3","rdkit2d","rdkit3d","delta","docking"],
		]
		for initial, features in itertools.product(list_initials,list_features):
			(
			avg_train_mae, 
			avg_test_mae, 
			avg_train_sd,
			avg_test_sd,
			avg_train_pearsonr,
			avg_test_pearsonr,
			avg_train_spearmanr,
			avg_test_spearmanr,
			avg_train_kendalltau,
			avg_test_kendalltau,
			) = test_models(
				models, parameter_ranges, data, 
				features=features,
				initial=initial, lowlevel=lowlevel,
				train_size=train_size, plot=False)
			dictionary = {
				"Model": model_names, 
				"Train MAE (log M)": np.round(avg_train_mae,3), 
				"Test MAE (log M)": np.round(avg_test_mae,3), 
				"Train STD (log M)": np.round(avg_train_sd,3),
				"Test STD (log M)": np.round(avg_test_sd,3),
				"Train Pearson's R": np.round(avg_train_pearsonr,3),
				"Test Pearson's R": np.round(avg_test_pearsonr,3),
				"Train Spearman's R": np.round(avg_train_spearmanr,3),
				"Test Spearman's R": np.round(avg_test_spearmanr,3),
				"Train Kendall's Tau": np.round(avg_train_kendalltau,3),
				"Test Kendall's Tau": np.round(avg_test_kendalltau,3),
			}
			dataframe = pd.DataFrame(dictionary)
			display(dataframe)
			init = "_"+initial
			if initial == "top": init += "_"+lowlevel+"_P90"
			elif initial == "random": init = ""
			savefile = dataset.split("_")[0]+"_"+"_".join(features)+"_"+str(train_size)+init+".csv"
			savedir = os.path.join(target,"supervised_learning")
			if not os.path.exists(savedir): os.makedirs(savedir)
			dataframe.to_csv(os.path.join(savedir,savefile),index=False)
			
