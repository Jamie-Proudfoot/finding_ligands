#%%
import os
import math
import time
import random
import logging
import traceback
from pathlib import Path
from zipfile import ZipFile
from tempfile import TemporaryDirectory

from IPython.display import display, SVG

import numpy as np
import pandas as pd

from chembl_webresource_client.new_client import new_client
from tqdm.auto import tqdm

from rdkit import Chem
from rdkit import rdBase
from rdkit import RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors, Draw, PandasTools, MACCSkeys, AllChem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
rdBase.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.*')

from unipressed import IdMappingClient

seed=42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)

#%%

def get_chembl_id(uniprot_id):
    targets_api = new_client.target
    targets = targets_api.get(target_components__accession=uniprot_id).only("target_chembl_id")
    target = targets[0]
    chembl_id = target["target_chembl_id"]
    print(f"ChEMBL ID: {chembl_id}")
    return chembl_id

def get_bioactivities(chembl_id,Y):
    bioactivities_api = new_client.activity
    bioactivities = bioactivities_api.filter(
        target_chembl_id=chembl_id, standard_type=Y).only(
        "molecule_chembl_id","standard_units","standard_value",
        "relation","activity_comment",) # data_validity_description
    print(f"Fetching bioactivities for {chembl_id} ...")
    df = pd.DataFrame.from_dict(tqdm(bioactivities))
    df.drop(["units","value"], axis=1, inplace=True)
    df = df.astype({"standard_value": "float64"})
    df = df.rename({"standard_value": Y, "standard_units": "units"}, axis="columns")
    # df["data_validity_description"] = df["data_validity_description"].fillna("No comment")
    display(df)
    print(f"Initial dataset size: {df.shape[0]}")
    print("Infer relation based on activity comments")
    df.loc[df.activity_comment.str.contains("inhibition < 50% @ 10 uM",na=False), 
           ['relation',Y,'units']] = ('>',10000,'nM')
    print("Set activity comment for missing values based on relation")
    df.loc[(df['activity_comment'].isna()) & (df['relation'] == '=') , 
           'activity_comment'] = 'active'
    df.loc[(df['activity_comment'].isna()) & (df['relation'] == '>') , 
           'activity_comment'] = 'inactive'
    df.dropna(axis=0, how="any", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Size after dropping NaN values: {df.shape[0]}")
    df = df[df["units"] == "nM"]
    df.reset_index(drop=True, inplace=True)
    print(f"Size after dropping non-standard (nM) units: {df.shape[0]}")
    df = df[df[Y] > 0]
    df.reset_index(drop=True, inplace=True)
    print(f"Size after dropping non-positive values: {df.shape[0]}")
    df["p"+Y] = df[Y].apply(lambda x: 9-np.log10(x))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['active'] = df['relation'].apply(lambda x: 1 if x == '=' else 0)
    # df.drop_duplicates("molecule_chembl_id", keep="first", inplace=True)
    # df.reset_index(drop=True, inplace=True)
    print(f"Size after dropping duplicate molecules: {df.shape[0]}")
    display(df)
    return df

def get_compounds(bioactivities_df):
    compounds_api = new_client.molecule
    compounds_provider = compounds_api.filter(
        molecule_chembl_id__in=list(bioactivities_df["molecule_chembl_id"])
    ).only("molecule_chembl_id","molecule_structures","molecule_properties")
    print(f"Fetching compounds ...")
    df = pd.DataFrame.from_records(list(tqdm(compounds_provider)))
    display(df)
    # Extract smiles strings
    df['smiles'] = df.loc[
        df['molecule_structures'].notnull(),
        'molecule_structures'].apply(lambda x: x['canonical_smiles'])
    df.drop("molecule_structures", axis=1, inplace=True)
    print(f"Dataset size after filtering NaN molecules: {df.shape[0]}")
    # Extract Ro5 violations data
    df['ro5_violations'] = df.loc[
        df['molecule_properties'].notnull(),
        'molecule_properties'].apply(lambda x: x['num_ro5_violations'])
    df['ro5_violations'] = df['ro5_violations'].fillna(0)
    df.drop("molecule_properties", axis=1, inplace=True)
    print(f"Initial dataset size: {df.shape[0]}")
    df.dropna(axis=0, how="any", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Dataset size after filtering NaN values: {df.shape[0]}")
    df.drop_duplicates("molecule_chembl_id", keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Dataset size after dropping duplicates: {df.shape[0]}")
    df = filter(df)
    # Must repeat the process with salt-separated SMILES
    PandasTools.AddMoleculeColumnToFrame(df, "smiles")
    display(df)
    return df

def filter(df):
    PandasTools.AddMoleculeColumnToFrame(df, "smiles")
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)
    remover = SaltRemover()
    df["smiles"] = df["ROMol"].apply(lambda x: Chem.MolToSmiles(remover.StripMol(x)))
    df = df[~df["smiles"].apply(lambda x: '.' in x)]
    print(f"Size after dropping multi-fragment molecules: {df.shape[0]}")
    # print(f"Filtering compounds by PAINS matches ...")
    # df = df[~df["ROMol"].progress_apply(catalog.HasMatch)]
    # print(f"Size after dropping PAINS hits: {df.shape[0]}")
    df = df.drop("ROMol", axis=1)
    df.reset_index(drop=True, inplace=True)
    return df

def get_2D_descriptors(mol):
    """
    Calculate the full set of 2D RDKit descriptors for a molecule
    missingVal is used if the descriptor cannot be calculated
    """
    res = {}
    for nm, fn in Descriptors._descList:
        try: val = fn(mol)
        except:
            traceback.print_exc()
            val = None
        val = fn(mol)
        res[nm] = val
    return res

def get_MACCS(mol):
    """
    Calculate MACCS keys (166-bit)
    """
    maccs = {f"maccs_{i}": xi for i, xi in enumerate(MACCSkeys.GenMACCSKeys(mol))}
    return maccs

def get_Morgan2(mol):
    """
    Calculate radius-2 Morgan fingerprints
    """
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
    morgan2 = {f"morgan2_{i}": xi for i, xi in enumerate(fpgen.GetFingerprint(mol))}
    return morgan2

def get_Morgan3(mol):
    """
    Calculate radius-3 Morgan fingerprints
    """
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=2048)
    morgan3 = {f"morgan3_{i}": xi for i, xi in enumerate(fpgen.GetFingerprint(mol))}
    return morgan3

def get_rdkitfp(mol):
    """
    Calculate RDKit fingerprints
    """
    fpgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=1024)
    rdkitfp = {f"rdkitfp_{i}": xi for i, xi in enumerate(fpgen.GetFingerprint(mol))}
    return rdkitfp

def api_calls(uniprot_id, output_csv, Y):
    """
    Acquire data from ChEMBL using the
    UniProt ID of a specific assay target
    """
    tqdm.pandas()
    print(f"UniProt ID: {uniprot_id}")
    chembl_id = get_chembl_id(uniprot_id)
    bioactivities_df = get_bioactivities(chembl_id,Y)
    compounds_df = get_compounds(bioactivities_df)
    df = pd.merge(
        compounds_df[["molecule_chembl_id","smiles","ro5_violations","ROMol"]], #qed_weighted
        bioactivities_df[["molecule_chembl_id","relation",Y,"units",
                          "activity_comment","p"+Y,"active"]], #data_validity_description
        on="molecule_chembl_id",
    )
    # Drop duplicates by taking the maximum reported values
    maxima = []
    _, idx = np.unique(df['molecule_chembl_id'].values, return_index=True)
    smiles = df['smiles'].values[np.sort(idx)]
    for smi in tqdm(smiles[:]):
        mol_df = df[df['smiles'] == smi]
        # If "=" equalities are available, ignore ineqaulities
        if mol_df['relation'].str.contains('=').any(): 
            mol_df = mol_df[mol_df['relation'] == '=']
            id = mol_df.index[np.argmax(mol_df["p"+Y].values)]
        # If only ">" inequalities, take lowest pY (highest Y) value
        else: id = mol_df.index[np.argmin(mol_df["p"+Y].values)]
        maxima.append(id)
    df = df.iloc[maxima]
    df.reset_index(drop=True, inplace=True)
    print(f"Size after dropping duplicate molecules: {df.shape[0]}")
    display(df)
    print(f"Shape before adding descriptors: {df.shape}")
    df.dropna(axis=0, how="any", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Computing 2D RDKit descriptors ...")
    rdkit2d = pd.DataFrame([get_2D_descriptors(m) for m in tqdm(df["ROMol"])])
    df = df.join(rdkit2d)
    df.dropna(axis=0, how="any", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Shape after adding 2D descriptors: {df.shape}")
    print(f"Computing MACCS keys ...")
    maccs = pd.DataFrame([get_MACCS(m) for m in tqdm(df["ROMol"])])
    df = df.join(maccs)
    print(f"Computing radius-3 Morgan fingerprints ...")
    morgan3 = pd.DataFrame([get_Morgan3(m) for m in tqdm(df["ROMol"])])
    df = df.join(morgan3)
    print(f"Shape after adding FP descriptors: {df.shape}")
    print(f"Saving data to {output_csv}")
    df = df.drop("ROMol", axis=1)
    df.dropna(axis=0, how="any", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(output_csv,index=False)


#%%

if __name__ == "__main__":
    Y = "Ki"
    target = "EGFR"
    request = IdMappingClient.submit(source="GeneCards",dest="UniProtKB",ids={target})
    time.sleep(2.0)
    uniprot_id = list(request.each_result())[0]["to"]
    output_csv = os.path.join(target,f"{target}_data_p{Y}.csv")
    api_calls(uniprot_id, output_csv, Y)

#%%
