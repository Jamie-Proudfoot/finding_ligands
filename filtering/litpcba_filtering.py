
#%%

import os, sys
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import time
import requests
from rcsbapi.data import DataQuery as Query
from rcsbapi.search import Sort
from unipressed import IdMappingClient
from rdkit import Chem

#%%

redocking_IDs = []
with open("rd_input_pairs.txt","r") as f:
    lines = f.readlines()
    for line in lines:
        data = line.split()
        protein_id = data[0].split("/")[1]
        ligand_id = protein_id
        redocking_IDs.append((protein_id, ligand_id))
redocking_IDs = np.array(redocking_IDs)

#%%

crossdocking_IDs = []
with open("ds_cd_input_pairs.txt","r") as f:
    lines = f.readlines()
    for line in lines:
        data = line.split()
        protein_id = data[0].split("/")[-1].split("_")[0]
        ligand_id = data[1].split("/")[-1].split("_")[0]
        crossdocking_IDs.append((protein_id, ligand_id))
crossdocking_IDs = np.array(crossdocking_IDs)

#%%

targets = {
    "ESR1ago":"2QZO", "ESR1ant": "5UFX",
    "PPARG": "3B1M", "TP53": "3ZME",
}
target_IDs = list(targets.values())

#%%

import time
import requests
from rcsbapi.data import DataQuery as Query
from rcsbapi.search import Sort
from unipressed import IdMappingClient
from rdkit import Chem


def get_uniprot_IDs(pdb_IDs):
    query = Query(
        input_type="polymer_entities",
        # Select the first (main) polymer in each pdb entry
        input_ids=list(map(lambda x: x+"_1", pdb_IDs)),
        return_data_list=[
            "polymer_entities.rcsb_polymer_entity_align.reference_database_accession"
        ],
    )
    result_dict = query.exec()
    uniprot_IDs = {}
    # Collect UniProt ID
    for entry in result_dict["data"]["polymer_entities"]:
        p_entity = entry["rcsb_polymer_entity_align"]
        if p_entity:
            p = p_entity[0]["reference_database_accession"]
            # uniprot_IDs.append(p)
            uniprot_IDs[entry["rcsb_id"][:-2]] = p
        # else: uniprot_IDs.append(None)
        else: uniprot_IDs[entry["rcsb_id"][:-2]] = None
    return uniprot_IDs

def get_ligand_IDs(pdb_IDs):
    query = Query(
        input_type="polymer_entities",
        # Select the first (main) polymer in each pdb entry
        input_ids=list(map(lambda x: x+"_1", pdb_IDs)),
        return_data_list=[
            "entry.nonpolymer_entities.rcsb_nonpolymer_entity_container_identifiers.nonpolymer_comp_id"
        ]
    )
    result_dict = query.exec()
    ligand_IDs = {}
    # print(result_dict["data"])
    for entry in result_dict["data"]["polymer_entities"]:
        p_entity = entry["entry"]["nonpolymer_entities"]
        # if not p_entity: pass
        if p_entity:
            ret = []
            # Collect all ligand IDs (PDB has variable ligand order)
            for ent in p_entity:
                p = ent[
                    "rcsb_nonpolymer_entity_container_identifiers"
                    ]["nonpolymer_comp_id"]
                # ligand_IDs.append(p)
                ret.append(p)
            ligand_IDs[entry["rcsb_id"][:-2]] = ret
        else: ligand_IDs[entry["rcsb_id"][:-2]] = []
    return ligand_IDs

def ligandID_to_smiles(lig_id):
    if not lig_id: return None
    r = requests.get(
        f"https://www.ebi.ac.uk/pdbe/api/pdb/compound/summary/{lig_id}"
    )
    # Convert ligand ID to SMILES using RDKit
    if lig_id in r.json().keys():
        # Collect InChi
        smi = r.json()[lig_id][0]["smiles"][0]["name"]
        mol = Chem.MolFromSmiles(smi)
        # RDKit returns Canonical SMILES
        if mol: return Chem.MolToSmiles(mol)

def uniprotID_to_geneID(uniprot_IDs):
    request = IdMappingClient.submit(
    source="UniProtKB_AC-ID", dest="Gene_Name", ids=uniprot_IDs
        )
    while True:
        status = request.get_status()
        if status in {"FINISHED", "ERROR"}: break
        else: time.sleep(5)
    return {x["from"]: str.upper(x["to"]) for x in request.each_result()}
    #return [str.upper(x["to"]) for x in request.each_result()]

#%%

# PDB_ID overlap
print(set(redocking_IDs[:,0]) & set(target_IDs))
print(set(crossdocking_IDs[:,0]) & set(target_IDs))

#%%

target_uniprotIDs = get_uniprot_IDs(np.unique(target_IDs))
redocking_uniprotIDs = get_uniprot_IDs(np.unique(redocking_IDs[:,0]))
crossdocking_uniprotIDs = get_uniprot_IDs(np.unique(crossdocking_IDs[:,0]))

#%%

# UniprotID and GeneID overlap
redocking_overlap = set(redocking_uniprotIDs.values()) & set(target_uniprotIDs.values())
redocking_overlap_names = set(uniprotID_to_geneID(redocking_overlap).values())
# print(redocking_overlap_names)
crossdocking_overlap = set(crossdocking_uniprotIDs.values()) & set(target_uniprotIDs.values())
crossdocking_overlap_names = set(uniprotID_to_geneID(crossdocking_overlap).values())
# print(crossdocking_overlap_names)
# GeneID overlap
total_overlap = redocking_overlap_names | crossdocking_overlap_names
print(total_overlap)
total_unseen = set(["ESR1","TP53","PPARG"]) ^ total_overlap
print(total_unseen)

#%%

# Create redocking dataframe 
redocking_df = pd.DataFrame(redocking_uniprotIDs.items(),columns=["pdb","UniprotID"])
redocking_geneIDs = uniprotID_to_geneID(np.unique([x for x in redocking_uniprotIDs.values() if x]))
print("Finding GeneIDs...")
redocking_df["GeneID"] = [
    redocking_geneIDs[x] if x in redocking_geneIDs.keys() else None 
    for x in redocking_df["UniprotID"].values
]
print("Finding LigandIDs...")
redocking_ligandIDs = get_ligand_IDs(np.unique(redocking_IDs))
redocking_df["LigandIDs"] = [
    redocking_ligandIDs[x] if x in redocking_ligandIDs.keys() else [] 
    for x in redocking_df["pdb"].values
]
redocking_df["LigandIDs"] = [" ".join([v for v in x if v]) for x in redocking_df["LigandIDs"].values]
print("Finding SMILES...")
redocking_SMILES = {}
for ligand_id in tqdm(np.unique([v for x in redocking_ligandIDs.values() for v in x])):
    redocking_SMILES[ligand_id] = ligandID_to_smiles(ligand_id)
redocking_df["SMILES"] = [
    [redocking_SMILES[v] if v in redocking_SMILES.keys() else None for v in x.split()]
    for x in redocking_df["LigandIDs"].values
]
redocking_df["SMILES"] = [" ".join([v for v in x if v]) for x in redocking_df["SMILES"].values]
redocking_df.to_csv("redocking_train_labelled.csv",index=False)

#%%

# Collect redocking ligand SMILES
redocking_SMILES_unique = np.unique([v for x in redocking_df["SMILES"]for v in x.split() if v])
with open("redocking_train.smi","w+") as f:
    for smi in redocking_SMILES_unique:
        if smi: f.write(smi+"\n")

#%%

# Create crossdocking dataframe 
crossdocking_df = pd.DataFrame(crossdocking_IDs,columns=["prot_pdb","lig_pdb"])
crossdocking_df["UniprotID"] = [
    crossdocking_uniprotIDs[x] if x in crossdocking_uniprotIDs.keys() else None 
    for x in crossdocking_df["prot_pdb"].values
]
print("Finding GeneIDs...")
crossdocking_geneIDs = uniprotID_to_geneID(np.unique([x for x in crossdocking_uniprotIDs.values() if x]))
crossdocking_df["GeneID"] = [
    crossdocking_geneIDs[x] if x in crossdocking_geneIDs.keys() else None 
    for x in crossdocking_df["UniprotID"].values
]
print("Finding LigandIDs...")
crossdocking_ligandIDs = get_ligand_IDs(np.unique(crossdocking_IDs[:,1]))
crossdocking_df["LigandIDs"] = [
    crossdocking_ligandIDs[x] if x in crossdocking_ligandIDs.keys() else [] 
    for x in crossdocking_df["lig_pdb"].values
]
crossdocking_df["LigandIDs"] = [" ".join([v for v in x if v]) for x in crossdocking_df["LigandIDs"].values]
print("Finding SMILES...")
crossdocking_SMILES = {}
for ligand_id in tqdm(np.unique([v for x in crossdocking_ligandIDs.values() for v in x])):
    crossdocking_SMILES[ligand_id] = ligandID_to_smiles(ligand_id)
crossdocking_df["SMILES"] = [
    [crossdocking_SMILES[v] if v in crossdocking_SMILES.keys() else None for v in x.split()]
    for x in crossdocking_df["LigandIDs"].values
]
crossdocking_df["SMILES"] = [" ".join([v for v in x if v]) for x in crossdocking_df["SMILES"].values]
crossdocking_df.to_csv("crossdocking_train_labelled.csv",index=False)

#%%

# Collect crossdocking ligand SMILES
crossdocking_SMILES_unique = np.unique([v for x in crossdocking_df["SMILES"] for v in x.split() if v])
with open("crossdocking_train.smi","w+") as f:
    for smi in crossdocking_SMILES_unique:
        if smi: f.write(smi+"\n")

#%%

# mode = "any": any ligands in the redocking / crossdocking training set regardless of target
# mode = "target": any ligands for the given target in the redocking / crossdocking training set

redocking_df = pd.read_csv("redocking_train_labelled.csv")
crossdocking_df = pd.read_csv("crossdocking_train_labelled.csv")
redocking_SMILES_unique = np.unique([v for x in redocking_df["SMILES"].values.astype(str) for v in x.split() if v])
crossdocking_SMILES_unique = np.unique([v for x in crossdocking_df["SMILES"].values.astype(str) for v in x.split() if v])
training_SMILES = set(redocking_SMILES_unique) | set(crossdocking_SMILES_unique)

mode = "target" # "any"

for target in targets.keys():
    data_csv = os.path.join("..","workdir","data",f"{target}_data_full.csv")
    if target in ["ESR1ago","ESR1ant"]: geneID = "ESR1"
    else: geneID = target
    df = pd.read_csv(data_csv)
    smiles = [Chem.CanonSmiles(smi) for smi in df["PUBCHEM_EXT_DATASOURCE_SMILES"].values] # load smiles and make canonical
    redocking_target_SMILES = [v for x in redocking_df[redocking_df["GeneID"]==geneID]["SMILES"].values for v in x.split()]
    crossdocking_target_SMILES = [v for x in crossdocking_df[crossdocking_df["GeneID"]==geneID]["SMILES"].values for v in x.split()]
    training_target_SMILES = set(redocking_target_SMILES) | set(crossdocking_target_SMILES)
    if mode == "target": filter_SMILES = training_target_SMILES
    elif mode == "any": filter_SMILES  = training_SMILES 
    else: filter_SMILES = training_SMILES
    overlapping = []
    for smi in smiles:
        if smi in filter_SMILES: overlapping.append(smi)
    print(target, len(overlapping), len(overlapping)/len(df) * 100)
    print(overlapping)
    if not os.path.exists("data_filtered"): os.mkdir("data_filtered")
    if len(overlapping) > 0:
        df = df[~pd.Series(smiles).isin(filter_SMILES)]
        df.to_csv(os.path.join(f"data_filtered",f"{target}_filtered_{mode}.csv"),index=False)
    print()

#%%