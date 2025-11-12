
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

LinF9_IDs = pd.read_csv("LinF9_train.csv")["pdb"].values.tolist()
DeltaLinF9XGB_IDs = pd.read_csv("DeltaLinF9XGB_train.csv")["pdb"].values.tolist()

#%%

targets = {
    "EGFR":"2rgp","JAK2":"3lpb","LCK":"2of2",
    "MAOB":"1s3b","NOS1":"1qw6","PARP1":"3l3m",
    "ACHE":"1e66","PDE5A":"1udt","PTGS2":"3ln1",
    "ESR1":"1sj0","NR3C1":"3bqd","AR":"2am9",
    "F10":"3kl6","ADRB2":"3ny8"
}
target_IDs = list(targets.values())

#%%


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
print(set(LinF9_IDs) & set(target_IDs))
print(set(DeltaLinF9XGB_IDs) & set(target_IDs))

#%%

target_uniprotIDs = get_uniprot_IDs(np.unique(target_IDs))
LinF9_uniprotIDs = get_uniprot_IDs(np.unique(LinF9_IDs))
DeltaLinF9XGB_uniprotIDs = get_uniprot_IDs(np.unique(DeltaLinF9XGB_IDs))

#%%

# UniprotID and GeneID overlap
LinF9_overlap = set(LinF9_uniprotIDs.values()) & set(target_uniprotIDs.values())
LinF9_overlap_names = set(uniprotID_to_geneID(LinF9_overlap).values())
# print(LinF9_overlap_names)
D9XGB_overlap = set(DeltaLinF9XGB_uniprotIDs.values()) & set(target_uniprotIDs.values())
D9XGB_overlap_names = set(uniprotID_to_geneID(D9XGB_overlap).values())
# print(D9XGB_overlap_names)
# GeneID overlap
total_overlap = D9XGB_overlap_names | LinF9_overlap_names
print(total_overlap)
total_unseen = set(targets.keys()) ^ total_overlap
print(total_unseen)

#%%

# Create LinF9 dataframe 
LinF9_df = pd.DataFrame(LinF9_uniprotIDs.items(),columns=["pdb","UniprotID"])
LinF9_geneIDs = uniprotID_to_geneID(np.unique([x for x in LinF9_uniprotIDs.values() if x]))
print("Finding GeneIDs...")
LinF9_df["GeneID"] = [
    LinF9_geneIDs[x] if x in LinF9_geneIDs.keys() else None 
    for x in LinF9_df["UniprotID"].values
]
print("Finding LigandIDs...")
LinF9_ligandIDs = get_ligand_IDs(np.unique(LinF9_IDs))
LinF9_df["LigandIDs"] = [
    LinF9_ligandIDs[x] if x in LinF9_ligandIDs.keys() else None 
    for x in LinF9_df["pdb"].values
]
LinF9_df["LigandIDs"] = [" ".join([v for v in x if v]) for x in LinF9_df["LigandIDs"].values]
print("Finding SMILES...")
LinF9_SMILES = {}
for ligand_id in tqdm(np.unique([v for x in LinF9_ligandIDs.values() for v in x])):
    LinF9_SMILES[ligand_id] = ligandID_to_smiles(ligand_id)
LinF9_df["SMILES"] = [
    [LinF9_SMILES[v] if v in LinF9_SMILES.keys() else None for v in x.split()]
    for x in LinF9_df["LigandIDs"].values
]
LinF9_df["SMILES"] = [" ".join([v for v in x if v]) for x in LinF9_df["SMILES"].values]
LinF9_df.to_csv("LinF9_train_labelled.csv",index=False)

#%%

# Collect LinF9 ligand SMILES
LinF9_SMILES_unique = np.unique([v for x in LinF9_df["SMILES"] for v in x.split() if v])
with open("LinF9_train.smi","w+") as f:
    for smi in LinF9_SMILES_unique:
        if smi: f.write(smi+"\n")

#%%

# Create DeltaLinF9XGB dataframe 
DeltaLinF9XGB_df = pd.DataFrame(DeltaLinF9XGB_uniprotIDs.items(),columns=["pdb","UniprotID"])
print("Finding GeneIDs...")
DeltaLinF9XGB_geneIDs = uniprotID_to_geneID(np.unique([x for x in DeltaLinF9XGB_uniprotIDs.values() if x]))
DeltaLinF9XGB_df["GeneID"] = [
    DeltaLinF9XGB_geneIDs[x] if x in DeltaLinF9XGB_geneIDs.keys() else None 
    for x in DeltaLinF9XGB_df["UniprotID"].values
]
print("Finding LigandIDs...")
DeltaLinF9XGB_ligandIDs = get_ligand_IDs(np.unique(DeltaLinF9XGB_IDs))
DeltaLinF9XGB_df["LigandIDs"] = [
    DeltaLinF9XGB_ligandIDs[x] if x in DeltaLinF9XGB_ligandIDs.keys() else None 
    for x in DeltaLinF9XGB_df["pdb"].values
]
DeltaLinF9XGB_df["LigandIDs"] = [" ".join([v for v in x if v]) for x in DeltaLinF9XGB_df["LigandIDs"].values]
print("Finding SMILES...")
DeltaLinF9XGB_SMILES = {}
for ligand_id in tqdm(np.unique([v for x in DeltaLinF9XGB_ligandIDs.values() for v in x])):
    DeltaLinF9XGB_SMILES[ligand_id] = ligandID_to_smiles(ligand_id)
DeltaLinF9XGB_df["SMILES"] = [
    [DeltaLinF9XGB_SMILES[v] if v in DeltaLinF9XGB_SMILES.keys() else None for v in x.split()]
    for x in DeltaLinF9XGB_df["LigandIDs"].values
]
DeltaLinF9XGB_df["SMILES"] = [" ".join([v for v in x if v]) for x in DeltaLinF9XGB_df["SMILES"].values]
DeltaLinF9XGB_df.to_csv("DeltaLinF9XGB_train_labelled.csv",index=False)

#%%

# Collect DeltaLinF9XGB ligand SMILES
DeltaLinF9XGB_SMILES_unique = np.unique([v for x in DeltaLinF9XGB_df["SMILES"] for v in x.split() if v])
with open("DeltaLinF9XGB_train.smi","w+") as f:
    for smi in DeltaLinF9XGB_SMILES_unique:
        if smi: f.write(smi+"\n")

#%%

# Mode = "any": any ligands in the LinF9 / DeltaLinF9XGB training set regardless of target
# Mode = "target": any ligands for the given target in the LinF9 / DeltaLinF9XGB training set

LinF9_df = pd.read_csv("LinF9_train_labelled.csv")
DeltaLinF9XGB_df = pd.read_csv("DeltaLinF9XGB_train_labelled.csv")
LinF9_SMILES_unique = np.unique([v for x in LinF9_df["SMILES"].values.astype(str) for v in x.split() if v])
DeltaLinF9XGB_SMILES_unique = np.unique([v for x in DeltaLinF9XGB_df["SMILES"].values.astype(str) for v in x.split() if v])
training_SMILES = set(LinF9_SMILES.values()) | set(DeltaLinF9XGB_SMILES.values())

mode = "target" # "any"

for target in targets.keys():
    data_csv = os.path.join("..","workdir","data",f"{target}-2048_data_3d_delta_pKi.csv")
    df = pd.read_csv(data_csv)
    smiles = [Chem.CanonSmiles(s) for s in df["smiles"].values] # canonical smiles from ChEMBL
    LinF9_target_SMILES = [v for x in LinF9_df[LinF9_df["GeneID"]==target]["SMILES"].values for v in x.split()]
    DeltaLinF9XGB_target_SMILES = [v for x in DeltaLinF9XGB_df[DeltaLinF9XGB_df["GeneID"]==target]["SMILES"].values for v in x.split()]
    training_target_SMILES = LinF9_target_SMILES + DeltaLinF9XGB_target_SMILES
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
        df = df[~df["smiles"].isin(filter_SMILES)]
        df.to_csv(os.path.join("data_filtered",f"{target}_filtered_{mode}.csv"),index=False)
    print()

#%%