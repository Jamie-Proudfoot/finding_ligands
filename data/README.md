Folder containing csv files of ligand data for each target  
Produced by the following process for ChEMBL data:  
1. (py3env) gen_chembl.py  
2. (py3env) cluster.py  
3. (py3env) gen_dockstring.py (e.g. via submit_docking.sh)  
4. (DXGB) gen_delta.py (e.g. via submit_rescore.sh)  

Note: Some files may need to be moved to the correct folder for the next script between steps
