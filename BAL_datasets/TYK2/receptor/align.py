

#%%


from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import py3Dmol


import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign

# Load molecules
ref_mol = Chem.MolFromXYZFile('lig_target.xyz')
prb_mol = Chem.MolFromXYZFile('lig_source.xyz')

#%%

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


# From here: https://hunterheidenreich.com/posts/kabsch-algorithm/
# Thank you ^_^ !
def kabsch_numpy(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.

    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    # Optimal translation
    t = centroid_Q - centroid_P

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = np.dot(p.T, q)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Validate right-handed coordinate system
    if np.linalg.det(np.dot(Vt.T, U.T)) < 0.0:
        Vt[-1, :] *= -1.0

    # Optimal rotation
    R = np.dot(Vt.T, U.T)

    # RMSD
    rmsd = np.sqrt(np.sum(np.square(np.dot(p, R.T) - q)) / P.shape[0])

    return R, t, rmsd


prb_coords = np.array([prb_mol.GetConformer().GetAtomPosition(i) for i in range(prb_mol.GetNumAtoms())])
ref_coords = np.array([ref_mol.GetConformer().GetAtomPosition(i) for i in range(ref_mol.GetNumAtoms())])
# Apply Kabsch algorithm for MSA
rotation_matrix, translation_vector, rmsd = kabsch_numpy(ref_coords, prb_coords)

print("Rotation Matrix:")
print(rotation_matrix)

print("Translation Vector:")
print(translation_vector)

transform = np.empty((4, 4))
transform[:3,:3] = rotation_matrix
transform[:3,3] = translation_vector

print("Transformation Matrix:")
print(transform)

print("RMSD:")
print(rmsd)

transformed_coords = np.dot(prb_coords - np.mean(prb_coords, axis=0), rotation_matrix) + np.mean(ref_coords, axis=0)

print("Transformed Probe Coordinates:")
print(transformed_coords)

#%%

# This also recovers (close to) ref_coord
prb_centroid=np.mean(prb_coords, axis=0)
ref_centroid=np.mean(ref_coords, axis=0)
(prb_coords-prb_centroid)@rotation_matrix + ref_centroid
# As does this
prb_coords@rotation_matrix + (ref_centroid - prb_centroid@rotation_matrix)

#%%

#%%

from Bio import PDB
import numpy as np

# Load structure
parser = PDB.PDBParser()
structure = parser.get_structure("my_protein", "4GIH.pdb")


true_translation = (ref_centroid-prb_centroid@rotation_matrix)

# Apply transformation
for atom in structure.get_atoms():
    # atom.transform(rotation_matrix, translation_vector)
    atom.transform(rotation_matrix, true_translation)

# Save result
io = PDB.PDBIO()
io.set_structure(structure)
io.save("4GIH_transformed.pdb")


#%%
