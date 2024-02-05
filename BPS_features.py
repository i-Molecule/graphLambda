import numpy as np
from collections import namedtuple
from math import cos, pi
from copy import copy, deepcopy
from sklearn.neighbors import KDTree
from itertools import product
import os
import deepdish as dd
from tqdm import tqdm
from multiprocessing import Pool
'''
Before runnning the script:
1- make sure you have the lignads saved in .xyz format with Hydrogen atoms removed
2- Remove water molecules from the PDB files.
3- Specify the correct paths to lignads and complexes.pdb in the function (compute)
'''
Mol = namedtuple('Mol', ['symbols', 'coords', 'message'])
idk=0
#symbols is a list of atomic name to be met in the pdb
#kdtrees - is a dictionary of the KD-trees for each element

PDB = namedtuple('PDB', ['symbols', 'kdtrees', 'coords'])


def read_xyz(filename):
    f = open(filename, 'r')
    data = f.readlines()
    f.close()
    print("read_xyz: ",filename)
    message = data[1].rstrip("\n")
    symbols = []
    coords = np.array([]).reshape((0, 3)).astype(float)
    for item in data[2:]:
        item = item.strip("\n")
        item = item.split(" ")
        item = list(filter(None, item))
        symbols.append(item[0])
        coords = np.concatenate((coords, np.array(item[1:]).reshape((1,3)).astype(float)), 0)
    mol = Mol(symbols = symbols, coords = coords, message = message)
    mol.coords.astype(float)
    return mol

def read_pdb(filename):
    global uk
    f = open(filename, 'r')
    data = f.readlines()
    f.close()
    print("read_pdb: ",filename)
    # retain only lines containing atomic coordinates
    data = [line.rstrip("\n") for line in data if line.startswith("ATOM") or line.startswith("HETATM")]
    symbols = ['C', 'O', 'N', 'S', 'P', ,'H','M1', 'M2']
    coords = {'C': [], 'O': [], 'N': [], 'S': [], 'P': [],'H': [], 'M1': [], 'M2': []}
    # read and keep coordinate per atom type
    for line in data:
        x = float(line[30:37])
        y = float(line[38:45])
        z = float(line[46:53])

        if len(line) < 80 or line[76:78] == "  ":
            if line.startswith("ATOM") or line[17:20] in ["CSO", "IAS", "PTR", "TPO", "SEP", "ACE", "PCA", "CSD", "LLP", "KCX"]:
                symbol = deepcopy(line[12:16]).strip()[0]
            elif line[17:20] == "MSE":
                symbol = deepcopy(line[12:16]).strip()
                if symbol != "SE":
                    symbol = symbol[0]
            else:
                symbol = deepcopy(line[12:16]).strip()

        elif line[76] == " ":
            symbol = line[77]

        else:
            symbol = line[76:78]

        if symbol in ['C', 'O', 'N', 'S', 'P','H']:
            coords[symbol].append([x, y, z])
        elif symbol in ['LI', 'NA', 'K', 'RB', 'CS']:
            coords['M1'].append([x, y, z])
        elif symbol in ['AU', 'BA', 'CA', 'CD', 'SR', 'BE', 'CO', 'MG', 'CU', 'FE', 'HG', 'MN', 'NI', 'ZN']:
            coords['M2'].append([x, y, z])
        elif symbol == 'SE':
            coords['S'].append([x, y, z])
        else:
            print(symbol, line[17:20])
            print("**************************** ",filename)
          #  raise ValueError("Unknown atomic symbol in pdb")

    kdtrees = {}

    # create a dict of KD trees
    for atom_type in symbols:
        coords[atom_type] = np.array(coords[atom_type])
        if len(coords[atom_type]) != 0:
            kdtrees[atom_type] = KDTree(coords[atom_type], leaf_size=10)

    mol = PDB(symbols=symbols, kdtrees=kdtrees, coords=coords)

    return mol


def f_c(r, rc):
    return np.tanh(1 - r / rc) ** 3
     

def BP_function(mol, pdb, rc = 16.0, rs_list = [2., 4., 6., 8., 10.], eta_list = [0.008, 0.04, 0.2, 1.], zeta_list = [1., 2., 4., 8.]):
    out = []
    global idk
    idk=idk+1
    for atom_type in pdb.coords.keys():
        # check if the atom is in the structure
        # if not add zeros for the vector
        if len(pdb.coords[atom_type]) != 0:
            ind, distances = pdb.kdtrees[atom_type].query_radius(mol.coords, rc, return_distance = True)
            atom_type_list = []
            for i in range(mol.coords.shape[0]):
                atom_env_coords = pdb.coords[atom_type][ind[i], :]
                current_distances = distances[i]
                atom_lig_coords = mol.coords[i]

                desc = []

                # compute all cosines and distances
                a1 = np.tile(atom_env_coords, (atom_env_coords.shape[0],1))
                a2 = np.repeat(atom_env_coords, atom_env_coords.shape[0], axis = 0)
                r_ij = np.sum((atom_lig_coords - a1)**2, axis=1) ** 0.5
                r_ik = np.sum((atom_lig_coords - a2)**2, axis=1) ** 0.5
                r_jk = np.sum((a1 - a2)**2, axis=1) ** 0.5
                cos_teta_ijk = np.sum((atom_lig_coords - a1) * (atom_lig_coords - a2), axis = 1) / r_ij / r_ik            
             
                # computing radial distribution functions
                for rs, eta in product(rs_list, eta_list):
                    temp = np.sum(np.exp(-eta*(current_distances - rs)**2) * f_c(current_distances, rc))
                    desc.append(temp)
            
                # computing angular part
                l = 1
                for eta, zeta in product(eta_list, zeta_list):
                    temp = 2**(1-zeta) * np.sum((1 + l * cos_teta_ijk)**zeta * np.exp(-eta*(r_ij**2 + r_ik**2 + r_jk**2)) * f_c(r_ij, rc) * f_c(r_ik, rc) * f_c(r_jk, rc))
                    desc.append(temp)
                
                l = -1
                for eta, zeta in product(eta_list, zeta_list):
                    temp = 2**(1-zeta) * np.sum((1 + l * cos_teta_ijk)**zeta * np.exp(-eta*(r_ij**2 + r_ik**2 + r_jk**2)) * f_c(r_ij, rc) * f_c(r_ik, rc) * f_c(r_jk, rc))
                    desc.append(temp)
            
                atom_type_list.append(desc)
                
        else:
            atom_type_list = []
            #if atom is absent is absent in pdb then we will fill its descriptor field with zeros
            for i in range(mol.coords.shape[0]):
                atom_type_list.append([0.0] * (len(rs_list) * len(eta_list) + 2 * len(eta_list) * len(zeta_list)))
        out.append(atom_type_list)
        
    out = np.transpose(np.array(out), (1, 0, 2))
    # add to the descriptors atomtype information to the beginning
    
    lig_atom_types = ['C', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I','H']
    out1 = np.zeros((len(mol.symbols), len(lig_atom_types)))
    
    for i in range(mol.coords.shape[0]):
        out1[i, lig_atom_types.index(mol.symbols[i])] += 1.
    out = out.reshape(out.shape[0], out.shape[1] * out.shape[2])
    
    return np.concatenate((out1, out), 1).astype(np.float32)



def compute(pdb_id):
    global idk
    mol = read_xyz("./" + pdb_id + "/" + pdb_id + ".xyz")  #specify the paths to lignads saved in .xyz format 
    pdb = read_pdb("./" + pdb_id + "/" + pdb_id + "_nowaterr.pdb") #Specify the paths to proteins contained in files: complex_code.pdb 
    out = BP_function(mol, pdb)
    return pdb_id, out


def main(): 
    p = Pool(6) 
    codes = [code for code in os.listdir() if len(code) == 4]
    data = p.map(compute, codes)
    print(" computation is finished")
    data = {key: value for key, value in data}
    dd.io.save('refined_set_features.h5', data)


if __name__ == "__main__":   
    main()









