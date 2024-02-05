''' Before running the script to score complexes make sure to place it in the same directory 
where the file containing docked molecules (*.sdf) and the protein file (*.pdb)  
and the model's saved state (*.pt) and then replace the paths in the main function.
'''
import torch
import torch_geometric
import deepdish as dd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tqdm import tqdm
import copy
from os.path import join
from rdkit.Chem import ChemicalFeatures, AllChem
from rdkit import RDConfig
from rdkit.Chem.rdmolfiles import MolFromMolFile
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import random
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, BatchNorm1d, Dropout
import torch_geometric.transforms as T
from torch_geometric.nn import NNConv, Set2Set, GCNConv, global_add_pool, global_mean_pool,GATConv,GINConv
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Data
from rdkit import Chem
from collections import namedtuple
from math import cos, pi
from copy import copy, deepcopy
from sklearn.neighbors import KDTree
from itertools import product
import os
import deepdish as dd
from tqdm import tqdm
from multiprocessing import Pool



###############################################################
Mol = namedtuple('Mol', ['symbols', 'coords', 'message'])

#symbols is a list of atomic name to be met in the pdb
#kdtrees - is a dictionary of the KD-trees for each element

PDB = namedtuple('PDB', ['symbols', 'kdtrees', 'coords'])


def read_xyz(filename):
    f = open(filename, 'r')
    data = f.readlines()
    f.close()
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
    f = open(filename, 'r')
    data = f.readlines()
    f.close()
    # retain only lines containing atomic coordinates
    data = [line.rstrip("\n") for line in data if line.startswith("ATOM") or line.startswith("HETATM")]
    symbols = ['C', 'O', 'N', 'S', 'P', 'M1', 'M2']
    coords = {'C': [], 'O': [], 'N': [], 'S': [], 'P': [], 'M1': [], 'M2': []}
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

        if symbol in ['C', 'O', 'N', 'S', 'P']:
            coords[symbol].append([x, y, z])
        elif symbol in ['LI', 'NA', 'K', 'RB', 'CS']:
            coords['M1'].append([x, y, z])
        elif symbol in ['AU', 'BA', 'CA', 'CD', 'SR', 'BE', 'CO', 'MG', 'CU', 'FE', 'HG', 'MN', 'NI', 'ZN']:
            coords['M2'].append([x, y, z])
        elif symbol == 'SE':
            coords['S'].append([x, y, z])
        else:
            print(symbol, line[17:20])
#             raise ValueError("Unknown atomic symbol in pdb")

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
     

def BP_function(mol, pdb, rc = 12.0, rs_list = [2., 4., 6., 8., 10.], eta_list = [0.008, 0.04, 0.2, 1.], zeta_list = [1., 2., 4., 8.]):
    out = []
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
    
    lig_atom_types = ['C', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I']
    out1 = np.zeros((len(mol.symbols), len(lig_atom_types)))
    
    for i in range(mol.coords.shape[0]):
        out1[i, lig_atom_types.index(mol.symbols[i])] += 1.
    out = out.reshape(out.shape[0], out.shape[1] * out.shape[2])
    
    return np.concatenate((out1, out), 1).astype(np.float32)


def compute(pdb_id):
    mol = read_xyz(pdb_id)
    pdb = read_pdb("pronolig.pdb") # path to the protein pdb file 
    out = BP_function(mol, pdb)
    return pdb_id, out

######################################################################

def add_edges_list(mol):
#     print(join(root, code, ligand_filename))
    m = mol
    atoms1 = [b.GetBeginAtomIdx() for b in m.GetBonds()]
    atoms2 = [b.GetEndAtomIdx() for b in m.GetBonds()]    
    # Edge attributes: distance; SINGLE; DOUBLE; TRIPLE; AROMATIC.
    edge_weights= []
    coords = m.GetConformers()[0].GetPositions()  # Get a const reference to the vector of atom positions
    for b in m.GetBonds():
        if str(b.GetBondType()) == "SINGLE":
            edge_weights.append(1)
        elif str(b.GetBondType()) == "DOUBLE":
            edge_weights.append(2)
        elif str(b.GetBondType()) == "TRIPLE":
            edge_weights.append(3)
        else:
            edge_weights.append(4)
    edge_features = np.array(edge_weights) 
    # since the torch-geometric graphs are directed add reverse direction of edges
    return np.array([atoms1 + atoms2, atoms2 + atoms1])

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #GCN-representation
        self.conv1 = GCNConv(373, 256, cached=False )
        self.bn01 = BatchNorm1d(256)
        self.conv2 = GCNConv(256, 128, cached=False )
        self.bn02 = BatchNorm1d(128)
        self.conv3 = GCNConv(128, 128, cached=False)
        self.bn03 = BatchNorm1d(128)
        #GAT-representation
        self.gat1 = GATConv(373, 256,heads=3)
        self.bn11 = BatchNorm1d(256*3)
        self.gat2 = GATConv(256*3, 128,heads=3)
        self.bn12 = BatchNorm1d(128*3)
        self.gat3 = GATConv(128*3, 128,heads=3)
        self.bn13 = BatchNorm1d(128*3)
        #GIN-representation
        fc_gin1=Sequential(Linear(373, 256), ReLU(), Linear(256, 256))
        self.gin1 = GINConv(fc_gin1)
        self.bn21 = BatchNorm1d(256)
        fc_gin2=Sequential(Linear(256, 128), ReLU(), Linear(128, 128))
        self.gin2 = GINConv(fc_gin2)
        self.bn22 = BatchNorm1d(128)
        fc_gin3=Sequential(Linear(128, 64), ReLU(), Linear(64, 64))
        self.gin3 = GINConv(fc_gin3)
        self.bn23 = BatchNorm1d(64)
        #Fully connected layers for concatinating outputs
        self.fc1=Linear(128*4 + 64, 256)
        self.dropout1=Dropout(p=0.2,)
        self.fc2=Linear(256, 64)
        self.dropout2=Dropout(p=0.2,)
        self.fc3=Linear(64, 1)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        y=x
        z=x
        #GCN-representation
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn01(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn02(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn03(x)
        x = global_add_pool(x, data.batch)
        #GAT-representation
        y = F.relu(self.gat1(y, edge_index))
        y = self.bn11(y)
        y = F.relu(self.gat2(y, edge_index))
        y = self.bn12(y)
        y = F.relu(self.gat3(y, edge_index))
        y = self.bn13(y)
        y = global_add_pool(y, data.batch)
        #GIN-representation
        z = F.relu(self.gin1(z, edge_index))
        z = self.bn21(z)
        z = F.relu(self.gin2(z, edge_index))
        z = self.bn22(z)
        z = F.relu(self.gin3(z, edge_index))
        z = self.bn23(z)
        z = global_add_pool(z, data.batch)
        #Concatinating_representations
        cr=torch.cat((x,y,z),1)
        cr = F.relu(self.fc1(cr))
        cr = self.dropout1(cr)
        cr = F.relu(self.fc2(cr))
        cr = self.dropout2(cr)
        cr = self.fc3(cr)
        cr = F.relu(cr).view(-1)
        return cr  

@torch.no_grad()
def test_predictions(model, loader):
    device = 'cpu'
    model.eval()
    pred = []
    true = []
    for data in loader:
        data = data.to(device)
        pred += model(data).detach().cpu().numpy().tolist()
    return pred

#######################################################################



def main(): 
    path = "CACHE_challnege/CACHE_mols_docked.sdf" #Path to the sdf file that contains docked molecules
    mols = Chem.SDMolSupplier(path)  
    for idx,m in tqdm(enumerate(mols)):
        mol = Chem.RemoveHs(m) 
        molxyz = Chem.rdmolfiles.MolToXYZFile(mol,f'Ligand_{idx}.xyz')
    p = Pool(6) 
    codes = [code for code in os.listdir() if (code[-4:]) == ".xyz"]  # list of ligand files (*.xyz)
    data = p.map(compute, codes)
    data = {key: value for key, value in data}
    dd.io.save('bps_features.h5', data)
    ######################################################################
    features = dd.io.load('bps_features.h5') # Load the computed features
    graphs = []
    edges = []
    node_features = []
    for idx,m in enumerate(mols):
        edges.append(add_edges_list(m))
        node_features.append(features[f'Ligand_{idx}.xyz'])
    for i in range(len(node_features)):
        graphs.append(Data(x = torch.FloatTensor(node_features[i]),
                            edge_index = torch.LongTensor(edges[i])))
    test_loader = DataLoader(graphs, batch_size=1, shuffle=False)

    model= Net()
    model.load_state_dict(torch.load('models/GNNs_fusion_TM_split.pt')) #path to the saved model dict
    preds = test_predictions(model, test_loader)
    print("Binding constants for the given list of docked complexes:", preds)
    sourceFile = open('score.txt', 'w')
    print(preds, file = sourceFile)
    sourceFile.close()


if __name__ == "__main__":   
    main()


