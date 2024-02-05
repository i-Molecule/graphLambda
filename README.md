# graphLambda: Binding Affinity Prediction Using Graph Neural Networks.
This repo contains is about graphLambda, a deep learning model to score the binding affinity of protein-ligand complexes in PyTorch and PyTorch Geometric, developed by Ghaith Mqawass and Petr Popov.
![alt text](https://github.com/i-Molecule/graphLambda/blob/main/illustration.jpg)

## Overview

We provide the implementation of the graphLambda model in [Pytorch](https://github.com/pytorch/pytorch) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) frameworks, along with the scripts that can be used to train the model and also replicate the results. All data used in this work can be downloaded from the [data_link](https://drive.google.com/drive/folders/1SYPxp2RqG68Q8cUWDVKL4wkeSM_RG3_z?usp=sharing) The repository is orignaized as follows:

- `models` contains:
  -  various GNN models implemented in Pytorch. All possible combinations of (GCN,GAT,GIN) are provided. 
  - `graphLambda.py` : The overall model implemented in Pytorch.


- `Data` contains:
  - `Dataset.py` : Dataset class that combines pre-computed BPS features. This dataset is to be passed to the dataloader to train the model.
  - `data.txt` : Description of the used data and benchmark. Also links to download the data are provided.
  - `refined_data2020.csv`: A **csv** file that contains **PDB** codes of protein-ligand complexes with the expiremental binding affinity.
  - `QSAR_set1.csv` , `QSAR_set1.csv`  and `coreset2016.csv` : **csv** files of used benchmarks containing  **PDB** codes of protein-ligand complexes with the expiremental binding affinity.


- `BPS_features.py` : A python script that computes BPS features. 
 
## Prepare the environment:

```sh
$ conda env create -f environment.yml
$ source activate myenv
$ conda env list
```
## Training the model:
- Download the refined set from the official website of PDBBind [link](http://www.pdbbind.org.cn/index.php)
- Compute BPS features using `BPS_features.py` (Or you can download pre-calculated features along with train/validation ids from Zenodo: https://???
- In the directory "refined_set" place the downloaded "refined_set.h5" file and then run the notebook `graphLambda_train.ipynb`
## Using the model:
- The final models can be downloaded from Zenodo: https://??? 
- To replicate the results you need to:
  - Download the test set "coreset" from Zenodo: https://???  and place it in the same directory of the notebook. The downloaded folder contains the coreset from PDBbind with BPS features precomputed using `BPS_features.py` and stored `*.h5 file` . You need to load paths to the work directory and *.h5 file and coreset2016.csv file in the notebook graphLambda_train. Some preprcessing was already carried out to the original data :  
  - Preprocessed the PDB samples by removing water molecules. 
  - Generated bps_features.h5 file using`BPS_features.py`.
  For inference you can use the script  `graphLambda_inference.py`. An example of how to use it in inference is provided in the `CACHE_Challenge` directory.


