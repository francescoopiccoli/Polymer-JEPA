"""Data loading and preprocessing for Polymer-JEPA.

This module handles:
- Loading polymer datasets (Aldeghi and diblock copolymer)
- Converting SMILES strings to molecular graphs
- Creating PyTorch Geometric datasets with transforms
- Data sampling and statistics

Supported datasets:
- Aldeghi: Conjugated copolymers with EA/IP properties
- Diblock: Diblock copolymers with phase behavior properties
"""

import collections
import os
import random
from typing import List, Optional, Tuple, Any, Dict

import numpy as np
import pandas as pd
import torch
import tqdm
from rdkit import Chem
from torch_geometric.data import InMemoryDataset

from src.featurization_utils.featurization import poly_smiles_to_graph
from src.transform import PositionalEncodingTransform, GraphJEPAPartitionTransform


class PolymerDataset(InMemoryDataset):
    """Custom PyTorch Geometric dataset for polymer graphs.
    
    This dataset handles in-memory storage of polymer molecular graphs
    with optional pre-processing transforms.
    
    Args:
        root: Root directory for processed data (optional)
        data_list: List of PyTorch Geometric Data objects
        transform: Transform applied on-the-fly during data loading
        pre_transform: Transform applied once during preprocessing
    """
    def __init__(self, root: Optional[str], data_list: List[Any], 
                 transform=None, pre_transform=None):
        self.data_list = data_list
        super().__init__(root or "", transform, pre_transform)
        # self.load(self.processed_paths[0])
        # For PyG<2.4:
        if root is not None:
            self.data, self.slices = torch.load(self.processed_paths[0])
        else: 
            self.data, self.slices = self.collate(data_list)
            

    @property
    def processed_file_names(self) -> str:
        return 'dataset.pt'

    def process(self) -> None:
        """Process the dataset and save to processed directory."""
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            self.data_list = [self.pre_transform(data) for data in self.data_list]
        # self.save(self.data_list, self.processed_paths[0])
        # For PyG<2.4:
        torch.save(self.collate(self.data_list), self.processed_paths[0])


def get_graphs(dataset: str = 'aldeghi') -> Tuple[List[Any], None]:
    """Load or create molecular graphs from polymer datasets.
    
    Args:
        dataset: Dataset name ('aldeghi' or 'diblock')
        
    Returns:
        Tuple of (graph_list, None) where graph_list contains PyG Data objects
        
    Raises:
        ValueError: If dataset name is not supported
    """
    all_graphs = []
    
    file_csv='Data/aldeghi_coley_ea_ip_dataset.csv' if dataset == 'aldeghi' else 'Data/diblock_copolymer_dataset.csv'
    file_graphs_list='Data/aldeghi_graphs_list.pt' if dataset == 'aldeghi' else 'Data/diblock_graphs_list.pt'

    if not os.path.isfile(file_graphs_list):
        print('Creating graphs pt file...')
        df = pd.read_csv(file_csv)
        # use tqdm to show progress bar
        if dataset == 'aldeghi':
            for i in tqdm.tqdm(range(len(df.loc[:, 'poly_chemprop_input']))):
                poly_strings = df.loc[i, 'poly_chemprop_input']
                ea_values = df.loc[i, 'EA vs SHE (eV)']
                ip_values = df.loc[i, 'IP vs SHE (eV)']
                # given the input polymer string, this function returns a pyg data object
                graph = poly_smiles_to_graph(
                    poly_strings=poly_strings, 
                    isAldeghiDataset=True,
                    y_EA=ea_values, 
                    y_IP=ip_values
                ) 
            
                all_graphs.append(graph)

        elif dataset == 'diblock':
            for i in tqdm.tqdm(range(len(df.loc[:, 'poly_chemprop_input']))):
                poly_strings = df.loc[i, 'poly_chemprop_input']
                lamellar_values = df.loc[i, 'lamellar']
                cylinder_values = df.loc[i, 'cylinder']
                sphere_values = df.loc[i, 'sphere']
                gyroid_values = df.loc[i, 'gyroid']
                disordered_values = df.loc[i, 'disordered']
                phase1 = df.loc[i, 'phase1']
                # given the input polymer string, this function returns a pyg data object
                graph = poly_smiles_to_graph(
                    poly_strings=poly_strings, 
                    isAldeghiDataset=False,
                    y_lamellar=lamellar_values, 
                    y_cylinder=cylinder_values,
                    y_sphere=sphere_values, 
                    y_gyroid=gyroid_values,
                    y_disordered=disordered_values,
                    phase1=phase1
                ) 
                all_graphs.append(graph)
        else:
            raise ValueError('Invalid dataset name')
    else: 
        all_graphs = torch.load(file_graphs_list)
                
   
    return all_graphs, None
        


def create_data(cfg) -> Tuple[Any, Any, Any]:
    """Create dataset with transforms for training pipeline.
    
    Args:
        cfg: Configuration object containing dataset and transform parameters
        
    Returns:
        Tuple of (dataset, train_transform, val_transform)
        
    Raises:
        ValueError: If dataset name is not supported
    """
    pre_transform = PositionalEncodingTransform(rw_dim=cfg.pos_enc.rw_dim)

    transform_train = GraphJEPAPartitionTransform(
        subgraphing_type=cfg.subgraphing.type,
        num_targets=cfg.jepa.num_targets,
        n_patches=cfg.subgraphing.n_patches,
        patch_rw_dim=cfg.pos_enc.patch_rw_dim,
        patch_num_diff=cfg.pos_enc.patch_num_diff,
        drop_rate=cfg.subgraphing.drop_rate,
        context_size=cfg.subgraphing.context_size,
        target_size=cfg.subgraphing.target_size,
        dataset=cfg.finetuneDataset
    )

    # no dropout for validation
    transform_val = GraphJEPAPartitionTransform(
        subgraphing_type=cfg.subgraphing.type,
        num_targets=cfg.jepa.num_targets,
        n_patches=cfg.subgraphing.n_patches,
        patch_rw_dim=cfg.pos_enc.patch_rw_dim,
        patch_num_diff=cfg.pos_enc.patch_num_diff,
        drop_rate=0.0, 
        context_size=cfg.subgraphing.context_size,
        target_size=cfg.subgraphing.target_size,
        dataset=cfg.finetuneDataset
    )
    
    if cfg.finetuneDataset == 'aldeghi' or cfg.finetuneDataset == 'diblock':
        all_graphs = []
        if cfg.finetuneDataset == 'diblock' and not os.path.isfile('Data/diblock_graphs_list.pt'):
            all_graphs, _ = get_graphs(dataset=cfg.finetuneDataset)
        if not os.path.isfile('Data/aldeghi/processed/dataset.pt'): # avoid loading the graphs, if dataset already exists
            all_graphs, _ = get_graphs(dataset=cfg.finetuneDataset)
        
        dataset = PolymerDataset(root='Data/aldeghi', data_list=all_graphs, pre_transform=pre_transform)
        
        # return full dataset and transforms, split in pretrain/finetune, train/test is done in the training script with k fold
        return dataset, transform_train, transform_val
    else:
        raise ValueError('Invalid dataset name')

def print_dataset_stats(graphs: List[Any]) -> None:
    """Print statistics about the molecular graph dataset.
    
    Args:
        graphs: List of PyTorch Geometric Data objects
    """
    avg_num_nodes = sum([g.num_nodes for g in graphs]) / len(graphs)
    avg_num_edges = sum([g.num_edges for g in graphs]) / len(graphs)
    print(f'Average number of nodes: {avg_num_nodes}')
    print(f'Average number of edges: {avg_num_edges}')

    # print n of graphs with more than 25 nodes
    print(f'Number of graphs with more than 25 nodes: {len([g for g in graphs if g.num_nodes > 25])}')
    print(f'Number of graphs with more than 30 nodes: {len([g for g in graphs if g.num_nodes > 30])}')

    # print max number of nodes
    print(f'Max number of nodes: {max([g.num_nodes for g in graphs])}')
    print(f'min number of nodes: {min([g.num_nodes for g in graphs])}')

def getFullAtomsList():
    # with open('full_atoms_list.txt', 'r') as f:
    #     full_atoms_list = f.read()
    # f.close()

    # full_atoms_list = set(full_atoms_list.split(','))

    return {'8', '0', '16', '17', '7', '35', '9', '53', '6'}

def get_random_data(ft_data: Any, size: int, seed: Optional[int] = None) -> List[Any]:
    """Sample random subset of data for training.
    
    Args:
        ft_data: Dataset or list of data points
        size: Number of samples to select
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled data points
    """

    # include the least possible number of different monomerA and monomerB while making sure that all possible atoms are present in the dataset
def get_lab_data(ft_data, size, seed=None):
    torch.manual_seed(seed)
    ft_data.shuffle()
    current_size = 0
    i = 0
     # keep track of how many occurrences of each monomerA, if for a monomerA we have 1/9 of the size, we can stop adding that monomerA
    monomerADict = collections.defaultdict(int)
    # stoichiometry and chain architecture dict (same as monomer A), but ratios here are 1/3
    stoichDict = collections.defaultdict(int)
    chainArchDict = collections.defaultdict(int)
    # just make sure that the monomerB is not repeating, so its not in the set
    monomerBDict = collections.defaultdict(int)
   
    dataset = []
    full_atoms_list = getFullAtomsList()
    subset_atoms = set()

    # keep track of how many keys in monomerADict and monomerBSet, 
    while i < len(ft_data) and current_size < size:
        m = Chem.MolFromSmiles(ft_data[i].smiles['polymer'])
        m_atoms = set([atom.GetAtomicNum() for atom in m.GetAtoms()])
        monomerA = ft_data[i].smiles['monomer1']
        monomerB = ft_data[i].smiles['monomer2']
        stoich = ft_data[i].stoichiometry

        # chainArchDict[chainArch] >= size // 3 or \
        if \
        (len(monomerADict) >= 4 and monomerA not in monomerADict) or \
        (len(monomerBDict) >= 20 and monomerB not in monomerBDict) or \
        stoichDict[stoich] >= size // 3 or \
        (len(subset_atoms) < len(full_atoms_list) and len(m_atoms) == len(m_atoms.intersection(subset_atoms))):
            i += 1
            continue

        monomerADict[monomerA] += 1
        monomerBDict[monomerB] += 1
        stoichDict[stoich] += 1
        subset_atoms.update(m_atoms)
        current_size += 1
        dataset.append(ft_data[i])
        i += 1
    
    # print("\Lab dataset stats:\n")
    # print("Size:", current_size)
    # print("Mon A dict:", monomerADict)
    # print("Mon B dict", len(monomerBDict))
    # print("Stoich dict:", stoichDict)
    # print("Subset atoms length:", len(subset_atoms), "; Full atoms length:", len(full_atoms_list))

    return dataset
    # select 'size' number of random data points from ft_data
    # randomly set torch seed based on the current time
    # torch.manual_seed(int(time.time()))
    # set random seed for python
    
    dataset = ft_data #.shuffle()
    if not isinstance(dataset, list):
        dataset = [x for x in dataset]
    
    random.seed(seed)
    if size != len(ft_data):
        dataset = random.sample(dataset, size)
    else:
        random.shuffle(dataset)

    return dataset



def analyze_diblock_dataset() -> None:
    """Analyze the diblock copolymer dataset structure and composition.
    
    Prints statistics about polymer strings, monomer counts, and displays
    sample molecular structures using RDKit.
    """
    csv_file = 'Data/diblock_copolymer_dataset.csv'
    df = pd.read_csv(csv_file)
    
    # check how many differen entries for the 'poly_chemprop_input' column
    poly_strings = df.loc[:, 'poly_chemprop_input']
    for i in range(len(poly_strings)):
        poly_strings[i] = poly_strings[i].split('|')[0]
    poly_set = set()
    for p in poly_strings:
        poly_set.add(p)
    print(poly_set)
    print('Number of different polymer strings:', len(poly_set))

    # for each polymer string, check how many different monomers are present (sepeartead by a dot)
    n_of_monomers = []
    for p in poly_set:
        n_of_monomers.append(len(p.split('.')))
    print('Number of monomers:', n_of_monomers)
    # count how many times each number of monomers appears
    monomer_dict = collections.defaultdict(int)
    for n in n_of_monomers:
        monomer_dict[n] += 1
    print('Monomer dict:', monomer_dict)

    # use rdkit to plot randomly 9 polymer strings
    for i in range(9):
        m = Chem.MolFromSmiles(poly_strings[i])
        Chem.Draw.MolToImage(m).show()


""" Slightly adjusted methods for the experiments with monomer A based splits """
def get_graphs_by_monomerA(dataset: str = 'aldeghi', monomerA_list: List[str] = []) -> Tuple[List[Any], Any]:
    """
    Load graphs for a subset of monomer A identities.
    Also returns the corresponding dataframe for reference with the rows that match the monomer A identities.
    
    Args:
        dataset: Dataset name ('aldeghi' only supported)
        monomerA_list: List of monomer A identities to include
    Returns:
        Tuple of (graph_list, df) where graph_list contains PyG Data objects and df contains the corresponding df
    """

    if dataset == 'diblock':
        raise NotImplementedError('Function not implemented for diblock dataset')
    graphs = []
    df = pd.read_csv('Data/aldeghi_coley_ea_ip_dataset.csv')
    df_filtered = df[df['poly_chemprop_input'].str.split('|').str[0].str.split('.').str[0].isin(monomerA_list)]
    for monomerA in monomerA_list:
        for i in tqdm.tqdm(range(len(df))):
            monomerA_id = df.loc[i, 'poly_chemprop_input'].split('|')[0].split('.')[0]
            # If the monomer A of the current row is not the one we want, skip
            if monomerA_id != monomerA:
                continue
            poly_strings = df.loc[i, 'poly_chemprop_input']
            ea_values = df.loc[i, 'EA vs SHE (eV)']
            ip_values = df.loc[i, 'IP vs SHE (eV)']
            graph = poly_smiles_to_graph(
                poly_strings=poly_strings, 
                isAldeghiDataset=True,
                y_EA=ea_values,
                y_IP=ip_values
            ) 
            graphs.append(graph)
            
    return graphs, df_filtered

def create_data_monomer_split(cfg, root, monomer_list) -> Tuple[Any, Any, Any]:
    """Create dataset with transforms for training pipeline.
    
    Args:
        cfg: Configuration object containing dataset and transform parameters
        root: Root directory for dataset storage. If not provided, defaults to 'Data/{dataset}'
        monomer_list: List of monomer A identities to include in the dataset
        
    Returns:
        Tuple of (dataset, train_transform, val_transform)
        
    Raises:
        ValueError: If dataset name is not supported
    """
    pre_transform = PositionalEncodingTransform(rw_dim=cfg.pos_enc.rw_dim)

    transform_train = GraphJEPAPartitionTransform(
        subgraphing_type=cfg.subgraphing.type,
        num_targets=cfg.jepa.num_targets,
        n_patches=cfg.subgraphing.n_patches,
        patch_rw_dim=cfg.pos_enc.patch_rw_dim,
        patch_num_diff=cfg.pos_enc.patch_num_diff,
        drop_rate=cfg.subgraphing.drop_rate,
        context_size=cfg.subgraphing.context_size,
        target_size=cfg.subgraphing.target_size,
        dataset=cfg.finetuneDataset
    )

    # no dropout for validation
    transform_val = GraphJEPAPartitionTransform(
        subgraphing_type=cfg.subgraphing.type,
        num_targets=cfg.jepa.num_targets,
        n_patches=cfg.subgraphing.n_patches,
        patch_rw_dim=cfg.pos_enc.patch_rw_dim,
        patch_num_diff=cfg.pos_enc.patch_num_diff,
        drop_rate=0.0, 
        context_size=cfg.subgraphing.context_size,
        target_size=cfg.subgraphing.target_size,
        dataset=cfg.finetuneDataset
    )
    
    if cfg.finetuneDataset == 'aldeghi' and cfg.split_type == 'MonomerA':
        all_graphs = []
        # if already created, the pt file should be in root_path+"processed/dataset.pt"
        if not os.path.isfile(f'{root}/processed/dataset.pt'):
            # save monomer list to a text file in the root directory
            # Make directory if it doesn't exist
            os.makedirs(root, exist_ok=True)
            with open(f'{root}/monomerA_list.txt', 'w') as f:
                f.write(','.join(monomer_list))    
            all_graphs, df_subset = get_graphs_by_monomerA(dataset=cfg.finetuneDataset, monomerA_list=monomer_list)
            # save the subset dataframe to a csv file in the root directory, useful for benchmarking against other models
            df_subset.to_csv(f'{root}/subset_dataframe.csv', index=False)
            
            
        dataset = PolymerDataset(root=root, data_list=all_graphs, pre_transform=pre_transform)
        
        # return full dataset and transforms, split in pretrain/finetune, train/test is done in the training script with k fold
        return dataset, transform_train, transform_val
    else:
        raise ValueError('Function only supports aldeghi dataset with MonomerA splits. Check configuration file. You might want to use create_data()')

