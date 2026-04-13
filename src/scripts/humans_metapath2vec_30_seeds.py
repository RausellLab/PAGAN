#!/usr/bin/env python

## intro

import numpy as np 
import pandas as pd
import random 
import os
from typing import Tuple, List
import time

# https://discuss.pytorch.org/t/how-to-enable-torch-use-cuda-dsa/202824/5
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
# https://discuss.pytorch.org/t/keep-getting-cuda-oom-error-with-pytorch-failing-to-allocate-all-free-memory/133896/10
# https://dev.to/shittu_olumide_/how-can-i-set-maxsplitsizemb-to-avoid-fragmentation-in-pytorch-37h9
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64 "

from tqdm import tqdm
import torch
import copy
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
import torch_geometric
# from torch_geometric import seed_everything
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader, ImbalancedSampler
import torch_geometric.transforms as T
import torch_geometric.utils as U
from torch_geometric.utils import coalesce
from torch_geometric.nn import summary, HeteroConv, GATv2Conv, SAGEConv, Linear
import pyg_lib
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, f1_score, auc, precision_recall_fscore_support, matthews_corrcoef 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import tempfile
# edit the line below to specify specify tmp_directory
# tempfile.tempdir = ''

torch.multiprocessing.set_sharing_strategy('file_system')
# https://stackoverflow.com/questions/71642653/how-to-resolve-the-error-runtimeerror-received-0-items-of-ancdata
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

print(f"numpy.__version__ = {np.__version__}")
print(f"pandas.__version__ = {pd.__version__}")
print(f"torch.__version__ = {torch.__version__}")
print(f"torch_geometric.__version__ = {torch_geometric.__version__}")
print(f"pyg_lib.__version__ = {pyg_lib.__version__}")

hyperparameters = {
    'node_mode' : 'degree', #'id' or 'degree' for abcm nodes_types
    'edge_features' : False, # True = those are the true edge_weights from my network
    'Heads' : 1, #2
    # 'conv_number' : 2, 
    # 'conv_type': 'SAGE',
    'sage_aggr': 'sum', # 'sum', 'var'
    'sage_norm': True, # False
    'sage_project': False,
    'heteroconv_aggr' : 'sum', # 'cat', 'max'
    'heteroconv_aggr_1' : 'sum', # 'sum', 'cat', 'max'
    'heteroconv_aggr_2' : 'sum', # 'sum', 'cat', 'max'
    'heteroconv_aggr_3' : 'sum', # 'cat', 'max'
    # 'activation': 'relu', # 'tanh'
    'hc_2_q' : 1, # hc_2 = int(i/hc_2_q)
    'hc_3_q': 1, # hc_3_q = int(i/hc_3_q)
    # 'size_of_batch': 128,
    'neighbors': 30,
    'disjoint_loader' : True, 
    'zero_out_batch_features': True,
    'epochs_to_train' : 500, # maximum number of epochs to train 
    'min_epochs_to_train': 3,
    'patience': 10,
    'min_delta': 0.005,
    'auto_loop': False,
    # 'nodes_pairs_features': 'zeros', # for p2p or n2p only
    # 'remove_forbidden_value': True, # for p2p or n2p only
    # 'ana_edges': 'common', # 'all
    'type_to_return': 'torch', # 'numpy'
    'final_type': 'torch', # 'numpy'
    'shuffle_train': True,
    'string_threshold': 700, # confidence = {'low': 150, 'medium': 400, 'high': 700, 'highest': 900} from STRING doc
    'setting': 'nodes_to_nodes',
    'target_to_use': 'essential',
    'features_to_use': 'basic' # features in ['basic', 'strict', 'large']
}

dict_j = {1:'01', 2:'02', 3:'03', 4:'04', 5:'05', 6:'06', 7:'07', 8:'08', 9:'09', 10:'10'}

sage_aggr = hyperparameters['sage_aggr'] # 'sum'
sage_norm = hyperparameters['sage_norm'] # 'sum'
heteroconv_aggr = hyperparameters['heteroconv_aggr'] #'sum', 'cat'
heteroconv_aggr_1 = hyperparameters['heteroconv_aggr_1'] #'sum', 'cat'
heteroconv_aggr_2 = hyperparameters['heteroconv_aggr_2'] #'sum', 'cat'
heteroconv_aggr_3 = hyperparameters['heteroconv_aggr_3'] #'sum', 'cat'
node_mode = hyperparameters['node_mode'] # 'id' or 'degree'
sage_norm = hyperparameters['sage_norm']
Heads = hyperparameters['Heads']
# ana_edges = hyperparameters['ana_edges']
# conv_number = hyperparameters['conv_number']
string_threshold = hyperparameters['string_threshold']

setting = hyperparameters['setting']
conv_type = hyperparameters['conv_type']
heads = hyperparameters['Heads']
neighbors = hyperparameters['neighbors']
patience = hyperparameters['patience']
min_delta = hyperparameters['min_delta']
zero_out_batch_features = hyperparameters['zero_out_batch_features']
epochs_to_train = hyperparameters['epochs_to_train']
size_of_batch = hyperparameters['size_of_batch']
target_of_script = hyperparameters['target_to_use']
features = 'basic' # 4 features, data['GENE'].x.shape will be (19330,4)

t0 = time.time()

old_path = os.getcwd()
new_path = old_path + '/../../'
os.chdir(new_path)
current_path = os.getcwd()
data_raw_path = current_path + '/data/raw/'
data_processed_path = current_path + '/data/processed/'
result_path = current_path + '/results/humans/baseline/'
model_path = current_path + '/models/humans/baseline/'

# open the Human_Knowledge_Graph
net = pd.read_csv(data_processed_path + f'human_knowledge_graph.tsv', sep = '\t', header = 0, dtype = {1: str, 3: str, 4: str})
net = net.drop(net[(net['EdgeType'].isin(['PPI'])) & (net['EdgeScore'] < string_threshold)].index, axis = 0)
net = net.reset_index(drop = True)
# I will not be using these edges,maybe in a future version ? 
edges_to_remove = [
    'CoExp', 'Neighbour', 
    'not_expressed_in', 'DIDA',
    'synthetic_lethal', 'synthetic_non_lethal', 
    'regulatory_proximal', 'regulatory_distal'
    ]
net = net.drop(net[net['EdgeType'].isin(edges_to_remove)].index, axis=0)
net = net.reset_index(drop=True)

all_genes = list(set().union(net[net['node1_type'] == 'GENE']['node1_ID'].unique(), net[net['node2_type'] == 'GENE']['node2_ID'].unique()))
# I could hardcode edges_names, edge_summary eveneutally
edges_names = net['EdgeType'].value_counts().to_dict()
edges_summary = []
for edge_name in edges_names.keys():
    tmp_list = [edge_name, 
                net[net['EdgeType'] == edge_name]['node1_type'].unique()[0], 
                net[net['EdgeType'] == edge_name]['node2_type'].unique()[0], 
                net[net['EdgeType'] == edge_name]['EdgeDirection'].unique()[0], 
                edges_names[edge_name]]
    # create a special case when there are edge_attributes
    # not currently using edges_attributes, but maybe in a future version 
    if sum(net[net['EdgeType'] == edge_name]['EdgeScore'].isna()) == 0:
        tmp_list.append(1)
    else:
        tmp_list.append(0)
    edges_summary.append(tmp_list) 

edges_types = list(set(net['EdgeType'].unique()))
nodes_types = list(set().union(net['node1_type'].unique(),net['node2_type'].unique()))
nodes_types.sort()
nodes_numbers = {}
for node_type in nodes_types:
    nodes_numbers[node_type] = len(set().union(net[net['node1_type'] == node_type]['node1_ID'].unique(), net[net['node2_type'] == node_type]['node2_ID'].unique()))

print(f"nodes_types: {nodes_types}")
print(f"nodes_numbers: {nodes_numbers}")

list_of_edges = net['EdgeType'].unique().tolist()

def Get_df_for_nodetype_id(df: pd.DataFrame, node_type : str) -> pd.DataFrame:
    node_type_list = list(set().union(df[df['node1_type'] == node_type]['node1_ID'].unique(), df[df['node2_type'] == node_type]['node2_ID'].unique()))
    node_type_list.sort()
    node_type_array = np.asarray(node_type_list)
    new_df = pd.DataFrame(data = {
        f"{node_type}_id": node_type_array,
        "mapped_id": pd.RangeIndex(len(node_type_array))
    })
    return(new_df)

nodes2id = {}
for type_of_node in nodes_types:
    nodes2id[type_of_node] = Get_df_for_nodetype_id(net, type_of_node)

edges_to_nodes = {}
for type_of_edge in edges_summary:
    edges_to_nodes[type_of_edge[0]] = [type_of_edge[1], type_of_edge[2]] 

def Get_df_for_edgetype(df: pd.DataFrame, type_of_edge: str, edge_score: bool = False) -> pd.DataFrame:
    node1_type = edges_to_nodes[type_of_edge][0]
    node2_type = edges_to_nodes[type_of_edge][1]
    new_df = df[df['EdgeType'] == type_of_edge].loc[:,["node1_ID", "node2_ID", "EdgeScore"]]
    new_df = pd.merge(new_df, nodes2id[node1_type], left_on = 'node1_ID', right_on = f"{node1_type}_id", how = 'left')
    new_df = new_df.rename(columns = {f"{node1_type}_id": f"{node1_type}1_id", 'mapped_id': 'node1_mapped_id'})
    new_df = pd.merge(new_df, nodes2id[node2_type], left_on = 'node2_ID', right_on = f"{node2_type}_id", how = 'left')
    new_df = new_df.rename(columns = {f"{node1_type}_id": f"{node1_type}2_id", 'mapped_id': 'node2_mapped_id'})
    if edge_score == False:
        new_df = new_df.drop('EdgeScore', axis = 1)
    return new_df

def Get_tensor_for_edges(df: pd.DataFrame, type_of_edge: str) -> torch.Tensor:
    node1_type = edges_to_nodes[type_of_edge][0]
    node2_type = edges_to_nodes[type_of_edge][1]
    new_df = df[df['EdgeType'] == type_of_edge]
    new_df = pd.merge(new_df, nodes2id[node1_type], left_on = 'node1_ID', right_on = f"{node1_type}_id", how = 'left')
    new_df = new_df.rename(columns = {"mapped_id": "node1_mapped_id"})
    new_df = pd.merge(new_df, nodes2id[node2_type], left_on = 'node2_ID', right_on = f"{node2_type}_id", how = 'left')
    new_df = new_df.rename(columns = {"mapped_id": "node2_mapped_id"})
    return torch.stack([torch.from_numpy(new_df['node1_mapped_id'].values), torch.from_numpy(new_df['node2_mapped_id'].values)], dim = 0)

def Get_tensor_and_rev_for_edges(df: pd.DataFrame, type_of_edge: str) -> torch.Tensor:
    node1_type = edges_to_nodes[type_of_edge][0]
    node2_type = edges_to_nodes[type_of_edge][1]
    new_df = df[df['EdgeType'] == type_of_edge]
    new_df = pd.merge(new_df, nodes2id[node1_type], left_on = 'node1_ID', right_on = f"{node1_type}_id", how = 'left')
    new_df = new_df.rename(columns = {"mapped_id": "node1_mapped_id"})
    new_df = pd.merge(new_df, nodes2id[node2_type], left_on = 'node2_ID', right_on = f"{node2_type}_id", how = 'left')
    new_df = new_df.rename(columns = {"mapped_id": "node2_mapped_id"})
    source, destination = torch.stack([torch.from_numpy(new_df['node1_mapped_id'].values), torch.from_numpy(new_df['node2_mapped_id'].values)], dim = 0)
    source_tot = torch.cat([source, destination])      
    destination_tot = torch.cat([destination, source])
    edge_index = torch.stack([source_tot, destination_tot], dim = 0)  
    return edge_index

edges2df = {}
for sublist in edges_summary:
    if sublist[5] == 1: 
        edges2df[sublist[0]] = Get_df_for_edgetype(net, sublist[0], True)
    else:
        edges2df[sublist[0]] = Get_df_for_edgetype(net, sublist[0], False)

# create columns selectors
numerical_columns_selector = selector(dtype_exclude = object)
categorical_columns_selector = selector(dtype_include = object)
numerical_preprocessor = StandardScaler()
categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")

def get_strict_target_tensors(target_to_use: str = 'essential', features_to_use: str = 'strict', remove_sex_chrom:bool = False):
    nodes2id['GENE'] = nodes2id['GENE'][['GENE_id', 'mapped_id']]
    # open the file with the features values 
    feat = pd.read_csv(data_processed_path + 'human_genes_features_20230414.tsv', sep = '\t', header = 0)
    feat = pd.merge(nodes2id['GENE'], feat, left_on = 'GENE_id', right_on = 'ENSG_ID', how = 'left')
    # open the file with the essential vs non_essential, and remove some of them: pseudogene etc.  
    # TODO update the essential_file
    vitro = pd.read_csv(data_raw_path + 'essentials_invitro_20240214.tsv', sep = '\t', names = ['gene_id'])
    list_to_remove = [
        'ENSG00000158483', 'ENSG00000172014', 'ENSG00000180953', 'ENSG00000183148', 'ENSG00000186825',
        'ENSG00000197927', 'ENSG00000198019', 'ENSG00000204677', 'ENSG00000213029', 'ENSG00000232962', 
        'ENSG00000263091', 'ENSG00000276816', 'ENSG00000282827', 'ENSG00000283472', 'SF3B14']
    # conversion_df = pd.read_csv('/data-cbl/rnicolle/GRL4DD/data/processed/essentials_conversion_dict_20240216.tsv', header = 0, sep = '\t')
    conversion_df = pd.read_csv(data_processed_path + 'essentials_conversion_dict_20240216.tsv', header = 0, sep = '\t')
    my_keys = conversion_df['old_ID'].tolist()
    my_values = conversion_df['new_ID'].tolist()
    conversion_dict = {}
    conversion_dict = {my_keys[i]: my_values[i] for i in range(conversion_df.shape[0])}
    vitro = vitro.drop(vitro[vitro['gene_id'].isin(list_to_remove)].index, axis = 0)
    vitro = vitro.reset_index(drop = True)         
    vitro.replace(to_replace=conversion_dict, inplace=True)   
    # get the essential/non-essential type for each GENE
    feat.loc[:,['type']] = 'non_essential'
    feat.loc[feat[feat['GENE_id'].isin(vitro['gene_id'].tolist())].index, 'type'] = 'essential'
    # choose the target: 'pLI' (0: haplo-sufficient, 1: haplo-sensitive, ), 'essential' (0: non_essential, 1: essential)
    if target_to_use in ['pLI', 'OLIDA', 'proven_SL']:
        feat.loc[feat[feat['pLI'] >= 0.9].index, 'y'] = 1
        feat.loc[feat[feat['pLI'] < 0.9].index, 'y'] = 0
        feat.loc[feat[feat['pLI'].isna()].index, 'y'] = 0
        feat.loc[feat[feat['pLI'] == -1].index, 'y'] = 0
    elif target_to_use == 'essential':
        feat.loc[:, 'y'] = 0
        feat.loc[feat[feat['type'] == 'essential'].index, 'y'] = 1
    elif target_to_use == 'essential_K562':
        essential = pd.read_csv(data_raw_path + 'essentials_K562.tsv', header = 0, sep = '\t')
        feat['type'] = 'non_essential'
        feat.loc[feat[feat['GENE_id'].isin(essential['ENSG_ID'].tolist())].index, 'type'] = 'essential'
        feat.loc[:, 'y'] = 0
        feat.loc[feat[feat['type'] == 'essential'].index, 'y'] = 1
    elif target_to_use == 'essential_A375':
        essential = pd.read_csv(data_raw_path + 'essentials_A375.tsv', header = 0, sep = '\t')
        feat['type'] = 'non_essential'
        feat.loc[feat[feat['GENE_id'].isin(essential['ENSG_ID'].tolist())].index, 'type'] = 'essential'
        feat.loc[:, 'y'] = 0
        feat.loc[feat[feat['type'] == 'essential'].index, 'y'] = 1
    elif target_to_use == 'essential_A549':
        essential = pd.read_csv(data_raw_path + 'essentials_A549.tsv', header = 0, sep = '\t')
        feat['type'] = 'non_essential'
        feat.loc[feat[feat['GENE_id'].isin(essential['ENSG_ID'].tolist())].index, 'type'] = 'essential'
        feat.loc[:, 'y'] = 0
        feat.loc[feat[feat['type'] == 'essential'].index, 'y'] = 1
    elif target_to_use == 'OMIM':
        omim = pd.read_csv(data_raw_path + 'humans_n2n_OMIM_20250826.tsv', header = 0, sep = '\t')
        feat = feat.merge(omim[['GENE_id', 'target']], how='left', left_on='GENE_id', right_on='GENE_id')
        feat = feat.rename(columns = {'target': 'y'})
    else:
        raise Exception(f"Please select either 'pLI' or 'essential' or 'essential_K562' or 'essential_A375' or 'essential_A549' for the target to use.")
    # remove sex_chromosom ? TODO
    if (remove_sex_chrom == True) and (set(['chrX','chrY']).issubset(set(feat['chrom'].unique().tolist())) == True):
        feat = feat.drop(feat[feat['chrom'].isin(['chrX', 'chrY'])].index, axis = 0)
        feat = feat.drop('chrom', axis = 1)
    # some GENES are missing values: get the median value of the GENES of the same class, i.e. haplo-sensitive/sufficient or essential/non_essential
    feat = feat.reset_index(drop = True)
    dict_0 = feat[feat['y'] == 0].median(axis = 0, skipna = True, numeric_only = True).to_dict()
    dict_1 = feat[feat['y'] == 1].median(axis = 0, skipna = True, numeric_only = True).to_dict()
    # some GENES with OMIM target have no features at all, they have target -1
    # for i in ['zscore_mis','zscore_syn','f_parameter']:
    for i in dict_0.keys():
        feat.loc[feat[(feat['y'] == 0) & (feat[i].isna())].index, i] = dict_0[i] 
        feat.loc[feat[(feat['y'] == 1) & (feat[i].isna())].index, i] = dict_1[i] 
    if target_to_use == 'OMIM':
        for i in dict_0.keys():
            feat.loc[feat[(feat['y'] == -1) & (feat[i].isna())].index, i] = dict_0[i] 

    if features_to_use in ['strict', 'basic']:
        feat = feat.drop([
        'GENE_id', 'mapped_id', 'ENSG_ID', 'GeneName', 'chrom', 'loeuf',
        'Haploinsufficiency_Score', 'Haploinsufficiency_Description',
        'Triplosensitivity_Score', 'Triplosensitivity_Description', 'omim_class', 'known_developmental_disorder_genes',
        'DDD_HI_percentage', 'RVIS', 'DOMINO', 'pHI', 'SCoNeS', 'shet_det', 'shet_drift', #'GDI', 
        'ncRVIS', 'ncGERP', 'type'], axis = 1)
    elif features_to_use == 'large':
        feat = feat.drop([
        'GENE_id', 'mapped_id', 'ENSG_ID', 'GeneName', 'chrom', #'loeuf',
        'Haploinsufficiency_Score', 'Haploinsufficiency_Description',
        'Triplosensitivity_Score', 'Triplosensitivity_Description', 'omim_class', 'known_developmental_disorder_genes',
        'DDD_HI_percentage', 
        # 'RVIS', 
        'DOMINO', 'pHI', 
        # 'SCoNeS', 'shet_det', 'shet_drift', #'GDI', 
        'ncRVIS', 'ncGERP', 'type'], axis = 1)
    
    if target_to_use in ['essential', 'essential_K562', 'essential_A375', 'essential_A549'] or features_to_use == 'basic':
        feat = feat.drop('FUSIL', axis = 1) # FUSIL is highly correlated to essentiality
    else: # target_to_use == 'pLI' = keep FUSIL category
        categorical_columns = ['FUSIL']
        feat = pd.get_dummies(feat, columns = categorical_columns, dtype = float)
    feat = feat.drop('pLI', axis = 1)
    numerical_columns = numerical_columns_selector(feat)
    numerical_columns.remove('y')
    feat[numerical_columns] = StandardScaler().fit_transform(feat[numerical_columns])
    gene_feat = feat.copy()
    ### get tensors
    target_tensor = torch.from_numpy(gene_feat.y.values)
    gene_feat = feat.drop('y', axis = 1)
    gene_feat_tensor = torch.from_numpy(gene_feat.values)
    
    return gene_feat_tensor, target_tensor, feat 

def add_strict_nodes(data: HeteroData, node_target: str = 'essential') -> HeteroData:
    gene_strict_tensor, target_tensor, _ = get_strict_target_tensors(target_to_use = node_target)
    data['GENE'].x = gene_strict_tensor.to(torch.float32)
    data['GENE'].y = target_tensor.type(torch.float32)
    for type_of_node in nodes_types:
        if type_of_node != 'GENE':
            data[type_of_node].num_nodes = len(nodes2id[type_of_node])
    return data

def create_masks_cross_val(data: HeteroData, split_seed: int = 1, cv_fold_to_test: int = 1) -> HeteroData:
    cv_folds = pd.read_csv(
        data_processed_path + 'humans_n2n_19330_splits_seeds_cv.tsv',
        sep = '\t', header = 0)
    col_to_use = f"cv_seed_{split_seed}"
    folds = cv_folds[[col_to_use]]
    train_folds = ['cv_1', 'cv_2', 'cv_3', 'cv_4', 'cv_5', 'cv_6', 'cv_7', 'cv_8', 'cv_9', 'cv_10']
    test_fold = f'cv_{cv_fold_to_test}'
    train_folds.remove(test_fold)
    # get the indexes
    train_idx = folds[folds[col_to_use].isin(train_folds)].index
    test_idx = folds[folds[col_to_use] == test_fold].index
    val_idx = folds[folds[col_to_use] == 'leave_out_test'].index
    # initialize masks
    train_mask = torch.zeros(data['GENE'].x.shape[0])
    val_mask = torch.zeros(data['GENE'].x.shape[0])
    test_mask = torch.zeros(data['GENE'].x.shape[0])
    # create 1 at the specified index
    train_mask[train_idx] = 1
    val_mask[val_idx] = 1
    test_mask[test_idx] = 1
    # get the boolean tensor
    train_mask = train_mask > 0
    val_mask = val_mask > 0
    test_mask = test_mask > 0
    # create the masks to Data
    data['GENE'].train_mask = train_mask
    data['GENE'].val_mask = val_mask
    data['GENE'].test_mask = test_mask
    # return
    return data

# TODO update function to use 'net' as an input
def get_edges_of_type(data: HeteroData, type_of_edge: str, edge_attr: bool = False) -> HeteroData:
    node1_type = edges_to_nodes[type_of_edge][0]
    node2_type = edges_to_nodes[type_of_edge][1]
    new_df = net[net['EdgeType'] == type_of_edge]
    new_df = pd.merge(new_df, nodes2id[node1_type], left_on = 'node1_ID', right_on = f"{node1_type}_id", how = 'left')
    new_df = new_df.rename(columns = {"mapped_id": "node1_mapped_id"})
    new_df = pd.merge(new_df, nodes2id[node2_type], left_on = 'node2_ID', right_on = f"{node2_type}_id", how = 'left')
    new_df = new_df.rename(columns = {"mapped_id": "node2_mapped_id"})

## create a special case where I also take the edges attributes = edges scores
    if (edge_attr == True) and (type_of_edge in ['expressed_in', 'not_expressed_in', 'PPI', 'PPI_STRING', 'Neighbour', 'CoExp', 'DIDA']): 
        if type_of_edge in ['expressed_in', 'not_expressed_in']:
        # the edges 'expressed_in' and 'not_expressed_in' are directed because it's a link between different nodes types, I have to use T.ToUndirected() later
            edge_index = torch.stack([torch.from_numpy(new_df['node1_mapped_id'].values), torch.from_numpy(new_df['node2_mapped_id'].values)], dim = 0)
            #edge_attr = torch.from_numpy(edges2df[type_of_edge].EdgeScore.values).view(-1,1).type('torch.DoubleTensor')
            edge_attr = torch.from_numpy(edges2df[type_of_edge].EdgeScore.values).view(-1,1).type(torch.float32)
            data[node1_type, type_of_edge, node2_type].edge_index = edge_index
            data[node1_type, type_of_edge, node2_type].edge_attr = edge_attr#.view(-1)
        else:
        # the edges 'PPI', 'Neighbour', 'CoExp' and 'DIDA' are symetrical and between the same node type      
            source, destination = torch.stack([torch.from_numpy(new_df['node1_mapped_id'].values), torch.from_numpy(new_df['node2_mapped_id'].values)], dim = 0)
            source_tot = torch.cat([source, destination])      
            destination_tot = torch.cat([destination, source])  
            edge_index = torch.stack([source_tot, destination_tot], dim = 0)
            edge_attr = torch.from_numpy(edges2df[type_of_edge].EdgeScore.values).view(-1,1).type(torch.float32)
            edge_attr = torch.cat([edge_attr, edge_attr], dim = 0)
            data[node1_type, type_of_edge, node2_type].edge_index = edge_index
            data[node1_type, type_of_edge, node2_type].edge_attr = edge_attr#.view(-1)
## no edge_attr but directed edges, I have to use T.ToUndirected(merge = True) later for 'expressed_in' and 'not_expressed_in'
## use T.ToUndirected(merge = False) for 'ana_ana', 'biological_process', 'cellular_component', 'molecular_function', 'regulatory_proximal', 'regulatory_distal'
    elif (edge_attr == False) and (type_of_edge in ['expressed_in', 'not_expressed_in', 'gene_to_BP', 'gene_to_CC', 'gene_to_MF', 
                                                    'ana_ana', 'biological_process', 'cellular_component', 'molecular_function',
                                                    'regulatory_proximal', 'regulatory_distal'
                                                    ]):
        edge_index = torch.stack([torch.from_numpy(new_df['node1_mapped_id'].values), torch.from_numpy(new_df['node2_mapped_id'].values)], dim = 0)
        data[node1_type, type_of_edge, node2_type].edge_index = edge_index
## those edges are undirected
    elif (edge_attr == False) and (type_of_edge in ['PPI', 'PPI_STRING', 'paralog', 'Neighbour', 'CoExp', 'synthetic_lethal', 'synthetic_non_lethal', 'DIDA']):
        source, destination = torch.stack([torch.from_numpy(new_df['node1_mapped_id'].values), torch.from_numpy(new_df['node2_mapped_id'].values)], dim = 0)
        source_tot = torch.cat([source, destination])      
        destination_tot = torch.cat([destination, source])  
        edge_index = torch.stack([source_tot, destination_tot], dim = 0)
        data[node1_type, type_of_edge, node2_type].edge_index = edge_index
## if edge_attr is set to True for edges without an EdgeScore    
    else:
        raise Exception(f"the edge of type: {type_of_edge} doesn't have an edge_attribute. Run this function again with edge_attr set to False")
    
    return data

# in this section I will create dummy features for the nodes types that are not 'GENES' based on their node-degree, according to one edge-type
data = HeteroData()
data = add_strict_nodes(data, node_target = target_of_script)
data = get_edges_of_type(data, 'ana_ana')
data = get_edges_of_type(data, 'biological_process')
data = get_edges_of_type(data, 'cellular_component')
data = get_edges_of_type(data, 'molecular_function')

ana = Data()
ana.num_nodes = len(nodes2id['ANATOMY']) 
ana.edge_index = data.edge_index_dict[('ANATOMY', 'ana_ana', 'ANATOMY')]
df_ana = pd.DataFrame(ana.edge_index.t(), columns = ['source', 'dst'])
max_degree = max(df_ana.dst.value_counts().tolist()[0], df_ana.source.value_counts().tolist()[0])
ana = T.OneHotDegree(max_degree = max_degree)(ana)
ana.x_id = torch.eye(ana.num_nodes, dtype = torch.float32, requires_grad = False)

bp = Data()
bp.num_nodes = len(nodes2id['BP']) 
bp.edge_index = data.edge_index_dict[('BP', 'biological_process', 'BP')]
df_bp = pd.DataFrame(bp.edge_index.t(), columns = ['source', 'dst'])
max_degree = max(df_bp.dst.value_counts().tolist()[0], df_bp.source.value_counts().tolist()[0])
bp = T.OneHotDegree(max_degree = max_degree)(bp)
bp.x_id = torch.eye(bp.num_nodes, dtype = torch.float32, requires_grad = False)

cc = Data()
cc.num_nodes = len(nodes2id['CC']) 
cc.edge_index = data.edge_index_dict[('CC', 'cellular_component', 'CC')]
df_cc = pd.DataFrame(cc.edge_index.t(), columns = ['source', 'dst'])
max_degree = max(df_cc.dst.value_counts().tolist()[0], df_cc.source.value_counts().tolist()[0])
cc = T.OneHotDegree(max_degree = max_degree)(cc)
cc.x_id = torch.eye(cc.num_nodes, dtype = torch.float32, requires_grad = False)

mf = Data()
mf.num_nodes = len(nodes2id['MF']) 
mf.edge_index = data.edge_index_dict[('MF', 'molecular_function', 'MF')]
df_mf = pd.DataFrame(mf.edge_index.t(), columns = ['source', 'dst'])
max_degree = max(df_mf.dst.value_counts().tolist()[0], df_mf.source.value_counts().tolist()[0])
mf = T.OneHotDegree(max_degree = max_degree)(mf)
mf.x_id = torch.eye(mf.num_nodes, dtype = torch.float32, requires_grad = False)

def add_features(data: HeteroData, mode: str = 'degree') -> HeteroData:
    '''create features for the nodes types that are not 'GENE', either one-hot-encoding their node_degree or their id'''
    if mode == 'id':
        data['ANATOMY'].x = ana.x_id
        data['BP'].x = bp.x_id
        data['CC'].x = cc.x_id
        data['MF'].x = mf.x_id
    elif mode == 'degree':
        data['ANATOMY'].x = ana.x
        data['BP'].x = bp.x
        data['CC'].x = cc.x
        data['MF'].x = mf.x
    else:
        raise Exception("Please select a mode 'id' or 'degree'")
    
    return data

gene_strict_tensor, target_tensor, _ = get_strict_target_tensors(target_to_use = target_of_script)

def create_abcmg_data(list_of_edges: list, use_edge_attr: bool = False, feature_mode: str = 'degree', merge_choice: bool = True, my_target: str = 'essential', my_features: str = 'essential'):
    '''Do not create 'regulatory_distal' or 'regulatory_proximal' edge'''
    edges_names = list_of_edges
    # in the list_of_edges, an edge is just a string
    edges_with_attr = ['PPI', 'PPI_STRING', 'Neighbour', 'CoExp', 'expressed_in'] #
    edges_with_same_node_type = [
        ('GENE', 'PPI', 'GENE'),
        ('GENE', 'PPI_STRING', 'GENE'),
        ('GENE', 'paralog', 'GENE'), 
        ('GENE', 'Neighbour', 'GENE'), 
        ('GENE', 'CoExp', 'GENE'), 
        ('ANATOMY', 'ana_ana', 'ANATOMY'), 
        ('BP', 'biological_process', 'BP'), 
        ('CC', 'cellular_component', 'CC'), 
        ('MF', 'molecular_function', 'MF')]
    # ('GENE', 'synthetic_lethal', 'GENE'), ('GENE', 'synthetic_non_lethal', 'GENE'), ('GENE', 'regulatory_proximal', 'GENE'), ('GENE', 'regulatory_distal', 'GENE')
    gene_strict_tensor, target_tensor, _ = get_strict_target_tensors(target_to_use = my_target, remove_sex_chrom = False)
    bool_attr = use_edge_attr
    data = HeteroData()
    data = add_strict_nodes(data, node_target = my_target)
    data = add_features(data, mode = feature_mode)

    if use_edge_attr == False:
        for edge_type in list_of_edges:   
            data = get_edges_of_type(data, edge_type)
    else: # use_edge_attr == True
        # for the edges in list_of_edges with edge_attr    
        for edge_type in list(set(edges_with_attr).intersection(edges_names)):
            data = get_edges_of_type(data, edge_type, edge_attr = True)
        # for the edges in list_of_edges without edge_attr
        for edge_type in list(set(edges_names) - set(edges_with_attr)):
            data = get_edges_of_type(data, edge_type, edge_attr = False)

    # do not remove 'self-loops' connecting different node_types = they are not 'self_loops'  
    # double check the edges concerned
    for key in data.edge_index_dict.keys():
        if key in edges_with_same_node_type:
            if U.contains_self_loops(data.edge_index_dict[key]) == True:
                data[key].edge_index = U.remove_self_loops(data.edge_index_dict[key])[0]

    data = T.ToUndirected(merge = merge_choice)(data)

    ana_list = ['ana_ana', 'expressed_in', 'not_expressed_in']
    bp_list = ['gene_to_BP', 'biological_process']
    cc_list = ['gene_to_CC', 'cellular_component']
    mf_list = ['gene_to_MF', 'molecular_function']

    if any(edge in edges_names for edge in ana_list):
        pass
    else:
        del data['ANATOMY']
    if any(edge in edges_names for edge in bp_list):
        pass
    else:
        del data['BP']
    if any(edge in edges_names for edge in cc_list):
        pass
    else:
        del data['CC']    
    if any(edge in edges_names for edge in mf_list):
        pass
    else:
        del data['MF']

    return edges_names, data

# clean edges but do not coalesce
def clean_data(data: HeteroData, edge_attr: bool = False) -> HeteroData:
    
    edges_with_attr = [
        ('GENE', 'PPI', 'GENE'), 
        ('GENE', 'PPI_STRING', 'GENE'), 
        ('GENE', 'Neighbour', 'GENE'), 
        ('GENE', 'CoExp', 'GENE')]
    edges_with_same_node_type = [
        ('GENE', 'PPI', 'GENE'),
        ('GENE', 'PPI_STRING', 'GENE'),
        ('GENE', 'paralog', 'GENE'), 
        ('GENE', 'Neighbour', 'GENE'), 
        ('GENE', 'CoExp', 'GENE'), 
        ('BP', 'biological_process', 'BP'), 
        ('BP', 'rev_biological_process', 'BP'), 
        ('CC', 'cellular_component', 'CC'), 
        ('CC', 'rev_cellular_component', 'CC'), 
        ('MF', 'molecular_function', 'MF'),
        ('MF', 'rev_molecular_function', 'MF')]
    # in data.edge_index_dict.keys(), an edge is a tuple of 3 strings: ('node1_type', 'relation_type', 'node2_type')    
    if edge_attr == False:
        for edge in data.edge_index_dict.keys():
            data[edge].edge_index = data[edge].edge_index.contiguous()

            if edge in edges_with_same_node_type:
                if U.contains_self_loops(data.edge_index_dict[edge]) == True:
                    data[edge].edge_index = U.remove_self_loops(data.edge_index_dict[edge])[0]    
    
    elif edge_attr == True:
        for edge in data.edge_index_dict.keys():
            if edge not in edges_with_attr:
                data[edge].edge_index = data[edge].edge_index.contiguous()

                if edge in edges_with_same_node_type:
                    if U.contains_self_loops(data.edge_index_dict[edge]) == True:
                        data[edge].edge_index = U.remove_self_loops(data.edge_index_dict[edge])[0]

            elif edge in edges_with_attr:
                data[edge].edge_index = data[edge].edge_index.contiguous()

                if edge in edges_with_same_node_type:
                    if U.contains_self_loops(data.edge_index_dict[edge]) == True:
                        data[edge].edge_index = U.remove_self_loops(data.edge_index_dict[edge])[0]
    
    return data 


## create data
features = 'basic'

# create the data now
node_mode = hyperparameters['node_mode'] # 'id' or 'degree'
edge_features = hyperparameters['edge_features'] # True or False

list_of_edges, data = create_abcmg_data(['expressed_in', 'ana_ana',
                                        'gene_to_BP', 'biological_process', 
                                        'gene_to_CC', 'cellular_component', 
                                        'gene_to_MF', 'molecular_function'
                                            ], feature_mode = node_mode, use_edge_attr = edge_features, merge_choice = False, my_target=target_of_script)
list_of_edges_2, data_2 = create_abcmg_data(['PPI', 'paralog' 
                                            ], feature_mode = node_mode, use_edge_attr = edge_features, merge_choice = True, my_target = target_of_script) 
for key in data_2.edge_index_dict.keys():
    data[key].edge_index = data_2[key].edge_index
if edge_features == True:
    for key in data_2.edge_attr_dict.keys():
        data[key].edge_attr = data_2[key].edge_attr
else:
    pass
list_of_edges = list_of_edges + list_of_edges_2
del data_2
del list_of_edges_2
data = clean_data(data)

# coalesce at this step, the edges for individual genes 
for edge in data.edge_index_dict.keys():
    data[edge].edge_index = coalesce(data[edge].edge_index)

data = data.to('cpu')

## split_seed = 1, j = 1
split_seed = 1
j = 1
data = create_masks_cross_val(data = data, split_seed = split_seed, cv_fold_to_test = j)

nodes = nodes2id['GENE']
gene_strict_tensor, target_tensor, feat = get_strict_target_tensors(target_to_use = target_of_script, remove_sex_chrom = False, features_to_use = features)
nodes = pd.concat((nodes, feat), axis=1)

col_to_use = list(feat.columns)
col_to_use.remove('y')
target_col = ['y']

nodes.loc[:, ['train']] = data['GENE'].train_mask.numpy().astype(float)
# I know 
nodes.loc[:, ['val']] = data['GENE'].test_mask.numpy().astype(float) # inner cross-val test set
nodes.loc[:, ['test']] = data['GENE'].val_mask.numpy().astype(float) # outter cross-val test set

nodes.loc[nodes[nodes['train']==1].index, ['mask']] = 'train'
nodes.loc[nodes[nodes['val']==1].index, ['mask']] = 'val'
# nodes.loc[nodes[nodes['val']==1].index, ['mask']] = 'train'
nodes.loc[nodes[nodes['test']==1].index, ['mask']] = 'test'

print(data['GENE'].train_mask.sum())
print(data['GENE'].test_mask.sum())

new_train_mask = (data['GENE'].train_mask + data['GENE'].test_mask)
new_test_mask = (data['GENE'].val_mask)
del data['GENE'].train_mask
del data['GENE'].val_mask
del data['GENE'].test_mask
data['GENE'].train_mask = new_train_mask
data['GENE'].test_mask = new_test_mask

print(data['GENE'].train_mask.sum())
print(data['GENE'].test_mask.sum())

## my_metapath2vec

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader

from torch_geometric.typing import EdgeType, NodeType, OptTensor
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.sparse import index2ptr

EPS = 1e-15

class my_MetaPath2Vec(torch.nn.Module):
    r"""The MetaPath2Vec model from the `"metapath2vec: Scalable Representation
    Learning for Heterogeneous Networks"
    <https://ericdongyx.github.io/papers/
    KDD17-dong-chawla-swami-metapath2vec.pdf>`_ paper where random walks based
    on a given :obj:`metapath` are sampled in a heterogeneous graph, and node
    embeddings are learned via negative sampling optimization.

    .. note::

        For an example of using MetaPath2Vec, see
        `examples/hetero/metapath2vec.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        hetero/metapath2vec.py>`_.

    Args:
        edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): Dictionary
            holding edge indices for each
            :obj:`(src_node_type, rel_type, dst_node_type)` edge type present
            in the heterogeneous graph.
        embedding_dim (int): The size of each embedding vector.
        metapath (List[Tuple[str, str, str]]): The metapath described as a list
            of :obj:`(src_node_type, rel_type, dst_node_type)` tuples.
        walk_length (int): The walk length.
        context_size (int): The actual context size which is considered for
            positive samples. This parameter increases the effective sampling
            rate by reusing samples across different source nodes.
        walks_per_node (int, optional): The number of walks to sample for each
            node. (default: :obj:`1`)
        num_negative_samples (int, optional): The number of negative samples to
            use for each positive sample. (default: :obj:`1`)
        num_nodes_dict (Dict[str, int], optional): Dictionary holding the
            number of nodes for each node type. (default: :obj:`None`)
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to the
            weight matrix will be sparse. (default: :obj:`False`)
    """
    def __init__(
        self,
        edge_index_dict: Dict[EdgeType, Tensor],
        embedding_dim: int,
        metapath: List[EdgeType],
        walk_length: int,
        context_size: int,
        walks_per_node: int = 1,
        num_negative_samples: int = 1,
        num_nodes_dict: Optional[Dict[NodeType, int]] = None,
        sparse: bool = False,
    ):
        super().__init__()

        if num_nodes_dict is None:
            num_nodes_dict = {}
            for keys, edge_index in edge_index_dict.items():
                key = keys[0]
                N = int(edge_index[0].max() + 1)
                num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

                key = keys[-1]
                N = int(edge_index[1].max() + 1)
                num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

        self.rowptr_dict, self.col_dict, self.rowcount_dict = {}, {}, {}
        for keys, edge_index in edge_index_dict.items():
            sizes = (num_nodes_dict[keys[0]], num_nodes_dict[keys[-1]])
            row, col = sort_edge_index(edge_index, num_nodes=max(sizes)).cpu()
            rowptr = index2ptr(row, size=sizes[0])
            self.rowptr_dict[keys] = rowptr
            self.col_dict[keys] = col
            self.rowcount_dict[keys] = rowptr[1:] - rowptr[:-1]

        for edge_type1, edge_type2 in zip(metapath[:-1], metapath[1:]):
            if edge_type1[-1] != edge_type2[0]:
                raise ValueError(
                    "Found invalid metapath. Ensure that the destination node "
                    "type matches with the source node type across all "
                    "consecutive edge types.")

        assert walk_length + 1 >= context_size
        if walk_length > len(metapath) and metapath[0][0] != metapath[-1][-1]:
            raise AttributeError(
                "The 'walk_length' is longer than the given 'metapath', but "
                "the 'metapath' does not denote a cycle")

        self.embedding_dim = embedding_dim
        self.metapath = metapath
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.num_nodes_dict = num_nodes_dict

        types = set([x[0] for x in metapath]) | set([x[-1] for x in metapath])
        types = sorted(list(types))

        count = 0
        self.start, self.end = {}, {}
        for key in types:
            self.start[key] = count
            count += num_nodes_dict[key]
            self.end[key] = count

        offset = [self.start[metapath[0][0]]]
        offset += [self.start[keys[-1]] for keys in metapath
                   ] * int((walk_length / len(metapath)) + 1)
        offset = offset[:walk_length + 1]
        assert len(offset) == walk_length + 1
        self.offset = torch.tensor(offset)

        # + 1 denotes a dummy node used to link to for isolated nodes.
        self.embedding = Embedding(count + 1, embedding_dim, sparse=sparse)
        self.dummy_idx = count

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.embedding.reset_parameters()

    def forward(self, node_type: str, batch: OptTensor = None) -> Tensor:
        r"""Returns the embeddings for the nodes in :obj:`batch` of type
        :obj:`node_type`.
        """
        emb = self.embedding.weight[self.start[node_type]:self.end[node_type]]
        return emb if batch is None else emb.index_select(0, batch)

    def loader(self, **kwargs):
        r"""Returns the data loader that creates both positive and negative
        random walks on the heterogeneous graph.

        Args:
            **kwargs (optional): Arguments of
                :class:`torch.utils.data.DataLoader`, such as
                :obj:`batch_size`, :obj:`shuffle`, :obj:`drop_last` or
                :obj:`num_workers`.
        """
        return DataLoader(range(self.num_nodes_dict[self.metapath[0][0]]),
                          collate_fn=self._sample, **kwargs)

    def _pos_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node)

        rws = [batch]
        for i in range(self.walk_length):
            edge_type = self.metapath[i % len(self.metapath)]
            batch = sample(
                self.rowptr_dict[edge_type],
                self.col_dict[edge_type],
                self.rowcount_dict[edge_type],
                batch,
                num_neighbors=1,
                dummy_idx=self.dummy_idx,
            ).view(-1)
            rws.append(batch)

        rw = torch.stack(rws, dim=-1)
        rw.add_(self.offset.view(1, -1))
        rw[rw > self.dummy_idx] = self.dummy_idx

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def _neg_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rws = [batch]
        for i in range(self.walk_length):
            keys = self.metapath[i % len(self.metapath)]
            batch = torch.randint(0, self.num_nodes_dict[keys[-1]],
                                  (batch.size(0), ), dtype=torch.long)
            rws.append(batch)

        rw = torch.stack(rws, dim=-1)
        rw.add_(self.offset.view(1, -1))

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def _sample(self, batch: List[int]) -> Tuple[Tensor, Tensor]:
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch, dtype=torch.long)
        return self._pos_sample(batch), self._neg_sample(batch)

    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        r"""Computes the loss given positive and negative random walks."""
        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(pos_rw.size(0), 1,
                                             self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(pos_rw.size(0), -1,
                                                    self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(neg_rw.size(0), 1,
                                             self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(neg_rw.size(0), -1,
                                                    self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

        return pos_loss + neg_loss

    def test(self, train_z: Tensor, train_y: Tensor, test_z: Tensor,
             test_y: Tensor, solver: str = "lbfgs", multi_class: str = "auto",
             *args, **kwargs) -> float:
        r"""Evaluates latent space quality via a logistic regression downstream
        task.
        """
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(solver=solver, multi_class=multi_class, *args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        return (
            clf.score(test_z.detach().cpu().numpy(), test_y.detach().cpu().numpy()), 
            clf.predict_proba(train_z.detach().cpu().numpy()), 
            train_y.detach().cpu(),
            clf.predict_proba(test_z.detach().cpu().numpy()), 
            test_y.detach().cpu(),
            )

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'{self.embedding.weight.size(0) - 1}, '
                f'{self.embedding.weight.size(1)})')


def sample(rowptr: Tensor, col: Tensor, rowcount: Tensor, subset: Tensor,
           num_neighbors: int, dummy_idx: int) -> Tensor:

    mask = subset >= dummy_idx
    subset = subset.clamp(min=0, max=rowptr.numel() - 2)
    count = rowcount[subset]

    rand = torch.rand((subset.size(0), num_neighbors), device=subset.device)
    rand *= count.to(rand.dtype).view(-1, 1)
    rand = rand.to(torch.long) + rowptr[subset].view(-1, 1)
    # based on the 2.6.1 release of pyg ? released on Sep. 26, 2024
    # https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/metapath2vec.py
    rand = rand.clamp(max=col.numel() - 1)  # If last node is isolated.

    col = col[rand] if col.numel() > 0 else rand
    # col = col[rand-1] if col.numel() > 0 else rand
    col[mask | (count == 0)] = dummy_idx
    return col

from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.sparse import index2ptr

data['GENE'].y_index = torch.from_numpy(pd.RangeIndex(19330).values)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

metapath = [
    ('GENE', 'PPI', 'GENE'),
    ('GENE', 'paralog', 'GENE'),
    ('GENE', 'PPI', 'GENE'),
    ('GENE', 'gene_to_BP', 'BP'), ('BP', 'rev_gene_to_BP', 'GENE'), # GBBG
    ('GENE', 'PPI', 'GENE'),
    # ('GENE', 'gene_to_BP', 'BP'), ('BP', 'biological_process', 'BP'), ('BP', 'rev_gene_to_BP', 'GENE'), # GBBBBG
    ('GENE', 'gene_to_CC', 'CC'), ('CC', 'rev_gene_to_CC', 'GENE'), # GCCG
    ('GENE', 'PPI', 'GENE'),
    # ('GENE', 'gene_to_CC', 'CC'), ('CC', 'cellular_component', 'CC'), ('CC', 'rev_gene_to_CC', 'GENE'), # GCCCCG
    ('GENE', 'gene_to_MF', 'MF'), ('MF', 'rev_gene_to_MF', 'GENE'), # GMMG
    ('GENE', 'PPI', 'GENE'),
    # ('GENE', 'gene_to_MF', 'MF'), ('MF', 'molecular_function', 'MF'), ('MF', 'rev_gene_to_MF', 'GENE') # GMMMMG
    ('GENE', 'expressed_in', 'ANATOMY'), ('ANATOMY', 'rev_expressed_in', 'GENE'), # GAAG
    ('GENE', 'PPI', 'GENE'),
    # ('GENE', 'expressed_in', 'ANATOMY'), ('ANATOMY', 'ana_ana', 'ANATOMY'), ('ANATOMY', 'rev_expressed_in', 'GENE'), # GAAAAG
]

model = my_MetaPath2Vec(
    data.edge_index_dict, embedding_dim=64,
    metapath = metapath, walk_length = 100, context_size = 10,
    walks_per_node = 20, num_negative_samples = 5,
    num_nodes_dict = {'ANATOMY': 14337, 'BP': 27993, 'CC': 4039, 'GENE': 19330, 'MF': 11271},
    sparse = True).to(device)

loader = model.loader(batch_size=64, shuffle=True, num_workers=6)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

def train(epoch, log_steps=100, eval_steps=2000):
    model.train()

    total_loss = 0
    for i, (pos_rw, neg_rw) in enumerate(loader):
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % log_steps == 0:
            print(f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                  f'Loss: {total_loss / log_steps:.4f}')
            total_loss = 0

        if (i + 1) % eval_steps == 0:
            acc = test()
            print(f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                  f'Acc: {acc:.4f}')


@torch.no_grad()
def test(train_ratio=0.1):
    model.eval()

    z = model('GENE', batch=data['GENE'].y_index.to(device))
    y = data['GENE'].y

    perm = torch.randperm(z.size(0))
    # train_perm = perm[:int(z.size(0) * train_ratio)]
    # test_perm = perm[int(z.size(0) * train_ratio):]
    train_perm = data['GENE'].train_mask #perm[:int(z.size(0) * train_ratio)]
    test_perm = data['GENE'].test_mask # perm[int(z.size(0) * train_ratio):]

    return model.test(z[train_perm], y[train_perm], z[test_perm], y[test_perm],max_iter=150) 


for epoch in range(1, 8):
    train(epoch)
    acc, train_preds, train_targets, test_preds, test_targets = test()
    print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')

print("Saving the model.")
model_path = '/models/humans/baseline/'
# wl = walk_length, cs = context_size, wpn = walk_per_node, nns = num_nnegative_sample
model_name = 'humans_baseline_metapath2vec_n2n_wl_100_cs_10_wpn_20_nns_5' + f'_split_seed_{split_seed}_epoch_{epoch}.pyg'
torch.save(model.state_dict(), model_path+model_name)

AUROC_test_metapath = roc_auc_score(test_targets, test_preds[:,1])
AUPR_test_metapath = average_precision_score(test_targets, test_preds[:,1])
metapath_tot = pd.DataFrame(data = {'AUROC_test': AUROC_test_metapath, 'AUPR_test': AUPR_test_metapath, 'model': 'Metapath', 'random_seed':1}, index=[split_seed]) 

## for loop 2,31

for split_seed in range(2,31):
    print(f"split_seed: {split_seed}")
    j = 1
    data = create_masks_cross_val(data = data, split_seed = split_seed, cv_fold_to_test = j)

    nodes = nodes2id['GENE']
    gene_strict_tensor, target_tensor, feat = get_strict_target_tensors(target_to_use = target_of_script, remove_sex_chrom = False, features_to_use = features)
    nodes = pd.concat((nodes, feat), axis=1)

    col_to_use = list(feat.columns)
    col_to_use.remove('y')
    target_col = ['y']

    nodes.loc[:, ['train']] = data['GENE'].train_mask.numpy().astype(float)
    # I know 
    nodes.loc[:, ['val']] = data['GENE'].test_mask.numpy().astype(float) # inner cross-val test set
    nodes.loc[:, ['test']] = data['GENE'].val_mask.numpy().astype(float) # outter cross-val test set

    nodes.loc[nodes[nodes['train']==1].index, ['mask']] = 'train'
    nodes.loc[nodes[nodes['val']==1].index, ['mask']] = 'val'
    # nodes.loc[nodes[nodes['val']==1].index, ['mask']] = 'train'
    nodes.loc[nodes[nodes['test']==1].index, ['mask']] = 'test'

    print(data['GENE'].train_mask.sum())
    print(data['GENE'].test_mask.sum())

    new_train_mask = (data['GENE'].train_mask + data['GENE'].test_mask)
    new_test_mask = (data['GENE'].val_mask)
    del data['GENE'].train_mask
    del data['GENE'].val_mask
    del data['GENE'].test_mask
    data['GENE'].train_mask = new_train_mask
    data['GENE'].test_mask = new_test_mask

    print(data['GENE'].train_mask.sum())
    print(data['GENE'].test_mask.sum())

    metapath = [
    ('GENE', 'PPI', 'GENE'),
    ('GENE', 'paralog', 'GENE'),
    ('GENE', 'PPI', 'GENE'),
    ('GENE', 'gene_to_BP', 'BP'), ('BP', 'rev_gene_to_BP', 'GENE'), # GBBG
    ('GENE', 'PPI', 'GENE'),
    # ('GENE', 'gene_to_BP', 'BP'), ('BP', 'biological_process', 'BP'), ('BP', 'rev_gene_to_BP', 'GENE'), # GBBBBG
    ('GENE', 'gene_to_CC', 'CC'), ('CC', 'rev_gene_to_CC', 'GENE'), # GCCG
    ('GENE', 'PPI', 'GENE'),
    # ('GENE', 'gene_to_CC', 'CC'), ('CC', 'cellular_component', 'CC'), ('CC', 'rev_gene_to_CC', 'GENE'), # GCCCCG
    ('GENE', 'gene_to_MF', 'MF'), ('MF', 'rev_gene_to_MF', 'GENE'), # GMMG
    ('GENE', 'PPI', 'GENE'),
    # ('GENE', 'gene_to_MF', 'MF'), ('MF', 'molecular_function', 'MF'), ('MF', 'rev_gene_to_MF', 'GENE') # GMMMMG
    ('GENE', 'expressed_in', 'ANATOMY'), ('ANATOMY', 'rev_expressed_in', 'GENE'), # GAAG
    ('GENE', 'PPI', 'GENE'),
    # ('GENE', 'expressed_in', 'ANATOMY'), ('ANATOMY', 'ana_ana', 'ANATOMY'), ('ANATOMY', 'rev_expressed_in', 'GENE'), # GAAAAG
    ]

    model = my_MetaPath2Vec(
        data.edge_index_dict, embedding_dim=64,
        metapath = metapath, walk_length = 100, context_size = 10,
        walks_per_node = 20, num_negative_samples = 5,
        num_nodes_dict = {'ANATOMY': 14337, 'BP': 27993, 'CC': 4039, 'GENE': 19330, 'MF': 11271},
        sparse = True).to(device)

    loader = model.loader(batch_size=64, shuffle=True, num_workers=6)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train(epoch, log_steps=100, eval_steps=2000):
        model.train()

        total_loss = 0
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (i + 1) % log_steps == 0:
                print(f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                    f'Loss: {total_loss / log_steps:.4f}')
                total_loss = 0

            if (i + 1) % eval_steps == 0:
                acc = test()
                print(f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                    f'Acc: {acc:.4f}')


    @torch.no_grad()
    def test(train_ratio=0.1):
        model.eval()

        z = model('GENE', batch=data['GENE'].y_index.to(device))
        y = data['GENE'].y

        perm = torch.randperm(z.size(0))
        # train_perm = perm[:int(z.size(0) * train_ratio)]
        # test_perm = perm[int(z.size(0) * train_ratio):]
        train_perm = data['GENE'].train_mask #perm[:int(z.size(0) * train_ratio)]
        test_perm = data['GENE'].test_mask # perm[int(z.size(0) * train_ratio):]

        return model.test(z[train_perm], y[train_perm], z[test_perm], y[test_perm],max_iter=150) 


    for epoch in range(1, 8):
        train(epoch)
        acc, train_preds, train_targets, test_preds, test_targets = test()
        print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')

    print("Saving the model.")
    model_path = "/data-tmp/rnicolle/models/baseline/"
    # wl = walk_length, cs = context_size, wpn = walk_per_node, nns = num_nnegative_sample
    model_name = "humans_baseline_metapath2vec_n2n_wl_100_cs_10_wpn_20_nns_5" f'_split_seed_{split_seed}_epoch_{epoch}.pyg'
    torch.save(model.state_dict(), model_path+model_name)

    AUROC_test_metapath = roc_auc_score(test_targets, test_preds[:,1])
    AUPR_test_metapath = average_precision_score(test_targets, test_preds[:,1])
    metapath_tmp = pd.DataFrame(data = {'AUROC_test': AUROC_test_metapath, 'AUPR_test': AUPR_test_metapath, 'model': 'Metapath', 'random_seed':1}, index=[split_seed]) 
    metapath_tot = pd.concat([metapath_tot, metapath_tmp], axis=0)
    print("xxx")

result_path = current_path + '/results/humans/baseline/'
df_name = 'humans_baseline_metapath2vec_30_seeds.tsv'
metapath_tot.to_csv(result_path+df_name, header=True, index=False, sep='\t')

## end