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
from torch_geometric import seed_everything
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader, ImbalancedSampler, LinkNeighborLoader
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

## hyperparameters

hyperparameters = {
    'node_mode' : 'degree', #'id' or 'degree' for abcm nodes_types
    'edge_features' : False, # True = those are the true edge_weights from my network
    'Heads' : 1, #2
    'conv_number' : 2, 
    'conv_type': 'SAGE',
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
    'disjoint_loader' : False, # p2p = no need to disjoint_loader, the features for pairs will be 0 
    'zero_out_batch_features': True,
    'epochs_to_train' : 500, # maximum number of epochs to train 
    'min_epochs_to_train': 3,
    'patience': 10,
    'min_delta': 0.005,
    'auto_loop': False,
    'nodes_pairs_features': 'zeros', # for p2p or n2p only
    'remove_forbidden_value': True, # for p2p or n2p only
    'ana_edges': 'common', # 'all
    'type_to_return': 'torch', # 'numpy'
    'final_type': 'torch', # 'numpy'
    'shuffle_train': True,
    'string_threshold': 700, # confidence = {'low': 150, 'medium': 400, 'high': 700, 'highest': 900} from STRING doc
    'setting': 'edge_prediction',
    # 'target_to_use': 'essential_K562',
    'features_to_use': 'basic', # features in ['basic', 'strict', 'large']
    'global_loader_neighbors': 'all', # neighbors
    'test_set_ratio': 0.3,
    'train_neg_ratio': 10.0,
    'test_neg_ratio': 10.0,
    'edge_pred_setting': 'all_edges_fixed' # 'random_sample_negative_edges_only', 'random_sample_split_negative', 'all_edges_fixed'
}

dict_j = {1:'01', 2:'02', 3:'03', 4:'04', 5:'05', 6:'06', 7:'07', 8:'08', 9:'09', 10:'10'}

sage_aggr = hyperparameters['sage_aggr'] # 'sum'
sage_norm = hyperparameters['sage_norm'] # 'sum'
heteroconv_aggr = hyperparameters['heteroconv_aggr'] #'sum', 'cat'
heteroconv_aggr_1 = hyperparameters['heteroconv_aggr_1'] #'sum', 'cat'
heteroconv_aggr_2 = hyperparameters['heteroconv_aggr_2'] #'sum', 'cat'
heteroconv_aggr_3 = hyperparameters['heteroconv_aggr_3'] #'sum', 'cat'
activation = hyperparameters['activation']
node_mode = hyperparameters['node_mode'] # 'id' or 'degree'
sage_norm = hyperparameters['sage_norm']
Heads = hyperparameters['Heads']
ana_edges = hyperparameters['ana_edges']
string_threshold = hyperparameters['string_threshold']

epochs_to_train = hyperparameters['epochs_to_train']
global_loader_neighbors = hyperparameters['global_loader_neighbors']
test_set_ratio = hyperparameters['test_set_ratio']
train_neg_ratio = hyperparameters['train_neg_ratio']
test_neg_ratio = hyperparameters['test_neg_ratio']
size_of_batch = hyperparameters['size_of_batch']
neighbors = hyperparameters['neighbors']
edge_pred_setting = hyperparameters['edge_pred_setting']

old_path = os.getcwd()
new_path = old_path + '/../../'
os.chdir(new_path)
current_path = os.getcwd()
data_raw_path = current_path + '/data/raw/'
data_processed_path = current_path + '/data/processed/'
result_path = current_path + '/results/humans/edge_pred/'
model_path = current_path + '/models/humans/edge_pred/'

# only use 'essential_K562' and 'proven_SL' there 
edge_to_predict = ('GENE', 'synthetic_lethality', 'GENE')

# 1st for_loop
for target_of_script in ['essential_K562']: # 'proven_SL' 
    # 2nd for_loop
    for SNL_type in ['unmatched', 'matched']:
        if target_of_script == 'essential_K562':
            SL_type = 'K562_SL'
            proofs = [-1, 0.7]
            # proofs = -1 corresponds to SL pairs in K562 proven by either CRISP_CRISPRi experiments or GenomeRNAi
            # proofs = 0.7 corresponds to SL pairs in K562 proven by CRISP_CRISPRi experiments
        elif target_of_script == 'proven_SL':
            SL_type = 'proven_SL'
            proofs = ['large', 'small', 'strict']
        # 3rd for_loop
        for proof in proofs: 
            if proof in ['large', -1]:
                balanced_train_set_list = [True]
            elif proof in ['small', 'strict', 0.7]:
                balanced_train_set_list = [True, False]
            # 4th for_loop
            for balanced_train_set in balanced_train_set_list:
                if balanced_train_set == True:
                    train_type = 'balanced'
                elif balanced_train_set == False:
                    train_type = 'unbalanced'
                for splits in ['50_50', '80_20']: 
                    
                    base_model_param = f"edge_pred_genes_19330_p2p_{SL_type}_{proof}_{SNL_type}_SNL_split_{splits}_disjoint_sets_train_{train_type}_STRING_{string_threshold}_{ana_edges}_ana_{activation}_abcm_{node_mode}_2HetConv_{heteroconv_aggr_1}_{heteroconv_aggr_2}_SAGE_{sage_aggr}_sagenorm_{sage_norm}"
                    
                    print(base_model_param, "\n", "XXXX"*20)
                        
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
                    # apply the columns selectors to the columns
                    # numerical_columns = numerical_columns_selector(feat)
                    # categorical_columns = categorical_columns_selector(data)
                    # create the preprocessors for the numerical and categorical columns
                    numerical_preprocessor = StandardScaler()
                    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
                    # create the global preprocessor for the data 
                    # preprocessor = ColumnTransformer([('standard_scaler', numerical_preprocessor, numerical_columns)])

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

                    def create_abcmg_data(list_of_edges: list, use_edge_attr: bool = False, feature_mode: str = 'degree', merge_choice: bool = True, my_target: str = 'essential'):
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
                    
                    def get_nop(df:pd.DataFrame, left_col: str, right_col:str):
                        df['nodes_are_ordered'] = df[left_col] < df[right_col]
                        ordered_index = df[df['nodes_are_ordered'] == True].index
                        reverse_ordered_index = df[df['nodes_are_ordered'] == False].index
                        df.loc[ordered_index, 'name_of_pair'] = df[left_col].astype(str) + '_' + df[right_col].astype(str)
                        df.loc[reverse_ordered_index, 'name_of_pair'] = df[right_col].astype(str) + '_' + df[left_col].astype(str)
                        return df

                    ## edge_pred model
                    # DO NOT USE edge_to_predict as a message_passing edge !!

                    class HeteroConv_2SAGE_act(torch.nn.Module):
                        def __init__(self, hidden_channels_1, hidden_channels_2, sage_aggr = 'sum', sage_norm = True, sage_project = False, hetero_aggr_1 = 'sum', hetero_aggr_2 = 'sum', activation = 'relu'): # 
                            super().__init__()
                            torch.manual_seed(1234567)
                            self.conv1 = HeteroConv({ 
                                    ('GENE', 'expressed_in', 'ANATOMY'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('ANATOMY', 'rev_expressed_in', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('GENE', 'PPI', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('GENE', 'CoExp', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('GENE', 'paralog', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('GENE', 'Neighbour', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('GENE', 'gene_to_BP', 'BP'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('BP', 'rev_gene_to_BP', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('GENE', 'gene_to_CC', 'CC'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('CC', 'rev_gene_to_CC', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('GENE', 'gene_to_MF', 'MF'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('MF', 'rev_gene_to_MF', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('ANATOMY', 'ana_ana', 'ANATOMY'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('ANATOMY', 'rev_ana_ana', 'ANATOMY'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('BP', 'biological_process', 'BP'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('BP', 'rev_biological_process', 'BP'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('CC', 'cellular_component', 'CC'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('CC', 'rev_cellular_component', 'CC'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('MF', 'molecular_function', 'MF'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('MF', 'rev_molecular_function', 'MF'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project),
                                    # ('GENE', 'synthetic_lethality', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project)
                                }, aggr=hetero_aggr_1)
                            self.conv2 = HeteroConv({ 
                                    ('GENE', 'expressed_in', 'ANATOMY'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('ANATOMY', 'rev_expressed_in', 'GENE'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('GENE', 'PPI', 'GENE'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('GENE', 'CoExp', 'GENE'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('GENE', 'paralog', 'GENE'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('GENE', 'Neighbour', 'GENE'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('GENE', 'gene_to_BP', 'BP'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('BP', 'rev_gene_to_BP', 'GENE'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('GENE', 'gene_to_CC', 'CC'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('CC', 'rev_gene_to_CC', 'GENE'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('GENE', 'gene_to_MF', 'MF'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('MF', 'rev_gene_to_MF', 'GENE'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('ANATOMY', 'ana_ana', 'ANATOMY'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('ANATOMY', 'rev_ana_ana', 'ANATOMY'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('BP', 'biological_process', 'BP'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('BP', 'rev_biological_process', 'BP'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('CC', 'cellular_component', 'CC'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('CC', 'rev_cellular_component', 'CC'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('MF', 'molecular_function', 'MF'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                    ('MF', 'rev_molecular_function', 'MF'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project),
                                    # ('GENE', 'synthetic_lethality', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project)
                                }, aggr=hetero_aggr_2)
                            if activation == 'relu':
                                self.activation = torch.relu
                            elif activation == 'tanh':
                                self.activation = torch.tanh
                        def forward(self, x_dict, edge_index_dict):
                            x_dict = self.conv1(x_dict, edge_index_dict)
                            x_dict = {key: self.activation(x) for key, x in x_dict.items()}
                            x_dict = {key: F.dropout(x, p = 0.5, training = self.training) for key, x in x_dict.items()}
                            x_dict = self.conv2(x_dict, edge_index_dict)
                            x_dict = {key: self.activation(x) for key, x in x_dict.items()}
                            x_dict = {key: F.dropout(x, p = 0.5, training = self.training) for key, x in x_dict.items()}
                            return x_dict

                    class EdgeDecoder(torch.nn.Module):
                        def __init__(self, hidden_channels):
                            super().__init__()
                            self.lin1 = Linear(2 * hidden_channels, hidden_channels)
                            self.lin2 = Linear(hidden_channels, 1)

                        def forward(self, z_dict, edge_label_index):
                            row, col = edge_label_index
                            z = torch.cat([z_dict['GENE'][row], z_dict['GENE'][col]], dim = -1)
                            z = self.lin1(z).relu()
                            z = self.lin2(z)
                            # return z.view(-1)
                            return torch.sigmoid(z.view(-1))
                        
                    class Model_act(torch.nn.Module):
                        def __init__(self, hc_1, hc_2, sage_aggr='mean', heteroconv_aggr_1='mean', heteroconv_aggr_2='mean', act='tanh'):
                            super().__init__()
                            # self.encoder = GATv2Conv_no_attr(hidden_channels_1 = hidden_channels_1, hidden_channels_2 = hidden_channels_2, number_of_heads = 1, hetero_aggr_1 = 'sum', hetero_aggr_2 = 'sum')
                            self.encoder = HeteroConv_2SAGE_act(hidden_channels_1=hc_1, hidden_channels_2=hc_2, sage_aggr=sage_aggr, hetero_aggr_1=heteroconv_aggr_1, hetero_aggr_2=heteroconv_aggr_2, activation=act)
                            self.decoder = EdgeDecoder(hidden_channels = hc_2)
                            # self.loss = torch.nn.BCEWithLogitsLoss()
                        def forward(self, x_dict, edge_index_dict, edge_label_index):
                            z_dict = self.encoder(x_dict, edge_index_dict)
                            return self.decoder(z_dict, edge_label_index)

                    ### plotting function 

                    def plot_df_edges(df_edges: pd.DataFrame, val_set: bool = False):
                        # In this setting, the val_set sometimes contains only positive (or only negative) class
                        # Use the val_set as a 'full_test_set' while the 'test_set' is the 'clean_dataset'
                        # get the prediction
                        out_train_arr = df_edges.loc[df_edges[df_edges['mask']=='train'].index, 'pred'].values
                        out_test_arr = df_edges.loc[df_edges[df_edges['mask']=='test'].index, 'pred'].values
                        if val_set == True:
                            out_val_arr = df_edges.loc[df_edges[df_edges['mask']=='val'].index, 'pred'].values
                            out_val_arr = np.concatenate((out_val_arr, out_test_arr), axis=0)

                        # get the ground truth
                        target_train_arr = df_edges.loc[df_edges[df_edges['mask']=='train'].index, 'target'].values
                        auroc_train = roc_auc_score(target_train_arr, out_train_arr)
                        aupr_train = average_precision_score(target_train_arr, out_train_arr)

                        target_test_arr = df_edges.loc[df_edges[df_edges['mask']=='test'].index, 'target'].values
                        auroc_test = roc_auc_score(target_test_arr, out_test_arr)
                        aupr_test = average_precision_score(target_test_arr, out_test_arr)

                        if val_set == True:
                            target_val_arr = df_edges.loc[df_edges[df_edges['mask']=='val'].index, 'target'].values
                            target_val_arr = np.concatenate((target_val_arr, target_test_arr), axis=0)
                            auroc_val = roc_auc_score(target_val_arr, out_val_arr)
                            aupr_val = average_precision_score(target_val_arr, out_val_arr)

                        ## get the predicted classes for a given threshold
                        '''F1 score, Matthews correlation coefficient, F0.01 score and Confusion Matrix'''
                        ## train_set
                        f1_scores_train = {}
                        mcc_scores_train = {}
                        beta = 0.01
                        f001_scores_train = {} 
                        for t in np.arange(0, 1, 0.01):
                            yhat_train = (out_train_arr > t).astype(float)
                            f1_train = f1_score(target_train_arr, yhat_train)
                            f1_scores_train[t] = f1_train
                            mcc_train = matthews_corrcoef(target_train_arr, yhat_train)
                            mcc_scores_train[t] = mcc_train  
                            _, _, f001_train, _ = precision_recall_fscore_support(target_train_arr, yhat_train, beta = beta, zero_division = 0)
                            f001_scores_train[t] = f001_train[1]
                        best_f1_t_train, best_f1_train = max(f1_scores_train.items(), key = lambda k: k[1])
                        best_mcc_t_train, best_mcc_train = max(mcc_scores_train.items(), key = lambda k: k[1])
                        best_f001_t_train, best_f001_train = max(f001_scores_train.items(), key = lambda k: k[1])
                        if best_f001_t_train < 0.99:
                            pass 
                        else:
                            f001_scores_train = {}
                            for t in np.arange(0.99, 1,  0.001):
                                yhat_train = (out_train_arr > t).astype(float)
                                _, _, f001_train, _ = precision_recall_fscore_support(target_train_arr, yhat_train, beta = beta, zero_division = 0)
                                f001_scores_train[t] = f001_train[1]
                            best_f001_t_train, best_f001_train = max(f001_scores_train.items(), key = lambda k: k[1])
                        ## val_set
                        if val_set == True:
                            f1_scores_val = {}
                            mcc_scores_val = {}
                            beta = 0.01
                            f001_scores_val = {} 
                            for t in np.arange(0, 1, 0.01):
                                yhat_val = (out_val_arr > t).astype(float)
                                f1_val = f1_score(target_val_arr, yhat_val)
                                f1_scores_val[t] = f1_val
                                mcc_val = matthews_corrcoef(target_val_arr, yhat_val)
                                mcc_scores_val[t] = mcc_val  
                                _, _, f001_val, _ = precision_recall_fscore_support(target_val_arr, yhat_val, beta = beta, zero_division = 0)
                                f001_scores_val[t] = f001_val[1]
                            best_f1_t_val, best_f1_val = max(f1_scores_val.items(), key = lambda k: k[1])
                            best_mcc_t_val, best_mcc_val = max(mcc_scores_val.items(), key = lambda k: k[1])
                            best_f001_t_val, best_f001_val = max(f001_scores_val.items(), key = lambda k: k[1])
                            if best_f001_t_val < 0.99:
                                pass 
                            else:
                                f001_scores_val = {}
                                for t in np.arange(0.99, 1,  0.001):
                                    yhat_val = (out_val_arr > t).astype(float)
                                    _, _, f001_val, _ = precision_recall_fscore_support(target_val_arr, yhat_val, beta = beta, zero_division = 0)
                                    f001_scores_val[t] = f001_val[1]
                                best_f001_t_val, best_f001_val = max(f001_scores_val.items(), key = lambda k: k[1])
                        ## test_set
                        f1_scores_test = {}
                        mcc_scores_test = {}
                        beta = 0.01
                        f001_scores_test = {}
                        for t in np.arange(0, 1, 0.01):
                            yhat_test = (out_test_arr > t).astype(float)
                            f1_test = f1_score(target_test_arr, yhat_test)
                            f1_scores_test[t] = f1_test
                            mcc_test = matthews_corrcoef(target_test_arr, yhat_test)
                            mcc_scores_test[t] = mcc_test
                            _, _, f001_test, _ = precision_recall_fscore_support(target_test_arr, yhat_test, beta = beta, zero_division = 0)
                            f001_scores_test[t] = f001_test[1]
                        best_f1_t_test, best_f1_test = max(f1_scores_test.items(), key = lambda k: k[1])
                        best_mcc_t_test, best_mcc_test = max(mcc_scores_test.items(), key = lambda k: k[1])
                        best_f001_t_test, best_f001_test = max(f001_scores_test.items(), key = lambda k: k[1])
                        if best_f001_t_test < 0.99:
                            pass
                        else:
                            f001_scores_test = {}
                            for t in np.arange(0.99, 1,  0.001):
                                yhat_test = (out_test_arr > t).astype(float)
                                _, _, f001_test, _ = precision_recall_fscore_support(target_test_arr, yhat_test, beta = beta, zero_division = 0)
                                f001_scores_test[t] = f001_test[1]
                            best_f001_t_test, best_f001_test = max(f001_scores_test.items(), key = lambda k: k[1])  

                        ## get the confusion matrices
                        # train_set
                        f1_train_yhat_arr = (out_train_arr > best_f1_t_train).astype(float)
                        f1_cm_train = confusion_matrix(target_train_arr, f1_train_yhat_arr.ravel())
                        f001_train_yhat_arr = (out_train_arr > best_f001_t_train).astype(float)
                        f001_cm_train = confusion_matrix(target_train_arr, f001_train_yhat_arr.ravel())
                        mcc_train_yhat_arr = (out_train_arr > best_mcc_t_train).astype(float)
                        mcc_cm_train = confusion_matrix(target_train_arr, mcc_train_yhat_arr.ravel())
                        # val_set
                        if val_set == True:
                            f1_val_yhat_arr = (out_val_arr > best_f1_t_val).astype(float)
                            f1_cm_val = confusion_matrix(target_val_arr, f1_val_yhat_arr.ravel())
                            f001_val_yhat_arr = (out_val_arr > best_f001_t_val).astype(float)
                            f001_cm_val = confusion_matrix(target_val_arr, f001_val_yhat_arr.ravel())
                            mcc_val_yhat_arr = (out_val_arr > best_mcc_t_val).astype(float)
                            mcc_cm_val = confusion_matrix(target_val_arr, mcc_val_yhat_arr.ravel())
                        # test_set
                        f1_test_yhat_arr = (out_test_arr > best_f1_t_test).astype(float)
                        f1_cm_test = confusion_matrix(target_test_arr, f1_test_yhat_arr.ravel())
                        f001_test_yhat_arr = (out_test_arr > best_f001_t_test).astype(float)
                        f001_cm_test = confusion_matrix(target_test_arr, f001_test_yhat_arr.ravel())
                        mcc_test_yhat_arr = (out_test_arr > best_mcc_t_test).astype(float)
                        mcc_cm_test = confusion_matrix(target_test_arr, mcc_test_yhat_arr.ravel())

                        '''ROC and PR curves'''
                        # on the train set
                        ns_probs_train = [0 for _ in range(len(out_train_arr))]
                        ns_fpr, ns_tpr, _ = roc_curve(target_train_arr, ns_probs_train)
                        model_fpr_train, model_tpr_train, _ = roc_curve(target_train_arr, out_train_arr)
                        # no_skill_train = len(data['GENE'].y[data['GENE'].train_mask][data['GENE'].y[data['GENE'].train_mask] == 1])/len(data['GENE'].y[data['GENE'].train_mask]) # no_skill = 0.15
                        no_skill_train = len(target_train_arr[target_train_arr==1])/len(target_train_arr)
                        precision_train, recall_train, _ = precision_recall_curve(target_train_arr, out_train_arr)
                        # on the val set
                        if val_set == True:
                            ns_probs_val = [0 for _ in range(len(out_train_arr))]
                            # ns_fpr_val, ns_tpr_val, _ = roc_curve(target_val_arr, ns_probs_val)
                            model_fpr_val, model_tpr_val, _ = roc_curve(target_val_arr, out_val_arr)
                            no_skill_val = len(data['GENE'].y[data['GENE'].val_mask][data['GENE'].y[data['GENE'].val_mask] == 1])/len(data['GENE'].y[data['GENE'].val_mask])
                            precision_val, recall_val, _ = precision_recall_curve(target_val_arr, out_val_arr) 
                        # on test set
                        ns_probs_test = [0 for _ in range(len(out_test_arr))]
                        # ns_fpr_test, ns_tpr_test, _ = roc_curve(target_test_arr, ns_probs_test)
                        model_fpr_test, model_tpr_test, _ = roc_curve(target_test_arr, out_test_arr)
                        # no_skill_test = len(data['GENE'].y[data['GENE'].test_mask][data['GENE'].y[data['GENE'].test_mask] == 1])/len(data['GENE'].y[data['GENE'].test_mask])
                        no_skill_test = len(target_test_arr[target_test_arr==1])/len(target_test_arr)
                        precision_test, recall_test, _ = precision_recall_curve(target_test_arr, out_test_arr) 

                        '''Plot the figure'''
                        # figsize = (float, float), (Width, Height) in inches
                        if val_set:
                            fig = plt.figure(figsize = (18, 24), constrained_layout = True)
                            fig.suptitle("Performances of the models", fontsize = 'xx-large')
                            # create 4 horizontal subfigures
                            subfigs = fig.subfigures(4, 1)
                            # subfigs[0].suptitle('AUROC and AUPR', fontsize = 'x-large')
                            subfigs[1].suptitle('Confusion Matrix MCC-score', fontsize = 'x-large')
                            subfigs[2].suptitle('Confusion Matrix F1-score', fontsize = 'x-large')
                            subfigs[3].suptitle('Confusion Matrix F0.01-score', fontsize = 'x-large')
                            # create 3 vertical sub-sub-figure for each sub-figure
                            ax_roc, ax_pr, ax_pairs = subfigs[0].subplots(1, 3)
                            ax_mcc_train, ax_mcc_val, ax_mcc_test = subfigs[1].subplots(1, 3)
                            ax_f1_train, ax_f1_val, ax_f1_test = subfigs[2].subplots(1, 3)
                            ax_f001_train, ax_f001_val, ax_f001_test = subfigs[3].subplots(1, 3)
                            ## do the first plot: AUROC
                            ax_roc.plot(ns_fpr, ns_tpr, linestyle='--', label='Random Model', color = 'black')
                            # ax_roc.plot(ns_fpr_val, ns_tpr_val, linestyle='--', label='Random Model on Val set', color = 'darkgreen')
                            # ax_roc.plot(ns_fpr_test, ns_tpr_test, linestyle='--', label='Random Model on Test set', color = 'darkgreen')
                            ax_roc.plot(model_fpr_train, model_tpr_train, marker='.', label=f'Train set: AUC = {auroc_train:0.4f}', color = 'tab:orange') #'#ff7f0e'
                            ax_roc.plot(model_fpr_val, model_tpr_val, marker='.', label=f'Full Test set: AUC = {auroc_val:0.4f}', color = 'tab:green')
                            ax_roc.plot(model_fpr_test, model_tpr_test, marker='.', label=f'Correct Test set: AUC = {auroc_test:0.4f}', color = 'tab:blue') # '#1f77b4'
                            # axis labels
                            ax_roc.set_xlabel('False Positive Rate')
                            ax_roc.set_ylabel('True Positive Rate')
                            ax_roc.legend(loc = 'lower right')
                            ax_roc.set_title("ROC curves")
                            ## do the second plot: AUPR
                            ax_pr.plot([0, 1], [no_skill_train, no_skill_train], linestyle='--', label='Random Model on Train set', color = 'tab:brown')
                            ax_pr.plot([0, 1], [no_skill_val, no_skill_val], linestyle='--', label='Random Model on Validation set', color = 'tab:olive')
                            ax_pr.plot([0, 1], [no_skill_test, no_skill_test], linestyle='--', label='Random Model on Test set', color = 'tab:cyan')
                            ax_pr.plot([0, 1], [0, 0], linestyle='--', label = None, color = 'purple', alpha = 0) # juste pour avoir la même échelle, tracer une ligne invisible, y = 0
                            ax_pr.plot(recall_train, precision_train, marker='.', label=f'Train set: AUPR = {aupr_train:0.4f}', color = 'tab:orange') # '#ff7f0e'
                            ax_pr.plot(recall_val, precision_val, marker='.', label=f'Full Test set: AUPR = {aupr_val:0.4f}', color = 'tab:green')
                            ax_pr.plot(recall_test, precision_test, marker='.', label=f'Correct Test set: AUPR = {aupr_test:0.4f}', color = 'tab:blue') # '#1f77b4'
                            # axis labels
                            ax_pr.set_xlabel('Recall')
                            ax_pr.set_ylabel('Precision')
                            ax_pr.legend(loc = 'lower left')
                            ax_pr.set_title("PR curves")
                            ## do the third plot: corrected prediction
                            ## 
                            
                            ## do the fourth plot: confusion matrix train dataset MCC
                            ConfusionMatrixDisplay(mcc_cm_train).plot(ax = ax_mcc_train)
                            ax_mcc_train.set_title(f"Train set: MCC = {best_mcc_train:0.4f} with t = {best_mcc_t_train:0.4f}")
                            ## do the fifth plot: confusion matrix val dataset MCC
                            ConfusionMatrixDisplay(mcc_cm_val).plot(ax = ax_mcc_val)
                            ax_mcc_val.set_title(f"Val set: MCC = {best_mcc_val:0.4f} with t = {best_mcc_t_val:0.4f}")
                            ## do the sixth plot: confusion matrix test dataset MCC
                            ConfusionMatrixDisplay(mcc_cm_test).plot(ax = ax_mcc_test)
                            ax_mcc_test.set_title(f"Test set: MCC = {best_mcc_test:0.4f} with t = {best_mcc_t_test:0.4f}")
                            
                            ## do the seventh plot: confusion matrix train dataset F1
                            ConfusionMatrixDisplay(f1_cm_train).plot(ax = ax_f1_train)
                            ax_f1_train.set_title(f"Train set: F1 = {best_f1_train:0.4f} with t = {best_f1_t_train:0.4f}")
                            ## do the eighth plot: confusion matrix val dataset F1
                            ConfusionMatrixDisplay(f1_cm_val).plot(ax = ax_f1_val)
                            ax_f1_val.set_title(f"Validation set: F1 = {best_f1_val:0.4f} with t = {best_f1_t_val:0.4f}")
                            ## do the ninth plot: confusion matrix test dataset F1
                            ConfusionMatrixDisplay(f1_cm_test).plot(ax = ax_f1_test)
                            ax_f1_test.set_title(f"Test set: F1 = {best_f1_test:0.4f} with t = {best_f1_t_test:0.4f}")

                            # do the tenth plot: confusion matrix train dataset F001
                            ConfusionMatrixDisplay(f001_cm_train).plot(ax = ax_f001_train)
                            ax_f001_train.set_title(f"Train set: F-001 = {best_f001_train:0.4f} with t = {best_f001_t_train:0.4f}")
                            # do the eleventh plot: confusion matrix val dataset F001
                            ConfusionMatrixDisplay(f001_cm_val).plot(ax = ax_f001_val)
                            ax_f001_val.set_title(f"Validation set: F-001 = {best_f001_val:0.4f} with t = {best_f001_t_val:0.4f}")    
                            # do the twelfth plot: confusion matrix test dataset F001
                            ConfusionMatrixDisplay(f001_cm_test).plot(ax = ax_f001_test)
                            ax_f001_test.set_title(f"Test set: F-001 = {best_f001_test:0.4f} with t = {best_f001_t_test:0.4f}")

                            # show the plot
                            plt.show()

                            data_df = {'results': [auroc_train, auroc_val, auroc_test, aupr_train, aupr_val, aupr_test,
                                                best_mcc_train, best_mcc_t_train, np.ravel(mcc_cm_train),
                                                best_mcc_val, best_mcc_t_val, np.ravel(mcc_cm_val),
                                                best_mcc_test, best_mcc_t_test, np.ravel(mcc_cm_test),
                                                best_f1_train, best_f1_t_train, np.ravel(f1_cm_train), 
                                                best_f1_val, best_f1_t_val, np.ravel(f1_cm_val), 
                                                best_f1_test, best_f1_t_test, np.ravel(f1_cm_test),
                                                best_f001_train, best_f001_t_train, np.ravel(f001_cm_train), 
                                                best_f001_val, best_f001_t_val, np.ravel(f001_cm_val), 
                                                best_f001_test, best_f001_t_test, np.ravel(f001_cm_test)]}
                            
                            col_names = [
                                'AUROC_train', 'AUROC_val', 'AUROC_test', 'AUPR_train', 'AUPR_val', 'AUPR_test',
                                'best_MCC_train', 'best_MCC_train_threshold', 'best_MCC_train_cm',
                                'best_MCC_val', 'best_MCC_val_threshold', 'best_MCC_val_cm',
                                'best_MCC_test', 'best_MCC_test_threshold', 'best_MCC_test_cm',
                                'best_F1_train', 'best_F1_train_threshold', 'best_F1_train_cm',
                                'best_F1_val', 'best_F1_val_threshold', 'best_F1_val_cm',
                                'best_F1_test', 'best_F1_test_threshold', 'best_F1_test_cm',
                                'best_F0.01_train', 'best_F0.01_train_threshold', 'best_F0.01_train_cm',
                                'best_F0.01_val', 'best_F0.01_val_threshold', 'best_F0.01_val_cm',
                                'best_F0.01_test', 'best_F0.01_test_threshold', 'best_F0.01_test_cm']
                            df = pd.DataFrame.from_dict(data_df, orient = 'index', columns = col_names)
                            
                        else: # use_val_set == False
                            fig = plt.figure(figsize = (12, 24), constrained_layout = True)
                            fig.suptitle("Performances of the models", fontsize = 'xx-large')
                            # create 4 horizontal subfigures
                            subfigs = fig.subfigures(4, 1)
                            # subfigs[0].suptitle('AUROC and AUPR', fontsize = 'x-large')
                            subfigs[1].suptitle('Confusion Matrix MCC-score', fontsize = 'x-large')
                            subfigs[2].suptitle('Confusion Matrix F1-score', fontsize = 'x-large')
                            subfigs[3].suptitle('Confusion Matrix F0.01-score', fontsize = 'x-large')
                            # create 2 vertical sub-sub-figure for each sub-figure
                            ax_roc, ax_pr = subfigs[0].subplots(1, 2)
                            ax_mcc_train, ax_mcc_test = subfigs[1].subplots(1, 2)
                            ax_f1_train, ax_f1_test = subfigs[2].subplots(1, 2)
                            ax_f001_train, ax_f001_test = subfigs[3].subplots(1, 2)
                            ## do the first plot: AUROC
                            ax_roc.plot(ns_fpr, ns_tpr, linestyle='--', label='Random Model', color = 'black')
                            # ax_roc.plot(ns_fpr_test, ns_tpr_test, linestyle='--', label='Random Model on Test set', color = 'darkgreen')
                            ax_roc.plot(model_fpr_train, model_tpr_train, marker='.', label=f'Train set: AUC = {auroc_train:0.4f}', color = 'tab:orange') #'#ff7f0e'
                            ax_roc.plot(model_fpr_test, model_tpr_test, marker='.', label=f'Test set: AUC = {auroc_test:0.4f}', color = 'tab:blue') # '#1f77b4'
                            # axis labels
                            ax_roc.set_xlabel('False Positive Rate')
                            ax_roc.set_ylabel('True Positive Rate')
                            ax_roc.legend(loc = 'lower right')
                            ax_roc.set_title("ROC curves")
                            ## do the second plot: AUPR
                            ax_pr.plot([0, 1], [no_skill_train, no_skill_train], linestyle='--', label='Random Model on Train set', color = 'tab:brown')
                            ax_pr.plot([0, 1], [no_skill_test, no_skill_test], linestyle='--', label='Random Model on Test set', color = 'tab:cyan')
                            ax_pr.plot([0, 1], [0, 0], linestyle='--', label = None, color = 'purple', alpha = 0) # juste pour avoir la même échelle, tracer une ligne invisible, y = 0
                            ax_pr.plot(recall_train, precision_train, marker='.', label=f'Train set: AUPR = {aupr_train:0.4f}', color = 'tab:orange') # '#ff7f0e'
                            ax_pr.plot(recall_test, precision_test, marker='.', label=f'Test set: AUPR = {aupr_test:0.4f}', color = 'tab:blue') # '#1f77b4'
                            # axis labels
                            ax_pr.set_xlabel('Recall')
                            ax_pr.set_ylabel('Precision')
                            ax_pr.legend(loc = 'lower left')
                            ax_pr.set_title("PR curves")

                            ## do the third plot: confusion matrix train dataset MCC
                            ConfusionMatrixDisplay(mcc_cm_train).plot(ax = ax_mcc_train)
                            ax_mcc_train.set_title(f"Train set: MCC = {best_mcc_train:0.4f} with t = {best_mcc_t_train:0.4f}")
                            ## do the fourth plot: confusion matrix test dataset MCC
                            ConfusionMatrixDisplay(mcc_cm_test).plot(ax = ax_mcc_test)
                            ax_mcc_test.set_title(f"Test set: MCC = {best_mcc_test:0.4f} with t = {best_mcc_t_test:0.4f}")
                            
                            ## do the fifth plot: confusion matrix train dataset F1
                            ConfusionMatrixDisplay(f1_cm_train).plot(ax = ax_f1_train)
                            ax_f1_train.set_title(f"Train set: F1 = {best_f1_train:0.4f} with t = {best_f1_t_train:0.4f}")
                            ## do the sixth plot: confusion matrix test dataset F1
                            ConfusionMatrixDisplay(f1_cm_test).plot(ax = ax_f1_test)
                            ax_f1_test.set_title(f"Validation set: F1 = {best_f1_test:0.4f} with t = {best_f1_t_test:0.4f}")

                            # do the seventh plot: confusion matrix train dataset F001
                            ConfusionMatrixDisplay(f001_cm_train).plot(ax = ax_f001_train)
                            ax_f001_train.set_title(f"Train set: F-001 = {best_f001_train:0.4f} with t = {best_f001_t_train:0.4f}")  
                            # do the eighth plot: confusion matrix test dataset F001
                            ConfusionMatrixDisplay(f001_cm_test).plot(ax = ax_f001_test)
                            ax_f001_test.set_title(f"Test set: F-001 = {best_f001_test:0.4f} with t = {best_f001_t_test:0.4f}")

                            # show the plot
                            plt.show()

                        return fig

                    ## create data

                    # create the data now
                    gene_mode = hyperparameters['gene_mode']
                    node_mode = hyperparameters['node_mode'] # 'id' or 'degree'
                    edge_features = hyperparameters['edge_features'] # True or False

                    if ana_edges in ['all', 'common']:
                        print(f"I am using ana_edges: {ana_edges}.")
                        list_of_edges, data = create_abcmg_data(['expressed_in', 'ana_ana',
                                                            'gene_to_BP', 'biological_process', 
                                                            'gene_to_CC', 'cellular_component', 
                                                            'gene_to_MF', 'molecular_function'
                                                                ], feature_mode = node_mode, use_edge_attr = edge_features, merge_choice = False, my_target = target_of_script) 
                    else: 
                        print("I am not using expression edges.")
                        list_of_edges, data = create_abcmg_data([#'expressed_in', 'ana_ana',
                                                            'gene_to_BP', 'biological_process', 
                                                            'gene_to_CC', 'cellular_component', 
                                                            'gene_to_MF', 'molecular_function'
                                                                ], feature_mode = node_mode, use_edge_attr = edge_features, merge_choice = False, my_target = target_of_script) # full_genes_features = True,
                        
                    list_of_edges_2, data_2 = create_abcmg_data(['PPI', 'paralog'#, 'synthetic_lethality' #, 'CoExp', 'Neighbour', 
                                                                ], use_edge_attr = edge_features, merge_choice = True, my_target = target_of_script) # full_genes_features = True,
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

                    # TO DO coalesce at this step, the edges for individual genes 
                    for edge in data.edge_index_dict.keys():
                        data[edge].edge_index = coalesce(data[edge].edge_index)

                    data = data.to('cpu')

                    if proof in [-1, 'large']:
                        df_pairs_name = f'20250717_SL_{SNL_type}_SNL_{target_of_script}_{proof}_disjoint_sets_splits_05_08.tsv'
                    elif proof in [0.7, 'small', 'strict']:
                        df_pairs_name = f'20250717_SL_{SNL_type}_SNL_{target_of_script}_{proof}_{train_type}_train_disjoint_sets_splits_05_08.tsv'
                    print(df_pairs_name)
                    fd = pd.read_csv(data_processed_path + df_pairs_name, header=0, sep='\t', dtype={6:str, 7:str})
                    print(f'The shape of the pairs_df is: {fd.shape}')

                    nodes = fd[~fd['GENE_id'].str.contains('_')]

                    if splits == '80_20':
                        fd.loc[:,'mask'] = fd['mask_08'].values
                    elif splits == '50_50':
                        fd.loc[:,'mask'] = fd['mask_05'].values

                    fd_basic_columns = fd.columns

                    ## create edge_index

                    # for the train_data
                    train_source = torch.from_numpy(fd[(fd['mask']=='train')]['mapped_id_left'].values)
                    train_destination = torch.from_numpy(fd[(fd['mask']=='train')]['mapped_id_right'].values)
                    train_labels = torch.from_numpy(fd[(fd['mask']=='train')]['target'].values)
                    train_labels_tot = torch.cat([train_labels, train_labels])
                    train_source_tot = torch.cat([train_source, train_destination])
                    train_destination_tot = torch.cat([train_destination, train_source])
                    train_edge_index_tot = torch.stack([train_source_tot, train_destination_tot], dim = 0)
                    # for the val_data
                    val_source = torch.from_numpy(fd[(fd['mask']=='val')]['mapped_id_left'].values)
                    val_destination = torch.from_numpy(fd[(fd['mask']=='val')]['mapped_id_right'].values)
                    val_labels = torch.from_numpy(fd[(fd['mask']=='val')]['target'].values)
                    val_source_tot = torch.cat([val_source, val_destination])
                    val_destination_tot = torch.cat([val_destination, val_source])
                    val_edge_label_index = torch.stack([val_source, val_destination], dim = 0)
                    val_edge_index_tot = torch.stack([val_source_tot, val_destination_tot], dim = 0)
                    # for the test_data
                    test_source = torch.from_numpy(fd[(fd['mask']=='test')]['mapped_id_left'].values)
                    test_destination = torch.from_numpy(fd[(fd['mask']=='test')]['mapped_id_right'].values)
                    test_labels = torch.from_numpy(fd[(fd['mask']=='test')]['target'].values)
                    test_source_tot = torch.cat([test_source, test_destination])
                    test_destination_tot = torch.cat([test_destination, test_source])
                    test_edge_label_index = torch.stack([test_source, test_destination], dim = 0)
                    test_edge_index_tot = torch.stack([test_source_tot, test_destination_tot], dim = 0)
                    # give the edges to data
                    data[edge_to_predict].edge_index = train_edge_index_tot.type(torch.LongTensor)
                    data[edge_to_predict].edge_label = train_labels

                    ##  RandomLinkSplit

                    print("To summarize, the parameters chosen are as follows:")
                    print(f"target_to_use: {target_of_script}")
                    print(f"edge_to_predict: {edge_to_predict}")

                    transform = T.RandomLinkSplit(
                        num_val=0,     # manually put val_edges if I use them
                        num_test=0,    # test_set_ratio
                        key = 'edge_label',
                        is_undirected = True, # prevent data_leakage when the edge to predict is undirected
                        disjoint_train_ratio=0, # put 0.0 = no disjoint between message_passing and supervision edges but do not use the edge SL/OLIDA in message_passing !!
                        neg_sampling_ratio=0, # 
                        add_negative_train_samples=False, # the negative edges are added only to val_data and test_data, not train_data 
                        edge_types=edge_to_predict,
                        rev_edge_types=edge_to_predict,
                    )

                    # train_data
                    train_data, val_data, test_data = transform(data)
                    # test_data
                    test_data[edge_to_predict].edge_label = test_labels
                    test_data[edge_to_predict].edge_label_index = test_edge_label_index.type(torch.LongTensor)
                    # val_data
                    val_data[edge_to_predict].edge_label = val_labels
                    val_data[edge_to_predict].edge_label_index = val_edge_label_index.type(torch.LongTensor)

                    use_val_set = True

                    train_loader = LinkNeighborLoader(
                        data = train_data,
                        num_neighbors = {key: [neighbors, neighbors] for key in data.edge_types},
                        neg_sampling_ratio = 0, # put a value there, to randomly sample on the fly in the train_data = negative edges are different across epochs
                        edge_label_index = (edge_to_predict, train_data[edge_to_predict].edge_label_index),
                        edge_label = train_data[edge_to_predict].edge_label,
                        batch_size = size_of_batch,
                        shuffle = True
                    )

                    if use_val_set:
                        val_loader = LinkNeighborLoader(
                            data = val_data,
                            num_neighbors = {key: [-1, -1] for key in data.edge_types},
                            neg_sampling = None,
                            edge_label_index = (edge_to_predict, val_data[edge_to_predict].edge_label_index),
                            edge_label = val_data[edge_to_predict].edge_label,
                            batch_size = size_of_batch,
                            shuffle = False
                        )

                    test_loader = LinkNeighborLoader(
                        data = test_data,
                        num_neighbors = {key: [-1, -1] for key in data.edge_types},
                        neg_sampling = None, # put None there, the negative samples are fixed in the initial RandomLinkSplit 
                        edge_label_index = (edge_to_predict, test_data[edge_to_predict].edge_label_index),
                        edge_label = test_data[edge_to_predict].edge_label,
                        batch_size = size_of_batch,
                        shuffle = False
                    )

                    for edge in data.edge_types:
                        data[edge].edge_index = U.sort_edge_index(data[edge].edge_index)
                    data = data.to('cpu')

                    ## create model

                    for i in [16,32,64]:
                        ## indent begins here 
                        hc_1 = i
                        hc_2_q = hyperparameters['hc_2_q']
                        hc_3_q = hyperparameters['hc_3_q']
                        hc_2 = int(i/hc_2_q)
                        # hc_3 = int(i/hc_3_q)

                        # hyperparameters
                        Heads = hyperparameters['Heads']             
                        conv_number = hyperparameters['conv_number']    
                        sage_aggr = hyperparameters['sage_aggr'] 
                        sage_norm = hyperparameters['sage_norm']
                        sage_project = hyperparameters['sage_project']
                        heteroconv_aggr_1 = hyperparameters['heteroconv_aggr_1'] #'sum', 'cat'
                        heteroconv_aggr_2 = hyperparameters['heteroconv_aggr_2'] #'sum', 'cat'
                        heteroconv_aggr_3 = hyperparameters['heteroconv_aggr_3'] #'sum', 'cat'
                        epochs_to_train = hyperparameters['epochs_to_train'] 
                        min_epochs_to_train = hyperparameters['min_epochs_to_train']
                        patience = hyperparameters['patience']
                        min_delta = hyperparameters['min_delta']

                        print("XXX" * 40)
                        print("This is Humans: link prediction setting on synthetic_lethality_edges.")
                        print(f"Now hidden_channels_1 = {hc_1} and hidden_channels_2 = {hc_2} and batch = {size_of_batch}.")

                        outter_model_param = base_model_param + f"_{hc_1}_{int(hc_2)}_batch_{size_of_batch}_neighbors_{neighbors}"

                        model = Model_act(hc_1=i, hc_2=i, sage_aggr=sage_aggr, heteroconv_aggr_1=heteroconv_aggr_1, heteroconv_aggr_2=heteroconv_aggr_2, act=activation)

                        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
                        # device = 'cpu'

                        model = model.to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                        criterion = torch.nn.BCELoss()

                        ## define train/test functions

                        patience = hyperparameters['patience']
                        min_delta = hyperparameters['min_delta']

                        class EarlyStopping:
                            def __init__(self, patience = 5, min_delta = 0, criterion_type: str = 'decreasing'):
                                self.patience = patience
                                self.min_delta = min_delta
                                self.counter = 0
                                self.criterion_type = criterion_type
                                self.min_criterion = 1000000000 # for values that are supposed to decrease during training, e.g. loss
                                self.max_criterion = 0 # for value that are supposed to increase during training, e.g. AUROC, AUPR, accuracy
                            # Return True when the loss doesn't decrease by the min_delta for at least 'patience' epochs
                            # or return True when the AUROC  doesn't increase by at least min_delta for at least 'patience' epochs
                            def early_stop_check(self, current_value):
                                if self.criterion_type == 'decreasing':
                                    if (current_value + self.min_delta) <= self.min_criterion:
                                        self.min_criterion = current_value # current_loss becomes the new "min_test_loss"
                                        self.counter = 0 # reset counter when test_loss decreases by at least min_delta
                                    elif (current_value + self.min_delta) > self.min_criterion:
                                        self.counter = self.counter + 1 # increase counter if test_loss is not decreased by at least min_delta
                                        if self.counter >= self.patience:
                                            return True
                                    return False
                                elif self.criterion_type == 'increasing':
                                    if current_value >= self.max_criterion + self.min_delta:
                                        self.max_criterion = current_value
                                        self.counter = 0
                                    elif current_value < self.max_criterion + self.min_delta:
                                        self.counter = self.counter + 1
                                        if self.counter >= self.patience:
                                            return True
                                    return False
                                else:
                                    raise Exception("Please select a criterion_type: 'decreasing' or 'increasing'.")
                                
                        early_stopping = EarlyStopping(patience = patience, min_delta = min_delta, criterion_type = 'decreasing')

                        def train(model, edge_to_predict, optimizer = optimizer):
                            total_loss = total_examples = 0
                            preds = []
                            ground_truths = []
                            for sampled_data in tqdm(train_loader):
                                model.train()
                                optimizer.zero_grad()
                                # Move `sampled_data` to the respective `device`
                                sampled_data = sampled_data.to(device)
                                # Run `forward` pass of the model
                                # pred = model(sampled_data.x_dict, sampled_data.edge_index_dict, sampled_data['GENE', 'synthetic_lethality', 'GENE'].edge_label_index)
                                pred = model(sampled_data.x_dict, sampled_data.edge_index_dict, sampled_data[edge_to_predict].edge_label_index)
                                # target = sampled_data['GENE', 'synthetic_lethality', 'GENE'].edge_label
                                target = sampled_data[edge_to_predict].edge_label
                                loss = criterion(pred.view(-1,1), target.view(-1,1).float())
                                loss.backward()
                                optimizer.step()
                                total_loss += float(loss) * pred.numel()
                                total_examples += pred.numel()
                                preds.append(pred.detach().cpu())
                                ground_truths.append(target.cpu())
                            all_pred = torch.cat(preds, dim=0).numpy()
                            all_ground_truth = torch.cat(ground_truths, dim=0).numpy()
                            auc = roc_auc_score(all_ground_truth, all_pred)
                            aupr = average_precision_score(all_ground_truth,all_pred)
                            return total_loss/total_examples, auc, aupr

                        def test(model, loader, edge_to_predict, mask_value):
                            total_loss = total_examples = 0
                            model.eval()
                            edges = []
                            preds = []
                            ground_truths = []
                            for sampled_data in tqdm(loader):
                                with torch.no_grad():
                                    sampled_data = sampled_data.to(device)
                                    # pred = model(sampled_data.x_dict, sampled_data.edge_index_dict, sampled_data['GENE', 'synthetic_lethality', 'GENE'].edge_label_index)
                                    pred = model(sampled_data.x_dict, sampled_data.edge_index_dict, sampled_data[edge_to_predict].edge_label_index)
                                    # target = sampled_data['GENE', 'synthetic_lethality', 'GENE'].edge_label
                                    target = sampled_data[edge_to_predict].edge_label
                                    loss = criterion(pred.view(-1,1), target.view(-1,1).float())
                                    total_loss += float(loss) * pred.numel()
                                    total_examples += pred.numel()
                                    ## get the index for the nodes in edges$
                                    # in each batch, the node have a local index, I need to get their global_index with batch['GENE'].n_id
                                    local_src, local_dst = sampled_data[edge_to_predict].edge_label_index
                                    global_src = sampled_data['GENE'].n_id[local_src].cpu()
                                    global_dst = sampled_data['GENE'].n_id[local_dst].cpu()
                                    global_edges = torch.stack([global_src, global_dst], dim=0)
                                    edges.append(global_edges)
                                    ## 
                                    preds.append(pred.cpu())
                                    ground_truths.append(target.cpu())

                            pred = torch.cat(preds, dim=0).numpy()
                            ground_truth = torch.cat(ground_truths, dim=0).numpy()
                            # in the case where I have only one class in the val_set
                            if ground_truth.shape[0] == ground_truth.sum():
                                auc = -1.0
                                aupr = -1.0
                            else:
                                auc = roc_auc_score(ground_truth, pred)
                                aupr = average_precision_score(ground_truth,pred)
                            pred_df = pd.DataFrame(torch.cat(edges,dim=1).cpu().t().numpy(), columns=['src', 'dst'])
                            pred_df.loc[:,'pred'] = pred
                            pred_df.loc[:,'target'] = ground_truth
                            pred_df.loc[:,'mask'] = mask_value
                            return total_loss/total_examples, auc, aupr, pred_df

                        def do_complete_edge_training(model_to_train, optimizer_to_use, criterion_to_stop, edge_type, val_set: bool = True, test_set: bool = True,
                                                        max_number_of_epochs: int = 100, min_epochs_to_train: int = 1):
                            best_epoch = 0
                            AUROC_df = pd.DataFrame(
                                {'epoch': int(), 'AUROC_train': float(), 'train_loss': float(), 
                                'AUROC_val': float(), 'val_loss': float(), 'AUROC_test': float(), 'test_loss': float()}, index = [])
                            for epoch in range(1, max_number_of_epochs + 1):
                                train_loss, auroc_train, _ = train(model_to_train, optimizer = optimizer_to_use, edge_to_predict=edge_type)
                                if val_set:
                                    val_loss, auroc_val, _ , _ = test(model_to_train, loader = val_loader, mask_value='val', edge_to_predict=edge_type)
                                else:
                                    auroc_val = 'not_concerned'
                                    val_loss = 'not_concerned'
                                if test_set:
                                    test_loss, auroc_test, _ , _ = test(model_to_train, loader = test_loader, mask_value='test', edge_to_predict=edge_type)
                                else:
                                    auroc_test = 'not_concerned'
                                    test_loss = 'not_concerned'
                                tmp_df = pd.DataFrame(
                                    {'epoch':epoch, 'AUROC_train': auroc_train, 'train_loss': train_loss, 
                                    'AUROC_val': auroc_val, 'val_loss': val_loss, 'AUROC_test':auroc_test, 'test_loss': test_loss}, index = [f'sampler_epoch_{epoch}'])#[f'mask_{j}_epoch_{epoch}'])
                                AUROC_df = pd.concat([AUROC_df, tmp_df], axis = 0)
                                if val_set and test_set:
                                    print(f'Epoch: {epoch:03d}, Train_loss: {train_loss:.4f}, AUROC_train: {auroc_train:.4f}, Val_loss: {val_loss:.4f}, AUROC_val: {auroc_val:.4f}, Test_loss: {test_loss:.4f} and AUROC_test: {auroc_test:.4f}')
                                elif val_set and not test_set:
                                    print(f'Epoch: {epoch:03d}, Train_loss: {train_loss:.4f}, AUROC_train: {auroc_train:.4f}, Val_loss: {val_loss:.4f}, AUROC_val: {auroc_val:.4f}')
                                elif not val_set and test_set:
                                    print(f'Epoch: {epoch:03d}, Train_loss: {train_loss:.4f}, AUROC_train: {auroc_train:.4f}, Test_loss: {test_loss:.4f} and AUROC_test: {auroc_test:.4f}')
                                else:
                                    print(f'Epoch: {epoch:03d}, Train_loss: {train_loss:.4f}, AUROC_train: {auroc_train:.4f}')

                                # early stopping if the criterion is met, begins checking it after at min_epochs_to_train
                                if epoch > min_epochs_to_train:
                                    if criterion_to_stop == 'train_loss':
                                        if early_stopping.early_stop_check(train_loss):
                                            print(f"We stop at epoch: {epoch}.")
                                            break
                                        else:
                                            if early_stopping.counter == 0:
                                                print(f"I am updating the best_model_state_dict with the parameters of epoch: {epoch}.")
                                                best_model_state_dict = copy.deepcopy(model_to_train.state_dict())
                                                best_epoch = epoch
                                            print(f"My current patience_counter is: {early_stopping.counter}.")
                                    elif criterion_to_stop == 'val_loss':
                                        if early_stopping.early_stop_check(val_loss):
                                            print(f"We stop at epoch: {epoch}.")
                                            break
                                        else:
                                            if early_stopping.counter == 0: 
                                                print(f"I am updating the best_model_state_dict with the parameters of epoch: {epoch}.")
                                                best_model_state_dict = copy.deepcopy(model_to_train.state_dict())
                                                best_epoch = epoch
                                            print(f"My current patience_counter is: {early_stopping.counter}.")
                                    elif criterion_to_stop == 'test_loss':
                                        if early_stopping.early_stop_check(test_loss):
                                            print(f"We stop at epoch: {epoch}.")
                                            break
                                        else:
                                            if early_stopping.counter == 0: 
                                                print(f"I am updating the best_model_state_dict with the parameters of epoch: {epoch}.")
                                                best_model_state_dict = copy.deepcopy(model_to_train.state_dict())
                                                best_epoch = epoch
                                            print(f"My current patience_counter is: {early_stopping.counter}.")
                                    elif criterion_to_stop == 'auroc_train':
                                        if early_stopping.early_stop_check(auroc_train):
                                            print(f"We stop at epoch: {epoch}.")
                                            break
                                        else:
                                            if early_stopping.counter == 0:
                                                print(f"I am updating the best_model_state_dict with the parameters of epoch: {epoch}.")
                                                best_model_state_dict = copy.deepcopy(model_to_train.state_dict())
                                                best_epoch = epoch
                                            print(f"My current patience_counter is: {early_stopping.counter}.")
                                    elif criterion_to_stop == 'auroc_val':
                                        if early_stopping.early_stop_check(auroc_val):
                                            print(f"We stop at epoch: {epoch}.")
                                            break
                                        else:
                                            if early_stopping.counter == 0:
                                                print(f"I am updating the best_model_state_dict with the parameters of epoch: {epoch}.")
                                                best_model_state_dict = copy.deepcopy(model_to_train.state_dict())
                                                best_epoch = epoch
                                            print(f"My current patience_counter is: {early_stopping.counter}.")
                                    elif criterion_to_stop == 'auroc_test':
                                        if early_stopping.early_stop_check(auroc_test):
                                            print(f"We stop at epoch: {epoch}.")
                                            break
                                        else:
                                            if early_stopping.counter == 0:
                                                print(f"I am updating the best_model_state_dict with the parameters of epoch: {epoch}.")
                                                best_model_state_dict = copy.deepcopy(model_to_train.state_dict())
                                                best_epoch = epoch
                                            print(f"My current patience_counter is: {early_stopping.counter}.")
                            print(f"I am restoring the model_parameters from those of best_model_state_dict at epoch: {best_epoch}.")
                            model_to_train.load_state_dict(best_model_state_dict)
                            return model_to_train, best_epoch, AUROC_df

                        ## do the training 
                        model, best_epoch, AUROC_df = do_complete_edge_training(model, optimizer_to_use = optimizer, criterion_to_stop = 'train_loss', 
                                                                                            val_set = use_val_set, test_set = True, edge_type = edge_to_predict,
                                                                                            max_number_of_epochs = epochs_to_train, min_epochs_to_train = 0)  

                        print("Training complete.")
                        print("Saving the model.")
                        model_name = model_path + outter_model_param + f'_epoch_{best_epoch}.pyg'
                        torch.save(model.state_dict(), model_name)

                        AUROC_df.loc[:,'best_epoch'] = best_epoch
                        auroc_df_name = result_path + outter_model_param + f'_epoch_{best_epoch}_auroc_df.tsv'
                        AUROC_df.to_csv(path_or_buf = auroc_df_name, header = True, index = False, sep = "\t")

                        edge_pred_setting = 'all_edges_fixed'

                        ## df_edges

                        def get_df_edges(edge_type:str,  df_to_use: pd.DataFrame = fd, val_set:bool = False):
                            
                            _, _, _, train_edges = test(model, edge_to_predict = edge_type, loader=train_loader, mask_value='train')
                            _, _, _, test_edges = test(model, edge_to_predict = edge_type, loader=test_loader, mask_value='test')
                            if val_set:
                                _, _, _, val_edges = test(model, edge_to_predict = edge_type, loader=val_loader, mask_value='val')
                                df_edges = pd.concat([train_edges, val_edges, test_edges], axis = 0)
                            else:
                                df_edges = pd.concat([train_edges, test_edges], axis = 0)

                            df_edges = df_edges.rename(columns={'src': 'mapped_id_left', 'dst': 'mapped_id_right'})
                            df = df_to_use
                            # this one works both for OLIDA and synthetic_lethality
                            df_edges = df_edges.merge(nodes[['GENE_id', 'mapped_id']], how='left', left_on='mapped_id_left', right_on='mapped_id')
                            df_edges = df_edges.drop('mapped_id', axis=1)
                            df_edges = df_edges.rename(columns={'GENE_id': 'left_side'})
                            df_edges = df_edges.merge(nodes[['GENE_id', 'mapped_id']], how='left', left_on='mapped_id_right', right_on='mapped_id')
                            df_edges = df_edges.drop('mapped_id', axis=1)
                            df_edges = df_edges.rename(columns={'GENE_id': 'right_side'})
                            df_edges = get_nop(df_edges, left_col='left_side', right_col='right_side')
                            df_edges = df.merge(df_edges[['name_of_pair', 'pred']], how='left', left_on='GENE_id', right_on='name_of_pair')
                            df_edges = df_edges.drop('name_of_pair', axis=1)
                                
                            return df_edges

                        df_edges = get_df_edges(edge_type = edge_to_predict, df_to_use = fd, val_set=use_val_set)

                        ## save df_edges_pred and fig
                        print("Saving the prediction df.")
                        df_edges_name = result_path + outter_model_param + f'_epoch_{best_epoch}_pred.tsv'
                        df_edges.to_csv(path_or_buf = df_edges_name, header = True, index = False, sep = "\t")

                        print("Drawing the figure.")
                        fig = plot_df_edges(df_edges, val_set=False)
                        print("Saving the figure.")
                        fig_name = result_path + outter_model_param + f'_epoch_{best_epoch}.png'
                        fig.savefig(fig_name)


                    ## explanation 
                    print("Writing the explanation file.")
                    # file_name = path + model_param + f"_max_epochs_{epochs_to_train}_explanation.txt"
                    _, _, dfeat = get_strict_target_tensors(target_to_use = target_of_script, remove_sex_chrom = False)
                    file_name = result_path + outter_model_param + f"_explanation.txt"
                    f = open(file_name, "w")
                    f.write(f"The edge_pred setting that we used is: {edge_pred_setting}.")
                    f.write(f"This edge_prediction model is trained on Humans, on {edge_to_predict} edges.")
                    f.write(f"The features used come from this dataframe: 'data/processed/human_genes_features_20230414.tsv'\n")
                    f.write(f"The features used are:")
                    f.write(str(dfeat.columns.tolist()))
                    f.write(f"For the models: {base_model_param}, for a maximum of {epochs_to_train}, the edges that were used are:\n")
                    for edge in data.edge_index_dict.keys():
                        f.write(f"{edge} \n")
                    f.write("\nThe precise data used are:\n")
                    f.write(str(data))
                    f.write("\nThe complete hyperparameters are:\n")
                    f.write(str(hyperparameters))
                    f.close()

                ## end 