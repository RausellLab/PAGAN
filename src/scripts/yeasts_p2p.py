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
    'gene_mode': 'degree',
    'node_mode' : 'degree', #'id' or 'degree' for abcm nodes_types
    'edge_features' : False, # True = those are the true edge_weights from my network
    'Heads' : 1, #2
    # 'conv_number' : 2, # 
    # 'conv_type': 'SAGE', # 
    'sage_aggr': 'sum', # 'sum', 'var'
    'sage_norm': True, # False
    'sage_project': False,
    'heteroconv_aggr' : 'sum', # 'cat', 'max'
    'heteroconv_aggr_1' : 'sum', # 'sum', 'cat', 'max'
    'heteroconv_aggr_2' : 'sum', # 'sum', 'cat', 'max'
    'heteroconv_aggr_3' : 'sum', # 'cat', 'max'
    # 'activation': 'relu', # 'relu', 'tanh'
    'hc_2_q' : 1, # hc_2 = int(i/hc_2_q)
    'hc_3_q': 1, # hc_3_q = int(i/hc_3_q)
    # 'size_of_batch': 32,
    'neighbors': 30, #30, 40, 50
    'disjoint_loader' : True, 
    'zero_out_batch_features': True,
    'epochs_to_train' : 500, #
    'min_epochs_to_train': 3,
    'patience': 10,
    'min_delta': 0.005,
    'auto_loop': False,
    'nodes_pairs_features': 'zeros', #
    'dnds_mode': 'sum',
    'type_to_return': 'torch', # 'numpy'
    'final_type': 'torch', # 'numpy'
    'remove_forbidden_value': True,
    'shuffle_train': True,
    'ana_edges': 'common', # 'all
    'string_threshold': 700, # confidence = {'low': 0.15, 'medium': 0.4, 'high': 0.7, 'highest': 0.9} from STRING doc, * 1000 to have EdgeScore
    'setting': 'pairs_to_pairs',
    'global_loader_neighbors': 'all', # neighbors
    'doublons_pairs': 'remove_all' # 'keep_first'
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
ana_edges = hyperparameters['ana_edges']
string_threshold = hyperparameters['string_threshold']
# conv_number = hyperparameters['conv_number']
setting = hyperparameters['setting']
# conv_type = hyperparameters['conv_type']
heads = hyperparameters['Heads']
# activation = hyperparameters['activation']
global_loader_neighbors = hyperparameters['global_loader_neighbors']
doublons_pairs = hyperparameters['doublons_pairs']

t0 = time.time()

old_path = os.getcwd()
new_path = old_path + '/../../'
os.chdir(new_path)
current_path = os.getcwd()
data_raw_path = current_path + '/data/raw/'
data_processed_path = current_path + '/data/processed/'
result_path = current_path + f'/results/yeasts/p2p/'
model_path = current_path + f'/models/yeasts/p2p/'

# 1st for_loop
for sl_genes in [4261, 3841]: # 
    # 2nd for_loop
    for DMF_t in [0.5, 0.4, 0.3]: # 
        # 3rd for_loop
        for splits in ['50_50', '80_20']:
            # 4th for_loop
            for activation in ['tanh', 'relu']:
                # if doublons_pairs == 'keep_first':
                #     base_model_param = f"yeasts_p2p_{sl_genes}_SL_genes_DMF_{DMF_t}_SNL_split_{splits}_STRING_{string_threshold}_act_{activation}_bcm_{node_mode}_2HetConv_{heteroconv_aggr_1}_{heteroconv_aggr_2}_SAGE_{sage_aggr}_sagenorm_{sage_norm}"
                # elif doublons_pairs == 'remove_all':
                base_model_param = f"yeasts_p2p_no_doublons_{sl_genes}_SL_genes_DMF_{DMF_t}_SNL_split_{splits}_STRING_{string_threshold}_act_{activation}_bcm_{node_mode}_2HetConv_{heteroconv_aggr_1}_{heteroconv_aggr_2}_SAGE_{sage_aggr}_sagenorm_{sage_norm}"

                # open the Yeasst_Knowledge_Graph
                net = pd.read_csv(data_processed_path + f'yeast_knowledge_graph.tsv', sep = '\t', header = 0, dtype = {1: str, 3: str, 4: str})
                net = net.drop(net[(net['EdgeType'].isin(['PPI'])) & (net['EdgeScore'] < string_threshold)].index, axis = 0)
                net = net.reset_index(drop = True)

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

                def get_target_tensor():
                    # for nodes, essential vs non_essential
                    essential = pd.read_csv(data_processed_path + 'yeast_essential_nonessential_20240612.tsv', sep = '\t', header = 0)
                    nodes2id['GENE'] = nodes2id['GENE'][['GENE_id', 'mapped_id']]
                    nodes = nodes2id['GENE'].copy()
                    nodes = nodes.merge(essential, how = 'left', left_on = 'GENE_id', right_on = 'gene_id')
                    nodes.loc[nodes[nodes['type'] == 'essential'].index, 'target'] = 1
                    nodes.loc[nodes[nodes['type'] == 'non_essential'].index, 'target'] = 0
                    target_tensor = torch.from_numpy(nodes['target'].values)
                    return target_tensor 

                def add_strict_nodes(data: HeteroData) -> HeteroData:
                    target_tensor = get_target_tensor()
                    data['GENE'].num_nodes = len(nodes2id['GENE'])
                    data['GENE'].y = target_tensor.type(torch.float32)
                    for type_of_node in nodes_types:
                        if type_of_node != 'GENE':
                            data[type_of_node].num_nodes = len(nodes2id[type_of_node])
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
                data = add_strict_nodes(data)
                data = get_edges_of_type(data, 'PPI')
                data = get_edges_of_type(data, 'biological_process')
                data = get_edges_of_type(data, 'cellular_component')
                data = get_edges_of_type(data, 'molecular_function')
                # those are dummy features for node_type == 'GENE', but I will not use it. TODO TOCLEAN
                gg = Data()
                gg.num_nodes = len(nodes2id['GENE'])
                gg.edge_index = data.edge_index_dict[('GENE', 'PPI', 'GENE')]
                df_gg = pd.DataFrame(gg.edge_index.t(), columns = ['source', 'dst'])
                max_degree = max(df_gg.dst.value_counts().tolist()[0], df_gg.source.value_counts().tolist()[0])
                gg = T.OneHotDegree(max_degree = max_degree)(gg)
                gg.x_id = torch.eye(gg.num_nodes, dtype = torch.float32, requires_grad = False)

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

                def add_features(data: HeteroData, mode: str = 'degree', gene_mode: str = 'id') -> HeteroData:
                    '''create features for the nodes types that are not 'GENE', either one-hot-encoding their node_degree or their id'''
                    if mode == 'id':
                        # data['ANATOMY'].x = ana.x_id
                        data['BP'].x = bp.x_id
                        data['CC'].x = cc.x_id
                        data['MF'].x = mf.x_id        
                    elif mode == 'degree':
                        # data['ANATOMY'].x = ana.x
                        data['BP'].x = bp.x
                        data['CC'].x = cc.x
                        data['MF'].x = mf.x
                    else:
                        raise Exception("Please select a mode 'id' or 'degree'")
                    if gene_mode == 'id':
                        data['GENE'].x = torch.eye(data['GENE'].num_nodes, dtype = torch.float32, requires_grad = False)
                    elif gene_mode == 'degree':
                        data['GENE'].x = gg.x
                    
                    return data

                def create_bcmg_data(list_of_edges: list, use_edge_attr: bool = False, feature_mode: str = 'id', merge_choice: bool = True):
                    edges_names = list_of_edges
                    edges_with_attr = ['PPI', 'Neighbour', 'CoExp'] #
                    # in data.edge_index_dict.keys(), an edge is a tuple (node1_type, relation, node2_type)
                    edges_with_same_node_type = [
                        ('GENE', 'PPI', 'GENE'),
                        ('GENE', 'paralog', 'GENE'), 
                        ('GENE', 'Neighbour', 'GENE'), 
                        ('GENE', 'CoExp', 'GENE'), 
                        ('BP', 'biological_process', 'BP'), 
                        ('CC', 'cellular_component', 'CC'), 
                        ('MF', 'molecular_function', 'MF')]
                    target_tensor = get_target_tensor()
                    bool_attr = use_edge_attr
                    data = HeteroData()
                    data = add_strict_nodes(data)
                    data = add_features(data, mode = feature_mode, gene_mode = gene_mode) # 

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

                    # ne pas remove les self-loops quand il y a des différents nodes types !! 
                    # double check the edges concerned
                    for key in data.edge_index_dict.keys():
                        if key in edges_with_same_node_type:
                            if U.contains_self_loops(data.edge_index_dict[key]) == True:
                                data[key].edge_index = U.remove_self_loops(data.edge_index_dict[key])[0]
                    # if merge = False, will create the reverse edge_type
                    data = T.ToUndirected(merge = merge_choice)(data)

                    bp_list = ['gene_to_BP', 'biological_process']
                    cc_list = ['gene_to_CC', 'cellular_component']
                    mf_list = ['gene_to_MF', 'molecular_function']

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
                        ('GENE', 'Neighbour', 'GENE'), 
                        ('GENE', 'CoExp', 'GENE')]
                    edges_with_same_node_type = [
                        ('GENE', 'PPI', 'GENE'),
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


                ## define models 

                class HeteroConv_1GATv2_tanh(torch.nn.Module):
                    def __init__(self, hidden_channels_1, out_channels, number_of_heads, hetero_aggr_1 = 'sum'):
                        super().__init__()
                        # torch.manual_seed(1234567)
                        # add_self_loops = False when using HeteroConv on different node types 
                        self.conv1 = HeteroConv({ 
                                ('GENE', 'expressed_in', 'ANATOMY'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('ANATOMY', 'rev_expressed_in', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'not_expressed_in', 'ANATOMY'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('ANATOMY', 'rev_not_expressed_in', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'PPI', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'CoExp', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'paralog', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'Neighbour', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'gene_to_BP', 'BP'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('BP', 'rev_gene_to_BP', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'gene_to_CC', 'CC'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('CC', 'rev_gene_to_CC', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'gene_to_MF', 'MF'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('MF', 'rev_gene_to_MF', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('ANATOMY', 'ana_ana', 'ANATOMY'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('ANATOMY', 'rev_ana_ana', 'ANATOMY'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('BP', 'biological_process', 'BP'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('BP', 'rev_biological_process', 'BP'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('CC', 'cellular_component', 'CC'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('CC', 'rev_cellular_component', 'CC'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('MF', 'molecular_function', 'MF'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('MF', 'rev_molecular_function', 'MF'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads)
                            }, aggr=hetero_aggr_1)
                        self.lin = Linear(-1, out_channels)
                    def forward(self, x_dict, edge_index_dict):
                        x_dict = self.conv1(x_dict, edge_index_dict)
                        x_dict = {key: torch.tanh(x) for key, x in x_dict.items()}
                        x_dict = {key: F.dropout(x, p = 0.5, training = self.training) for key, x in x_dict.items()}
                        x = x_dict['GENE']
                        x = self.lin(x)         
                        return torch.sigmoid(x)
                    
                class HeteroConv_1GATv2_relu(torch.nn.Module):
                    def __init__(self, hidden_channels_1, out_channels, number_of_heads, hetero_aggr_1 = 'sum'):
                        super().__init__()
                        # torch.manual_seed(1234567)
                        # add_self_loops = False when using HeteroConv on different node types 
                        self.conv1 = HeteroConv({ 
                                ('GENE', 'expressed_in', 'ANATOMY'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('ANATOMY', 'rev_expressed_in', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'not_expressed_in', 'ANATOMY'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('ANATOMY', 'rev_not_expressed_in', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'PPI', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'CoExp', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'paralog', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'Neighbour', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'gene_to_BP', 'BP'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('BP', 'rev_gene_to_BP', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'gene_to_CC', 'CC'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('CC', 'rev_gene_to_CC', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'gene_to_MF', 'MF'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('MF', 'rev_gene_to_MF', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('ANATOMY', 'ana_ana', 'ANATOMY'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('ANATOMY', 'rev_ana_ana', 'ANATOMY'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('BP', 'biological_process', 'BP'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('BP', 'rev_biological_process', 'BP'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('CC', 'cellular_component', 'CC'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('CC', 'rev_cellular_component', 'CC'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('MF', 'molecular_function', 'MF'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('MF', 'rev_molecular_function', 'MF'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads)
                            }, aggr=hetero_aggr_1)
                        self.lin = Linear(-1, out_channels)
                    def forward(self, x_dict, edge_index_dict):
                        x_dict = self.conv1(x_dict, edge_index_dict)
                        x_dict = {key: x.relu() for key, x in x_dict.items()}
                        # x_dict = {key: torch.tanh(x) for key, x in x_dict.items()}
                        x_dict = {key: F.dropout(x, p = 0.5, training = self.training) for key, x in x_dict.items()}
                        x = x_dict['GENE']
                        x = self.lin(x)         
                        return torch.sigmoid(x)

                class HeteroConv_2GATv2_tanh(torch.nn.Module):
                    def __init__(self, hidden_channels_1, hidden_channels_2, out_channels, number_of_heads, hetero_aggr_1 = 'sum', hetero_aggr_2 = 'sum'): # num_layers, hidden_linear_2
                        super().__init__()
                        # torch.manual_seed(1234567)
                        # add_self_loops = False when using HeteroConv on different node types 
                        self.conv1 = HeteroConv({ 
                                ('GENE', 'expressed_in', 'ANATOMY'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('ANATOMY', 'rev_expressed_in', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'not_expressed_in', 'ANATOMY'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('ANATOMY', 'rev_not_expressed_in', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'PPI', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'CoExp', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'paralog', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'Neighbour', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'gene_to_BP', 'BP'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('BP', 'rev_gene_to_BP', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'gene_to_CC', 'CC'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('CC', 'rev_gene_to_CC', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'gene_to_MF', 'MF'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('MF', 'rev_gene_to_MF', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('ANATOMY', 'ana_ana', 'ANATOMY'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('ANATOMY', 'rev_ana_ana', 'ANATOMY'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('BP', 'biological_process', 'BP'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('BP', 'rev_biological_process', 'BP'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('CC', 'cellular_component', 'CC'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('CC', 'rev_cellular_component', 'CC'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('MF', 'molecular_function', 'MF'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('MF', 'rev_molecular_function', 'MF'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads)
                            }, aggr=hetero_aggr_1)
                        self.conv2 = HeteroConv({ 
                                ('GENE', 'expressed_in', 'ANATOMY'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('ANATOMY', 'rev_expressed_in', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'not_expressed_in', 'ANATOMY'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('ANATOMY', 'rev_not_expressed_in', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'PPI', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'CoExp', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'paralog', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('GENE', 'Neighbour', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'gene_to_BP', 'BP'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('BP', 'rev_gene_to_BP', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'gene_to_CC', 'CC'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('CC', 'rev_gene_to_CC', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'gene_to_MF', 'MF'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('MF', 'rev_gene_to_MF', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('ANATOMY', 'ana_ana', 'ANATOMY'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('ANATOMY', 'rev_ana_ana', 'ANATOMY'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('BP', 'biological_process', 'BP'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('BP', 'rev_biological_process', 'BP'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('CC', 'cellular_component', 'CC'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('CC', 'rev_cellular_component', 'CC'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('MF', 'molecular_function', 'MF'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('MF', 'rev_molecular_function', 'MF'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads)
                            }, aggr=hetero_aggr_2)
                        self.lin = Linear(-1, out_channels)
                    def forward(self, x_dict, edge_index_dict):
                        x_dict = self.conv1(x_dict, edge_index_dict)
                        x_dict = {key: torch.tanh(x) for key, x in x_dict.items()}
                        x_dict = {key: F.dropout(x, p = 0.5, training = self.training) for key, x in x_dict.items()}
                        x_dict = self.conv2(x_dict, edge_index_dict)
                        x_dict = {key: torch.tanh(x) for key, x in x_dict.items()}
                        x_dict = {key: F.dropout(x, p = 0.5, training = self.training) for key, x in x_dict.items()}
                        x = x_dict['GENE']
                        x = self.lin(x)         
                        return torch.sigmoid(x)
                    
                class HeteroConv_2GATv2_relu(torch.nn.Module):
                    def __init__(self, hidden_channels_1, hidden_channels_2, out_channels, number_of_heads, hetero_aggr_1 = 'sum', hetero_aggr_2 = 'sum'): # num_layers, hidden_linear_2
                        super().__init__()
                        # torch.manual_seed(1234567)
                        # add_self_loops = False when using HeteroConv on different node types 
                        self.conv1 = HeteroConv({ 
                                ('GENE', 'expressed_in', 'ANATOMY'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('ANATOMY', 'rev_expressed_in', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'not_expressed_in', 'ANATOMY'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('ANATOMY', 'rev_not_expressed_in', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'PPI', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'CoExp', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'paralog', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'Neighbour', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'gene_to_BP', 'BP'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('BP', 'rev_gene_to_BP', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'gene_to_CC', 'CC'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('CC', 'rev_gene_to_CC', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'gene_to_MF', 'MF'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('MF', 'rev_gene_to_MF', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('ANATOMY', 'ana_ana', 'ANATOMY'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('ANATOMY', 'rev_ana_ana', 'ANATOMY'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('BP', 'biological_process', 'BP'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('BP', 'rev_biological_process', 'BP'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('CC', 'cellular_component', 'CC'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('CC', 'rev_cellular_component', 'CC'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('MF', 'molecular_function', 'MF'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('MF', 'rev_molecular_function', 'MF'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads)
                            }, aggr=hetero_aggr_1)
                        self.conv2 = HeteroConv({ 
                                ('GENE', 'expressed_in', 'ANATOMY'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('ANATOMY', 'rev_expressed_in', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'not_expressed_in', 'ANATOMY'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('ANATOMY', 'rev_not_expressed_in', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'PPI', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'CoExp', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'paralog', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('GENE', 'Neighbour', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'gene_to_BP', 'BP'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('BP', 'rev_gene_to_BP', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'gene_to_CC', 'CC'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('CC', 'rev_gene_to_CC', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'gene_to_MF', 'MF'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('MF', 'rev_gene_to_MF', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('ANATOMY', 'ana_ana', 'ANATOMY'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('ANATOMY', 'rev_ana_ana', 'ANATOMY'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('BP', 'biological_process', 'BP'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('BP', 'rev_biological_process', 'BP'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('CC', 'cellular_component', 'CC'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('CC', 'rev_cellular_component', 'CC'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('MF', 'molecular_function', 'MF'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('MF', 'rev_molecular_function', 'MF'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads)
                            }, aggr=hetero_aggr_2)
                        self.lin = Linear(-1, out_channels)
                    def forward(self, x_dict, edge_index_dict):
                        x_dict = self.conv1(x_dict, edge_index_dict)
                        x_dict = {key: x.relu() for key, x in x_dict.items()}
                        x_dict = {key: F.dropout(x, p = 0.5, training = self.training) for key, x in x_dict.items()}
                        x_dict = self.conv2(x_dict, edge_index_dict)
                        x_dict = {key: x.relu() for key, x in x_dict.items()}
                        x_dict = {key: F.dropout(x, p = 0.5, training = self.training) for key, x in x_dict.items()}
                        x = x_dict['GENE']
                        x = self.lin(x)         
                        return torch.sigmoid(x)
                    
                class HeteroConv_3GATv2_tanh(torch.nn.Module):
                    def __init__(self, hidden_channels_1, hidden_channels_2, hidden_channels_3, out_channels, number_of_heads, hetero_aggr_1 = 'sum', hetero_aggr_2 = 'sum', hetero_aggr_3 = 'sum'): # num_layers, hidden_linear_2
                        super().__init__()
                        # torch.manual_seed(1234567)
                        self.conv1 = HeteroConv({ 
                                ('GENE', 'expressed_in', 'ANATOMY'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('ANATOMY', 'rev_expressed_in', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'not_expressed_in', 'ANATOMY'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('ANATOMY', 'rev_not_expressed_in', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'PPI', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'CoExp', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'paralog', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'Neighbour', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'gene_to_BP', 'BP'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('BP', 'rev_gene_to_BP', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'gene_to_CC', 'CC'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('CC', 'rev_gene_to_CC', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'gene_to_MF', 'MF'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('MF', 'rev_gene_to_MF', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('ANATOMY', 'ana_ana', 'ANATOMY'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('ANATOMY', 'rev_ana_ana', 'ANATOMY'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('BP', 'biological_process', 'BP'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('BP', 'rev_biological_process', 'BP'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('CC', 'cellular_component', 'CC'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('CC', 'rev_cellular_component', 'CC'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('MF', 'molecular_function', 'MF'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('MF', 'rev_molecular_function', 'MF'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads)
                            }, aggr=hetero_aggr_1)
                        self.conv2 = HeteroConv({ 
                                ('GENE', 'expressed_in', 'ANATOMY'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('ANATOMY', 'rev_expressed_in', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'not_expressed_in', 'ANATOMY'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('ANATOMY', 'rev_not_expressed_in', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'PPI', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'CoExp', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'paralog', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('GENE', 'Neighbour', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'gene_to_BP', 'BP'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('BP', 'rev_gene_to_BP', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'gene_to_CC', 'CC'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('CC', 'rev_gene_to_CC', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'gene_to_MF', 'MF'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('MF', 'rev_gene_to_MF', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('ANATOMY', 'ana_ana', 'ANATOMY'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('ANATOMY', 'rev_ana_ana', 'ANATOMY'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('BP', 'biological_process', 'BP'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('BP', 'rev_biological_process', 'BP'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('CC', 'cellular_component', 'CC'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('CC', 'rev_cellular_component', 'CC'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('MF', 'molecular_function', 'MF'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('MF', 'rev_molecular_function', 'MF'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads)
                            }, aggr=hetero_aggr_2)
                        self.conv3 = HeteroConv({ 
                                ('GENE', 'expressed_in', 'ANATOMY'): GATv2Conv(-1, hidden_channels_3, add_self_loops = False, heads = number_of_heads),
                                ('ANATOMY', 'rev_expressed_in', 'GENE'): GATv2Conv(-1, hidden_channels_3, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'not_expressed_in', 'ANATOMY'): GATv2Conv(-1, hidden_channels_3, add_self_loops = False, heads = number_of_heads),
                                ('ANATOMY', 'rev_not_expressed_in', 'GENE'): GATv2Conv(-1, hidden_channels_3, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'PPI', 'GENE'): GATv2Conv(-1, hidden_channels_3, add_self_loops = True, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'CoExp', 'GENE'): GATv2Conv(-1, hidden_channels_3, add_self_loops = True, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'paralog', 'GENE'): GATv2Conv(-1, hidden_channels_3, add_self_loops = True, heads = number_of_heads),
                                ('GENE', 'Neighbour', 'GENE'): GATv2Conv(-1, hidden_channels_3, add_self_loops = True, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'gene_to_BP', 'BP'): GATv2Conv(-1, hidden_channels_3, add_self_loops = False, heads = number_of_heads),
                                ('BP', 'rev_gene_to_BP', 'GENE'): GATv2Conv(-1, hidden_channels_3, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'gene_to_CC', 'CC'): GATv2Conv(-1, hidden_channels_3, add_self_loops = False, heads = number_of_heads),
                                ('CC', 'rev_gene_to_CC', 'GENE'): GATv2Conv(-1, hidden_channels_3, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'gene_to_MF', 'MF'): GATv2Conv(-1, hidden_channels_3, add_self_loops = False, heads = number_of_heads),
                                ('MF', 'rev_gene_to_MF', 'GENE'): GATv2Conv(-1, hidden_channels_3, add_self_loops = False, heads = number_of_heads),
                                ('ANATOMY', 'ana_ana', 'ANATOMY'): GATv2Conv(-1, hidden_channels_3, add_self_loops = True, heads = number_of_heads),
                                ('ANATOMY', 'rev_ana_ana', 'ANATOMY'): GATv2Conv(-1, hidden_channels_3, add_self_loops = True, heads = number_of_heads),
                                ('BP', 'biological_process', 'BP'): GATv2Conv(-1, hidden_channels_3, add_self_loops = True, heads = number_of_heads),
                                ('BP', 'rev_biological_process', 'BP'): GATv2Conv(-1, hidden_channels_3, add_self_loops = True, heads = number_of_heads),
                                ('CC', 'cellular_component', 'CC'): GATv2Conv(-1, hidden_channels_3, add_self_loops = True, heads = number_of_heads),
                                ('CC', 'rev_cellular_component', 'CC'): GATv2Conv(-1, hidden_channels_3, add_self_loops = True, heads = number_of_heads),
                                ('MF', 'molecular_function', 'MF'): GATv2Conv(-1, hidden_channels_3, add_self_loops = True, heads = number_of_heads),
                                ('MF', 'rev_molecular_function', 'MF'): GATv2Conv(-1, hidden_channels_3, add_self_loops = True, heads = number_of_heads)
                            }, aggr=hetero_aggr_3)
                        self.lin = Linear(-1, out_channels)
                    def forward(self, x_dict, edge_index_dict):
                        x_dict = self.conv1(x_dict, edge_index_dict)
                        x_dict = {key: torch.tanh(x) for key, x in x_dict.items()}
                        x_dict = {key: F.dropout(x, p = 0.5, training = self.training) for key, x in x_dict.items()}
                        x_dict = self.conv2(x_dict, edge_index_dict)
                        x_dict = {key: torch.tanh(x) for key, x in x_dict.items()}
                        x_dict = {key: F.dropout(x, p = 0.5, training = self.training) for key, x in x_dict.items()}
                        x_dict = self.conv3(x_dict, edge_index_dict)
                        x_dict = {key: torch.tanh(x) for key, x in x_dict.items()}
                        x_dict = {key: F.dropout(x, p = 0.5, training = self.training) for key, x in x_dict.items()}
                        x = x_dict['GENE']
                        x = self.lin(x)         
                        return torch.sigmoid(x)

                class HeteroConv_3GATv2_relu(torch.nn.Module):
                    def __init__(self, hidden_channels_1, hidden_channels_2, hidden_channels_3, out_channels, number_of_heads, hetero_aggr_1 = 'sum', hetero_aggr_2 = 'sum', hetero_aggr_3 = 'sum'): # num_layers, hidden_linear_2
                        super().__init__()
                        # torch.manual_seed(1234567)
                        self.conv1 = HeteroConv({ 
                                ('GENE', 'expressed_in', 'ANATOMY'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('ANATOMY', 'rev_expressed_in', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'not_expressed_in', 'ANATOMY'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('ANATOMY', 'rev_not_expressed_in', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'PPI', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'CoExp', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'paralog', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'Neighbour', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'gene_to_BP', 'BP'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('BP', 'rev_gene_to_BP', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'gene_to_CC', 'CC'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('CC', 'rev_gene_to_CC', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'gene_to_MF', 'MF'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('MF', 'rev_gene_to_MF', 'GENE'): GATv2Conv(-1, hidden_channels_1, add_self_loops = False, heads = number_of_heads),
                                ('ANATOMY', 'ana_ana', 'ANATOMY'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('ANATOMY', 'rev_ana_ana', 'ANATOMY'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('BP', 'biological_process', 'BP'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('BP', 'rev_biological_process', 'BP'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('CC', 'cellular_component', 'CC'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('CC', 'rev_cellular_component', 'CC'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('MF', 'molecular_function', 'MF'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads),
                                ('MF', 'rev_molecular_function', 'MF'): GATv2Conv(-1, hidden_channels_1, add_self_loops = True, heads = number_of_heads)
                            }, aggr=hetero_aggr_1)
                        self.conv2 = HeteroConv({ 
                                ('GENE', 'expressed_in', 'ANATOMY'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('ANATOMY', 'rev_expressed_in', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'not_expressed_in', 'ANATOMY'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('ANATOMY', 'rev_not_expressed_in', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'PPI', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'CoExp', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'paralog', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('GENE', 'Neighbour', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'gene_to_BP', 'BP'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('BP', 'rev_gene_to_BP', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'gene_to_CC', 'CC'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('CC', 'rev_gene_to_CC', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'gene_to_MF', 'MF'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('MF', 'rev_gene_to_MF', 'GENE'): GATv2Conv(-1, hidden_channels_2, add_self_loops = False, heads = number_of_heads),
                                ('ANATOMY', 'ana_ana', 'ANATOMY'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('ANATOMY', 'rev_ana_ana', 'ANATOMY'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('BP', 'biological_process', 'BP'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('BP', 'rev_biological_process', 'BP'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('CC', 'cellular_component', 'CC'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('CC', 'rev_cellular_component', 'CC'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('MF', 'molecular_function', 'MF'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads),
                                ('MF', 'rev_molecular_function', 'MF'): GATv2Conv(-1, hidden_channels_2, add_self_loops = True, heads = number_of_heads)
                            }, aggr=hetero_aggr_2)
                        self.conv3 = HeteroConv({ 
                                ('GENE', 'expressed_in', 'ANATOMY'): GATv2Conv(-1, hidden_channels_3, add_self_loops = False, heads = number_of_heads),
                                ('ANATOMY', 'rev_expressed_in', 'GENE'): GATv2Conv(-1, hidden_channels_3, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'not_expressed_in', 'ANATOMY'): GATv2Conv(-1, hidden_channels_3, add_self_loops = False, heads = number_of_heads),
                                ('ANATOMY', 'rev_not_expressed_in', 'GENE'): GATv2Conv(-1, hidden_channels_3, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'PPI', 'GENE'): GATv2Conv(-1, hidden_channels_3, add_self_loops = True, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'CoExp', 'GENE'): GATv2Conv(-1, hidden_channels_3, add_self_loops = True, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'paralog', 'GENE'): GATv2Conv(-1, hidden_channels_3, add_self_loops = True, heads = number_of_heads),
                                ('GENE', 'Neighbour', 'GENE'): GATv2Conv(-1, hidden_channels_3, add_self_loops = True, heads = number_of_heads), # edge_dim = edge_dim,
                                ('GENE', 'gene_to_BP', 'BP'): GATv2Conv(-1, hidden_channels_3, add_self_loops = False, heads = number_of_heads),
                                ('BP', 'rev_gene_to_BP', 'GENE'): GATv2Conv(-1, hidden_channels_3, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'gene_to_CC', 'CC'): GATv2Conv(-1, hidden_channels_3, add_self_loops = False, heads = number_of_heads),
                                ('CC', 'rev_gene_to_CC', 'GENE'): GATv2Conv(-1, hidden_channels_3, add_self_loops = False, heads = number_of_heads),
                                ('GENE', 'gene_to_MF', 'MF'): GATv2Conv(-1, hidden_channels_3, add_self_loops = False, heads = number_of_heads),
                                ('MF', 'rev_gene_to_MF', 'GENE'): GATv2Conv(-1, hidden_channels_3, add_self_loops = False, heads = number_of_heads),
                                ('ANATOMY', 'ana_ana', 'ANATOMY'): GATv2Conv(-1, hidden_channels_3, add_self_loops = True, heads = number_of_heads),
                                ('ANATOMY', 'rev_ana_ana', 'ANATOMY'): GATv2Conv(-1, hidden_channels_3, add_self_loops = True, heads = number_of_heads),
                                ('BP', 'biological_process', 'BP'): GATv2Conv(-1, hidden_channels_3, add_self_loops = True, heads = number_of_heads),
                                ('BP', 'rev_biological_process', 'BP'): GATv2Conv(-1, hidden_channels_3, add_self_loops = True, heads = number_of_heads),
                                ('CC', 'cellular_component', 'CC'): GATv2Conv(-1, hidden_channels_3, add_self_loops = True, heads = number_of_heads),
                                ('CC', 'rev_cellular_component', 'CC'): GATv2Conv(-1, hidden_channels_3, add_self_loops = True, heads = number_of_heads),
                                ('MF', 'molecular_function', 'MF'): GATv2Conv(-1, hidden_channels_3, add_self_loops = True, heads = number_of_heads),
                                ('MF', 'rev_molecular_function', 'MF'): GATv2Conv(-1, hidden_channels_3, add_self_loops = True, heads = number_of_heads)
                            }, aggr=hetero_aggr_3)
                        self.lin = Linear(-1, out_channels)
                    def forward(self, x_dict, edge_index_dict):
                        x_dict = self.conv1(x_dict, edge_index_dict)
                        x_dict = {key: x.relu() for key, x in x_dict.items()}
                        x_dict = {key: F.dropout(x, p = 0.5, training = self.training) for key, x in x_dict.items()}
                        x_dict = self.conv2(x_dict, edge_index_dict)
                        x_dict = {key: x.relu() for key, x in x_dict.items()}
                        x_dict = {key: F.dropout(x, p = 0.5, training = self.training) for key, x in x_dict.items()}
                        x_dict = self.conv3(x_dict, edge_index_dict)
                        x_dict = {key: x.relu() for key, x in x_dict.items()}
                        x_dict = {key: F.dropout(x, p = 0.5, training = self.training) for key, x in x_dict.items()}
                        x = x_dict['GENE']
                        x = self.lin(x)         
                        return torch.sigmoid(x)

                class HeteroConv_1SAGE_tanh(torch.nn.Module):
                    def __init__(self, hidden_channels_1, out_channels, sage_aggr = 'sum', sage_norm = True, sage_project = True, hetero_aggr_1 = 'sum'): # num_layers, hidden_linear_2
                        super().__init__()
                        # torch.manual_seed(1234567)
                        self.conv1 = HeteroConv({ 
                                ('GENE', 'expressed_in', 'ANATOMY'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('ANATOMY', 'rev_expressed_in', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'not_expressed_in', 'ANATOMY'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('ANATOMY', 'rev_not_expressed_in', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'PPI', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'CoExp', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'paralog', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'Neighbour', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'gene_to_BP', 'BP'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('BP', 'rev_gene_to_BP', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'gene_to_CC', 'CC'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('CC', 'rev_gene_to_CC', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'gene_to_MF', 'MF'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('MF', 'rev_gene_to_MF', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('ANATOMY', 'ana_ana', 'ANATOMY'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('ANATOMY', 'rev_ana_ana', 'ANATOMY'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('BP', 'biological_process', 'BP'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('BP', 'rev_biological_process', 'BP'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('CC', 'cellular_component', 'CC'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('CC', 'rev_cellular_component', 'CC'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('MF', 'molecular_function', 'MF'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('MF', 'rev_molecular_function', 'MF'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project)
                            }, aggr=hetero_aggr_1)
                        self.lin = Linear(hidden_channels_1, out_channels)
                    def forward(self, x_dict, edge_index_dict):
                        x_dict = self.conv1(x_dict, edge_index_dict)
                        x_dict = {key: torch.tanh(x) for key, x in x_dict.items()}
                        x_dict = {key: F.dropout(x, p = 0.5, training = self.training) for key, x in x_dict.items()}
                        x = x_dict['GENE']
                        x = self.lin(x)         
                        return torch.sigmoid(x)
                    
                class HeteroConv_1SAGE_relu(torch.nn.Module):
                    def __init__(self, hidden_channels_1, out_channels, sage_aggr = 'max', sage_norm = True, sage_project = True, hetero_aggr_1 = 'sum'): # num_layers, hidden_linear_2
                        super().__init__()
                        # torch.manual_seed(1234567)
                        self.conv1 = HeteroConv({ 
                                ('GENE', 'expressed_in', 'ANATOMY'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('ANATOMY', 'rev_expressed_in', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'not_expressed_in', 'ANATOMY'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('ANATOMY', 'rev_not_expressed_in', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'PPI', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'CoExp', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'paralog', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'Neighbour', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'gene_to_BP', 'BP'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('BP', 'rev_gene_to_BP', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'gene_to_CC', 'CC'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('CC', 'rev_gene_to_CC', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'gene_to_MF', 'MF'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('MF', 'rev_gene_to_MF', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('ANATOMY', 'ana_ana', 'ANATOMY'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('ANATOMY', 'rev_ana_ana', 'ANATOMY'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('BP', 'biological_process', 'BP'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('BP', 'rev_biological_process', 'BP'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('CC', 'cellular_component', 'CC'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('CC', 'rev_cellular_component', 'CC'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('MF', 'molecular_function', 'MF'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('MF', 'rev_molecular_function', 'MF'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project)
                            }, aggr=hetero_aggr_1)
                        self.lin = Linear(hidden_channels_1, out_channels)
                    def forward(self, x_dict, edge_index_dict):
                        x_dict = self.conv1(x_dict, edge_index_dict)
                        x_dict = {key: x.relu() for key, x in x_dict.items()}
                        x_dict = {key: F.dropout(x, p = 0.5, training = self.training) for key, x in x_dict.items()}
                        x = x_dict['GENE']
                        x = self.lin(x)         
                        return torch.sigmoid(x)
                    
                class HeteroConv_2SAGE_tanh(torch.nn.Module):
                    def __init__(self, hidden_channels_1, hidden_channels_2, out_channels, sage_aggr = 'sum', sage_norm = True, sage_project = False, hetero_aggr_1 = 'sum', hetero_aggr_2 = 'sum'): # num_layers, hidden_linear_2
                        super().__init__()
                        # torch.manual_seed(1234567)
                        self.conv1 = HeteroConv({ 
                                ('GENE', 'expressed_in', 'ANATOMY'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('ANATOMY', 'rev_expressed_in', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'not_expressed_in', 'ANATOMY'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('ANATOMY', 'rev_not_expressed_in', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'PPI', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'CoExp', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'paralog', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'Neighbour', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'gene_to_BP', 'BP'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('BP', 'rev_gene_to_BP', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'gene_to_CC', 'CC'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('CC', 'rev_gene_to_CC', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'gene_to_MF', 'MF'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('MF', 'rev_gene_to_MF', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('ANATOMY', 'ana_ana', 'ANATOMY'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('ANATOMY', 'rev_ana_ana', 'ANATOMY'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('BP', 'biological_process', 'BP'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('BP', 'rev_biological_process', 'BP'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('CC', 'cellular_component', 'CC'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('CC', 'rev_cellular_component', 'CC'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('MF', 'molecular_function', 'MF'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('MF', 'rev_molecular_function', 'MF'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project)
                            }, aggr=hetero_aggr_1)
                        self.conv2 = HeteroConv({ 
                                ('GENE', 'expressed_in', 'ANATOMY'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('ANATOMY', 'rev_expressed_in', 'GENE'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('GENE', 'not_expressed_in', 'ANATOMY'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('ANATOMY', 'rev_not_expressed_in', 'GENE'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
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
                                ('MF', 'rev_molecular_function', 'MF'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project)
                            }, aggr=hetero_aggr_2)
                        self.lin = Linear(hidden_channels_2, out_channels)
                    def forward(self, x_dict, edge_index_dict):
                        x_dict = self.conv1(x_dict, edge_index_dict)
                        x_dict = {key: torch.tanh(x) for key, x in x_dict.items()}
                        x_dict = {key: F.dropout(x, p = 0.5, training = self.training) for key, x in x_dict.items()}
                        x_dict = self.conv2(x_dict, edge_index_dict)
                        x_dict = {key: torch.tanh(x) for key, x in x_dict.items()}
                        x_dict = {key: F.dropout(x, p = 0.5, training = self.training) for key, x in x_dict.items()}
                        x = x_dict['GENE']
                        x = self.lin(x)         
                        return torch.sigmoid(x)
                    
                class HeteroConv_2SAGE_relu(torch.nn.Module):
                    def __init__(self, hidden_channels_1, hidden_channels_2, out_channels, sage_aggr = 'sum', sage_norm = True, sage_project = False, hetero_aggr_1 = 'sum', hetero_aggr_2 = 'sum'): # num_layers, hidden_linear_2
                        super().__init__()
                        # torch.manual_seed(1234567)
                        self.conv1 = HeteroConv({ 
                                ('GENE', 'expressed_in', 'ANATOMY'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('ANATOMY', 'rev_expressed_in', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'not_expressed_in', 'ANATOMY'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('ANATOMY', 'rev_not_expressed_in', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'PPI', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'CoExp', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'paralog', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'Neighbour', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'gene_to_BP', 'BP'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('BP', 'rev_gene_to_BP', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'gene_to_CC', 'CC'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('CC', 'rev_gene_to_CC', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'gene_to_MF', 'MF'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('MF', 'rev_gene_to_MF', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('ANATOMY', 'ana_ana', 'ANATOMY'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('ANATOMY', 'rev_ana_ana', 'ANATOMY'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('BP', 'biological_process', 'BP'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('BP', 'rev_biological_process', 'BP'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('CC', 'cellular_component', 'CC'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('CC', 'rev_cellular_component', 'CC'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('MF', 'molecular_function', 'MF'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('MF', 'rev_molecular_function', 'MF'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project)
                            }, aggr=hetero_aggr_1)
                        self.conv2 = HeteroConv({ 
                                ('GENE', 'expressed_in', 'ANATOMY'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('ANATOMY', 'rev_expressed_in', 'GENE'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('GENE', 'not_expressed_in', 'ANATOMY'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('ANATOMY', 'rev_not_expressed_in', 'GENE'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
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
                                ('MF', 'rev_molecular_function', 'MF'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project)
                            }, aggr=hetero_aggr_2)
                        self.lin = Linear(hidden_channels_2, out_channels)
                    def forward(self, x_dict, edge_index_dict):
                        x_dict = self.conv1(x_dict, edge_index_dict)
                        x_dict = {key: torch.relu(x) for key, x in x_dict.items()}
                        x_dict = {key: F.dropout(x, p = 0.5, training = self.training) for key, x in x_dict.items()}
                        x_dict = self.conv2(x_dict, edge_index_dict)
                        x_dict = {key: torch.relu(x) for key, x in x_dict.items()}
                        x_dict = {key: F.dropout(x, p = 0.5, training = self.training) for key, x in x_dict.items()}
                        x = x_dict['GENE']
                        x = self.lin(x)         
                        return torch.sigmoid(x)
                    
                class HeteroConv_3SAGE_relu(torch.nn.Module):
                    def __init__(self, hidden_channels_1, hidden_channels_2, hidden_channels_3, out_channels, sage_aggr = 'sum', sage_norm = True, sage_project = False, hetero_aggr_1 = 'sum', hetero_aggr_2 = 'sum', hetero_aggr_3='sum'): # num_layers, hidden_linear_2
                        super().__init__()
                        # torch.manual_seed(1234567)
                        self.conv1 = HeteroConv({ 
                                ('GENE', 'expressed_in', 'ANATOMY'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('ANATOMY', 'rev_expressed_in', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'not_expressed_in', 'ANATOMY'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('ANATOMY', 'rev_not_expressed_in', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'PPI', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'CoExp', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'paralog', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'Neighbour', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'gene_to_BP', 'BP'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('BP', 'rev_gene_to_BP', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'gene_to_CC', 'CC'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('CC', 'rev_gene_to_CC', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'gene_to_MF', 'MF'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('MF', 'rev_gene_to_MF', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('ANATOMY', 'ana_ana', 'ANATOMY'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('ANATOMY', 'rev_ana_ana', 'ANATOMY'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('BP', 'biological_process', 'BP'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('BP', 'rev_biological_process', 'BP'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('CC', 'cellular_component', 'CC'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('CC', 'rev_cellular_component', 'CC'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('MF', 'molecular_function', 'MF'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('MF', 'rev_molecular_function', 'MF'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project)
                            }, aggr=hetero_aggr_1)
                        self.conv2 = HeteroConv({ 
                                ('GENE', 'expressed_in', 'ANATOMY'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('ANATOMY', 'rev_expressed_in', 'GENE'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('GENE', 'not_expressed_in', 'ANATOMY'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('ANATOMY', 'rev_not_expressed_in', 'GENE'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
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
                                ('MF', 'rev_molecular_function', 'MF'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project)
                            }, aggr=hetero_aggr_2)
                        self.conv3 = HeteroConv({ 
                                ('GENE', 'expressed_in', 'ANATOMY'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('ANATOMY', 'rev_expressed_in', 'GENE'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('GENE', 'not_expressed_in', 'ANATOMY'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('ANATOMY', 'rev_not_expressed_in', 'GENE'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('GENE', 'PPI', 'GENE'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('GENE', 'CoExp', 'GENE'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('GENE', 'paralog', 'GENE'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('GENE', 'Neighbour', 'GENE'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('GENE', 'gene_to_BP', 'BP'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('BP', 'rev_gene_to_BP', 'GENE'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('GENE', 'gene_to_CC', 'CC'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('CC', 'rev_gene_to_CC', 'GENE'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('GENE', 'gene_to_MF', 'MF'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('MF', 'rev_gene_to_MF', 'GENE'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('ANATOMY', 'ana_ana', 'ANATOMY'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('ANATOMY', 'rev_ana_ana', 'ANATOMY'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('BP', 'biological_process', 'BP'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('BP', 'rev_biological_process', 'BP'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('CC', 'cellular_component', 'CC'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('CC', 'rev_cellular_component', 'CC'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('MF', 'molecular_function', 'MF'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('MF', 'rev_molecular_function', 'MF'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project)
                            }, aggr=hetero_aggr_3)
                        self.lin = Linear(hidden_channels_3, out_channels)
                    def forward(self, x_dict, edge_index_dict):
                        x_dict = self.conv1(x_dict, edge_index_dict)
                        x_dict = {key: torch.relu(x) for key, x in x_dict.items()}
                        x_dict = {key: F.dropout(x, p = 0.5, training = self.training) for key, x in x_dict.items()}
                        x_dict = self.conv2(x_dict, edge_index_dict)
                        x_dict = {key: torch.relu(x) for key, x in x_dict.items()}
                        x_dict = {key: F.dropout(x, p = 0.5, training = self.training) for key, x in x_dict.items()}
                        x_dict = self.conv3(x_dict, edge_index_dict)
                        x_dict = {key: torch.relu(x) for key, x in x_dict.items()}
                        x_dict = {key: F.dropout(x, p = 0.5, training = self.training) for key, x in x_dict.items()}
                        x = x_dict['GENE']
                        x = self.lin(x)         
                        return torch.sigmoid(x)
                    
                class HeteroConv_3SAGE_tanh(torch.nn.Module):
                    def __init__(self, hidden_channels_1, hidden_channels_2, hidden_channels_3, out_channels, sage_aggr = 'sum', sage_norm = True, sage_project = False, hetero_aggr_1 = 'sum', hetero_aggr_2 = 'sum', hetero_aggr_3='sum'): # num_layers, hidden_linear_2
                        super().__init__()
                        # torch.manual_seed(1234567)
                        self.conv1 = HeteroConv({ 
                                ('GENE', 'expressed_in', 'ANATOMY'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('ANATOMY', 'rev_expressed_in', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'not_expressed_in', 'ANATOMY'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('ANATOMY', 'rev_not_expressed_in', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'PPI', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'CoExp', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'paralog', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'Neighbour', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'gene_to_BP', 'BP'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('BP', 'rev_gene_to_BP', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'gene_to_CC', 'CC'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('CC', 'rev_gene_to_CC', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('GENE', 'gene_to_MF', 'MF'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('MF', 'rev_gene_to_MF', 'GENE'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = False, project = sage_project), 
                                ('ANATOMY', 'ana_ana', 'ANATOMY'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('ANATOMY', 'rev_ana_ana', 'ANATOMY'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('BP', 'biological_process', 'BP'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('BP', 'rev_biological_process', 'BP'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('CC', 'cellular_component', 'CC'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('CC', 'rev_cellular_component', 'CC'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('MF', 'molecular_function', 'MF'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('MF', 'rev_molecular_function', 'MF'): SAGEConv(-1, hidden_channels_1, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project)
                            }, aggr=hetero_aggr_1)
                        self.conv2 = HeteroConv({ 
                                ('GENE', 'expressed_in', 'ANATOMY'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('ANATOMY', 'rev_expressed_in', 'GENE'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('GENE', 'not_expressed_in', 'ANATOMY'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('ANATOMY', 'rev_not_expressed_in', 'GENE'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
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
                                ('MF', 'rev_molecular_function', 'MF'): SAGEConv(-1, hidden_channels_2, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project)
                            }, aggr=hetero_aggr_2)
                        self.conv3 = HeteroConv({ 
                                ('GENE', 'expressed_in', 'ANATOMY'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('ANATOMY', 'rev_expressed_in', 'GENE'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('GENE', 'not_expressed_in', 'ANATOMY'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('ANATOMY', 'rev_not_expressed_in', 'GENE'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('GENE', 'PPI', 'GENE'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('GENE', 'CoExp', 'GENE'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('GENE', 'paralog', 'GENE'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('GENE', 'Neighbour', 'GENE'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('GENE', 'gene_to_BP', 'BP'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('BP', 'rev_gene_to_BP', 'GENE'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('GENE', 'gene_to_CC', 'CC'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('CC', 'rev_gene_to_CC', 'GENE'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('GENE', 'gene_to_MF', 'MF'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('MF', 'rev_gene_to_MF', 'GENE'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('ANATOMY', 'ana_ana', 'ANATOMY'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('ANATOMY', 'rev_ana_ana', 'ANATOMY'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('BP', 'biological_process', 'BP'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('BP', 'rev_biological_process', 'BP'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('CC', 'cellular_component', 'CC'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('CC', 'rev_cellular_component', 'CC'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('MF', 'molecular_function', 'MF'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project), 
                                ('MF', 'rev_molecular_function', 'MF'): SAGEConv(-1, hidden_channels_3, aggr = sage_aggr, normalize = sage_norm, root_weight = True, project = sage_project)
                            }, aggr=hetero_aggr_3)
                        self.lin = Linear(hidden_channels_3, out_channels)
                    def forward(self, x_dict, edge_index_dict):
                        x_dict = self.conv1(x_dict, edge_index_dict)
                        x_dict = {key: torch.tanh(x) for key, x in x_dict.items()}
                        x_dict = {key: F.dropout(x, p = 0.5, training = self.training) for key, x in x_dict.items()}
                        x_dict = self.conv2(x_dict, edge_index_dict)
                        x_dict = {key: torch.tanh(x) for key, x in x_dict.items()}
                        x_dict = {key: F.dropout(x, p = 0.5, training = self.training) for key, x in x_dict.items()}
                        x_dict = self.conv3(x_dict, edge_index_dict)
                        x_dict = {key: torch.tanh(x) for key, x in x_dict.items()}
                        x_dict = {key: F.dropout(x, p = 0.5, training = self.training) for key, x in x_dict.items()}
                        x = x_dict['GENE']
                        x = self.lin(x)         
                        return torch.sigmoid(x)


                ## plotting function 

                def get_ratio_of_pred_row(row):
                    pred = row['pred']
                    left_pred = row['pred_node_A']
                    right_pred = row['pred_node_B']
                    base_product_pred = left_pred * right_pred
                    if base_product_pred == 0:
                        base_product_pred = 0.0001
                    else:
                        base_product_pred = base_product_pred
                    corrected_pred = pred / base_product_pred
                    return corrected_pred

                ### new_one

                # TODO update to use the train loader and the test_loader = will be faster when predicting on pairs
                def plot_model_df(model_to_test, global_loader, zero_out_batch: bool = True, val_set: bool = True, test_set: bool = True, setting: str = 'nodes_to_nodes'):
                    # plant random seeds for reproductibility
                    torch.manual_seed(31415926535)
                    np.random.seed(31415)
                    random.seed(31415)

                    model = model_to_test
                    global_loader = global_loader
                    model.eval()
                    print("I am getting the output of everything, it will take some time.")
                    # https://stackoverflow.com/questions/64398900/how-to-concatenate-like-list-comprehension
                    total_out = torch.tensor([], dtype = torch.float64)
                    for batch in tqdm(global_loader):
                        batch = batch.to(device)
                        for edge in batch.edge_index_dict.keys():
                            batch[edge].edge_index = batch[edge].edge_index.to(device)
                        batch_size = batch['GENE'].batch_size
                        assert int(sum(batch['GENE'].input_id == batch['GENE'].n_id[:batch_size])) == batch_size
                        if zero_out_batch == True:
                            batch['GENE'].x[:batch['GENE'].batch_size] = batch['GENE'].x[:batch['GENE'].batch_size].zero_()
                        else:
                            pass
                        tmp_out = model(batch.x_dict, batch.edge_index_dict)[:batch_size]
                        tmp_out = tmp_out.cpu().detach()
                        total_out = torch.cat((total_out, tmp_out), axis = 0)   

                    out_train = total_out[data['GENE'].train_mask]
                    if val_set == True:
                        out_val = total_out[data['GENE'].val_mask]
                    if test_set == True:
                        out_test = total_out[data['GENE'].test_mask]

                    print("Ok, I have the final prediction tensor, now I am making some calculations.")
                    # get the probabilities from my model
                    out_train_arr = out_train.cpu().detach().numpy() 
                    if val_set == True:
                        out_val_arr = out_val.cpu().detach().numpy() 
                    if test_set == True:
                        out_test_arr = out_test.cpu().detach().numpy()

                    out_tot = total_out.cpu().detach().numpy()
                    del total_out
                    tot_pred = pd.DataFrame(out_tot.ravel(), columns = ['pred'])
                    # TODO update when there are pairs
                    if setting == 'nodes_to_pairs':
                        fusion = fd[['GENE_id', 'mapped_id', 'mapped_id_left', 'mapped_id_right', 'type', 'target', 'mask']]
                        fusion = pd.concat([fusion, tot_pred], axis = 1)
                        pred_id = fusion[['mapped_id', 'pred']]
                        fusion = fusion.merge(pred_id, how = 'left', left_on = 'mapped_id_left', right_on = 'mapped_id', suffixes = (None, '_node_A'))
                        fusion = fusion.merge(pred_id, how = 'left', left_on = 'mapped_id_right', right_on = 'mapped_id', suffixes = (None, '_node_B'))
                        fusion = fusion.drop(['mapped_id_node_A', 'mapped_id_node_B'], axis = 1)
                        fusion.loc[:, 'pred(A)*pred(B)'] = fusion['pred_node_A'] * fusion['pred_node_B']
                        fusion.loc[:, 'mean[p(A)+p(B)]'] = fusion[['pred_node_A', 'pred_node_B']].mean(axis = 1)
                        new_fusion = fusion.iloc[len(nodes2id['GENE']):,]
                    elif setting == 'pairs_to_pairs':
                        fusion = fd[['GENE_id', 'mapped_id', 'mapped_id_left', 'mapped_id_right', 'type', 'target', 'mask']]
                        fusion = pd.concat([fusion, tot_pred], axis = 1)
                    elif setting == 'nodes_to_nodes':
                        fusion = nodes2id['GENE'][['GENE_id', 'mapped_id']]
                        fusion = pd.concat([fusion, tot_pred], axis = 1)
                    else:
                        raise Exception("Please select a setting between 'node_to_nodes', 'pairs_to_pairs', 'nodes_to_pairs'.")

                    ## get the predicted classes for a given threshold
                    # get the ground truth
                    target_train_arr = data['GENE'].y[data['GENE'].train_mask].cpu().detach().numpy()
                    auroc_train = roc_auc_score(target_train_arr, out_train_arr)
                    aupr_train = average_precision_score(target_train_arr, out_train_arr)

                    if val_set == True:
                        target_val_arr = data['GENE'].y[data['GENE'].val_mask].cpu().detach().numpy()
                        auroc_val = roc_auc_score(target_val_arr, out_val_arr)
                        aupr_val = average_precision_score(target_val_arr, out_val_arr)
                    if test_set == True:
                        target_test_arr = data['GENE'].y[data['GENE'].test_mask].cpu().detach().numpy()
                        auroc_test = roc_auc_score(target_test_arr, out_test_arr)
                        aupr_test = average_precision_score(target_test_arr, out_test_arr)

                    '''F1 score, Matthews correlation coefficient, F0.01 score and Confusion Matrix'''
                    ## train_set
                    f1_scores_train = {}
                    mcc_scores_train = {}
                    beta = 0.01
                    f001_scores_train = {} 
                    for t in np.arange(0, 1, 0.01):
                        yhat_train = (out_train > t).float().cpu().detach().numpy()
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
                            yhat_train = (out_train > t).float().cpu().detach().numpy()
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
                            yhat_val = (out_val > t).float().cpu().detach().numpy()
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
                                yhat_val = (out_val > t).float().cpu().detach().numpy()
                                _, _, f001_val, _ = precision_recall_fscore_support(target_val_arr, yhat_val, beta = beta, zero_division = 0)
                                f001_scores_val[t] = f001_val[1]
                            best_f001_t_val, best_f001_val = max(f001_scores_val.items(), key = lambda k: k[1])
                    ## test_set
                    if test_set == True:
                        f1_scores_test = {}
                        mcc_scores_test = {}
                        beta = 0.01
                        f001_scores_test = {}
                        for t in np.arange(0, 1, 0.01):
                            yhat_test = (out_test > t).float().cpu().detach().numpy()
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
                                yhat_test = (out_test > t).float().cpu().detach().numpy()
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
                    if test_set == True:
                        f1_test_yhat_arr = (out_test_arr > best_f1_t_test).astype(float)
                        f1_cm_test = confusion_matrix(target_test_arr, f1_test_yhat_arr.ravel())
                        f001_test_yhat_arr = (out_test_arr > best_f001_t_test).astype(float)
                        f001_cm_test = confusion_matrix(target_test_arr, f001_test_yhat_arr.ravel())
                        mcc_test_yhat_arr = (out_test_arr > best_mcc_t_test).astype(float)
                        mcc_cm_test = confusion_matrix(target_test_arr, mcc_test_yhat_arr.ravel())

                    '''ROC and PR curves'''
                    # on the train set
                    ns_probs_train = [0 for _ in range(len(data['GENE'].y[data['GENE'].train_mask]))]
                    ns_fpr, ns_tpr, _ = roc_curve(target_train_arr, ns_probs_train)
                    model_fpr_train, model_tpr_train, _ = roc_curve(target_train_arr, out_train_arr)
                    no_skill_train = len(data['GENE'].y[data['GENE'].train_mask][data['GENE'].y[data['GENE'].train_mask] == 1])/len(data['GENE'].y[data['GENE'].train_mask]) # no_skill = 0.15
                    precision_train, recall_train, _ = precision_recall_curve(target_train_arr, out_train_arr)
                    # on the val set
                    if val_set == True:
                        ns_probs_val = [0 for _ in range(len(data['GENE'].y[data['GENE'].val_mask]))]
                        # ns_fpr_val, ns_tpr_val, _ = roc_curve(target_val_arr, ns_probs_val)
                        model_fpr_val, model_tpr_val, _ = roc_curve(target_val_arr, out_val_arr)
                        no_skill_val = len(data['GENE'].y[data['GENE'].val_mask][data['GENE'].y[data['GENE'].val_mask] == 1])/len(data['GENE'].y[data['GENE'].val_mask])
                        precision_val, recall_val, _ = precision_recall_curve(target_val_arr, out_val_arr) 
                    # on test set
                    if test_set == True:
                        ns_probs_test = [0 for _ in range(len(data['GENE'].y[data['GENE'].test_mask]))]
                        # ns_fpr_test, ns_tpr_test, _ = roc_curve(target_test_arr, ns_probs_test)
                        model_fpr_test, model_tpr_test, _ = roc_curve(target_test_arr, out_test_arr)
                        no_skill_test = len(data['GENE'].y[data['GENE'].test_mask][data['GENE'].y[data['GENE'].test_mask] == 1])/len(data['GENE'].y[data['GENE'].test_mask])
                        precision_test, recall_test, _ = precision_recall_curve(target_test_arr, out_test_arr) 
                    
                    '''Plot the figure'''
                    # figsize = (float, float), (Width, Height) in inches
                    if val_set and test_set:
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
                        ax_roc.plot(model_fpr_val, model_tpr_val, marker='.', label=f'Validation set: AUC = {auroc_val:0.4f}', color = 'tab:green')
                        ax_roc.plot(model_fpr_test, model_tpr_test, marker='.', label=f'Test set: AUC = {auroc_test:0.4f}', color = 'tab:blue') # '#1f77b4'
                        # axis labels
                        ax_roc.set_xlabel('False Positive Rate')
                        ax_roc.set_ylabel('True Positive Rate')
                        ax_roc.legend(loc = 'lower right')
                        ax_roc.set_title("ROC curves")
                        ## do the second plot: AUPR
                        ax_pr.plot([0, 1], [no_skill_train, no_skill_train], linestyle='--', label='Random Model on Train set', color = 'tab:brown')
                        ax_pr.plot([0, 1], [no_skill_val, no_skill_val], linestyle='--', label='Random Model on Validation set', color = 'tab:olive')
                        ax_pr.plot([0, 1], [no_skill_test, no_skill_test], linestyle='--', label='Random Model on Test set', color = 'tab:cyan')
                        ax_pr.plot([0, 1], [0, 0], linestyle='--', label = None, color = 'purple', alpha = 0) # draw an invisible line, just for scaling purpose, y = 0
                        ax_pr.plot(recall_train, precision_train, marker='.', label=f'Train set: AUPR = {aupr_train:0.4f}', color = 'tab:orange') # '#ff7f0e'
                        ax_pr.plot(recall_val, precision_val, marker='.', label=f'Validation set: AUPR = {aupr_val:0.4f}', color = 'tab:green')
                        ax_pr.plot(recall_test, precision_test, marker='.', label=f'Test set: AUPR = {aupr_test:0.4f}', color = 'tab:blue') # '#1f77b4'
                        # axis labels
                        ax_pr.set_xlabel('Recall')
                        ax_pr.set_ylabel('Precision')
                        ax_pr.legend(loc = 'lower left')
                        ax_pr.set_title("PR curves")
                        ## do the third plot: corrected prediction
                        if setting == 'nodes_to_pairs':
                            ax_pairs.scatter(new_fusion['pred(A)*pred(B)'].tolist(), new_fusion['pred'].tolist(), marker = 'x')
                            # axis labels
                            ax_pairs.set_xlabel('Product of the predictions of individual nodes: p(A)*p(B)')
                            ax_pairs.set_ylabel('Prediction for the fusion-nodes: p([AB])')
                            # show the legend
                            # ax_pairs.legend(loc = 'lower left')
                            ax_pairs.set_title("Prediction for the new-nodes")
                        elif setting == 'pairs_to_pairs':
                            fusion.boxplot(column = 'pred', by = 'type', ax = ax_pairs)
                            ax_pairs.set_title("Distribution of the predictions")
                        
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
                        
                    elif not val_set and test_set: # use_val_set == False:
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
                        ax_pr.plot([0, 1], [0, 0], linestyle='--', label = None, color = 'purple', alpha = 0) # draw an invisible line, just for scaling purpose, y = 0
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

                        data_df = {'results': [auroc_train, auroc_test, aupr_train, aupr_test,
                                            best_mcc_train, best_mcc_t_train, np.ravel(mcc_cm_train),
                                            best_mcc_test, best_mcc_t_test, np.ravel(mcc_cm_test),
                                            best_f1_train, best_f1_t_train, np.ravel(f1_cm_train),  
                                            best_f1_test, best_f1_t_test, np.ravel(f1_cm_test),
                                            best_f001_train, best_f001_t_train, np.ravel(f001_cm_train), 
                                            best_f001_test, best_f001_t_test, np.ravel(f001_cm_test)]}
                        
                        col_names = [
                            'AUROC_train', 'AUROC_test', 'AUPR_train', 'AUPR_test',
                            'best_MCC_train', 'best_MCC_train_threshold', 'best_MCC_train_cm',
                            'best_MCC_test', 'best_MCC_test_threshold', 'best_MCC_test_cm',
                            'best_F1_train', 'best_F1_train_threshold', 'best_F1_train_cm',
                            'best_F1_test', 'best_F1_test_threshold', 'best_F1_test_cm',
                            'best_F0.01_train', 'best_F0.01_train_threshold', 'best_F0.01_train_cm',
                            'best_F0.01_test', 'best_F0.01_test_threshold', 'best_F0.01_test_cm']
                        df = pd.DataFrame.from_dict(data_df, orient = 'index', columns = col_names)

                    else:
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
                        # ax_roc.plot(model_fpr_test, model_tpr_test, marker='.', label=f'Test set: AUC = {auroc_test:0.4f}', color = 'tab:blue') # '#1f77b4'
                        # axis labels
                        ax_roc.set_xlabel('False Positive Rate')
                        ax_roc.set_ylabel('True Positive Rate')
                        ax_roc.legend(loc = 'lower right')
                        ax_roc.set_title("ROC curves")
                        ## do the second plot: AUPR
                        ax_pr.plot([0, 1], [no_skill_train, no_skill_train], linestyle='--', label='Random Model on Train set', color = 'tab:brown')
                        # ax_pr.plot([0, 1], [no_skill_test, no_skill_test], linestyle='--', label='Random Model on Test set', color = 'tab:cyan')
                        ax_pr.plot([0, 1], [0, 0], linestyle='--', label = None, color = 'purple', alpha = 0) # juste pour avoir la même échelle, tracer une ligne invisible, y = 0
                        ax_pr.plot(recall_train, precision_train, marker='.', label=f'Train set: AUPR = {aupr_train:0.4f}', color = 'tab:orange') # '#ff7f0e'
                        # ax_pr.plot(recall_test, precision_test, marker='.', label=f'Test set: AUPR = {aupr_test:0.4f}', color = 'tab:blue') # '#1f77b4'
                        # axis labels
                        ax_pr.set_xlabel('Recall')
                        ax_pr.set_ylabel('Precision')
                        ax_pr.legend(loc = 'lower left')
                        ax_pr.set_title("PR curves")

                        ## do the third plot: confusion matrix train dataset MCC
                        ConfusionMatrixDisplay(mcc_cm_train).plot(ax = ax_mcc_train)
                        ax_mcc_train.set_title(f"Train set: MCC = {best_mcc_train:0.4f} with t = {best_mcc_t_train:0.4f}")
                        ## do the fourth plot: confusion matrix test dataset MCC
                        # ConfusionMatrixDisplay(mcc_cm_test).plot(ax = ax_mcc_test)
                        # ax_mcc_test.set_title(f"Test set: MCC = {best_mcc_test:0.4f} with t = {best_mcc_t_test:0.4f}")
                        
                        ## do the fifth plot: confusion matrix train dataset F1
                        ConfusionMatrixDisplay(f1_cm_train).plot(ax = ax_f1_train)
                        ax_f1_train.set_title(f"Train set: F1 = {best_f1_train:0.4f} with t = {best_f1_t_train:0.4f}")
                        ## do the sixth plot: confusion matrix test dataset F1
                        # ConfusionMatrixDisplay(f1_cm_test).plot(ax = ax_f1_test)
                        # ax_f1_test.set_title(f"Validation set: F1 = {best_f1_test:0.4f} with t = {best_f1_t_test:0.4f}")

                        # do the seventh plot: confusion matrix train dataset F001
                        ConfusionMatrixDisplay(f001_cm_train).plot(ax = ax_f001_train)
                        ax_f001_train.set_title(f"Train set: F-001 = {best_f001_train:0.4f} with t = {best_f001_t_train:0.4f}")  
                        # do the eighth plot: confusion matrix test dataset F001
                        # ConfusionMatrixDisplay(f001_cm_test).plot(ax = ax_f001_test)
                        # ax_f001_test.set_title(f"Test set: F-001 = {best_f001_test:0.4f} with t = {best_f001_t_test:0.4f}")

                        # show the plot
                        plt.show()

                        data_df = {'results': [auroc_train, aupr_train, 
                                            best_mcc_train, best_mcc_t_train, np.ravel(mcc_cm_train),
                                            best_f1_train, best_f1_t_train, np.ravel(f1_cm_train),  
                                            best_f001_train, best_f001_t_train, np.ravel(f001_cm_train), 
                                            ]}
                        
                        col_names = [
                            'AUROC_train', 'AUPR_train',
                            'best_MCC_train', 'best_MCC_train_threshold', 'best_MCC_train_cm',
                            'best_F1_train', 'best_F1_train_threshold', 'best_F1_train_cm',
                            'best_F0.01_train', 'best_F0.01_train_threshold', 'best_F0.01_train_cm',
                            ]
                        df = pd.DataFrame.from_dict(data_df, orient = 'index', columns = col_names)


                    return fig, df, fusion
                
                def get_df_results(model_to_test, loader, zero_out_batch: bool = True):
                    # plant random seeds for reproducibility
                    # torch.manual_seed(31415926535)
                    # np.random.seed(31415)
                    # random.seed(31415)
                    # seed_everything(31415)

                    model = model_to_test
                    model.eval()
                    print("I am getting the output for the test set, with all neighbors, it will take some time.")
                    # when there are edge_attr
                    # when there are no edge_attr        
                        # https://stackoverflow.com/questions/64398900/how-to-concatenate-like-list-comprehension
                    total_id = torch.tensor([], dtype = torch.float64)
                    total_out = torch.tensor([], dtype = torch.float64)
                    for batch in tqdm(loader):
                        batch = batch.to(device)
                        for edge in batch.edge_index_dict.keys():
                            batch[edge].edge_index = batch[edge].edge_index.to(device)
                        batch_size = batch['GENE'].batch_size
                        assert int(sum(batch['GENE'].input_id == batch['GENE'].n_id[:batch_size])) == batch_size
                        if zero_out_batch == True:
                            batch['GENE'].x[:batch['GENE'].batch_size] = batch['GENE'].x[:batch['GENE'].batch_size].zero_()
                        else:
                            pass
                        tmp_out = model(batch.x_dict, batch.edge_index_dict)[:batch_size]
                        tmp_out = tmp_out.cpu().detach()
                        total_out = torch.cat((total_out, tmp_out), axis = 0)   
                        tmp_id = batch['GENE'].input_id
                        tmp_id = tmp_id.cpu().detach()
                        total_id = torch.cat((total_id, tmp_id), axis = 0)
                    
                    # total out = just the test_set
                    # get the probabilities from my model
                    out_test_arr = total_out.cpu().detach().numpy()

                    print("Ok, I have the final prediction tensor for the test set, now I am making some calculations.")
                    results = torch.cat(tensors = (total_id.view(-1,1), total_out), dim = 1).cpu().numpy()
                    tot_pred = pd.DataFrame(results, columns = ['mapped_id', 'pred'])
                    tot_pred['mapped_id'] = tot_pred['mapped_id'].astype(int)

                    ## get the predicted classes for a given threshold
                    target_test_arr = data['GENE'].y[data['GENE'].test_mask].cpu().detach().numpy()
                    auroc_test = roc_auc_score(target_test_arr, out_test_arr)
                    aupr_test = average_precision_score(target_test_arr, out_test_arr)

                    '''F1 score, Matthews correlation coefficient, F0.01 score and Confusion Matrix'''
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
                            yhat_test = (out_test_arr > t).float().cpu().detach().numpy()
                            _, _, f001_test, _ = precision_recall_fscore_support(target_test_arr, yhat_test, beta = beta, zero_division = 0)
                            f001_scores_test[t] = f001_test[1]
                        best_f001_t_test, best_f001_test = max(f001_scores_test.items(), key = lambda k: k[1])  

                    ## get the confusion matrices
                    # test_set
                    f1_test_yhat_arr = (out_test_arr > best_f1_t_test).astype(float)
                    f1_cm_test = confusion_matrix(target_test_arr, f1_test_yhat_arr.ravel())
                    f001_test_yhat_arr = (out_test_arr > best_f001_t_test).astype(float)
                    f001_cm_test = confusion_matrix(target_test_arr, f001_test_yhat_arr.ravel())
                    mcc_test_yhat_arr = (out_test_arr > best_mcc_t_test).astype(float)
                    mcc_cm_test = confusion_matrix(target_test_arr, mcc_test_yhat_arr.ravel())

                    data_df = {'results': [
                        auroc_test, aupr_test,
                        best_mcc_test, best_mcc_t_test, np.ravel(mcc_cm_test),
                        best_f1_test, best_f1_t_test, np.ravel(f1_cm_test),
                        best_f001_test, best_f001_t_test, np.ravel(f001_cm_test)
                        ]}
                        
                    col_names = [
                        'AUROC_test', 'AUPR_test',
                        'best_MCC_test', 'best_MCC_test_threshold', 'best_MCC_test_cm',
                        'best_F1_test', 'best_F1_test_threshold', 'best_F1_test_cm',
                        'best_F0.01_test', 'best_F0.01_test_threshold', 'best_F0.01_test_cm']
                    df = pd.DataFrame.from_dict(data_df, orient = 'index', columns = col_names)

                    return tot_pred, df
                
                ## create data

                # create the data now
                gene_mode = hyperparameters['gene_mode']
                node_mode = hyperparameters['node_mode'] # 'id' or 'degree'
                edge_features = hyperparameters['edge_features'] # True or False

                list_of_edges, data = create_bcmg_data([
                    'gene_to_BP', 'biological_process', 
                    'gene_to_CC', 'cellular_component', 
                    'gene_to_MF', 'molecular_function'
                    ], feature_mode = node_mode, use_edge_attr = edge_features, merge_choice = False) 
                list_of_edges_2, data_2 = create_bcmg_data([
                    'PPI', 'paralog' 
                    ], use_edge_attr = edge_features, merge_choice = True) 
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

                # create the gene features
                feat_1000 = pd.read_csv(data_processed_path + 'yeast_features_1000_20240717.tsv', header = 0, sep = '\t')
                yeast_features_to_use = [
                    'dn_ds', 'chemical_compound_accumulation','chronological_lifespan', 'competitive_fitness',
                    'desiccation_resistance', 'haploinsufficient', 'heat_sensitivity', 'metal_resistance', 
                    'oxidative_stress_resistance', 'replicative_lifespan', 'resistance_to_chemicals', 'respiratory_growth',
                    'stress_resistance', 'toxin_resistance', 'utilization_of_nitrogen_source', 'vacuolar_morphology', 'vegetative_growth']
                data['GENE'].x = torch.from_numpy(feat_1000[yeast_features_to_use].astype('float32').values)
                target_tensor = get_target_tensor()
                data['GENE'].y = target_tensor 

                print(f"The gene features have shape: {data['GENE'].x.shape}")

                # TODO refaire mieux 
                yeast_genes = pd.read_csv(data_processed_path + 'yeast_essential_nonessential_20240612.tsv', header = 0, sep = '\t')
                yeast_genes['mapped_id'] = pd.RangeIndex(len(yeast_genes))
                yeast_genes.loc[yeast_genes[yeast_genes['type'] == 'essential'].index, 'target'] = 1
                yeast_genes.loc[yeast_genes[yeast_genes['type'] == 'non_essential'].index, 'target'] = 0
                yeast_genes = yeast_genes.rename(columns = {'gene_id': 'GENE_id'})
                fd = yeast_genes.copy()

                data['GENE'].num_nodes = len(data['GENE'].x)

                for edge in data.edge_types:
                    data[edge].edge_index = U.sort_edge_index(data[edge].edge_index)
                data = data.to('cpu')

                # TODO clean the 'keep_first' setting is not done anymore
                # if doublons_pairs == 'keep_first':
                #     df_pairs_name = f'20250724_yeasts_SL_matched_SNL_DMF_{DMF_t}_{sl_genes}_genes_disjoint_sets_splits_05_08.tsv'
                # elif doublons_pairs == 'remove_all':
                df_pairs_name = f'20250829_yeasts_SL_matched_SNL_no_interaction_DMF_{DMF_t}_{sl_genes}_genes_no_doublons_disjoint_sets_splits_05_08.tsv'
                fd = pd.read_csv(data_processed_path + df_pairs_name, header=0, sep='\t', dtype={3:str, 4:str})
                print(f'The shape of the pairs_df is: {fd.shape}')

                if splits == '80_20':
                    fd.loc[:,'mask'] = fd['mask_08'].values
                elif splits == '50_50':
                    fd.loc[:,'mask'] = fd['mask_05'].values

                fd_basic_columns = fd.columns

                ## new features and target
                new_features = torch.zeros(size = (fd.shape[0] - data['GENE'].x.shape[0], data['GENE'].x.shape[1]))
                new_target = torch.from_numpy(fd.target.values)
                data['GENE'].x = torch.cat(tensors = (data['GENE'].x, new_features), axis = 0)
                data['GENE'].y = new_target

                ## create new_edges

                # create the edges for the new_nodes
                def get_new_edges_for_fusion_nodes(relation: tuple, id: str, source_node: str, remove_forbidden_value: bool = True, return_type: str = 'torch'):
                    edges = data[relation].edge_index
                    fusion_id = id
                    #ex: mapped_id = 19592, left_id = 129, right_id = 7277
                    if source_node == 'left':
                        src_id = fd_end[fd_end['mapped_id'] == fusion_id]['mapped_id_left'].values[0]
                        forbidden_value = fd_end[fd_end['mapped_id'] == fusion_id]['mapped_id_right'].values[0]
                    elif source_node == 'right':
                        src_id = fd_end[fd_end['mapped_id'] == fusion_id]['mapped_id_right'].values[0]
                        forbidden_value = fd_end[fd_end['mapped_id'] == fusion_id]['mapped_id_left'].values[0]
                    else: 
                        raise Exception("You have to choose either 'left' or 'right' as a source node for the fusion node.")
                    all_potential_edges = edges[:, edges[0] == src_id] #value_of_interest
                    # remove the forbidden value
                    if remove_forbidden_value == True:
                        new_dst = all_potential_edges[:,all_potential_edges[1] != forbidden_value][1:,:]
                    else:
                        new_dst = all_potential_edges[1:,:]
                    if return_type == 'numpy':
                        return new_dst.numpy()
                    elif return_type == 'torch':
                        return new_dst

                def get_dst_for_source(relation: tuple , id: str, return_type: str = 'torch'):
                    edges = data[relation].edge_index
                    dst = edges[1:, edges[0] == id]
                    if return_type == 'numpy':
                        return dst.numpy()
                    elif return_type == 'torch':
                        return dst
                    
                def get_tensors_for_fusion_fdend(edge_type: str, id:str, input_type: str = 'numpy', output_type: str = 'numpy'):
                # the node id for my fusion nodes are the destination !!
                # all the nodes that individual nodes point to are the sources nodes for the fusion ones
                    left_col = 'left_' + edge_type + '_dst_nodes'     
                    right_col = 'right_' + edge_type + '_dst_nodes'  
                    # base_value = yeast_genes.shape[0]
                    # src = torch.cat((fd_end.loc[id-base_value, left_col], fd_end.loc[id-base_value, right_col]), dim = 1)
                    if input_type == 'numpy':
                        src = np.concatenate((fd_end.loc[fd_end[fd_end['mapped_id'] == id].index[0], left_col], fd_end.loc[fd_end[fd_end['mapped_id'] == id].index[0], right_col]), axis = 1)
                        dst = np.full((src.shape), id)
                        edges = np.concatenate((src, dst), axis = 0)
                        if output_type == 'numpy':
                            edges = edges
                        elif output_type == 'torch':
                            edges = torch.from_numpy(edges)
                    elif input_type == 'torch':
                        src = torch.cat((fd_end.loc[fd_end[fd_end['mapped_id'] == id].index[0], left_col], fd_end.loc[fd_end[fd_end['mapped_id'] == id].index[0], right_col]), dim = 1)
                        dst = torch.full((src.shape), id)
                        edges = torch.cat((src,dst), dim = 0)
                        if output_type == 'torch':
                            edges = edges
                        elif output_type == 'numpy':
                            edges = edges.numpy()
                    return edges

                fd_end = fd.iloc[fd[~fd['GENE_id'].str.contains("_")].shape[0]:,:]
                type_to_return = hyperparameters['type_to_return']
                final_type = hyperparameters['final_type']
                remove_forbidden_value = hyperparameters['remove_forbidden_value']

                def get_common_tensors_for_fusion_fdend(edge_type: str, id:str, input_type: str = 'numpy', output_type: str = 'numpy'):
                # the node_id for my fusion nodes are the destination !!
                # only the common nodes that the individual nodes both points to are the sources nodes for the fusion ones
                    left_col = 'left_' + edge_type + '_dst_nodes'     
                    right_col = 'right_' + edge_type + '_dst_nodes' 
                    if input_type == 'numpy':
                        # left = fd_end.loc[fd_end[fd_end['mapped_id'] == id].index[0], left_col]
                        # right = fd_end.loc[fd_end[fd_end['mapped_id'] == id].index[0], right_col]
                        # src = np.intersect1d(left, right).reshape(1,-1)
                        src = np.intersect1d(fd_end.loc[fd_end[fd_end['mapped_id'] == id].index[0], left_col], fd_end.loc[fd_end[fd_end['mapped_id'] == id].index[0], right_col]).reshape(1,-1)
                        dst = np.full((src.shape), id)
                        edges = np.concatenate((src, dst), axis = 0)
                        if output_type == 'numpy':
                            edges = edges
                        elif output_type == 'torch':
                            edges = torch.from_numpy(edges)
                    elif input_type == 'torch':
                        # left = fd_end.loc[fd_end[fd_end['mapped_id'] == id].index[0], left_col].numpy()
                        # right = fd_end.loc[fd_end[fd_end['mapped_id'] == id].index[0], right_col].numpy()
                        # src = np.intersect1d(left, right).reshape(1,-1).reshape(1,-1)
                        src = np.intersect1d(fd_end.loc[fd_end[fd_end['mapped_id'] == id].index[0], left_col].numpy(), fd_end.loc[fd_end[fd_end['mapped_id'] == id].index[0], right_col].numpy()).reshape(1,-1)
                        dst = np.full((src.shape), id)
                        edges = np.concatenate((src, dst), axis = 0)
                        if output_type == 'torch':
                            edges = torch.from_numpy(edges)
                        elif output_type == 'numpy':
                            edges = edges
                    return edges

                ## remove forbidden value in the functions !!

                print("Creating the edges for the new_nodes:")

                if 'PPI' in list_of_edges:
                    print("    - getting the PPI edges for the left and right nodes composing the pairs.")
                    fd_end.loc[:, ['left_ppi_dst_nodes']] = fd_end['mapped_id'].map(lambda x: get_new_edges_for_fusion_nodes(relation = ('GENE', 'PPI', 'GENE'), id = x, source_node = 'left', remove_forbidden_value = True, return_type = type_to_return))
                    fd_end.loc[:, ['right_ppi_dst_nodes']] = fd_end['mapped_id'].map(lambda x: get_new_edges_for_fusion_nodes(relation = ('GENE', 'PPI', 'GENE'), id = x, source_node = 'right', remove_forbidden_value = True, return_type = type_to_return))
                if 'paralog' in list_of_edges:
                    print("    - getting the paralog edges for the left and right nodes composing the pairs.")
                    fd_end.loc[:, ['left_paralog_dst_nodes']] = fd_end['mapped_id'].map(lambda x: get_new_edges_for_fusion_nodes(relation = ('GENE', 'paralog', 'GENE'), id = x, source_node = 'left', remove_forbidden_value = True, return_type = type_to_return))
                    fd_end.loc[:, ['right_paralog_dst_nodes']] = fd_end['mapped_id'].map(lambda x: get_new_edges_for_fusion_nodes(relation = ('GENE', 'paralog', 'GENE'), id = x, source_node = 'right', remove_forbidden_value = True, return_type = type_to_return))
                if 'Neighbour' in list_of_edges:
                    print("    - getting the Neighbour edges for the left and right nodes composing the pairs.")
                    fd_end.loc[:, ['left_neighbour_dst_nodes']] = fd_end['mapped_id'].map(lambda x: get_new_edges_for_fusion_nodes(relation = ('GENE', 'Neighbour', 'GENE'), id = x, source_node = 'left', remove_forbidden_value = True, return_type = type_to_return))
                    fd_end.loc[:, ['right_neighbour_dst_nodes']] = fd_end['mapped_id'].map(lambda x: get_new_edges_for_fusion_nodes(relation = ('GENE', 'Neighbour', 'GENE'), id = x, source_node = 'right', remove_forbidden_value = True, return_type = type_to_return))
                if 'CoExp' in list_of_edges:
                    print("    - getting the CoExp edges for the left and right nodes composing the pairs.")
                    fd_end.loc[:, ['left_coexp_dst_nodes']] = fd_end['mapped_id'].map(lambda x: get_new_edges_for_fusion_nodes(relation = ('GENE', 'CoExp', 'GENE'), id = x, source_node = 'left', remove_forbidden_value = True, return_type = type_to_return))
                    fd_end.loc[:, ['right_coexp_dst_nodes']] = fd_end['mapped_id'].map(lambda x: get_new_edges_for_fusion_nodes(relation = ('GENE', 'CoExp', 'GENE'), id = x, source_node = 'right', remove_forbidden_value = True, return_type = type_to_return))

                # passer par nodesid['GENE'] pour les heterogeneous edge_type = pas de valeurs interdites
                # no forbidden values there 
                if 'gene_to_BP' in list_of_edges:
                    print("    - getting the gene_to_BP edges for individual genes.")
                    nodes2id['GENE']['gene_to_BP_dst_nodes'] = nodes2id['GENE']['mapped_id'].map(lambda x: get_dst_for_source(relation = ('GENE', 'gene_to_BP', 'BP'), id = x, return_type = type_to_return))
                if 'gene_to_CC' in list_of_edges:
                    print("    - getting the gene_to_CC edges for individual genes.")
                    nodes2id['GENE']['gene_to_CC_dst_nodes'] = nodes2id['GENE']['mapped_id'].map(lambda x: get_dst_for_source(relation = ('GENE', 'gene_to_CC', 'CC'), id = x, return_type = type_to_return))
                if 'gene_to_MF' in list_of_edges:
                    print("    - getting the gene_to_MF edges for individual genes.")
                    nodes2id['GENE']['gene_to_MF_dst_nodes'] = nodes2id['GENE']['mapped_id'].map(lambda x: get_dst_for_source(relation = ('GENE', 'gene_to_MF', 'MF'), id = x, return_type = type_to_return))
                if 'expressed_in' in list_of_edges:
                    print("    - getting the expressed_in edges for individual genes.")
                    nodes2id['GENE']['gene_to_ANATOMY_dst_nodes'] = nodes2id['GENE']['mapped_id'].map(lambda x: get_dst_for_source(relation = ('GENE', 'expressed_in', 'ANATOMY'), id = x, return_type = type_to_return))

                if 'gene_to_BP' in list_of_edges:
                    print("    - getting the sources nodes for the rev_gene_to_BP_edges going to the new_nodes.")
                    fd_end = fd_end.merge(nodes2id['GENE'][['mapped_id', 'gene_to_BP_dst_nodes']], how = 'left', left_on = 'mapped_id_left', right_on = 'mapped_id', suffixes = (None, '_fusion'))
                    fd_end = fd_end.rename(columns = {'gene_to_BP_dst_nodes': 'left_gene_to_BP_dst_nodes'})
                    fd_end = fd_end.drop(['mapped_id_fusion'], axis = 1)
                    fd_end = fd_end.merge(nodes2id['GENE'][['mapped_id', 'gene_to_BP_dst_nodes']], how = 'left', left_on = 'mapped_id_right', right_on = 'mapped_id', suffixes = (None, '_fusion'))
                    fd_end = fd_end.rename(columns = {'gene_to_BP_dst_nodes': 'right_gene_to_BP_dst_nodes'})
                    fd_end = fd_end.drop(['mapped_id_fusion'], axis = 1)
                if 'gene_to_CC' in list_of_edges:
                    print("    - getting the sources nodes for the rev_gene_to_CC_edges going to the new_nodes.")
                    fd_end = fd_end.merge(nodes2id['GENE'][['mapped_id', 'gene_to_CC_dst_nodes']], how = 'left', left_on = 'mapped_id_left', right_on = 'mapped_id', suffixes = (None, '_fusion'))
                    fd_end = fd_end.rename(columns = {'gene_to_CC_dst_nodes': 'left_gene_to_CC_dst_nodes'})
                    fd_end = fd_end.drop(['mapped_id_fusion'], axis = 1)
                    fd_end = fd_end.merge(nodes2id['GENE'][['mapped_id', 'gene_to_CC_dst_nodes']], how = 'left', left_on = 'mapped_id_right', right_on = 'mapped_id', suffixes = (None, '_fusion'))
                    fd_end = fd_end.rename(columns = {'gene_to_CC_dst_nodes': 'right_gene_to_CC_dst_nodes'})
                    fd_end = fd_end.drop(['mapped_id_fusion'], axis = 1)
                if 'gene_to_MF' in list_of_edges:
                    print("    - getting the sources nodes for the rev_gene_to_MF_edges going to the new_nodes.")
                    fd_end = fd_end.merge(nodes2id['GENE'][['mapped_id', 'gene_to_MF_dst_nodes']], how = 'left', left_on = 'mapped_id_left', right_on = 'mapped_id', suffixes = (None, '_fusion'))
                    fd_end = fd_end.rename(columns = {'gene_to_MF_dst_nodes': 'left_gene_to_MF_dst_nodes'})
                    fd_end = fd_end.drop(['mapped_id_fusion'], axis = 1)
                    fd_end = fd_end.merge(nodes2id['GENE'][['mapped_id', 'gene_to_MF_dst_nodes']], how = 'left', left_on = 'mapped_id_right', right_on = 'mapped_id', suffixes = (None, '_fusion'))
                    fd_end = fd_end.rename(columns = {'gene_to_MF_dst_nodes': 'right_gene_to_MF_dst_nodes'})
                    fd_end = fd_end.drop(['mapped_id_fusion'], axis = 1)
                if 'expressed_in' in list_of_edges:
                    print("    - getting the sources nodes for the rev_expressed_in_edges going to the new_nodes.")
                    fd_end = fd_end.merge(nodes2id['GENE'][['mapped_id', 'gene_to_ANATOMY_dst_nodes']], how = 'left', left_on = 'mapped_id_left', right_on = 'mapped_id', suffixes = (None, '_fusion'))
                    fd_end = fd_end.rename(columns = {'gene_to_ANATOMY_dst_nodes': 'left_gene_to_ANATOMY_dst_nodes'})
                    fd_end = fd_end.drop(['mapped_id_fusion'], axis = 1)
                    fd_end = fd_end.merge(nodes2id['GENE'][['mapped_id', 'gene_to_ANATOMY_dst_nodes']], how = 'left', left_on = 'mapped_id_right', right_on = 'mapped_id', suffixes = (None, '_fusion'))
                    fd_end = fd_end.rename(columns = {'gene_to_ANATOMY_dst_nodes': 'right_gene_to_ANATOMY_dst_nodes'})
                    fd_end = fd_end.drop(['mapped_id_fusion'], axis = 1)

                # create edges going from basic nodes to new_nodes
                if 'PPI' in list_of_edges:
                    print("    - getting the PPI edges for the new nodes.")
                    fd_end['ppi_edges']       = fd_end['mapped_id'].map(lambda x: get_tensors_for_fusion_fdend(edge_type = 'ppi', id = x, input_type = type_to_return, output_type = final_type))
                if 'paralog' in list_of_edges:
                    print("    - getting the paralog edges for the new nodes.")
                    fd_end['paralog_edges']   = fd_end['mapped_id'].map(lambda x: get_tensors_for_fusion_fdend(edge_type = 'paralog', id = x, input_type = type_to_return, output_type = final_type))
                if 'Neighbour' in list_of_edges:  
                    print("    - getting the Neighbour edges for the new nodes.")  
                    fd_end['neighbour_edges'] = fd_end['mapped_id'].map(lambda x: get_tensors_for_fusion_fdend(edge_type = 'neighbour', id = x, input_type = type_to_return, output_type = final_type))
                if 'CoExp' in list_of_edges:  
                    print("    - getting the CoExp edges for the new nodes.")  
                    fd_end['coexp_edges'] = fd_end['mapped_id'].map(lambda x: get_tensors_for_fusion_fdend(edge_type = 'coexp', id = x, input_type = type_to_return, output_type = final_type))

                # create the edges going from BP/CC/MF to new_nodes 
                if 'gene_to_BP' in list_of_edges:
                    print("    - getting the rev_genes_to_BP_edges for the new nodes.")
                    fd_end['rev_gene_to_BP_edges']   = fd_end['mapped_id'].map(lambda x: get_tensors_for_fusion_fdend(edge_type = 'gene_to_BP', id = x, input_type = type_to_return, output_type = final_type))
                if 'gene_to_CC' in list_of_edges:
                    print("    - getting the rev_genes_to_CC_edges for the new nodes.")
                    fd_end['rev_gene_to_CC_edges']   = fd_end['mapped_id'].map(lambda x: get_tensors_for_fusion_fdend(edge_type = 'gene_to_CC', id = x, input_type = type_to_return, output_type = final_type))
                if 'gene_to_MF' in list_of_edges:
                    print("    - getting the rev_genes_to_MF_edges for the new nodes.")
                    fd_end['rev_gene_to_MF_edges']   = fd_end['mapped_id'].map(lambda x: get_tensors_for_fusion_fdend(edge_type = 'gene_to_MF', id = x, input_type = type_to_return, output_type = final_type))
                if 'expressed_in' in list_of_edges:
                    print("    - getting the rev_expressed_in_edges for the new nodes.")
                    if ana_edges == 'all':
                        fd_end['rev_gene_to_ANATOMY_edges'] = fd_end['mapped_id'].map(lambda x: get_tensors_for_fusion_fdend(edge_type = 'gene_to_ANATOMY', id = x, input_type = type_to_return, output_type = final_type))
                    elif ana_edges == 'common':
                        fd_end['rev_gene_to_ANATOMY_edges'] = fd_end['mapped_id'].map(lambda x: get_common_tensors_for_fusion_fdend(edge_type = 'gene_to_ANATOMY', id = x, input_type = type_to_return, output_type = final_type))

                if 'PPI' in list_of_edges:
                    old_ppi_edges = data['GENE', 'PPI', 'GENE'].edge_index
                if 'paralog' in list_of_edges:
                    old_paralog_edges = data['GENE', 'paralog', 'GENE'].edge_index
                if 'Neighbour' in list_of_edges:
                    old_neighbour_edges = data['GENE', 'Neighbour', 'GENE'].edge_index
                if 'CoExp' in list_of_edges:
                    old_coexp_edges = data['GENE', 'CoExp', 'GENE'].edge_index  
                if 'gene_to_BP' in list_of_edges:
                    old_rev_gene_to_BP_edges = data['BP', 'rev_gene_to_BP', 'GENE'].edge_index
                if 'gene_to_CC' in list_of_edges:
                    old_rev_gene_to_CC_edges = data['CC', 'rev_gene_to_CC', 'GENE'].edge_index
                if 'gene_to_MF' in list_of_edges:
                    old_rev_gene_to_MF_edges = data['MF', 'rev_gene_to_MF', 'GENE'].edge_index
                if 'expressed_in' in list_of_edges:
                    old_rev_expressed_in_edges = data['ANATOMY', 'rev_expressed_in', 'GENE'].edge_index

                print("Creating the new tensors.")

                if final_type == 'numpy':
                    if 'PPI' in list_of_edges:
                        new_ppi_edges = torch.from_numpy(np.concatenate(fd_end['ppi_edges'].tolist(), axis = 1))
                    if 'paralog' in list_of_edges:
                        new_paralog_edges = torch.from_numpy(np.concatenate(fd_end['paralog_edges'].tolist(), axis = 1))
                    if 'Neighbour' in list_of_edges:    
                        new_neighbour_edges = torch.from_numpy(np.concatenate(fd_end['neighbour_edges'].tolist(), axis = 1))
                    if 'CoExp' in list_of_edges:    
                        new_coexp_edges = torch.from_numpy(np.concatenate(fd_end['coexp_edges'].tolist(), axis = 1))
                    new_rev_gene_to_BP_edges = torch.from_numpy(np.concatenate(fd_end['rev_gene_to_BP_edges'].tolist(), axis = 1))
                    new_rev_gene_to_CC_edges = torch.from_numpy(np.concatenate(fd_end['rev_gene_to_CC_edges'].tolist(), axis = 1))
                    new_rev_gene_to_MF_edges = torch.from_numpy(np.concatenate(fd_end['rev_gene_to_MF_edges'].tolist(), axis = 1))
                    if 'expressed_in' in list_of_edges:
                        new_rev_gene_to_ANATOMY_edges = torch.from_numpy(np.concatenate(fd_end['rev_gene_to_ANATOMY_edges'].tolist(), axis = 1))
                elif final_type == 'torch':
                    if 'PPI' in list_of_edges:
                        new_ppi_edges = torch.cat(fd_end['ppi_edges'].tolist(), dim = 1)
                    if 'paralog' in list_of_edges:
                        new_paralog_edges = torch.cat(fd_end['paralog_edges'].tolist(), dim = 1)
                    if 'Neighbour' in list_of_edges:
                        new_neighbour_edges = torch.cat(fd_end['neighbour_edges'].tolist(), dim = 1)
                    if 'CoExp' in list_of_edges:
                        new_coexp_edges = torch.cat(fd_end['coexp_edges'].tolist(), dim = 1)
                    if 'gene_to_BP' in list_of_edges:
                        new_rev_gene_to_BP_edges = torch.cat(fd_end['rev_gene_to_BP_edges'].tolist(), dim = 1)
                    if 'gene_to_CC' in list_of_edges:
                        new_rev_gene_to_CC_edges = torch.cat(fd_end['rev_gene_to_CC_edges'].tolist(), dim = 1)
                    if 'gene_to_MF' in list_of_edges:
                        new_rev_gene_to_MF_edges = torch.cat(fd_end['rev_gene_to_MF_edges'].tolist(), dim = 1)
                    if 'expressed_in' in list_of_edges:
                        new_rev_gene_to_ANATOMY_edges = torch.cat(fd_end['rev_gene_to_ANATOMY_edges'].tolist(), dim = 1)    

                edge_features = hyperparameters['edge_features']

                if edge_features == True:
                    old_ppi_attr = data['GENE', 'PPI', 'GENE'].edge_attr
                    new_ppi_attr = torch.from_numpy(np.concatenate(fd_end['ppi_attr'].tolist(), axis = 0)) # axis = 0 to concat edge_attr
                    old_neighbour_attr = data['GENE', 'Neighbour', 'GENE'].edge_attr
                    new_neighbour_attr = torch.from_numpy(np.concatenate(fd_end['neighbour_attr'].tolist(), axis = 0)) # axis = 0 to concat edge_attr
                    old_coexp_attr = data['GENE', 'CoExp', 'GENE'].edge_attr
                    new_coexp_attr = torch.from_numpy(np.concatenate(fd_end['coexp_attr'].tolist(), axis = 0)) # axis = 0 to concat edge_attr
                else:
                    pass

                print("Passing the new tensors to the data object.")

                if 'PPI' in list_of_edges:
                    data['GENE', 'PPI', 'GENE'].edge_index = torch.cat((old_ppi_edges, new_ppi_edges), dim = 1)
                if 'paralog' in list_of_edges:
                    data['GENE', 'paralog', 'GENE'].edge_index = torch.cat((old_paralog_edges, new_paralog_edges), dim = 1)
                if 'Neighbour' in list_of_edges:
                    data['GENE', 'Neighbour', 'GENE'].edge_index = torch.cat((old_neighbour_edges, new_neighbour_edges), dim = 1)
                if 'CoExp' in list_of_edges:
                    data['GENE', 'CoExp', 'GENE'].edge_index = torch.cat((old_coexp_edges, new_coexp_edges), dim = 1)
                # edges going from BP/CC/MF/ANATOMY -> genes 
                if 'gene_to_BP' in list_of_edges:
                    data['BP', 'rev_gene_to_BP', 'GENE'].edge_index = torch.cat((old_rev_gene_to_BP_edges, new_rev_gene_to_BP_edges), dim = 1)
                if 'gene_to_CC' in list_of_edges:
                    data['CC', 'rev_gene_to_CC', 'GENE'].edge_index = torch.cat((old_rev_gene_to_CC_edges, new_rev_gene_to_CC_edges), dim = 1)
                if 'gene_to_MF' in list_of_edges:
                    data['MF', 'rev_gene_to_MF', 'GENE'].edge_index = torch.cat((old_rev_gene_to_MF_edges, new_rev_gene_to_MF_edges), dim = 1)
                if 'expressed_in' in list_of_edges:
                    data['ANATOMY', 'rev_expressed_in', 'GENE'].edge_index = torch.cat((old_rev_expressed_in_edges, new_rev_gene_to_ANATOMY_edges), dim = 1)

                if edge_features == True:
                    data['GENE', 'PPI', 'GENE'].edge_attr = torch.cat((old_ppi_attr, new_ppi_attr), dim = 0)
                    data['GENE', 'Neighbour', 'GENE'].edge_attr = torch.cat((old_neighbour_attr, new_neighbour_attr), dim = 0)
                else:
                    pass

                data['GENE'].num_nodes = len(data['GENE'].x)

                for edge in data.edge_types:
                    data[edge].edge_index = U.sort_edge_index(data[edge].edge_index)
                data = data.to('cpu')

                ## outter_loop, outter-loop
                ## 5th for_loop
                for i in [64, 32, 16]:
                ## indent begins here 
                    hc_1 = i
                    hc_2_q = hyperparameters['hc_2_q']
                    hc_3_q = hyperparameters['hc_3_q']
                    hc_2 = int(i/hc_2_q)
                    hc_3 = int(i/hc_3_q)

                    # hyperparameters
                    Heads = hyperparameters['Heads']               
                    conv_number = hyperparameters['conv_number']    
                    sage_aggr = hyperparameters['sage_aggr'] 
                    sage_norm = hyperparameters['sage_norm']
                    sage_project = hyperparameters['sage_project']
                    heteroconv_aggr = hyperparameters['heteroconv_aggr'] 
                    heteroconv_aggr_1 = hyperparameters['heteroconv_aggr_1'] 
                    heteroconv_aggr_2 = hyperparameters['heteroconv_aggr_2'] 
                    heteroconv_aggr_3 = hyperparameters['heteroconv_aggr_3']
                    size_of_batch = hyperparameters['size_of_batch'] 
                    neighbors = hyperparameters['neighbors'] 
                    disjoint_loader = hyperparameters['disjoint_loader'] # True
                    epochs_to_train = hyperparameters['epochs_to_train'] 
                    min_epochs_to_train = hyperparameters['min_epochs_to_train']
                    patience = hyperparameters['patience']
                    min_delta = hyperparameters['min_delta']
                    auto_loop = hyperparameters['auto_loop']
                    zero_out_batch_features = hyperparameters['zero_out_batch_features']

                    print("XXX" * 40)
                    print("This is Yeasts: pairs_to_pairs setting, target = SL vs SNL for pairs.")
                    print(f"Now hidden_channels_1 = {hc_1} and hidden_channels_2 = {hc_2} and batch = {size_of_batch}.")

                    if global_loader_neighbors == 'all':
                        outter_model_param = base_model_param + f"_{hc_1}_{int(hc_2)}_batch_{size_of_batch}_neighbors_{neighbors}_global_all"
                    else:
                        outter_model_param = base_model_param + f"_{hc_1}_{int(hc_2)}_batch_{size_of_batch}_neighbors_{neighbors}_global_sample"

                    # https://colab.research.google.com/drive/1v-46ZzliH3cSjY9Z6pQu7M8MJ4ONVdrq#scrollTo=U2M4c-3GeqX8
                    # Already send node features/labels to GPU for faster access during sampling:
                    # device = 'cpu'
                    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
                    data = data.to(device, 'x', 'y')

                    df = pd.DataFrame(
                        {'AUROC_train': float(), 'AUROC_val': float(), 'AUROC_test': float(), 
                         'AUPR_train': float(), 'AUPR_val': float(), 'AUPR_test': float(),
                         'best_MCC_train': float(), 'best_MCC_train_threshold': float(), 'best_MCC_train_cm': '',
                         'best_MCC_val': float(), 'best_MCC_val_threshold': float(), 'best_MCC_val_cm': '',
                         'best_MCC_test': float(), 'best_MCC_test_threshold': float(), 'best_MCC_test_cm': '',
                         'best_F1_train': float(), 'best_F1_train_threshold': float(), 'best_F1_train_cm': '',
                         'best_F1_val': float(), 'best_F1_val_threshold': float(), 'best_F1_val_cm': '',
                         'best_F1_test': float(), 'best_F1_test_threshold': float(), 'best_F1_test_cm': '',
                         'best_F0.01_train': float(), 'best_F0.01_train_threshold': float(), 'best_F0.01_train_cm': '',
                         'best_F0.01_val': float(), 'best_F0.01_val_threshold': float(), 'best_F0.01_val_cm': '',
                         'best_F0.01_test': float(), 'best_F0.01_test_threshold': float(), 'best_F0.01_test_cm': '',
                         'mask': str(), 'epoch': str(), 'dim': int()
                         }, index = [])

                    numeric_col = [
                        'AUROC_train', 'AUROC_val', 'AUROC_test', 
                        'AUPR_train', 'AUPR_val', 'AUPR_test', 
                        'best_MCC_train', 'best_MCC_train_threshold', 
                        'best_MCC_val', 'best_MCC_val_threshold', 
                        'best_MCC_test', 'best_MCC_test_threshold',
                        'best_F1_train', 'best_F1_train_threshold', 
                        'best_F1_val', 'best_F1_val_threshold', 
                        'best_F1_test', 'best_F1_test_threshold',
                        'best_F0.01_train', 'best_F0.01_train_threshold', 
                        'best_F0.01_val', 'best_F0.01_val_threshold', 
                        'best_F0.01_test', 'best_F0.01_test_threshold'
                        ]    

                    pred_df = fd[fd_basic_columns]

                    full_AUROC_df = pd.DataFrame(
                        {'epoch': int(), 'AUROC_train': float(), 'train_loss': float(), 'AUROC_val': float(), 'val_loss': float(),
                    'AUROC_test': float(), 'test_loss': float(), 'mask': int(), 'epoch': int(), 'best_auroc_test_epoch': int() #
                    }, index = [])

                    col_names_of_pred_df = []
                    best_col_names_of_pred_df = []

                    def create_masks_essential_pairs(data: HeteroData) -> HeteroData:
                        # get the indexes
                        train_idx = fd[fd['mask'] == 'train'].index.values
                        val_idx = fd[fd['mask'] == 'val'].index.values
                        test_idx = fd[fd['mask'] == 'test'].index.values
                        # initialize the masks
                        train_mask = torch.zeros(data['GENE'].x.shape[0])
                        val_mask = torch.zeros(data['GENE'].x.shape[0])
                        test_mask = torch.zeros(data['GENE'].x.shape[0])
                        # create 1 at the specified indexes
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

                    print("xxx"*20)
                    print(fd['mask'].value_counts())
                    print("xxx"*20)
                    data = create_masks_essential_pairs(data)

                    ind_size = fd[~fd['GENE_id'].str.contains('_')].shape[0]

                    print(data)

                    ## create the model 

                    if activation == 'tanh':
                        my_model = HeteroConv_2SAGE_tanh(hidden_channels_1 = hc_1, hidden_channels_2 = hc_2, out_channels = 1, sage_aggr = sage_aggr, sage_norm = sage_norm, sage_project = False, hetero_aggr_1 = heteroconv_aggr_1, hetero_aggr_2 = heteroconv_aggr_2)  
                    elif activation == 'relu':
                        my_model = HeteroConv_2SAGE_relu(hidden_channels_1 = hc_1, hidden_channels_2 = hc_2, out_channels = 1, sage_aggr = sage_aggr, sage_norm = sage_norm, sage_project = False, hetero_aggr_1 = heteroconv_aggr_1, hetero_aggr_2 = heteroconv_aggr_2)  
                    my_model = my_model.to(device)
                    my_optimizer = torch.optim.Adam(my_model.parameters(), lr=0.01, weight_decay=5e-4)
                    criterion = torch.nn.BCELoss()

                    # let's plant some seeds to collect reproducibility
                    random.seed(31415)
                    np.random.seed(31415)
                    torch.manual_seed(31415)
                    torch.cuda.manual_seed(31415)
                    torch.cuda.manual_seed_all(31415)
                    torch.backends.cudnn.deterministic = True

                    for edge in data.edge_index_dict.keys():
                        data[edge].edge_index = data[edge].edge_index.contiguous()
                        data[edge].edge_index = U.sort_edge_index(data[edge].edge_index)

                    ## create loaders

                    size_of_batch = hyperparameters['size_of_batch']

                    if global_loader_neighbors == 'all':
                        print('The global_loader will take all of the neighbors.')
                        global_loader = NeighborLoader(data, 
                                            num_neighbors = {key: [-1] * conv_number for key in data.edge_types},
                                            batch_size = size_of_batch, #
                                            input_nodes=('GENE', None),
                                            disjoint = False,
                                            num_workers = 0,
                                            replace = False, 
                                            filter_per_worker = None 
                                            )
                    
                    else: # global_loader_neighbors == 'Neighbors'
                        print('The global_loader is sampling Neighbors. It will not take all of them.')
                        global_loader = NeighborLoader(data, 
                                            num_neighbors = {key: [neighbors] * conv_number for key in data.edge_types},
                                            batch_size = size_of_batch,
                                            input_nodes=('GENE', None),
                                            disjoint = True, 
                                            num_workers = 0,
                                            replace = False, 
                                            filter_per_worker = None 
                                            )

                    val_loader = NeighborLoader(data, 
                                            num_neighbors = {key: [neighbors] * conv_number for key in data.edge_types},
                                            batch_size = size_of_batch,
                                            input_nodes=('GENE', data['GENE'].val_mask.to("cpu")),
                                            disjoint = disjoint_loader,
                                            num_workers = 6,
                                            shuffle = False,
                                            replace = False, 
                                            filter_per_worker = None
                                            )

                    test_loader = NeighborLoader(data, 
                                            num_neighbors = {key: [neighbors] * conv_number for key in data.edge_types},
                                            batch_size = int(size_of_batch/2), 
                                            input_nodes=('GENE', data['GENE'].test_mask.to("cpu")),
                                            disjoint = disjoint_loader, # 
                                            num_workers = 0, #
                                            shuffle = False,
                                            replace = False, 
                                            filter_per_worker = None
                                            )

                    train_loader = NeighborLoader(data, 
                        num_neighbors = {key: [neighbors] * conv_number for key in data.edge_types},
                        batch_size = size_of_batch,
                        input_nodes=('GENE', data['GENE'].train_mask.to("cpu")),
                        disjoint = disjoint_loader,
                        num_workers = 6,
                        filter_per_worker = None 
                        )


                    ## initialize

                    print("Initializing model.")
                    @torch.no_grad()
                    def init_params():
                        # Initialize lazy parameters via forwarding a single batch to the model:
                        batch = next(iter(train_loader))
                        batch = batch.to(device, 'edge_index')
                        # my_model(batch.x_dict, batch.edge_index_dict)
                        out = my_model(batch.x_dict, batch.edge_index_dict)[:batch['GENE'].batch_size]
                        print(out.t())
                        print(out.shape)

                    init_params()
                    print("Initialization complete.")

                    data = data.to(device, 'x', 'y')

                    ## create the train, val and test functions 

                    def train(model, optimizer = my_optimizer, use_edge_attr:bool = False, zero_out_batch: bool = True):
                        model.train()
                        optimizer.zero_grad()
                        # Perform a single forward pass.
                        all_train_pred = np.array([])
                        all_train_target = np.array([])
                        train_loss_total = total_examples = 0
                        for batch in tqdm(train_loader):
                            optimizer.zero_grad()
                            batch = batch.to(device)
                            for edge in batch.edge_index_dict.keys():
                                batch[edge].edge_index = batch[edge].edge_index.to(device)
                            if use_edge_attr == True:
                                for edge in batch.edge_attr_dict.keys():
                                    batch[edge].edge_attr = batch[edge].edge_attr.to(device)
                            batch_size = batch['GENE'].batch_size
                            assert int(sum(batch['GENE'].input_id == batch['GENE'].n_id[:batch_size])) == batch_size
                            if zero_out_batch == True:
                                batch['GENE'].x[:batch['GENE'].batch_size] = batch['GENE'].x[:batch['GENE'].batch_size].zero_()
                            else:
                                pass

                            if use_edge_attr == False:
                                out = model(batch.x_dict, batch.edge_index_dict)
                            elif use_edge_attr == True: 
                                out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)

                            train_loss = criterion(out[:batch_size], batch['GENE'].y[:batch_size].view(-1,1).float()) 
                            train_loss.backward()
                            optimizer.step()

                            total_examples += batch_size
                            train_loss_total += float(train_loss) * batch_size
                            all_train_pred = np.concatenate([all_train_pred, out[:batch_size].view(-1,).cpu().detach().numpy()], axis = 0)
                            all_train_target = np.concatenate([all_train_target, batch['GENE'].y[:batch_size].view(-1,).int().cpu().detach().numpy()], axis = 0)

                        auroc_train = roc_auc_score(all_train_target, all_train_pred)
                        
                        return train_loss_total / total_examples, auroc_train

                    def val(model, use_edge_attr:bool = False, zero_out_batch: bool = True):
                        model.eval()
                        # Perform a single forward pass.
                        all_val_pred = np.array([])
                        all_val_target = np.array([])
                        val_loss_total = total_examples = 0
                        for batch in tqdm(val_loader):
                            batch = batch.to(device)
                            for edge in batch.edge_index_dict.keys():
                                batch[edge].edge_index = batch[edge].edge_index.to(device)

                            if use_edge_attr == True:
                                for edge in batch.edge_attr_dict.keys():
                                    batch[edge].edge_attr = batch[edge].edge_attr.to(device)
                            batch_size = batch['GENE'].batch_size
                            assert int(sum(batch['GENE'].input_id == batch['GENE'].n_id[:batch_size])) == batch_size
                            if zero_out_batch == True:
                                batch['GENE'].x[:batch['GENE'].batch_size] = batch['GENE'].x[:batch['GENE'].batch_size].zero_()
                            else:
                                pass

                            if use_edge_attr == False:
                                out = model(batch.x_dict, batch.edge_index_dict)
                            elif use_edge_attr == True: 
                                out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)

                            val_loss = criterion(out[:batch_size], batch['GENE'].y[:batch_size].view(-1,1).float()) 
                            total_examples += batch_size
                            val_loss_total += float(val_loss) * batch_size
                            
                            all_val_pred = np.concatenate([all_val_pred, out[:batch_size].view(-1,).cpu().detach().numpy()], axis = 0)
                            all_val_target = np.concatenate([all_val_target, batch['GENE'].y[:batch_size].view(-1,).int().cpu().detach().numpy()], axis = 0)
                            
                        auroc_val = roc_auc_score(all_val_target, all_val_pred)
                        
                        return val_loss_total / total_examples, auroc_val

                    def test(model, use_edge_attr:bool = False, zero_out_batch: bool = True):
                        model.eval()
                        # Perform a single forward pass.
                        all_test_pred = np.array([])
                        all_test_target = np.array([])
                        test_loss_total = total_examples = 0
                        for batch in tqdm(test_loader):
                            batch = batch.to(device)
                            for edge in batch.edge_index_dict.keys():
                                batch[edge].edge_index = batch[edge].edge_index.to(device)
                            if use_edge_attr == True:
                                for edge in batch.edge_attr_dict.keys():
                                    batch[edge].edge_attr = batch[edge].edge_attr.to(device)
                            batch_size = batch['GENE'].batch_size
                            assert int(sum(batch['GENE'].input_id == batch['GENE'].n_id[:batch_size])) == batch_size
                            if zero_out_batch == True:
                                batch['GENE'].x[:batch['GENE'].batch_size] = batch['GENE'].x[:batch['GENE'].batch_size].zero_()
                            else:
                                pass
                            
                            if use_edge_attr == False:
                                out = model(batch.x_dict, batch.edge_index_dict)
                            elif use_edge_attr == True: 
                                out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
                                out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch.normalization_dict)
                            
                            test_loss = criterion(out[:batch_size], batch['GENE'].y[:batch_size].view(-1,1).float()) 
                            total_examples += batch_size
                            test_loss_total += float(test_loss) * batch_size
                            all_test_pred = np.concatenate([all_test_pred, out[:batch_size].view(-1,).cpu().detach().numpy()], axis = 0)
                            all_test_target = np.concatenate([all_test_target, batch['GENE'].y[:batch_size].view(-1,).int().cpu().detach().numpy()], axis = 0)
                            
                        auroc_test = roc_auc_score(all_test_target, all_test_pred)
                        
                        return test_loss_total / total_examples, auroc_test

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
                    # early_stopping = EarlyStopping(patience = patience, min_delta = min_delta, criterion_type = 'increasing')

                    def do_complete_training_loader(model_to_train, optimizer_to_use, criterion_to_stop, 
                                                    val_set: bool = True, test_set: bool = True, remove_self_features: bool = True, 
                                                    max_number_of_epochs: int = 100, min_epochs_to_train: int = 1):
                        best_epoch = 0
                        AUROC_df = pd.DataFrame(
                            {'epoch': int(), 'AUROC_train': float(), 'train_loss': float(), 
                            'AUROC_val': float(), 'val_loss': float(), 'AUROC_test': float(), 'test_loss': float()}, index = [])
                        for epoch in range(1, max_number_of_epochs + 1):
                            train_loss, auroc_train = train(model_to_train, optimizer = optimizer_to_use, use_edge_attr = edge_features, zero_out_batch = remove_self_features)
                            if val_set:
                                val_loss, auroc_val = val(model_to_train, use_edge_attr = edge_features, zero_out_batch = remove_self_features)
                            else:
                                auroc_val = 'not_concerned'
                                val_loss = 'not_concerned'
                            if test_set:
                                test_loss, auroc_test = test(model_to_train, use_edge_attr = edge_features, zero_out_batch = remove_self_features)
                            else:
                                auroc_test = 'not_concerned'
                                test_loss = 'not_concerned'
                            tmp_df = pd.DataFrame(
                                {'epoch':epoch, 'AUROC_train': auroc_train, 'train_loss': train_loss, 
                                'AUROC_val': auroc_val, 'val_loss': val_loss, 'AUROC_test':auroc_test, 'test_loss': test_loss}, index = [f'sampler_epoch_{epoch}']) #[f'mask_{j}_epoch_{epoch}'])
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

                    # ## train 

                    print("Begin training.")
                    my_model, best_epoch, AUROC_df = do_complete_training_loader(my_model, optimizer_to_use = my_optimizer, criterion_to_stop = 'train_loss', 
                                                                                val_set = False, test_set = False, remove_self_features = zero_out_batch_features,
                                                                                max_number_of_epochs = epochs_to_train, min_epochs_to_train = 0)             

                    ## plot and save figures and dataframes

                    print("Training complete.")

                    AUROC_df.loc[:,'best_epoch'] = best_epoch
                    full_AUROC_df = pd.concat([full_AUROC_df, AUROC_df], axis = 0)

                    print("Saving the model.")
                    model_name = model_path + outter_model_param + f'_epoch_final_{best_epoch}.pyg'
                    torch.save(my_model.state_dict(), model_name)

                    print("Computing the scores and drawing the figure.")
                    fig, df_tmp, fusion_tmp = plot_model_df(my_model, global_loader = global_loader, zero_out_batch = zero_out_batch_features, 
                                                            val_set = True, test_set = True, setting = 'pairs_to_pairs')
                    print("Saving the figure.")
                    fig_name = result_path + outter_model_param + f'_epoch_final_{best_epoch}.png'
                    fig.savefig(fig_name)

                    df_tmp.index = [f"train_epoch_{best_epoch}"]
                    df_tmp.loc[:, 'epoch'] = f'best_{best_epoch}'
                    df_tmp.loc[:,'dim'] = i
                    df = pd.concat([df, df_tmp], axis = 0)

                    current_pred = fusion_tmp[['pred']]
                    current_pred_new_col_name = f"best_epoch_{best_epoch}"
                    current_pred.columns = [current_pred_new_col_name]
                    pred_df = pd.concat([pred_df, current_pred], axis = 1)

                    print("Saving the dataframes.")
                    full_auroc_df_name = result_path + outter_model_param + f'_max_epochs_{epochs_to_train}_auroc_df.tsv'
                    full_AUROC_df.to_csv(path_or_buf = full_auroc_df_name, header = True, index = False, sep = "\t")

                    print("Saving the big dataframes.")
                    pred_df_name = result_path + outter_model_param + f'_max_epochs_{epochs_to_train}_pred.tsv'

                    pred_df.to_csv(path_or_buf = pred_df_name, header = True, index = False, sep = "\t")
                    print('Saving the summary dataframes.')
                    df_name = result_path + outter_model_param + f'_max_epochs_{epochs_to_train}_summary.tsv'

                    df.to_csv(path_or_buf = df_name, header = True, index = True, sep = "\t", index_label = 'parameters')

                ## indent ends here
                ## explanation 

                print("Writing the explanation file.")
                file_name = result_path + outter_model_param + f"_explanation.txt"
                f = open(file_name, "w")
                f.write(f"This pairs_to_pairs model is trained on Yeasts, with DMF score < {DMF_t} for SL pairs and random sampling for SNL pairs:")
                f. write(f"The features used come from this dataframe: 'yeast_features_1000_20240717.tsv'\n")
                f.write(f"The features used are:")
                f.write(str(yeast_features_to_use))
                f.write(f"For the models: {base_model_param}, for a maximum of {epochs_to_train}, the edges that were used are:\n")
                for edge in data.edge_index_dict.keys():
                    f.write(f"{edge} \n")
                f.write("\nThe precise data used are:\n")
                f.write(str(data))
                f.write("\nThe complete hyperparameters are:\n")
                f.write(str(hyperparameters))
                f.close()