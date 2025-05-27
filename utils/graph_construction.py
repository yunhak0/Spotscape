import torch
from torch_geometric.data import HeteroData
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
from utils.utils import print_homophily_ratio

def construct_sp_graph(slices, adata_concat, gt_colname, data_name):
    num_nodes = adata_concat.shape[0]
    node_types = adata_concat.obs['batch_name'].values.astype(int)
    main_graph = HeteroData()

    for i in range(len(slices)):
        # Main Graph
        main_graph[f'{i}'].num_nodes = slices[i].n_obs

        try:
            X = slices[i].X.todense()
        except:
            X = slices[i].X
        main_graph[f'{i}'].x = torch.Tensor(X)

        if data_name.lower() == '10x_dlpfc':
            Cal_Spatial_Net(slices[i], rad_cutoff=150)
        else:
            raise ValueError(f'The dataset {data_name} is not support. Please specify the way of graph construction!')
        
        adj = slices[i].uns['adj'].tocoo()
        edge_attr = torch.tensor(adj.data)
        edge_index = torch.tensor(np.vstack((adj.row, adj.col)))
        main_graph[f'{i}', 'to', f'{i}'].edge_index = edge_index
        main_graph[f'{i}', 'to', f'{i}'].edge_attr = edge_attr.to(torch.float32)

    # Main Graph
    main_graph['node_indices'] = torch.arange(num_nodes)
    main_graph['node_type'] = torch.tensor(node_types)

    graph_data = main_graph.to_homogeneous()
    if 'norm_spatial' in adata_concat.obsm:
        graph_data.coord = torch.Tensor(adata_concat.obsm['norm_spatial'])
    else:
        graph_data.coord = torch.Tensor(adata_concat.obsm['spatial'])

    print_homophily_ratio(graph_data, adata_concat.obs[gt_colname])

    return graph_data


def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None,
                    max_neigh=50, model='radius', selfloop=True,
                    verbose=True):
    """
    From STAGATE

    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.

    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """
    assert (model.lower() in ['radius', 'knn'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    nbrs = NearestNeighbors(n_neighbors=max_neigh + 1, algorithm='ball_tree').fit(coor)
    distances, indices = nbrs.kneighbors(coor)

    if model.lower() == 'knn':
        indices = indices[:, 1:k_cutoff + 1]
        distances = distances[:, 1:k_cutoff + 1]
    if model.lower() == 'radius':
        indices = indices[:, 1:]
        distances = distances[:, 1:]

    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    if model.lower() == 'radius':
        Spatial_Net = KNN_df.loc[KNN_df['Distance'] < rad_cutoff,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(adata.obs.index), ))
    Spatial_Net.loc[:, 'Cell1'] = Spatial_Net.loc[:, 'Cell1'].map(id_cell_trans)
    Spatial_Net.loc[:, 'Cell2'] = Spatial_Net.loc[:, 'Cell2'].map(id_cell_trans)

    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))
    adata._uns['Spatial_Net'] = Spatial_Net # Warning

    #########
    X = pd.DataFrame(adata.X.toarray()[:, ], index=adata.obs.index, columns=adata.var.index)
    cells = np.array(X.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
        
    Spatial_Net = adata.uns['Spatial_Net']
    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    
    if selfloop:
        G = G + sp.eye(G.shape[0])  # self-loop
    adata._uns['adj'] = G



