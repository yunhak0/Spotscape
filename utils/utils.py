import os
import random
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.utils import homophily
from sklearn.base import TransformerMixin, BaseEstimator
from anndata import AnnData


def str2bool(s):
    if s not in {'False', 'True', 'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return (s == 'True') or (s == 'true')


def list_of_ints(arg):
    return list(map(int, arg.split(',')))


def set_seed(seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def config2string(args):

    def enumerateConfig(args):
        args_names = []
        args_vals = []
        for arg in vars(args):
            args_names.append(arg)
            args_vals.append(getattr(args, arg))

        return args_names, args_vals

    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        if val == 'integration':
            continue
        if name not in ['project', 'wb_name', 'trials', 'batch_size', 'n_hvg', 'decay', 'clustering_method',
                        'device', 'enc_type', 'ee', 'wandb', 'result_dir', 'data_dir', 'estimate_n_clusters',
                        'dropout', 'size_factors', 'logtrans_input', 'layers', 'neighs',
                        'graph_type', 'graphenc', 'de1', 'de2', 'df1', 'df2', 'lam_bt', 'eval_only',
                        'normalize_coord',
                        'proto_granularity']:
            if isinstance(val, list):
                val = str(val).replace(" ", "").replace(",", "_")
                st_ = '{}_{}_'.format(name, val)
            else:
                st_ = '{}_{}_'.format(name, val)
            st += st_

    return st[:-1]


def train_reports(args, main):
    results_str = ''

    model_list = []
    n_clusters_list = []
    each_clusters_list = []

    each_silhouette_list = []
    each_ari_list = []
    each_nmi_list = []
    each_ca_list = []
    each_maf1_list = []
    each_mif1_list = []

    total_silhouette_list = []
    total_ari_list = []
    total_nmi_list = []
    total_ca_list = []
    total_maf1_list = []
    total_mif1_list = []

    for trial in range(args.trials):
        each_str = ''
        set_seed(trial)
        model, res = main(args, trial)
        model_list.append(model)
        n_clusters_list.append(res['n_clusters'])
        each_clusters_list.append(res['each_n_clusters'])

        each_str += "| slices: " + ', '.join(res['batch_name'])
        each_str += " | n_clusters: " + ', '.join([str(n) for n in res['each_n_clusters']])
        each_str += " | silhouette: " + ', '.join([f'{s:.4f}' for s in res['silhouette'][:-1]])
        each_str += " | ari: " + ', '.join([f'{a:.4f}' for a in res['ari'][:-1]])
        each_str += " | nmi: " + ', '.join([f'{n:.4f}' for n in res['nmi'][:-1]])
        each_str += " | ca: " + ', '.join([f'{c:.4f}' for c in res['ca'][:-1]])
        each_str += " | ma-f1: " + ', '.join([f'{m:.4f}' for m in res['maf1'][:-1]])
        each_str += " | mi-f1: " + ', '.join([f'{m:.4f}' for m in res['mif1'][:-1]])
        each_str += " |\n"

        each_silhouette_list.append(res['silhouette'][:-1])
        each_ari_list.append(res['ari'][:-1])
        each_nmi_list.append(res['nmi'][:-1])
        each_ca_list.append(res['ca'][:-1])
        each_maf1_list.append(res['maf1'][:-1])
        each_mif1_list.append(res['mif1'][:-1])

        total_silhouette_list.append(res['silhouette'][-1])
        total_ari_list.append(res['ari'][-1])
        total_nmi_list.append(res['nmi'][-1])
        total_ca_list.append(res['ca'][-1])
        total_maf1_list.append(res['maf1'][-1])
        total_mif1_list.append(res['mif1'][-1])

        if trial == 0:
            results_str += f"{args.config_str}\n"
        results_str += f"Trial: {trial}\n"
        results_str += each_str
        results_str += f"** | n_clusters : {res['n_clusters']}"
        results_str += f" | silhouette : {res['silhouette'][-1]:.4f}"
        results_str += f" | ari : {res['ari'][-1]:.4F} | nmi : {res['nmi'][-1]:.4F} | ca : {res['ca'][-1]:.4f} | ma-f1 : {res['maf1'][-1]:.4f} | mi-f1 : {res['mif1'][-1]:.4f} | **\n"

    n_clusters_mean = torch.FloatTensor(n_clusters_list).mean().item()
    each_clusters_mean = torch.FloatTensor(each_clusters_list).mean(0).tolist()

    each_silhouette_mean = torch.tensor(each_silhouette_list).mean(0).tolist()
    each_ari_mean = torch.tensor(each_ari_list).mean(0).tolist()
    each_nmi_mean = torch.tensor(each_nmi_list).mean(0).tolist()
    each_ca_mean = torch.tensor(each_ca_list).mean(0).tolist()
    each_maf1_mean = torch.tensor(each_maf1_list).mean(0).tolist()
    each_mif1_mean = torch.tensor(each_mif1_list).mean(0).tolist()

    each_silhouette_std = torch.tensor(each_silhouette_list).std(0).tolist()
    each_ari_std = torch.tensor(each_ari_list).std(0).tolist()
    each_nmi_std = torch.tensor(each_nmi_list).std(0).tolist()
    each_ca_std = torch.tensor(each_ca_list).std(0).tolist()
    each_maf1_std = torch.tensor(each_maf1_list).std(0).tolist()
    each_mif1_std = torch.tensor(each_mif1_list).std(0).tolist()

    total_silhouette_mean = torch.tensor(total_silhouette_list).mean().item()
    total_ari_mean = torch.tensor(total_ari_list).mean().item()
    total_nmi_mean = torch.tensor(total_nmi_list).mean().item()
    total_ca_mean = torch.tensor(total_ca_list).mean().item()
    total_maf1_mean = torch.tensor(total_maf1_list).mean().item()
    total_mif1_mean = torch.tensor(total_mif1_list).mean().item() 

    total_silhouette_std = torch.tensor(total_silhouette_list).std().item()
    total_ari_std = torch.tensor(total_ari_list).std().item()
    total_nmi_std = torch.tensor(total_nmi_list).std().item()
    total_ca_std = torch.tensor(total_ca_list).std().item()
    total_maf1_std = torch.tensor(total_maf1_list).std().item()
    total_mif1_std = torch.tensor(total_mif1_list).std().item()

    each_str = ''
    for i in range(len(each_clusters_mean)):
        each_str += f"| slices: {res['batch_name'][i]} | n_clusters: {each_clusters_mean[i]}"
        each_str += f" | silhouette: {each_silhouette_mean[i]:.4f}({each_silhouette_std[i]:.4f})"
        each_str += f" | ari: {each_ari_mean[i]:.4f}({each_ari_std[i]:.4f})"
        each_str += f" | nmi: {each_nmi_mean[i]:.4f}({each_nmi_std[i]:.4f})"
        each_str += f" | ca: {each_ca_mean[i]:.4f}({each_ca_std[i]:.4f})"
        each_str += f" | ma-f1: {each_maf1_mean[i]:.4f}({each_maf1_std[i]:.4f})"
        each_str += f" | mi-f1: {each_mif1_mean[i]:.4f}({each_mif1_std[i]:.4f}) |\n"

    print(f'======================================= Summarize =======================================')
    print(each_str)
    print("** | n_clusters : {} | silhouette : {:.4f}({:.4f}) | ari : {:.4f}({:.4f}) | nmi : {:.4F}({:.4f}) | ca : {:.4F}({:.4f}) | ma-f1 : {:.4f}({:.4f}) | mi-f1 : {:.4f}({:.4f}) | **\n".format(
                n_clusters_mean, total_silhouette_mean, total_silhouette_std,
                total_ari_mean, total_ari_std, total_nmi_mean, total_nmi_std, total_ca_mean, total_ca_std,
                total_maf1_mean, total_maf1_std, total_mif1_mean, total_mif1_std))
    print(f'=========================================================================================')

    results_str += each_str
    results_str += f"** | n_clusters : {n_clusters_mean}"
    results_str += f" | silhouette : {total_silhouette_mean:.4f}({total_silhouette_std:.4f})"
    results_str += f" | ari : {total_ari_mean:.4f}({total_ari_std:.4f})"
    results_str += f" | nmi : {total_nmi_mean:.4F}({total_nmi_std:.4f})"
    results_str += f" | ca : {total_ca_mean:.4F}({total_ca_std:.4f})"
    results_str += f" | ma-f1 : {total_maf1_mean:.4f}({total_maf1_std:.4f}) | mi-f1 : {total_mif1_mean:.4f}({total_mif1_std:.4f}) | **\n"
    results_str += "=================================================================================================================================================\n"

    with open(args.result_path, 'a+') as f:
        f.write(results_str)


def print_homophily_ratio(graph, celltype):
    print(f"Number of Edges : {graph.edge_index.shape[1]}")
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(celltype)
    target = torch.tensor(target)
    homophiliy_ratio = homophily(graph.edge_index, target)
    print(f" Homophil Ratio : {homophiliy_ratio:.4f}")


# def mclust_R(latent, n_cluster, model_names='EEE', random_seed=42):
#     """
#     Clustering using the mclust algorithm.
#     The parameters are the same as those in the R package mclust.
#     """
#     import rpy2.robjects as robjects
#     robjects.r.library("mclust")

#     import rpy2.robjects.numpy2ri
#     rpy2.robjects.numpy2ri.activate()
#     r_random_seed = robjects.r['set.seed']
#     r_random_seed(random_seed)
#     rmclust = robjects.r['Mclust']

#     res = rmclust(latent, n_cluster, model_names)
#     mclust_res = np.array(res[-2])
#     mclust_res = mclust_res.astype('int')

#     return mclust_res


def normalize_coordinates(coord):
    std_scaler = StandardScaler()
    norm_coord = std_scaler.fit_transform(coord)

    return norm_coord

    

# From SLAT
def norm_to_raw(
    adata: AnnData, 
    library_size = 'total_counts',
    check_size = 100
) -> AnnData:
    r"""
    Convert normalized adata.X to raw counts
    
    Parameters
    ----------
    adata
        adata to be convert
    library_size
        raw library size of every cells, can be a key of `adata.obs` or a array
    check_size
        check the head `[0:check_size]` row and column to judge if adata normed
    
    Note
    ----------
    Adata must follow scanpy official norm step 
    """
    check_chunk = adata.X[0:check_size,0:check_size].todense()
    assert not all(isinstance(x, int) for x in check_chunk)
    
    from scipy import sparse
    scale_size = np.array(adata.X.expm1().sum(axis=1).round()).flatten()
    if isinstance(library_size, str):
        scale_factor = np.array(adata.obs[library_size])/scale_size
    elif isinstance(library_size, np.ndarray):
        scale_factor = library_size/scale_size
    else:
        try:
            scale_factor = np.array(library_size)/scale_size
        except:
            raise ValueError('Invalid `library_size`')
    scale_factor.resize((scale_factor.shape[0],1))
    raw_count = sparse.csr_matrix.multiply(sparse.csr_matrix(adata.X).expm1(), sparse.csr_matrix(scale_factor))
    raw_count = sparse.csr_matrix(np.round(raw_count))
    adata.X = raw_count
    # adata.layers['counts'] = raw_count
    return adata