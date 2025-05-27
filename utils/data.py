import os
import scanpy as sc
import pandas as pd
import anndata
from utils.utils import normalize_coordinates


def load_slices(data_dir, data_name, patient_idx=0, slice_idx=-1):
    if data_name.lower() == '10x_dlpfc':
        slices = load_10x_dlpfc(data_dir, patient_idx, slice_idx)
    else:
        raise ValueError(f'Cannot load the dataset: {data_name}')

    return slices


def preprocess_slices(slices, n_hvg=5000, normalize=True, log1p=True, scale=False, pca=False,
                      normalize_coord=True, n_comps=200):
    for i in range(len(slices)):
        adata = slices[i].copy()

        if n_hvg > 0:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat_v3")
            adata = adata[:, adata.var['highly_variable']]

        if normalize:
            sc.pp.normalize_total(adata, target_sum=1e4)

        if log1p:
            sc.pp.log1p(adata)

        if scale:
            sc.pp.scale(adata)

        if pca:
            sc.pp.pca(adata, n_comps=n_comps)

        if normalize_coord:
            adata.obsm['norm_spatial'] = normalize_coordinates(adata.obsm['spatial'])

        slices[i] = adata

    return slices


def concat_slices(slices, common_genes=True):
    adata_concat = anndata.concat(slices, label='slice_name', join='inner')

    adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
    print('adata_concat.shape: ', adata_concat.shape)
    total_common_genes = adata_concat.var.index
    print('Filtered all slices for common genes. There are ' + str(len(total_common_genes)) + ' common genes.')

    if common_genes:
        for i in range(len(slices)):
            common_genes = adata_concat.var.index[adata_concat.var.index.isin(slices[i].var.index)]
            slices[i] = slices[i][:, common_genes]

    return slices, adata_concat, total_common_genes



def load_10x_dlpfc(data_dir='./datasets', patient_idx=0, slice_idx=-1):
    section_ids_list = [["151673", "151674", "151675", "151676"],
                        ["151507", "151508", "151509", "151510"],
                        ["151669", "151670", "151671", "151672"]]
    
    section_ids = section_ids_list[patient_idx]
    # section_ids = ['151673']

    if slice_idx == -1: # Multiple slices
        section_ids = section_ids_list[patient_idx]
    else:
        section_ids = section_ids_list[patient_idx][slice_idx]
        section_ids = [section_ids]

    slices = []
    for section_id in section_ids:
        print(section_id)
        input_dir = os.path.join(data_dir, '10x_dlpfc', section_id)
        adata = sc.read_visium(path=input_dir,
                            count_file=section_id + '_filtered_feature_bc_matrix.h5',
                            load_images=True)  # Warning
        adata.var_names_make_unique(join="++")

        # read the annotation
        Ann_df = pd.read_csv(os.path.join(input_dir, section_id + '_truth.txt'), sep='\t', header=None, index_col=0)
        Ann_df.columns = ['Gold_Standard']
        Ann_df[Ann_df.isna()] = "unknown"
        adata.obs['Gold_Standard'] = Ann_df.loc[adata.obs_names, 'Gold_Standard'].astype('category')
        
        # make spot name unique
        adata.obs_names = [x+'_'+section_id for x in adata.obs_names]
        adata.obs['batch'] = section_id
        
        slices.append(adata)

    return slices