import os
import scanpy as sc
import numpy as np
import pandas as pd
import anndata
from utils.utils import normalize_coordinates, norm_to_raw


def load_slices(data_dir, data_name, patient_idx=0, slice_idx=-1):
    if data_name.lower() == '10x_dlpfc':
        slices = load_10x_dlpfc(data_dir, patient_idx, slice_idx)
    elif data_name.lower() == 'mtg':
        slices = load_mtg(data_dir, slice_idx=slice_idx)
    elif data_name.lower() == 'mouse_embryo':
        slices = load_mouse_embryo(data_dir)
    elif data_name.lower() == 'nsclc':
        slices = load_nsclc(data_dir)
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


def load_mtg(data_dir='./datasets', slice_idx=-1):
    data_path = os.path.join(data_dir, 'mtg_ad')
    
    metadata = pd.read_csv(os.path.join(data_path, 'metadata.csv'), index_col=0)
    metadata['Layer'] = np.where(metadata['Layer'] == 'Noise', 'unknown', metadata['Layer'])
    metadata = metadata[['patientID', 'Layer']].rename(columns={'Layer': 'Gold_Standard'})

    folder_list = ['2-5', 'T4857']
    index_list = ['2_5', 'T4857']

    if slice_idx != -1:
        folder_list = [folder_list[slice_idx]]
        index_list = [index_list[slice_idx]]

    slices = []
    for f, i in zip(folder_list, index_list):
        adata = sc.read_visium(path=os.path.join(data_path, f),
                                count_file='filtered_feature_bc_matrix.h5',
                                load_images=True)
        adata.var_names_make_unique(join="++")
        adata.obs.index = adata.obs.index.to_series().apply(lambda x: x.split('-')[0]) + '-' + i

        adata.obs['Gold_Standard'] = metadata.loc[adata.obs_names, 'Gold_Standard'].astype('category')
        adata.obs['batch'] = f
        adata.uns['batch'] = f
        slices.append(adata)

    return slices

def load_mouse_embryo(data_dir='./datasets'):
    data_path = os.path.join(data_dir, 'dev_stage_align')
    slice_list = ['E11.5_E1S1']

    slices = []
    for i, s in enumerate(slice_list):
        print(s)
        path = os.path.join(data_path, s)
        adata = sc.read_h5ad(path+'.MOSTA.h5ad')
        adata = norm_to_raw(adata, 'total_counts')
        adata.obs = adata.obs.iloc[:,0:5]
        adata.var = adata.var.iloc[:,0:6]
        del adata.layers['count']

        adata.var_names_make_unique(join="++")
        adata.obs_names = [x+'_'+str(i) for x in adata.obs_names]
        adata.obs['batch'] = s
        adata.uns['batch'] = s
        adata.obs['Gold_Standard'] = adata.obs['annotation'].astype('category')
        del adata.obs['annotation']

        slices.append(adata)
    return slices

def load_nsclc(data_dir='./datasets'):
    data_path = os.path.join(data_dir, 'NSCLC')
    section_ids = ['Lung5_3']

    slices = []
    for f in section_ids:
        adata = sc.read_h5ad(os.path.join(data_path, f+'.h5ad'))
        adata.var_names_make_unique(join='++')

        adata.obs_names = [x+'_'+str(f) for x in adata.obs_names]
        adata.obs['batch'] = f
        adata.uns['batch'] = f

        adata.obsm['spatial'] = adata.obs[['coord_x', 'coord_y']].values
        adata.obs['Gold_Standard'] = adata.obs['region'].astype('category')

        slices.append(adata)

    return slices