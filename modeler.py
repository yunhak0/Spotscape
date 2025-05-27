import wandb
import numpy as np
import torch
from torch_geometric.loader import NeighborLoader

import faiss
# from utils.utils import mclust_R
# from sklearn.mixture import GaussianMixture

from utils.data import load_slices, preprocess_slices, concat_slices
from utils.graph_construction import construct_sp_graph
from utils.metric import perf_clustering_metric


class Modeler:
    def __init__(self, args, trial):
        self.args = args
        self.trial = trial

        self.device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"

        # Read data
        self.slices = load_slices(data_dir=args.data_dir,
                                  data_name=args.dataset,
                                  patient_idx=args.patient_idx,
                                  slice_idx=args.slice_idx)
        
        self.slices = preprocess_slices(self.slices, n_hvg=args.n_hvg,
                                        normalize=args.normalize, log1p=args.log1p,
                                        scale=args.scale, pca=args.pca,
                                        normalize_coord=args.normalize_coord,
                                        n_comps=self.args.n_pca_comps if hasattr(self.args, 'n_pca_comps') else 200)

        self.slices, self.adata_concat, self.common_genes = concat_slices(self.slices, common_genes=True)

        self.gt_colname = 'Gold_Standard'
        self.input_dim = [s.shape[1] for s in self.slices]
        self.n_slices = len(self.input_dim)
        self.batch_names = [s.obs['batch'].values[0] for s in self.slices]

        if ~self.args.estimate_n_clusters:
            self.detect_n_clusters(self.slices, self.adata_concat, self.gt_colname)


    def detect_n_clusters(self, slices, adata_concat, gt_colname):
        n_clusters = []

        for i in range(len(slices)):
            u_c = np.unique(slices[i].obs[gt_colname])
            if 'unknown' in u_c:
                n_clusters.append(u_c.shape[0] - 1)
            else:
                n_clusters.append(u_c.shape[0])
        self.each_n_clusters = n_clusters
        self.args.each_n_clusters = n_clusters

        unique_clusters = np.unique(adata_concat.obs[gt_colname])
        if 'unknown' in unique_clusters:
            self.n_clusters = unique_clusters.shape[0] - 1
        else:
            self.n_clusters = unique_clusters.shape[0]
        self.args.n_clusters = self.n_clusters


    def construct_graph(self, slices, adata_concat, gt_colname, data_name, graph='spatial', **kwargs):
        if graph.lower() == 'spatial':
            self.graph = construct_sp_graph(slices, adata_concat,
                                            gt_colname, data_name)
        else:
            raise ValueError(f"Invalid graph construction method: {graph}")


    def init_loader(self, graph_data, adata_concat):
        input_nodes = torch.ones(adata_concat.n_obs).bool()

        if self.args.use_minibatch:
            self.data_loader = NeighborLoader(graph_data,
                                            num_neighbors=eval(self.args.neighs),
                                            input_nodes=input_nodes,
                                            shuffle=True, 
                                            batch_size=self.args.batch_size, drop_last=True)
            self.eval_loader = NeighborLoader(graph_data,
                                            num_neighbors=eval(self.args.neighs),
                                            input_nodes=input_nodes,
                                            shuffle=False,
                                            batch_size=self.args.batch_size, drop_last=False)
        else:
            self.data_loader = NeighborLoader(graph_data,
                                            num_neighbors=[-1],
                                            input_nodes=input_nodes,
                                            shuffle=True, 
                                            batch_size=adata_concat.n_obs)
            self.eval_loader = NeighborLoader(graph_data,
                                            num_neighbors=[-1],
                                            input_nodes=input_nodes,
                                            shuffle=False,
                                            batch_size=adata_concat.n_obs)
            
    #############
    # For Model #
    #############
    def train(self):
        self.init_model()
        self.train_integration()
        res = self.iter_eval_integration()

        return res


    def init_model(self):
        pass


    def train_integration(self):
        pass


    def get_representation(self):
        self.model.eval()
        Z = []
        for batch in self.eval_loader:
            batch = batch.to(self.device)
            _, z = self.model(batch)
            Z.append(z[:batch.batch_size].detach())
        Z = torch.cat(Z, dim=0).cpu()
        return Z


    def eval_integration(self, epoch, loss):
        self.model.eval()
        Z = self.get_representation()
        GT = self.adata_concat.obs[self.gt_colname]

        for i in range(self.n_slices):
            b_name = self.batch_names[i]
            cond = self.adata_concat.obs['slice_name'] == str(i)
            z = Z[cond]
            gt = GT[cond]
            silhouette, ari, nmi, ca, maf1, mif1 = self.clustering_metric(z, gt, k=self.each_n_clusters[i],
                                                                          clustering_method=self.args.clustering_method)

            wandb.log({f'silhouette_{b_name}': silhouette,
                        f'ari_{b_name}': ari, f'nmi_{b_name}': nmi, f'ca_{b_name}': ca})

            st = '| slice: {} | silhouette : {:.4f} | ari : {:.4f} | nmi : {:.4f} | ca : {:.4f} |'.format(
                b_name, silhouette, ari, nmi, ca)
            print(st)

        silhouette, ari, nmi, ca, maf1, mif1 = self.clustering_metric(Z, GT, k=self.n_clusters,
                                                                      clustering_method=self.args.clustering_method)
        
        wandb.log({'silhouette': silhouette, 'ari': ari, 'nmi': nmi, 'ca': ca})

        st = '** | epochs:{} | loss : {:.4f} | silhouette : {:.4f} | ari : {:.4f} | nmi : {:.4f} | ca : {:.4f} | **'.format(
            str(epoch+1).zfill(4), loss.item(), silhouette, ari, nmi, ca)
        print(st)

    
    def iter_eval_integration(self):
        self.model.eval()
        Z = self.get_representation()
        GT = self.adata_concat.obs[self.gt_colname]

        print(f'===================================== Iteration : {self.trial+1} =====================================')
        silhouette_list = []; ari_list = []; nmi_list = []; ca_list = []; maf1_list = []; mif1_list = []
        for i in range(self.n_slices):
            b_name = self.batch_names[i]
            cond = self.adata_concat.obs['slice_name'] == str(i)
            z = Z[cond]
            gt = GT[cond]
            silhouette, ari, nmi, ca, maf1, mif1 = self.clustering_metric(z, gt, k=self.each_n_clusters[i],
                                                                          clustering_method=self.args.clustering_method)

            wandb.log({f'silhouette_{b_name}': silhouette,
                       f'ari_{b_name}': ari, f'nmi_{b_name}': nmi, f'ca_{b_name}': ca})
            
            print("| slice: {} | n_clusters : {} | silhouette : {:.4f} | ari : {:.4F} | nmi : {:.4F} | ca : {:.4f}  | ma-f1 : {:.4F} | mi-f1 : {:.4F} |".format(
                b_name, self.each_n_clusters[i], silhouette, ari, nmi, ca, maf1, mif1))
            
            silhouette_list.append(silhouette)
            ari_list.append(ari)
            nmi_list.append(nmi)
            ca_list.append(ca)
            maf1_list.append(maf1)
            mif1_list.append(mif1)

        silhouette, ari, nmi, ca, maf1, mif1 = self.clustering_metric(Z, GT, k=self.each_n_clusters[i],
                                                                      clustering_method=self.args.clustering_method)

        wandb.log({'silhouette': silhouette, 'ari': ari, 'nmi': nmi, 'ca': ca})
            
        # save_results
        print("** | n_clusters : {} | silhouette : {:.4f} | ari : {:.4F} | nmi : {:.4F} | ca : {:.4f}  | ma-f1 : {:.4F} | mi-f1 : {:.4F} | **".format(
                self.n_clusters, silhouette, ari, nmi, ca, maf1, mif1))
        print(f'==========================================================================================')

        silhouette_list.append(silhouette)
        ari_list.append(ari)
        nmi_list.append(nmi)
        ca_list.append(ca)
        maf1_list.append(maf1)
        mif1_list.append(mif1)

        res = {'each_n_clusters': self.each_n_clusters, 'n_clusters': self.n_clusters, 'batch_name': self.batch_names,
               'silhouette': silhouette_list, 
               'ari': ari_list, 'nmi': nmi_list, 'ca': ca_list,
               'maf1': maf1_list, 'mif1': mif1_list}

        return res


    def clustering_metric(self, Z, gt, k, clustering_method='kmeans'):
        if 'unknown' in gt.values:
            unknown_idx = gt.values == 'unknown'
            Z = Z[~unknown_idx]
            gt = gt[~unknown_idx]


        if clustering_method.lower() == 'kmeans':
            kmeans = faiss.Kmeans(d=Z.shape[1], k=k, niter=20, nredo=10)
            kmeans.train(Z)
            y_preds = np.squeeze(kmeans.index.search(Z, 1)[1])


        # elif clustering_method == 'mclust':
        #     y_preds = mclust_R(Z, n_cluster=k)


        # elif clustering_method == 'gmm':
        #     gmm = GaussianMixture(n_components=k,
        #                             covariance_type="tied", n_init=20, tol=2e-4,
        #                             max_iter=300, reg_covar=1.5e-4, random_state=42)
        #     y_preds = gmm.fit_predict(Z)
        else:
            raise ValueError(f"Invalid clustering method: {clustering_method}")


        silhouette, ari, nmi, ca, maf1, mif1 = perf_clustering_metric(Z, y_preds, gt, device=self.device)

        return silhouette, ari, nmi, ca, maf1, mif1
