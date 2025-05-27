import wandb
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from modeler import Modeler
from models.backbone.ae import GAE
from utils.transform import get_graph_drop_transform
from models.loss.selfsupervised import (
    ReconstructionLoss, 
    RelationConsistencyLoss,
    PCL
)

class Spotscape_Trainer(Modeler):
    def __init__(self, args, trial):
        Modeler.__init__(self, args, trial)
        self.construct_graph(self.slices, self.adata_concat,
                             self.gt_colname, self.args.dataset,
                             graph=self.args.graph)
        self.init_loader(self.graph, self.adata_concat)


    def init_model(self):
        self.model = GAE(input_dim=self.graph.x.shape[1],
                         layer_sizes=self.args.layers,
                         enc_type=self.args.enc_type,
                         dropout=self.args.dropout).to(self.device)
        param_group = [{'params':self.model.parameters(), 'lr':self.args.lr, 'weight_decay':self.args.decay}]
        self.optimizer = optim.Adam(param_group)

        self.recon_loss = ReconstructionLoss().to(self.device)
        self.rc_loss = RelationConsistencyLoss().to(self.device)

        self.proto_loss = PCL(
            self.n_clusters,
            self.args.proto_granularity,
            self.args.tau
        ).to(self.device)

    def train_integration(self):
        # Augmentation
        transform_1 = get_graph_drop_transform(drop_edge_p=self.args.de1, drop_feat_p=self.args.df1)
        transform_2 = get_graph_drop_transform(drop_edge_p=self.args.de2, drop_feat_p=self.args.df2)

        for epoch in tqdm(range(self.args.epochs)):
            for batch in self.data_loader:
                self.model.train()
                batch = batch.to(self.device)

                # Augmentation
                view1 = transform_1(batch)
                view2 = transform_2(batch)

                # Forward
                recon1, z1 = self.model(view1)
                recon1 = recon1[:batch.batch_size]; z1 = z1[:batch.batch_size]
                recon2, z2 = self.model(view2)
                recon2 = recon2[:batch.batch_size]; z2 = z2[:batch.batch_size]

                # Reconstruction Loss
                if self.args.lam_re > 0:
                    recon_loss = self.recon_loss(batch, recon1, recon2)
                else:
                    recon_loss = 0

                # Relation Consistency Loss
                if self.args.lam_sc > 0:
                    rc_loss, similarity = self.rc_loss(z1, z2)
                else:
                    _, similarity = self.rc_loss(z1, z2)
                    rc_loss = 0

                # Similarity Loss
                sim_loss = 0
                slice_ids = batch.node_type[:batch.batch_size]
                if self.args.lam_ss > 0 and self.n_slices >= 2:
                    for i in range(self.n_slices):
                        for j in range(self.n_slices):
                            if i == j:
                                continue
                            inside_cond = (slice_ids == i)[:, None] == (slice_ids == i)[None, :]
                            inside_sim_topk1 = torch.topk(similarity * inside_cond, self.args.sim_k)[0].mean(1)

                            outside_cond = (slice_ids == i)[:, None] == (slice_ids == j)[None, :]
                            outside_sim_topk1 = torch.topk(similarity * outside_cond, self.args.sim_k)[0].mean(1)

                            sim_loss += F.mse_loss(inside_sim_topk1, outside_sim_topk1)

                # Prototypical Contrastive Loss
                if epoch >= self.args.warmup and self.n_slices >= 2:
                    proto_loss, _, _ = self.proto_loss(z1, z2)
                else:
                    proto_loss = 0

                loss = self.args.lam_re * recon_loss + \
                    self.args.lam_sc * rc_loss + self.args.lam_ss * sim_loss + \
                    self.args.lam_pcl * proto_loss
                

                wandb.log({'total_loss': loss, 'recon_loss': recon_loss, 'rc_loss': rc_loss,
                           'sim_loss': sim_loss, 'proto_loss': proto_loss})

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch % self.args.ee == 0:
                self.eval_integration(epoch, loss)
