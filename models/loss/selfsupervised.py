import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from faiss import Kmeans


def infer_n_protos(k, proto_granularity):
    n_proto_gran = len(proto_granularity)
    n_cluster_list = []
    for i in range(n_proto_gran):
        if i == 0:
            n_cluster_list.append(int(np.floor(k * proto_granularity[i])))
        else:
            n_cluster_list.append(int(max(n_cluster_list[i-1] + 1, np.ceil(k * proto_granularity[i]))))
    return n_cluster_list


def exp_func(x, tau):
    return torch.exp(x / tau)


def cos_sim_p(z1, z2):
    z1 = F.normalize(z1, dim=-1, p=2)
    z2 = F.normalize(z2, dim=-1, p=2)

    return torch.matmul(z1, z2.transpose(-1, -2))


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, batch, recon1, recon2):
        mse_loss = (F.mse_loss(batch.x[:batch.batch_size], recon1) +
                    F.mse_loss(batch.x[:batch.batch_size], recon2)) / 2
        return mse_loss


class RelationConsistencyLoss(nn.Module):
    def __init__(self):
        super(RelationConsistencyLoss, self).__init__()

    def forward(self, z1, z2):
        normalized_z1 = F.normalize(z1, dim=-1, p=2)
        normalized_z2 = F.normalize(z2, dim=-1, p=2)
        similarity = torch.matmul(normalized_z1, normalized_z2.transpose(1, 0))

        rc_loss = (F.mse_loss(similarity, similarity.t()) + 
                   F.mse_loss(similarity.t(), similarity)) / 2
        
        return rc_loss, similarity


class PCL(nn.Module):
    def __init__(self, n_clusters, proto_granularity, proto_tau):
        super(PCL, self).__init__()
        self.k = n_clusters
        if isinstance(proto_granularity, str):
            self.proto_granularity = eval(proto_granularity)
        else:
            if isinstance(proto_granularity, list):
                self.proto_granularity = [float(p) for p in proto_granularity]
            else:
                self.proto_granularity = float(proto_granularity)
        self.proto_tau = proto_tau
        self.n_cluster_list = infer_n_protos(self.k, self.proto_granularity)


    def forward(self, z1, z2):
        cluster_assignment = []
        centroids_list = []
        proto_loss = 0
        rep = z2.detach().cpu().numpy()
        for n_cluster in self.n_cluster_list:
            # Get Centroids
            kmeans = Kmeans(d=rep.shape[1], k=n_cluster, niter=20)
            kmeans.train(rep)
            D, I = kmeans.index.search(rep, 1)
            y_pred = torch.LongTensor(np.squeeze(I)).to(z1.device)
            centroids = torch.tensor(kmeans.centroids, dtype=torch.float32, device=z1.device)

            # Prototypical Contrastive Learning
            proto_loss += self.proto_loss(z1, centroids, y_pred)

            cluster_assignment.append(y_pred)
            centroids_list.append(centroids)

        proto_loss /= len(self.n_cluster_list)

        return proto_loss.mean(), cluster_assignment, centroids_list


    def proto_loss(self, z1, centroids, y_pred):
        scores = exp_func(cos_sim_p(z1, centroids), self.proto_tau)

        pos_mask = torch.zeros_like(scores, dtype=torch.bool)
        pos_mask[torch.arange(len(scores)).unsqueeze(1), y_pred.unsqueeze(1)] = True

        pos_scores = scores[pos_mask]
        pos_neg_scores = scores.sum(1)
        proto_loss = -torch.log(pos_scores / pos_neg_scores)
        return proto_loss
