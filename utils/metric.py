import torch
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn import metrics
import torch


"""Clustering Metrics"""

def perf_clustering_metric(latent, y_pred, y, device):
    sil_f = metric_silhouette(device)
    silhouette = np.round(sil_f.score(latent, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ca, maf1, mif1 = cluster_acc(y, y_pred)
    ca = np.round(ca, 5) ; maf1 = np.round(maf1, 5) ; mif1 = np.round(mif1, 5)

    return silhouette, ari, nmi, ca, maf1, mif1


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    try:
        y_true = y_true.astype(np.int64)
    except:
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_true = label_encoder.fit_transform(y_true)
        y_true = y_true.astype(np.int64)

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    pred_indices, true_indices = linear_assignment(w.max() - w)

    new_predict = np.zeros(len(y_pred))
    for pred_ind, true_ind in zip(pred_indices, true_indices):
        new_predict[y_pred == pred_ind] = true_ind


    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')

    return acc, f1_macro, f1_micro


class metric_silhouette():
    def __init__(self, device):
        self.device = device

    def score(self, X, labels, loss=False):
        """Compute the mean Silhouette Coefficient of all samples.
        The Silhouette Coefficient is calculated using the mean intra-cluster
        distance (a) and the mean nearest-cluster distance (b) for each sample.
        The Silhouette Coefficient for a sample is `(b - a) / max(a, b).
        To clarrify, b is the distance between a sample and the nearest cluster
        that b is not a part of.
        This function returns the mean Silhoeutte Coefficient over all samples.
        The best value is 1 and the worst value is -1. Values near 0 indicate
        overlapping clusters. Negative values generally indicate that a sample has
        been assigned to the wrong cluster, as a different cluster is more similar.

        Code developed in NumPy by Alexandre Abraham:
        https://gist.github.com/AlexandreAbraham/5544803  Avatar
        Parameters
        ----------
        X : array [n_samples_a, n_features]
            Feature array.
        labels : array, shape = [n_samples]
                 label values for each sample
        loss : Boolean
                If True, will return negative silhouette score as 
                torch tensor without moving it to the CPU. Can therefore 
                be used to calculate the gradient using autograd.
                If False positive silhouette score as float 
                on CPU will be returned.
        Returns
        -------
        silhouette : float
            Mean Silhouette Coefficient for all samples.
        References
        ----------
        Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
            Interpretation and Validation of Cluster Analysis". Computational
            and Applied Mathematics 20: 53-65. doi:10.1016/0377-0427(87)90125-7.
        http://en.wikipedia.org/wiki/Silhouette_(clustering)
        """

        if type(labels) != type(torch.HalfTensor()):
            labels = torch.HalfTensor(labels)
        if not labels.is_cuda:
            labels = labels.to(self.device)

        if type(X) != type(torch.HalfTensor()):
            X = torch.HalfTensor(X)
        if not X.is_cuda:
            X = X.to(self.device)

        unique_labels = torch.unique(labels)

        A = self._intra_cluster_distances_block(X, labels, unique_labels)
        B = self._nearest_cluster_distance_block(X, labels, unique_labels)
        sil_samples = (B - A) / torch.maximum(A, B)

        # nan values are for clusters of size 1, and should be 0
        mean_sil_score = torch.mean(torch.nan_to_num(sil_samples))
        if loss:
            return - mean_sil_score
        else:
            return float(mean_sil_score.cpu().numpy())

    def _intra_cluster_distances_block(self, X, labels, unique_labels):
        """Calculate the mean intra-cluster distance.
        Parameters
        ----------
        X : array [n_samples_a, n_features]
            Feature array.
        labels : array, shape = [n_samples]
            label values for each sample
        Returns
        -------
        a : array [n_samples_a]
            Mean intra-cluster distance
        """
        intra_dist = torch.zeros(labels.size(), dtype=torch.float32).to(self.device)
        values = [self._intra_cluster_distances_block_(
                    X[torch.where(labels == label)[0]])
                    for label in unique_labels]
        for label, values_ in zip(unique_labels, values):
            intra_dist[torch.where(labels == label)[0]] = values_
        return intra_dist

    def _intra_cluster_distances_block_(self, subX):
        distances = torch.cdist(subX, subX)
        return distances.sum(axis=1) / (distances.shape[0] - 1)

    def _nearest_cluster_distance_block(self, X, labels, unique_labels):
        """Calculate the mean nearest-cluster distance for sample i.
        Parameters
        ----------
        X : array [n_samples_a, n_features]
            Feature array.
        labels : array, shape = [n_samples]
            label values for each sample
        X : array [n_samples_a, n_features]
            Feature array.
        Returns
        -------
        b : float
            Mean nearest-cluster distance for sample i
        """
        inter_dist = torch.full(labels.size(), torch.inf,
                                 dtype=torch.float32).to(self.device)
        # Compute cluster distance between pairs of clusters

        label_combinations = torch.combinations(unique_labels, 2)

        values = [self._nearest_cluster_distance_block_(
                    X[torch.where(labels == label_a)[0]],
                    X[torch.where(labels == label_b)[0]])
                    for label_a, label_b in label_combinations]

        for (label_a, label_b), (values_a, values_b) in \
                zip(label_combinations, values):

                indices_a = torch.where(labels == label_a)[0]
                inter_dist[indices_a] = torch.minimum(values_a, inter_dist[indices_a])
                del indices_a
                indices_b = torch.where(labels == label_b)[0]
                inter_dist[indices_b] = torch.minimum(values_b, inter_dist[indices_b])
                del indices_b
        return inter_dist

    def _nearest_cluster_distance_block_(self, subX_a, subX_b):
        dist = torch.cdist(subX_a, subX_b)
        dist_a = dist.mean(axis=1)
        dist_b = dist.mean(axis=0)
        return dist_a, dist_b
