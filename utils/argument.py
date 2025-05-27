import argparse
import time
from utils.utils import str2bool


def parse_args():
    parser = argparse.ArgumentParser()

    # For Wandb
    timestr = time.strftime('%m%d')
    parser.add_argument('--project', type=str, default=timestr,
                        help='Project name for Wandb')
    parser.add_argument('--wb_name', type=str, default=None,
                        help='Name for Wandb')
    parser.add_argument('--wandb', type=str2bool, default=True,
                        help='Wandb activation')

    # Data
    parser.add_argument('--result_dir', type=str, default='./results/',
                        help="Location of result files. if 'reports' or 'report', the trained model weight will be saved.") # reports
    parser.add_argument('--data_dir', type=str, default='/home/users/yunhak/mount198/0_BIO/STA_ORG/datasets',
                        help='Location of datasets')
    parser.add_argument('--dataset', type=str, default='10X_DLPFC',
                        help='Dataset name: 10X_DLPFC')
    parser.add_argument('--patient_idx', type=int, default=0,
                        help='Patient index of datasets')
    parser.add_argument('--slice_idx', type=int, default=0,
                        help='Slice index of datasets. If the value is -1, it loads whole slices.')


    # Experiments
    parser.add_argument('--device', type=str, default='1',
                        help='GPU. It supports single gpu mode.')
    parser.add_argument('--trials', type=int, default=1,
                        help='No. of experiments')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Training epochs (we fixed)')
    parser.add_argument('--ee', type=int, default=100,
                        help='Evaluation epoch per training epochs')
    parser.add_argument('--use_minibatch', type=str2bool, default=False,
                        help='Minibatch activation')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size (we fixed for large datasets)')
    parser.add_argument('--neighs', nargs='?', default='[10, 5]',
                        help='No. of neighbors for NeighborLoader, ex) [10, 5, 5], [10, 5] (we fixed)')
    parser.add_argument('--estimate_n_clusters', type=str2bool, default=False,
                        help='Activation of cluster number estimation. If False, it uses ground-truth k in kmeans clustering (we fixed)')

    parser.add_argument('--embedder', type=str, default='Spotscape',
                        help='Model name: Spotscape')
    parser.add_argument('--enc_type', type=str, default='gcn',
                        help='Graph AutoEncoder Type (we fixed)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout for model. (we fixed 0)')

    # Data Preprocessing
    parser.add_argument('--n_hvg', type=int, default=5000,
                        help='No. of highly variable genes (we fixed)')
    parser.add_argument('--normalize', type=str2bool, default=True,
                        help='Activation of input normalization (we fixed)')
    parser.add_argument('--log1p', type=str2bool, default=True,
                        help='Activation of log1p transform (we fixed)')
    parser.add_argument('--scale', type=str2bool, default=False,
                        help='Activation of input scaling (we fixed)')
    parser.add_argument('--pca', type=str2bool, default=False,
                        help='Activation of PCA (we fixed)')
    parser.add_argument('--normalize_coord', type=str2bool, default=False,
                        help='Activation of normalization of spatial coordinates (we fixed)')
    parser.add_argument('--graph', type=str, default='spatial',
                        help='Graph construction method: spatial')

    # For model
    parser.add_argument('--lr', type=float, default=0.00005,
                        help="Learning Rate")
    parser.add_argument('--decay', type=float, default=0.0001,
                        help="Weight Decay (we fixed)")
    parser.add_argument('--layers', nargs='?', default='[256, 64]',
                        help='Hidden dimensions of encoder or decoder: [256, 128, 64], [256, 64] (we fixed)')
    parser.add_argument('--clustering_method', type=str, default='kmeans', 
                        help='Clustering methods: kmeans, mclust, gmm (we fixed)')

    parser.add_argument("--de1", type=float, default=0.2,
                        help='Graph augmentation - drop edges of view1 (we fixed)')
    parser.add_argument("--de2", type=float, default=0.2,
                        help='Graph augmentation - drop edges of view2 (we fixed)')
    parser.add_argument("--df1", type=float, default=0.2,
                        help='Graph augmentation - drop features of view1 (we fixed)')
    parser.add_argument("--df2", type=float, default=0.2,
                        help='Graph augmentation - drop features of view2 (we fixed)')

    parser.add_argument("--lam_re", type=float, default=0.1,
                        help="Loss control hyperparameter of Reconstruction Loss (we fixed)")
    parser.add_argument("--lam_sc", type=float, default=1.0,
                        help="Loss control hyperparameter of Similarity Telescope (SC) Loss (we fixed)")

    parser.add_argument("--lam_pcl", type=float, default=0.01,
                        help="Loss control hyperparameter of Prototypical Contrastive Loss(PCL) (we fixed)")
    parser.add_argument("--warmup", type=int, default=500,
                        help='Warmup epochs for PCL (we fixed)')
    parser.add_argument("--tau", type=float, default=0.75, help='Temperature for PCL (we fixed)')
    parser.add_argument('--proto_granularity', metavar='N', type=str, nargs='*', default='[1, 1.5, 2]',
                        help='Prototypes granularity. It multiples with the number of clusters (we fixed)')
    
    parser.add_argument("--lam_ss", type=float, default=1.0,
                        help="Loss control hyperparameter of Similarity Scaling (SS) Loss (we fixed)")
    parser.add_argument("--sim_k", type=int, default=5,
                        help="Top-k hyperparameter for Similarity Scaling (SS) Loss (we fixed)")


    return parser.parse_known_args()
