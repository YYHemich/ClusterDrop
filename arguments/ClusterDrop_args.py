import argparse


def parse_args():
    ap = argparse.ArgumentParser()
    # training hyper params
    ap.add_argument('--raw_dir', default='data', help='directory of datasets.')
    ap.add_argument('--dataset', default='cora', help='name of the used datasets.')
    ap.add_argument('--epoch', type=int, default=450, help='training epochs')
    ap.add_argument('--lr', type=float, default=0.01, help='learning rate')
    ap.add_argument('--weight_decay', type=float, default=5e-4)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--random_seed', type=int, default=0)
    ap.add_argument('--verbose', action='store_true', default=False)
    ap.add_argument('--drop_rate', type=float, default=0.15, help='drop edge rate')
    ap.add_argument('--sinkhorn_iterations', type=int, default=3, help='iteration number of sinkhorn-knopp algorithm')
    ap.add_argument('--sinkhorn_epsilon', type=float, default=0.2, help='coefficient of entropy regularization')
    ap.add_argument('--not_save_model', action='store_true', default=False,
                    help='only useful for optuna_ClusterDrop. If do not want to save checkpoint model, can specify this argument.')
    ap.add_argument('--trials_num', type=int, default=400, help='number of trials for optuna')
    ap.add_argument('--normalize_feature', default=False, action='store_true', help='If true, the input feature will be normalized.')
    ap.add_argument('--inner_iterations', type=int, default=1)
    ap.add_argument('--multi_runs', default=False, action='store_true', help='multiple random runs.')

    # GAT hyperparameters
    ap.add_argument('--gat_head_num', type=int, default=8, help='number of GAT attention heads.')

    # trainable cluster loss parameters
    ap.add_argument('--cluster_loss_ratio', type=float, default=0.0542, help='coefficient of cluster loss')
    ap.add_argument('--assign_loss_ratio', type=float, default=73.186, help='coefficient of assign loss')

    # assignment loss parameters
    ap.add_argument('--assign_sigma', type=float, default=1.)
    ap.add_argument('--assign_beta', type=float, default=0.5,
                    help='coefficient of the in cluster center regularization.')

    # model hyper params
    ap.add_argument('--gnn_backbone', default='GCN')
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--hidden_dim', type=int, default=256)
    ap.add_argument('--gnn_dropout', type=float, default=0.8)
    ap.add_argument('--n_clusters', type=int, default=150)
    ap.add_argument('--set_zero_except_first', default=True, action='store_true')
    ap.add_argument('--orthogonal', default=True, action='store_true')
    ap.add_argument('--bn', default=False, action='store_true')

    # others
    ap.add_argument('--comment', default='')

    return ap.parse_args()
