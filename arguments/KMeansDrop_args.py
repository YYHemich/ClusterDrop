import argparse


def parse_args():
    ap = argparse.ArgumentParser()
    # training hyper params
    ap.add_argument('--raw_dir', default='../data')
    ap.add_argument('--dataset', default='cora')
    ap.add_argument('--epoch', type=int, default=150)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--gnn_lr', type=float, default=0.01)
    ap.add_argument('--dropedge_lr', type=float, default=0.01)
    ap.add_argument('--weight_decay', type=float, default=5e-4)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--random_seed', type=int, default=0)
    ap.add_argument('--verbose', action='store_true', default=False)
    # ap.add_argument('--window_size', type=int, default=10)
    ap.add_argument('--drop_rate', type=float, default=0.1)
    ap.add_argument('--sinkhorn_iterations', type=int, default=5)
    ap.add_argument('--sinkhorn_epsilon', type=float, default=0.05)
    ap.add_argument('--not_save_model', action='store_true', default=False,
                    help='only useful for optuna_KLDrop. If do not want to save the best model, can specify this argument.')
    ap.add_argument('--trials_num', type=int, default=400)
    ap.add_argument('--normalize_feature', default=False, action='store_true')

    ap.add_argument('--gnn_pretrain', type=int, default=0)
    ap.add_argument('--cluster_pretrain', type=int, default=0)
    ap.add_argument('--inner_iterations', type=int, default=1)

    ap.add_argument('--fold_num', type=int, default=0, help='the fold num of MCF7_CPDB dataset.')
    ap.add_argument('--multi_runs', default=False, action='store_true')

    # GAT hyperparameters
    ap.add_argument('--gat_head_num', type=int, default=8)

    # trainable cluster loss parameters
    ap.add_argument('--prototype_dim', type=int, default=64)
    ap.add_argument('--cluster_loss_ratio', type=float, default=0.)
    ap.add_argument('--assign_loss_ratio', type=float, default=1.0)

    # assignment loss parameters
    ap.add_argument('--assign_sigma', type=float, default=1.)
    ap.add_argument('--assign_beta', type=float, default=0.5,
                    help='coefficient of the in clluster center regularization.')

    # model hyper params
    ap.add_argument('--gnn_backbone', default='GCN')
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--hidden_dim', type=int, default=256)
    ap.add_argument('--gnn_dropout', type=float, default=0.5)
    ap.add_argument('--n_clusters', type=int, default=100)
    ap.add_argument('--set_zero_except_first', default=False, action='store_true')
    ap.add_argument('--distance_type', default='euclidean', choices=['euclidean', 'cosine'])
    ap.add_argument('--bn', default=False, action='store_true')

    # others
    ap.add_argument('--comment', default='')
    return ap.parse_args()
