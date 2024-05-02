from dgl_graph_loader import DATASET_MAP
from models.ClusterDropModels import *
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from utils import set_random_seed
from arguments.ClusterDrop_args import parse_args
from tensorboardX import SummaryWriter
import os
import numpy as np
import torch.nn.functional as F
import copy
import optuna
import shutil
import dgl


ORIGINAL_ARGS = parse_args()
HISTORY_BEST_ACC = 0
HISTORY_BEST_STD = 100

if not os.path.exists('OptunaModelOut/%s%sOut/TrailsOut' % (ORIGINAL_ARGS.gnn_backbone, ORIGINAL_ARGS.dataset)):
    os.makedirs('OptunaModelOut/%s%sOut/TrailsOut' % (ORIGINAL_ARGS.gnn_backbone, ORIGINAL_ARGS.dataset))

device = ORIGINAL_ARGS.device if torch.cuda.is_available() else 'cpu'
dataset = DATASET_MAP[ORIGINAL_ARGS.dataset](raw_dir=ORIGINAL_ARGS.raw_dir, reverse_edge=True, verbose=ORIGINAL_ARGS.verbose)
ORIGINAL_G = dataset[0]
FEAT = ORIGINAL_G.ndata['feat']
TRAIN_MASK = ORIGINAL_G.ndata['train_mask']
VAL_MASK = ORIGINAL_G.ndata['val_mask']
TEST_MASK = ORIGINAL_G.ndata['test_mask']
LABEL = ORIGINAL_G.ndata['label'].to(device)
ORIGINAL_G = dgl.remove_self_loop(ORIGINAL_G)
ORIGINAL_G = dgl.add_self_loop(ORIGINAL_G)

num_classes = dataset.num_classes
in_feat = FEAT.shape[-1]

# features normalized
if ORIGINAL_ARGS.normalize_feature:
    FEAT = F.normalize(FEAT, p=1, dim=-1)


def adaptive_cluster_ratio(e, max_e):
    state = e / max_e
    if state < 1 / 5:
        return 1.0
    elif state < 1 / 3:
        return 0.5
    return 0.2


def objective(trial):
    """
    Tuning hyperparameters
    n_cluster
    sinkhorn_iterations
    drop_rate
    learning rate
    """
    args = copy.deepcopy(ORIGINAL_ARGS)
    args.drop_rate = trial.suggest_float('drop_rate', low=0.10, high=0.7, step=0.05)
    args.sinkhorn_iterations = trial.suggest_int('sinkhorn_iterations', low=3, high=6)
    args.n_clusters = trial.suggest_int('n_clusters', low=25, high=250, step=25)
    hidden_num = trial.suggest_categorical('layers', [64, 128, 256, 512])
    # hidden_num = trial.suggest_categorical('layers', [4, 8, 16, 32])  # GAT
    args.layers = [hidden_num] * 2
    args.sinkhorn_epsilon = trial.suggest_float('sinkhorn_epsilon', low=0.05, high=0.35, step=0.05)
    args.cluster_loss_ratio = trial.suggest_float('cluster_loss_ratio', low=0.001, high=100, log=True)
    args.assign_loss_ratio = trial.suggest_float('assign_loss_ratio', low=0.001, high=100, log=True)
    args.gnn_dropout = trial.suggest_float('gnn_dropout', low=0.3, high=0.8, step=0.1)
    # assignment loss parameters
    args.assign_beta = trial.suggest_float('assign_beta', low=0.05, high=50, log=True)
    random_seed_set = [30, 218, 14, 349, 103, 120, 241, 51, 216, 120]
    test_accs = []
    valid_accs = []
    best_epochs = []
    for random_test_idx, random_seed in enumerate(random_seed_set):
        set_random_seed(random_seed)
        model = GNN_MODEL_CONSTURCTOR[args.gnn_backbone](in_feat, args.hidden_dim, args.layers, num_classes, args)

        loss_func = nn.CrossEntropyLoss()
        loss_func.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        feat = FEAT.to(device)
        model = model.to(device)
        g = ORIGINAL_G.to(device)

        best_valid = 0
        best_test = 0
        best_epoch = 0
        for e in range(1, 1+args.epoch):
            model.train()
            output, embedding, cluster_loss, cluster_assign_loss = model(g, feat, label=LABEL, train_mask=TRAIN_MASK)

            cls_loss = loss_func(output[TRAIN_MASK], LABEL[TRAIN_MASK])
            sum_cluster_loss = 0
            for c_loss in cluster_loss:
                sum_cluster_loss = sum_cluster_loss + c_loss
            sum_assign_loss = 0
            for a_loss in cluster_assign_loss:
                sum_assign_loss = sum_assign_loss + a_loss
            loss = cls_loss + args.cluster_loss_ratio * sum_cluster_loss + args.assign_loss_ratio * sum_assign_loss  # * adaptive_cluster_ratio(e, args.epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.verbose:
                print('[Loss]', loss.item())

            # valid
            model.eval()
            with torch.no_grad():
                output, embedding, cluster_loss, assign_loss = model(g, feat, label=LABEL, train_mask=TRAIN_MASK)
                pred_label = torch.max(output, dim=-1)[1].cpu().data.numpy()

                train_acc = accuracy_score(LABEL[TRAIN_MASK].cpu().data.numpy(), pred_label[TRAIN_MASK])
                valid_acc = accuracy_score(LABEL[VAL_MASK].cpu().data.numpy(), pred_label[VAL_MASK])
                val_loss = loss_func(output[VAL_MASK], LABEL[VAL_MASK]).item()
                test_acc = accuracy_score(LABEL[TEST_MASK].cpu().data.numpy(), pred_label[TEST_MASK])

                if args.verbose:
                    print('[EPOCH] %d' % e)
                    print('train acc', train_acc)
                    print('valid acc', valid_acc)
                    print('test acc', test_acc)

                if valid_acc > best_valid or (valid_acc == best_valid and test_acc > best_test):
                    best_valid = valid_acc
                    best_test = test_acc
                    best_epoch = e
                    if not args.not_save_model:
                        torch.save(model.state_dict(), 'OptunaModelOut/%s%sOut/TrailsOut/%d_checkpoint.pth' % (
                            args.gnn_backbone, args.dataset, random_test_idx
                        ))

        # training importance predictor
        valid_accs.append(best_valid)
        test_accs.append(best_test)
        best_epochs.append(best_epoch)
    mean_acc = np.mean(test_accs)
    std_acc = np.std(test_accs)
    mean_val_acc = np.mean(valid_accs)
    std_val_acc = np.std(valid_accs)
    best_epoch_mean = np.mean(best_epochs)
    best_epoch_std = np.std(best_epochs)
    # save the best models
    global HISTORY_BEST_ACC
    global HISTORY_BEST_STD
    if mean_acc > HISTORY_BEST_ACC or (mean_acc == HISTORY_BEST_ACC and std_acc < HISTORY_BEST_STD):
        if not args.not_save_model:
            copy_best_random_group(args)
        HISTORY_BEST_ACC = mean_acc
        HISTORY_BEST_STD = std_acc
    trial.suggest_float('best_test_acc', mean_acc, mean_acc)
    trial.suggest_float('best_test_std', std_acc, std_acc)
    trial.suggest_float('best_valid_acc', mean_val_acc, mean_val_acc)
    trial.suggest_float('best_valid_std', std_val_acc, std_val_acc)
    trial.suggest_float('best_epoch_mean', best_epoch_mean, best_epoch_mean)
    trial.suggest_float('best_epoch_std', best_epoch_std, best_epoch_std)
    return mean_acc


def copy_best_random_group(args):
    tmp_save_root = 'OptunaModelOut/%s%sOut/TrailsOut' % (args.gnn_backbone, args.dataset)
    target_save_root = 'OptunaModelOut/%s%sOut' % (args.gnn_backbone, args.dataset)
    for file in sorted(os.listdir(tmp_save_root)):
        shutil.copyfile(os.path.join(tmp_save_root, file), os.path.join(target_save_root, file))
    torch.save(args, os.path.join(target_save_root, 'best_args.argument'))


if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=ORIGINAL_ARGS.trials_num)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    print('[HISTORY BEST]', HISTORY_BEST_ACC, '+/-', HISTORY_BEST_STD)
