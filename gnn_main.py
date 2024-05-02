from dgl_graph_loader import DATASET_MAP
from models.gnn import *
import argparse
import dgl
import torch
from sklearn.metrics import accuracy_score
import torch.nn as nn
from utils import set_random_seed
import torch.nn.functional as F
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', default='../data')
    ap.add_argument('--model', default='GCN')
    ap.add_argument('--epoch', type=int, default=200)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--weight_decay', type=float, default=5e-4)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dataset', default='cora')
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--random_seed', type=int, default=42)
    ap.add_argument('--verbose', default=False, action='store_true')
    ap.add_argument('--normalize', default=False, action='store_true')
    ap.add_argument('--multi_runs', default=False, action='store_true')
    ap.add_argument('--bn', default=False, action='store_true')
    return ap.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    dataset = DATASET_MAP[args.dataset](raw_dir=args.raw_dir, reverse_edge=True, verbose=False)
    g = dataset[0]
    feat = g.ndata['feat']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    label = g.ndata['label'].to(device)
    num_classes = dataset.num_classes
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    if args.verbose:
        print("input feat num", feat.shape[-1])

    if args.normalize:
        feat = F.normalize(feat, p=1, dim=-1)

    if args.multi_runs:
        random_seed_set = [30, 218, 14, 349, 103, 120, 241, 51, 216, 120]
    else:
        random_seed_set = [args.random_seed]

    valid_container = []
    test_container = []
    for random_seed in random_seed_set:
        set_random_seed(random_seed)
        model = GNN_MODEL_CONSTURCTOR[args.model](in_feat=feat.shape[-1], hidden_dim=args.hidden, layers=args.layers,
                                                  num_class=num_classes, dropout=args.dropout, norm_type='both', batch_norm=args.bn)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_func = nn.CrossEntropyLoss()

        model.to(device)
        feat = feat.to(device)
        label = label.to(device)
        g = g.to(device)

        best_valid = best_test = 0
        for e in range(1, 1+args.epoch):
            model.train()
            output = model(g, feat)
            loss = loss_func(output[train_mask], label[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # valid
            model.eval()
            with torch.no_grad():
                output = model(g, feat)
                pred_label = torch.max(output, dim=-1)[1].cpu().data.numpy()

                train_acc = accuracy_score(label[train_mask].cpu().data.numpy(), pred_label[train_mask])
                valid_acc = accuracy_score(label[val_mask].cpu().data.numpy(), pred_label[val_mask])
                valid_loss = loss_func(output[val_mask], label[val_mask]).item()
                test_acc = accuracy_score(label[test_mask].cpu().data.numpy(), pred_label[test_mask])
                if args.verbose:
                    print('[train acc]', train_acc, '[valid acc]', valid_acc, '[test acc]', test_acc)
                if valid_acc > best_valid:
                    best_valid = valid_acc
                    best_test = test_acc

        # training importance predictor
        valid_container.append(best_valid)
        test_container.append(best_test)
    print("%s Runs\n[BEST] Valid ACC %.5f +/- %.5f\n[BEST] TEST ACC %.5f +/- %.5f" % (
        len(random_seed_set), np.mean(valid_container), np.std(valid_container), np.mean(test_container), np.std(test_container)
    ))


if __name__ == '__main__':
    main()
