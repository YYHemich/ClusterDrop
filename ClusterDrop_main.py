from dgl_graph_loader import DATASET_MAP
from models.ClusterDropModels import *
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from utils import set_random_seed
from arguments.ClusterDrop_args import parse_args
import numpy as np
import torch.nn.functional as F
import dgl


def main():
    args = parse_args()
    if args.verbose:
        print(args)
    device = args.device if torch.cuda.is_available() else 'cpu'
    dataset = DATASET_MAP[args.dataset](raw_dir=args.raw_dir, reverse_edge=True, verbose=args.verbose)
    g = dataset[0]
    feat = g.ndata['feat']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    label = g.ndata['label'].to(device)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    num_classes = dataset.num_classes
    in_feat = feat.shape[-1]

    # features normalized
    if args.normalize_feature:
        feat = F.normalize(feat, p=1, dim=-1)

    if args.multi_runs:
        random_seed_set = [30, 218, 14, 349, 103, 120, 241, 51, 216, 120]
    else:
        random_seed_set = [args.random_seed]

    valid_container = []
    test_container = []
    epoch_container = []
    for random_seed in random_seed_set:
        set_random_seed(random_seed)
        model = GNN_MODEL_CONSTURCTOR[args.gnn_backbone](in_feat, args.hidden_dim, args.layers, num_classes, args)
        loss_func = nn.CrossEntropyLoss()
        loss_func.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        feat = feat.to(device)
        model = model.to(device)
        g = g.to(device)

        best_valid = 0
        best_test = 0
        best_epoch = 0
        for e in range(1, 1+args.epoch):
            model.train()
            total_loss = total_cluster_loss = total_cls_loss = total_assign_loss = 0
            optimizer.zero_grad()
            for inner_loop in range(args.inner_iterations):
                output, embedding, cluster_loss, cluster_assign_loss = model(g, feat, label=label, train_mask=train_mask)

                cls_loss = loss_func(output[train_mask], label[train_mask])
                sum_cluster_loss = 0
                for c_loss in cluster_loss:
                    sum_cluster_loss = sum_cluster_loss + c_loss
                sum_assign_loss = 0
                for a_loss in cluster_assign_loss:
                    sum_assign_loss = sum_assign_loss + a_loss

                loss = cls_loss + args.cluster_loss_ratio * sum_cluster_loss + args.assign_loss_ratio * sum_assign_loss  # * adaptive_cluster_ratio(e, args.epoch)
                loss = loss / args.inner_iterations
                loss.backward()

                total_cluster_loss += sum_cluster_loss.item() / args.inner_iterations
                total_loss += loss.item()
                total_cls_loss += cls_loss.item() / args.inner_iterations
                total_assign_loss += sum_assign_loss.item() / args.inner_iterations
            optimizer.step()
            if args.verbose:
                print('[Cls Loss]', total_cls_loss, '[Self-Label Loss]', total_cluster_loss, '[Assign Loss]', total_assign_loss, '[Total loss]',
                      total_loss)

            # valid
            model.eval()
            with torch.no_grad():
                output, embedding, cluster_loss, assign_loss = model(g, feat, label=label, train_mask=train_mask)
                pred_label = torch.max(output, dim=-1)[1].cpu().data.numpy()

                train_acc = accuracy_score(label[train_mask].cpu().data.numpy(), pred_label[train_mask])
                valid_acc = accuracy_score(label[val_mask].cpu().data.numpy(), pred_label[val_mask])
                val_loss = loss_func(output[val_mask], label[val_mask]).item()
                test_acc = accuracy_score(label[test_mask].cpu().data.numpy(), pred_label[test_mask])

                if args.verbose:
                    print('[EPOCH] %d' % e)
                    print('train acc', train_acc)
                    print('valid acc', valid_acc)
                    print('test acc', test_acc)

                if valid_acc > best_valid or (valid_acc == best_valid and test_acc > best_test):
                    best_valid = valid_acc
                    best_test = test_acc
                    best_epoch = e
        print("Best Epoch %s Valid ACC %s TEST ACC %s" % (best_epoch, best_valid, best_test))
        print()
        valid_container.append(best_valid)
        test_container.append(best_test)
        epoch_container.append(best_epoch)
    print("%s Runs\n[BEST] Valid ACC %.5f +/- %.5f\n[BEST] TEST ACC %.5f +/- %.5f" % (
        len(random_seed_set), np.mean(valid_container), np.std(valid_container), np.mean(test_container),
        np.std(test_container)
    ))


if __name__ == '__main__':
    main()
