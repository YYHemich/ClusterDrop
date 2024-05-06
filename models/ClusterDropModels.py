import os
import sys
import torch.nn as nn
import torch
import torch.nn.functional as F
import gnn
from dgl.nn import EdgeWeightNorm


class ClusterDropV2(nn.Module):
    weight_normalizer = EdgeWeightNorm(norm='both')

    def __init__(self, n_clusters, drop_rate, prototype_dim, input_dim, epsilon=0.05, sinkhorn_iterations=3,
                 assign_sigma=1.0, assign_beta=0.5, orthogonal=False):
        super(ClusterDropV2, self).__init__()
        self.c_prototype = nn.Linear(prototype_dim, n_clusters, bias=False)
        self.project_head = nn.Linear(input_dim, prototype_dim)
        self.epsilon = epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        self.drop_rate = drop_rate
        self.assign_loss_func = ClusterAssignmentLoss(assign_sigma, assign_beta)
        self.cluster_var_loss_func = ClusterVarLoss()
        self.align_loss_func = AlignLoss()
        if orthogonal:
            self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.orthogonal_(self.c_prototype.weight)

    def sinkhorn_knopp(self, out):
        """
        This part of codes borrows from https://github.com/facebookresearch/swav. Thanks very much.
        """
        Q = torch.exp(out / self.epsilon).t()
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(self.sinkhorn_iterations):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(self, graph, emb, nodrop_emb, label, train_mask, normalize=True):
        if self.training:
            return self._drop(graph, emb, nodrop_emb, label, train_mask, normalize)
        return self._inference(graph, emb, label, train_mask)

    def _drop(self, graph, dropout_emb, nodrop_emb, label, train_mask, normalize=True):
        if self.drop_rate <= 0:
            weight = self.weight_normalizer(graph, torch.ones((graph.num_edges(),), device=graph.device))
            return graph, weight, torch.tensor(0), torch.tensor(0)

        graph = graph.clone()
        nodrop_emb_detach = nodrop_emb.detach()
        # normalize prototype
        with torch.no_grad():
            w = self.c_prototype.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.c_prototype.weight.copy_(w)
            nodrop_emb_detach = F.normalize(nodrop_emb_detach, dim=-1, p=2)
            no_drop_cluster_out = self.c_prototype(nodrop_emb_detach)

        project_emb = dropout_emb
        if normalize:
            project_emb = F.normalize(project_emb, dim=-1, p=2)

        cluster_out = self.c_prototype(project_emb)

        with torch.no_grad():
            # assignment_code = self.sinkhorn_knopp(no_drop_cluster_out.detach()).detach()
            assignment_code = self.sinkhorn_knopp(cluster_out.detach()).detach()

        # Hard code loss
        self_label_loss = F.cross_entropy(cluster_out, torch.max(assignment_code, dim=-1)[1])
        # soft code loss
        # self_label_loss = -(assignment_code * F.log_softmax(cluster_out, dim=-1)).sum(dim=-1).mean()

        cluster_assignment_loss = self.assign_loss_func(cluster_out, project_emb, label, train_mask)
        # cluster_assignment_loss = self.assign_loss_func(no_drop_cluster_out, nodrop_emb, label, train_mask)

        u, v = graph.edges()
        edge_idx = torch.arange(graph.num_edges(), device=graph.device)
        src2dst = edge_idx
        dst2src = graph.edge_ids(v, u)
        single_edge_mask = (src2dst == torch.vstack([src2dst, dst2src]).min(dim=0)[0])
        # ignore self loop
        not_self_loop_mask = u != v
        single_edge_mask = single_edge_mask & not_self_loop_mask
        single_edge_idx = edge_idx[single_edge_mask]

        with torch.no_grad():
            guided_scores = self._compute_assignment_score(graph, no_drop_cluster_out.detach(), single_edge_idx)
            # guided_scores = self._compute_assignment_score(graph, cluster_out.detach())

        sample_prob = guided_scores[single_edge_mask]
        sample_prob /= sample_prob.sum()
        # print('sample prob', sample_prob.max(), sample_prob.mean(), sample_prob.min())
        remove_size = int(len(sample_prob) * self.drop_rate)
        # random remove
        remove_agent_ids = torch.multinomial(sample_prob, num_samples=remove_size, replacement=False)
        # Fix Remove
        # remove_agent_ids = torch.sort(sample_prob, descending=True)[1][:remove_size]

        remove_agent_mask = torch.zeros(len(sample_prob)).bool()
        remove_agent_mask[remove_agent_ids] = True

        edge_weight = torch.ones(graph.num_edges(), device=graph.device)
        remove_eids = edge_idx[single_edge_mask][remove_agent_mask]
        edge_weight[remove_eids] = 0
        edge_weight = edge_weight * edge_weight[dst2src]
        remove_eids = edge_idx[edge_weight == 0]
        # hard remove
        graph.remove_edges(remove_eids)
        weight = self.weight_normalizer(graph, torch.ones((graph.num_edges(),), device=graph.device))
        # soft remove
        # weight[remove_eids] = 0.
        # else:
        #     weight = self.weight_normalizer(graph, torch.ones((graph.num_edges(),), device=graph.device))
        #     self_label_loss = torch.tensor(0)
        #     cluster_var_loss = torch.tensor(0)
        #     # return graph, weight, torch.tensor(0), torch.tensor(0)
        return graph, weight, self_label_loss, cluster_assignment_loss

    def _inference(self, graph, emb, label, train_mask):
        graph = graph.clone()
        weight = self.weight_normalizer(graph, torch.ones((graph.num_edges(),), device=graph.device))
        with torch.no_grad():
            project_emb = emb
            project_emb = F.normalize(project_emb, dim=-1, p=2)
            cluster_out = self.c_prototype(project_emb)
        return graph, weight, torch.tensor(0), torch.tensor(0)

    def _compute_assignment_score(self, graph, assignment_code, single_edge_idx, mini_batch_num=10):
        def js_divergence(edges):
            median = ((edges.src['assignment_prob'] + edges.dst['assignment_prob']) / 2).log()
            # div = F.kl_div(median, edges.src['assignment_prob'], reduction='none').sum(dim=-1) +\
            #       F.kl_div(median, edges.dst['assignment_prob'], reduction='none').sum(dim=-1)
            # inplace operation
            div = F.kl_div(median, edges.src['assignment_prob'], reduction='none').sum(dim=-1)
            div.add_(F.kl_div(median, edges.dst['assignment_prob'], reduction='none').sum(dim=-1))
            return {'js_divergence': div.clamp(min=0) / 2}
        with graph.local_scope():
            assignment_prob = F.softmax(assignment_code, dim=-1)
            graph.ndata['assignment_prob'] = assignment_prob
            mini_batch_size = len(single_edge_idx) // mini_batch_num + int(len(single_edge_idx) % mini_batch_num != 0)
            for b_idx in range(mini_batch_num):
                graph.apply_edges(
                    js_divergence,
                    edges=single_edge_idx[b_idx * mini_batch_size: min((b_idx + 1) * mini_batch_size, graph.num_edges())]
                )
            prob = graph.edata['js_divergence']
        return prob


class ClusterVarLoss(nn.Module):
    def __init__(self):
        super(ClusterVarLoss, self).__init__()

    def forward(self, embedding, cluster_out):
        cluster_predict_label = torch.max(cluster_out, dim=-1)[1]
        cluster_predict_onehot = F.one_hot(cluster_predict_label, num_classes=cluster_out.shape[-1]).float().detach()
        cluster_mean = cluster_predict_onehot.t() @ embedding / cluster_predict_onehot.t().sum(dim=-1, keepdim=True).clamp(min=1).detach()
        second_momentum = torch.square(embedding).sum(dim=-1, keepdim=True)
        second_momentum = cluster_predict_onehot.t() @ second_momentum / cluster_predict_onehot.t().sum(dim=-1, keepdim=True).clamp(min=1).detach()
        weighted = F.normalize(cluster_predict_onehot.sum(dim=0), dim=0, p=1).detach()
        return weighted @ (second_momentum - torch.square(cluster_mean).sum(dim=-1, keepdim=True))


class AlignLoss(nn.Module):
    def __init__(self):
        super(AlignLoss, self).__init__()

    def forward(self, embedding, cluster_out):
        cluster_predict_label = torch.max(cluster_out, dim=-1)[1]
        same_cluster_mask = cluster_predict_label.reshape((-1, 1)) == cluster_predict_label.reshape((1, -1))
        self_dot_mask = (1 - torch.eye(len(embedding), device=embedding.device)).bool()
        return (embedding @ embedding.t())[same_cluster_mask & self_dot_mask].mean()


class ClusterAssignmentLoss(nn.Module):
    def __init__(self, sigma=1., beta=0.5):
        super(ClusterAssignmentLoss, self).__init__()
        self.sigma = sigma
        self.beta = beta

    def forward(self, sample2cluster_out, embedding, label, train_mask, mini_batch_num=1):
        label = label.clone()
        with torch.no_grad():
            predict_label = torch.max(sample2cluster_out, dim=-1)[1]
            predict_one_hot = F.one_hot(predict_label, num_classes=sample2cluster_out.shape[-1]).float()
            num_classes = torch.max(label) + 1
            label[~train_mask] = num_classes
            train_label_onehot = F.one_hot(label, num_classes=num_classes + 1).float()
            cluster_composition = predict_one_hot.t() @ train_label_onehot  # shape (n_cluster, num_classes + 1)
            has_known_cluster_idx = torch.arange(sample2cluster_out.shape[-1], device=embedding.device)[cluster_composition[:, :-1].sum(dim=-1) > 0]
        train_emb = embedding[train_mask]
        supervised_cluster_center = (train_emb.t() @ train_label_onehot[train_mask][:, :-1]) / train_label_onehot[train_mask][:, :-1].sum(dim=0)  # shape (emb_d, num_classes)
        supervised_cluster_center = supervised_cluster_center.t().detach()

        # compute the known anchor of each class
        container = []
        for known_cluster in has_known_cluster_idx:
            cluster_train_mask = (predict_label == known_cluster) & train_mask
            cluster_train_emb = embedding[cluster_train_mask]
            cluster_train_label_one_hot = train_label_onehot[cluster_train_mask][:, :-1]  # N * n_classes
            in_cluster_centers = (cluster_train_label_one_hot.t() @ cluster_train_emb) / cluster_train_label_one_hot.t().sum(dim=-1, keepdim=True).clamp(min=1)
            container += [in_cluster_centers.unsqueeze(0)]
        cluster_inner_centers = torch.vstack(container).detach()  # shape [n_known_cluster * n_class * emb_d]

        cluster_center_container = []
        for i, known_cluster_idx in enumerate(has_known_cluster_idx):
            cluster_emb = embedding[(predict_label == known_cluster_idx) & (~train_mask)]
            weighted = self.probability_func(cluster_emb, cluster_inner_centers[i])
            tmp_cluster_center = weighted.t() @ cluster_emb  # shape [n_classes, d_emb]
            cluster_center_container += [tmp_cluster_center.unsqueeze(0)]
        cluster_centers = torch.vstack(cluster_center_container)  # [num_known_cluster, n_classes, d_emb]

        l2_loss = (cluster_centers * cluster_centers).sum(dim=-1) + (supervised_cluster_center * supervised_cluster_center).sum(dim=-1) - 2 * (cluster_centers * supervised_cluster_center).sum(dim=-1)  # [n_known_cluster, n_classes]
        # soft cluster weight
        known_cluster_composition = cluster_composition[has_known_cluster_idx][:, :-1]  # [n_known_cluster, n_classes]
        known_cluster_composition = F.normalize(known_cluster_composition, p=1, dim=-1)
        # hard code
        loss = (l2_loss * known_cluster_composition).sum(dim=-1).mean()
        # regularizer
        in_cluster_class_dist = torch.bmm(cluster_centers, cluster_centers.transpose(-1, -2))
        dist_mask = 1 - torch.eye(in_cluster_class_dist.shape[-1], device=in_cluster_class_dist.device)
        dist_mask = dist_mask.bool()
        in_cluster_class_dist = torch.square(in_cluster_class_dist[:, dist_mask])
        dist_regularize_error = in_cluster_class_dist.mean()
        return loss + self.beta * dist_regularize_error

    def probability_func(self, cluster_embs, cluster_anchors):
        distance_matrix = -(cluster_embs * cluster_embs).sum(dim=-1, keepdim=True) - (cluster_anchors * cluster_anchors).sum(dim=-1) + 2 * cluster_embs @ cluster_anchors.t()
        distance_matrix = distance_matrix / self.sigma
        probability = F.softmax(distance_matrix, dim=0)
        return probability


class ClusterDropGCN(gnn.GCN):
    def __init__(self, in_feat, hidden_dim, layers, num_classes, args):
        super().__init__(in_feat, hidden_dim, layers, num_classes, dropout=args.gnn_dropout, norm_type='none', batch_norm=args.bn)
        drop_edges_li = [
            ClusterDropV2(
                n_clusters=args.n_clusters,
                drop_rate=args.drop_rate,
                # prototype_dim=min(tmp[i - 1], args.prototype_dim),
                prototype_dim=in_feat,
                input_dim=in_feat,
                epsilon=args.sinkhorn_epsilon,
                sinkhorn_iterations=args.sinkhorn_iterations,
                assign_sigma=args.assign_sigma,
                assign_beta=args.assign_beta,
                orthogonal=args.orthogonal
            )
        ] + [
            ClusterDropV2(
                n_clusters=args.n_clusters,
                drop_rate=args.drop_rate,
                # prototype_dim=min(tmp[i - 1], args.prototype_dim),
                prototype_dim=hidden_dim,
                input_dim=hidden_dim,
                epsilon=args.sinkhorn_epsilon,
                sinkhorn_iterations=args.sinkhorn_iterations,
                assign_sigma=args.assign_sigma,
                assign_beta=args.assign_beta,
                orthogonal=args.orthogonal
            ) for _ in range(layers-1)
        ]
        self.drop_edges = nn.ModuleList(drop_edges_li)
        if args.set_zero_except_first:
            self.drop_edges[0].drop_rate = 0.
            for i in range(2, len(self.drop_edges)):
                self.drop_edges[i].drop_rate = 0.
        assert len(self.drop_edges) == len(self.convs) == len(self.bns) + 1

    def forward(self, graph, h, weight=None, **kwargs):
        label = kwargs['label'].to(h.device)
        train_mask = kwargs['train_mask'].to(h.device)
        cluster_loss_sum = []
        cluster_assign_loss_sum = []
        # init drop
        dropout_h = F.dropout(h, p=0, training=self.training)
        g, weight, loss, cluster_assign_loss = self.drop_edges[0](graph, dropout_h, h, label, train_mask)
        cluster_loss_sum.append(loss)
        cluster_assign_loss_sum.append(cluster_assign_loss)
        # GNN conv
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(g, dropout_h, edge_weight=weight)
            h = self.bns[i](h)
            h = F.relu(h)
            dropout_h = F.dropout(h, p=self.dropout, training=self.training)
            g, weight, loss, cluster_assign_loss = self.drop_edges[i+1](graph, dropout_h, h, label, train_mask)
            cluster_loss_sum.append(loss)
            cluster_assign_loss_sum.append(cluster_assign_loss)
        # output
        h = self.convs[-1](g, dropout_h, edge_weight=weight)
        return h, h, cluster_loss_sum, cluster_assign_loss_sum


class ClusterDropGCNWithLinear(gnn.GCNWithLinear):
    def __init__(self, in_feat, hidden_dim, layers, num_classes, args):
        super().__init__(in_feat, hidden_dim, layers, num_classes, dropout=args.gnn_dropout, norm_type='none', batch_norm=args.bn)
        drop_edges_li = [
            ClusterDropV2(
                n_clusters=args.n_clusters,
                drop_rate=args.drop_rate,
                # prototype_dim=min(tmp[i - 1], args.prototype_dim),
                prototype_dim=hidden_dim,
                input_dim=hidden_dim,
                epsilon=args.sinkhorn_epsilon,
                sinkhorn_iterations=args.sinkhorn_iterations,
                assign_sigma=args.assign_sigma,
                assign_beta=args.assign_beta,
                orthogonal=args.orthogonal
            ) for _ in range(layers)
        ]
        self.drop_edges = nn.ModuleList(drop_edges_li)
        if args.set_zero_except_first:
            self.drop_edges[0].drop_rate = 0.
            for i in range(2, len(self.drop_edges)):
                self.drop_edges[i].drop_rate = 0.
        assert len(self.drop_edges) == len(self.convs) == len(self.bns) - 1

    def forward(self, graph, h, weight=None, **kwargs):
        label = kwargs['label'].to(h.device)
        train_mask = kwargs['train_mask'].to(h.device)
        cluster_loss_sum = []
        cluster_assign_loss_sum = []

        # init drop
        h = self.input_proj(h)
        h = self.bns[0](h)
        h = F.relu(h, inplace=True)
        dropout_h = F.dropout(h, p=self.dropout, training=self.training)
        g, weight, loss, cluster_assign_loss = self.drop_edges[0](graph, dropout_h, h, label, train_mask)
        cluster_loss_sum.append(loss)
        cluster_assign_loss_sum.append(cluster_assign_loss)
        # GNN conv
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(g, dropout_h, edge_weight=weight)
            h = self.bns[i + 1](h)
            h = F.relu(h)
            dropout_h = F.dropout(h, p=self.dropout, training=self.training)
            g, weight, loss, cluster_assign_loss = self.drop_edges[i + 1](graph, dropout_h, h, label, train_mask)
            cluster_loss_sum.append(loss)
            cluster_assign_loss_sum.append(cluster_assign_loss)
        # output
        h = self.convs[-1](g, dropout_h, edge_weight=weight)
        h = self.bns[-1](h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.output_proj(h)
        return h, h, cluster_loss_sum, cluster_assign_loss_sum


GNN_MODEL_CONSTURCTOR = {
    'GCN': ClusterDropGCN,
    'GCNLinear': ClusterDropGCNWithLinear
    # 'GraphSAGE': GraphSAGEModel,
    # 'GAT': GATModel,
    # 'JKNet': JKNetModel,
    # 'NoLinSAGE': NoLinearGraphSAGEModel
}
