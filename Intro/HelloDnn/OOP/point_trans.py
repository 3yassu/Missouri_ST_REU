#point_trans.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, K]
    Output:
        new_points:, indexed points data, [B, S, K, C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)

def square_distance(src, dst):
    """
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


class PointTransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.phi = nn.Linear(d_model, d_model, bias=False) # queries
        self.psi = nn.Linear(d_model, d_model, bias=False) # keys
        self.alpha = nn.Linear(d_model, d_model, bias=False) # values
        self.k = k

    # xyz: b x n x 3, features: b x n x f (f=d_points)
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)  # b x n x n
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx) # b x n x k x 3

        pre = features # b x n x f
        x = self.fc1(features) # b x n x d_model

        q = self.phi(x) # b x n x d_model
        k = index_points(self.psi(x), knn_idx) # b x n x k x d_model
        v = index_points(self.alpha(x), knn_idx) # b x n x k x d_model

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x d_model

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc) # b x n x k x d_model
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x d_model

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc) # b x n x d_model
        res = self.fc2(res) + pre # b x n x f
        return res, attn

"""## Transition down & Transition up"""

def farthest_point_sample_batch(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids.to(device)


class TransitionDown(nn.Module):
    def __init__(self, npoint, k, input_dim, output_dim) -> None:
        """
        npoint: target number of points after transition down
        nneighbor: number of neighbors to max pool the new features from
        input_dim: dimension of input features for each point
        outut_dim: dimension of output features for each point
        """
        super().__init__()
        self.npoint = npoint
        self.k = k
        # self.fc_linear = nn.Sequential(
        #     nn.Conv1d(input_dim, output_dim, 1),
        #     nn.BatchNorm1d(output_dim),
        #     nn.ReLU()
        # )
        self.mlp_convs = nn.ModuleList([
            nn.Conv2d(input_dim, output_dim, 1),
            nn.Conv2d(output_dim, output_dim, 1)
        ])
        self.mlp_bns = nn.ModuleList([
            nn.BatchNorm2d(output_dim),
            nn.BatchNorm2d(output_dim)
        ])

    def forward(self, xyz, features):
        """
        Input:
            xyz: input points position data, [B, N, 3]
            features: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, S, 3]
            new_features: new points feature data, [B, S, D']
        """
        fps_idx = farthest_point_sample_batch(xyz, self.npoint) # B x npoint
        torch.cuda.empty_cache()
        new_xyz = index_points(xyz, fps_idx) # B x npoint x 3
        torch.cuda.empty_cache()
        dists = square_distance(new_xyz, xyz)  # B x npoint x N
        idx = dists.argsort()[:, :, :self.k]  # B x npoint x k
        torch.cuda.empty_cache()
        grouped_xyz = index_points(xyz, idx) # B x npoint x k x 3
        torch.cuda.empty_cache()

        B, N, D = features.shape

        # new_features = features.transpose(1,2) # B x D x N
        # new_features = self.fc_linear(new_features) # B x D' x N
        # new_features = new_features.transpose(1,2) # B x N x D'
        # grouped_features = index_points(new_features, idx) # B x npoint x k x D'
        # new_features, _ = torch.max(grouped_features, 2) # B x npoint x D'

        new_features = index_points(features, idx) # B x npoint x k x D
        new_features = new_features.permute(0, 3, 2, 1) # B x D x k x npoint
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_features =  F.relu(bn(conv(new_features))) # B x D' x k x npoint
        new_features, _ = torch.max(new_features, 2) # B x D' x npoint
        new_features = new_features.transpose(1,2) # B x npoint x D'

        return new_xyz, new_features
        
class PointTransformerClassif(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = PointTransformerBlock(32, cfg.model.transformer_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(TransitionDown(npoints // 4 ** (i + 1), nneighbor, channel // 2, channel))
            self.transformers.append(PointTransformerBlock(channel, cfg.model.transformer_dim, nneighbor))

        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
        self.nblocks = nblocks

    def forward(self, x):
        xyz = x[..., :3]
        features = self.transformer1(xyz, self.fc1(x))[0]
        for i in range(self.nblocks):
            xyz, features = self.transition_downs[i](xyz, features)
            features = self.transformers[i](xyz, features)[0]
        res = self.fc2(features.mean(1))
        return res

