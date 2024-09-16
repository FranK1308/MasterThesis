import math
import torch
import torch.nn as nn
import torch_geometric as tg
import torch.nn.functional as F
from torch_scatter import scatter

class RFF(nn.Module):
    def __init__(self, in_features, out_features, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        self.in_features = in_features
        self.out_features = out_features

        if out_features % 2 != 0:
            self.compensation = 1
        else:
            self.compensation = 0

        B = torch.randn(int(out_features / 2) + self.compensation, in_features) * sigma
        B /= math.sqrt(2)
        self.register_buffer("B", B)

    def forward(self, x):
        x = F.linear(x, self.B)
        x = torch.cat((x.sin(), x.cos()), dim=-1)
        if self.compensation:
            x = x[..., :-1]
        return x

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, sigma={}".format(
            self.in_features, self.out_features, self.sigma
        )



class EGNNLayer(tg.nn.MessagePassing):
    def __init__(self, emb_dim, activation="relu", norm="layer", aggr="add", RFF_dim=None, RFF_sigma=None, mask=None):
        super().__init__(aggr=aggr)
        self.emb_dim = emb_dim
        self.activation = {"swish": nn.SiLU(), "relu": nn.ReLU()}[activation]
        self.norm = {"layer": torch.nn.LayerNorm, "batch": torch.nn.BatchNorm1d, "none": nn.Identity}[norm]
        self.RFF_dim = RFF_dim
        self.RFF_sigma = RFF_sigma
        self.mask = mask

        self.mlp_msg = nn.Sequential(
            nn.Linear(2 * emb_dim + 1 if self.RFF_dim is None else 2 * emb_dim + RFF_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
            nn.Linear(emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
        )

        self.mlp_upd = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
            nn.Linear(emb_dim, emb_dim),
            self.norm(emb_dim) if norm != "none" else nn.Identity(),
            self.activation,
        )
        if self.RFF_dim is not None:
            self.RFF = RFF(1, RFF_dim, RFF_sigma)

    def forward(self, h, edge_index, distances, mask=None):
        self.mask = mask
        out = self.propagate(edge_index, h=h, distances=distances, mask=mask)
        return out

    def message(self, h_i, h_j, distances):
        if self.RFF_dim is not None:
            distances = self.RFF(distances)

        msg = torch.cat([h_i, h_j, distances], dim=-1)
        msg = self.mlp_msg(msg)
        return msg

    def update(self, aggr_out, h):
        msg_aggr = aggr_out
        upd_out = self.mlp_upd(torch.cat([h, msg_aggr], dim=-1))
        if self.mask is not None:
            upd_out = torch.where(self.mask.unsqueeze(-1), upd_out, h)
        return upd_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})"



class EGNN_Full(nn.Module):
    def __init__(
            self,
            depth=5,
            hidden_features=128,
            node_features=1,
            out_features=1,
            activation="relu",
            norm="layer",
            aggr="sum",
            pool="add",
            residual=True,
            RFF_dim=None,
            RFF_sigma=None,
            mask=None,
            **kwargs
    ):
        super().__init__()

        # Store the parameters as attributes
        self.depth = depth
        self.hidden_features = hidden_features
        self.node_features = node_features
        self.out_features = out_features
        self.activation = activation
        self.norm = norm
        self.aggr = aggr
        self.pool = pool
        self.residual = residual
        self.RFF_dim = RFF_dim
        self.RFF_sigma = RFF_sigma
        self.mask = mask

        # Name of the network
        self.name = "EGNN"
        
        # Embedding lookup for initial node features
        self.emb_in = nn.Linear(node_features, hidden_features)
        
        self.make_dist = PBCConvLayer()
        
        # Stack of GNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(depth):
            self.convs.append(EGNNLayer(hidden_features, activation, norm, aggr, RFF_dim, RFF_sigma, mask))

        # Global pooling/readout function
        self.pool = {"mean": tg.nn.global_mean_pool, "add": tg.nn.global_add_pool}[pool]

        # Predictor MLP
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features, out_features)
        )
        self.residual = residual


    def forward(self, batch):
        h = self.emb_in(batch.x)  # (n,) -> (n, d)
        #pos = batch.pos  # (n, 3)
        batch.pos = torch.autograd.Variable(batch.pos, requires_grad=True)
        distances = self.make_dist(batch.pos, batch.edge_index, batch.cell_offset ,batch.unit_cell[:3]).unsqueeze(1)
        
        for conv in self.convs:
            # Message passing layer
            h_update = conv(h, batch.edge_index, distances)

            # Update node features (n, d) -> (n, d)
            h = h + h_update if self.residual else h_update

            # Update node coordinates (no residual) (n, 3) -> (n, 3)
        out = h
        if self.pool is not None:
            out = self.pool(h, batch.batch)
        
        return self.pred(out)



class PBCConvLayer(nn.Module):
    def __init__(self):
        super(PBCConvLayer, self).__init__()

    def forward(self, pos, edge_index, offsets, cell_vectors):
            # pos: Positions of nodes (N, 3)
            # edge_index: Indices of edges (2, E)
            # offsets: Offsets for PBC (E, 3), values like -1, 0, 1 for each edge considering PBC
            # cell_vectors: Cell vectors defining the unit cell (3, 3)

            # Calculate edge vectors considering initial positions
            to_move = pos[edge_index[1]]  # Shape (E, 3)

            # Apply PBC corrections using offsets and cell vectors
            pbc_adjustments = torch.matmul(offsets, cell_vectors)
            corrected = to_move - pbc_adjustments        
            # Compute distances
            distances = torch.linalg.vector_norm(corrected - pos[edge_index[0]],dim=-1)

            return distances

