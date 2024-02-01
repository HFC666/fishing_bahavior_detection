import torch
import torch.nn as nn
from tcn import TemporalConvNet
from attention import Attention
from utils import compute_similarity
from sklearn.cluster import KMeans
from drnn import DRNN


class DTC(nn.Module):
    def __init__(self, input_dim, num_channels, hidden_dim, num_heads, cluster_num, similarity, base_type=0,
                 combination=True):
        super().__init__()
        self.combination = combination
        self.similarity = similarity
        self.cluster_num = cluster_num
        reverse_channels = num_channels.copy()
        reverse_channels.reverse()
        reverse_channels.append(input_dim)
        reverse_channels = reverse_channels[1:]
        reverse_channels.insert(0, hidden_dim)
        if base_type == 0:
            self.encoder = TemporalConvNet(input_dim, num_channels)
            self.decoder = TemporalConvNet(num_channels[-1], reverse_channels, dilation=False)
        else:
            self.encoder = DRNN(input_dim, num_channels)
            self.decoder = DRNN(num_channels[-1], reverse_channels, dilation=False)
        if self.combination:
            dim_sum = sum(num_channels)
            self.com = nn.Linear(dim_sum, num_channels[-1])
        self.rnn = nn.RNN(hidden_dim, num_channels[-1], batch_first=True)
        self.attention = Attention(num_heads, num_channels[-1], hidden_dim)

        self.cluster = nn.Parameter(torch.randn(cluster_num, num_channels[-1]), requires_grad=True)
        self.cluster_dim = num_channels[-1]

    def forward(self, xh):
        hn, hs = self.encoder(xh)

        if self.combination:
            hn = self.com(hs)

        hn, _ = self.rnn(self.attention(hn))
        return hn

    def sim(self, f):
        # f的维度：batch_size x  latent_dim
        # center的维度:  n_cluster x latent_dim
        q = compute_similarity(f, self.cluster, self.similarity)
        q = torch.pow(1 + q, -1)
        return q / torch.sum(q, dim=-1).unsqueeze(1)
    
    def init_centroids(self, x):
        
        z = self(x.detach())[:, -1, :].squeeze()
        
        z_np = z.detach().numpy()

        cluster = KMeans(n_clusters = self.cluster_num, random_state = 0).fit(z_np)

        self.cluster = nn.Parameter(torch.tensor((cluster.cluster_centers_)))

    def loss(self, x):
        feature = self(x)
        f = feature[:, -1, :].squeeze()
        q = self.sim(f)
        p = self.p(q)
        loss1 = torch.sum(p * torch.log(p / q)) / p.shape[0]
        recon, _ = self.decoder(feature)
        loss2 = torch.mean(torch.sum(torch.pow(x - recon, 2), dim=-1))
        loss3 =  - torch.mean(torch.log(torch.pow(1 - 2 * p[:,0], 2)))
        return loss1, loss2, loss3

    def p(self, similarity):
        f = torch.sum(similarity, dim=0).unsqueeze(0)
        p = torch.pow(similarity, 2) / f
        p = p / torch.sum(p, dim=-1).unsqueeze(-1)
        return p
