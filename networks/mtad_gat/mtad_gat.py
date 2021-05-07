import dgl
import torch
from torch import nn
from IPython import embed
from dgl.nn.pytorch.conv import GATConv


class MTAD_GAT(nn.Module):
    def __init__(
        self,
        batch_size,
        window_size,
        n_dim,
        hidden_dim=32,
        num_heads=7,
        num_layers=1,
        feat_drop=0,
        attn_drop=0,
        residual=False,
    ):
        super().__init__()

        self.n_dim = n_dim
        self.window_size = window_size
        self.batch_size = batch_size
        self.feat_graph = self.__build_graph(n_dim, batch_size)
        self.time_graph = self.__build_graph(window_size, batch_size)

        self.feat_gat = GATConv(
            in_feats=window_size,
            out_feats=hidden_dim,
            num_heads=num_heads,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            residual=residual,
        )
        self.time_gat = GATConv(
            in_feats=n_dim,
            out_feats=hidden_dim,
            num_heads=num_heads,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=0.2,
            residual=residual,
        )

        self.rnn = nn.GRU(
            input_size=2 * hidden_dim + n_dim,
            hidden_size=hidden_dim,
            # num_layers=num_layers,
            batch_first=True,
        )

        self.reconst_ln == nn.Linear(hidden_dim, n_dim * window_size)
        self.forcast_ln == nn.Linear(hidden_dim, window_size)

    def __build_graph(self, num_nodes, batch_size):
        edges = []
        for batch_id in range(batch_size):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    edges.append((i + batch_id * num_nodes, j + batch_id * num_nodes))
        return dgl.graph(edges)

    def forward(self, abatch):
        real_batch_size = abatch.size()[0]
        if self.batch_size != real_batch_size:
            feat_graph = self.__build_graph(n_dim, real_batch_size)
            time_graph = self.__build_graph(window_size, real_batch_size)
        else:
            feat_graph = self.feat_graph
            time_graph = self.time_graph

        time_out = self.time_gat(time_graph, abatch.view(-1, self.n_dim)).mean(dim=1)
        feat_out = self.feat_gat(
            feat_graph, abatch.transpose(2, 1).reshape(-1, self.window_size)
        ).mean(dim=1)

        print(time_out.shape, feat_out.shape)
        time_out = time_out.reshape(real_batch_size, self.window_size, -1)
        feat_out = feat_out.reshape(real_batch_size, self.n_dim, -1)

        # print(time_out.shape, feat_out.shape)
        # time_out = time_out.mean(dim=1)
        # feat_out = feat_out.mean(dim=1)
        print(time_out.shape, feat_out.shape, abatch.shape)

        merged_out = torch.cat([time_out, feat_out, abatch], dim=-1)

        print(merged_out.shape)

        gru_out, _ = self.rnn(merged_out)
        gru_out = gru_out.mean(dim=1)

        embed()


if __name__ == "__main__":
    n_dim = 4
    window_size = 4
    batch_size = 3
    model = MTAD_GAT(batch_size=batch_size, window_size=window_size, n_dim=n_dim)

    abatch = torch.randn((batch_size, window_size, n_dim))
    model(abatch)
    # print(abatch)

    # 3 x 1