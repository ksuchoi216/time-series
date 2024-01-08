import torch.nn as nn
import torch


"""
b: batch
s: sequence
d: feature dim
p: prediction length
"""


class _MixingBlock(nn.Module):
    def __init__(self, seq_len, n_feature, hidden_size, dropout):
        super(_MixingBlock, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=1)
        self.lin_time = nn.Linear(in_features=seq_len, out_features=seq_len)
        self.lin_feat1 = nn.Linear(in_features=n_feature, out_features=hidden_size)
        self.lin_feat2 = nn.Linear(in_features=hidden_size, out_features=n_feature)

        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        # Time mixing
        y = x.unsqueeze(1)  # b, 1, s, d
        y = self.bn(y)
        y = y.squeeze(1)  # b, s, d
        y = y.permute(0, 2, 1)  # b, d, s -> same time
        y = self.act(self.lin_time(y))
        y = y.permute(0, 2, 1)  # b, s, d
        y = self.dropout(y)
        x = x + y  # residual

        # Feature mixing
        y = x.unsqueeze(1)  # b, 1, s, d
        y = self.bn(y)
        y = y.squeeze(1)  # b, s, d
        y = self.act(self.lin_feat1(y))
        y = self.dropout(y)
        y = self.lin_feat2(y)
        y = self.dropout(y)
        x = x + y  # residual

        return x


class TSMixerModel(nn.Module):
    def __init__(
        self,
        seq_len,
        n_feature,
        hidden_dims,
        dropout,
        n_block,
        pred_len,
        target_idx,
        options,
    ):
        super(TSMixerModel, self).__init__()
        hidden_size = hidden_dims[0]

        mixing_layers = []
        for _ in range(n_block):
            mixing_layers.append(_MixingBlock(seq_len, n_feature, hidden_size, dropout))

        self.mixing_layers = nn.ModuleList(mixing_layers)
        self.temp_proj_layer = nn.Linear(in_features=seq_len, out_features=pred_len)
        self.target_idx = target_idx

    def forward(self, x):
        for mixing_layer in self.mixing_layers:
            x = mixing_layer(x)

        if self.target_idx:
            x = x[:, :, self.target_idx]
            x = self.temp_proj_layer(x)
        else:
            x = x.permute(0, 2, 1)  # b, d, s
            x = self.temp_proj_layer(x)
            x = x.permute(0, 2, 1)  # b, p, d

        return x


def build_model(
    seq_len,
    n_feature,
    hidden_dims,
    dropout,
    n_block,
    pred_len,
    target_idx,
    options=None,
):
    model = TSMixerModel(
        seq_len, n_feature, hidden_dims, dropout, n_block, pred_len, target_idx, options
    )
    return model
