import torch.nn as nn
import torch


"""
b: batch
s: sequence
d: feature dim
p: prediction length
"""


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_size,
        dropout,
        use_layer_norm,
    ):
        """Pytorch module implementing the Residual Block from the TiDE paper."""
        super().__init__()

        # dense layer with ReLU activation with dropout
        self.dense = nn.Sequential(
            nn.Linear(input_dim, hidden_size),  # d -> h
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),  # h -> d'
            nn.Dropout(dropout),
        )
        # linear skip connection from input to output of self.dense
        self.skip = nn.Linear(input_dim, output_dim)

        # layer normalization as output
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None

    def forward(self, x):
        # residual connection
        # print("x:", x.shape)
        y1 = self.dense(x)
        # print("y1:", y1.shape)
        y2 = self.skip(x)
        # print("y2:", y2.shape)

        x = y1 + y2

        # layer normalization
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return x


class TideModel(nn.Module):
    def __init__(
        self,
        seq_len,
        n_feature,
        hidden_dims,
        output_dims,
        dropout,
        n_block,
        pred_len,
        target_idx,
        options,
    ):
        super(TideModel, self).__init__()
        use_layer_norm = options["use_layer_norm"]
        feat_proj_hidden_size = hidden_dims[0]  # for feature projection
        feat_proj_output_size = output_dims[0]

        self.feature_projection = _ResidualBlock(
            input_dim=n_feature,
            output_dim=feat_proj_output_size,
            hidden_size=feat_proj_hidden_size,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )

        encoder_layers = []
        # for _ in range(n_block):
        #     encoder_layers.append(
        #         _ResidualBlock(
        #             seq_len,
        #         )
        #     )

        # self.encoder_layers =

        # self.decoder = None
        # self.temporal_decoder = None
        # self.residual = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.feature_projection(x)
        print("x:", x.shape)
        
        
        return x


def build_model(
    seq_len,
    n_feature,
    hidden_dims,
    output_dims,
    dropout,
    n_block,
    pred_len,
    target_idx,
    options=None,
):
    model = TideModel(
        seq_len,
        n_feature,
        hidden_dims,
        output_dims,
        dropout,
        n_block,
        pred_len,
        target_idx,
        options,
    )
    return model
