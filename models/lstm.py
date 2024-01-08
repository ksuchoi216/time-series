import torch.nn as nn
import torch


"""
b: batch
s: sequence
d: feature dim
p: prediction length
"""


class LstmModel(nn.Module):
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
        super(LstmModel, self).__init__()
        if not target_idx:
            raise Exception("LSTM is required with MS or S, so need target_idx")
        self.n_block = n_block
        self.target_idx = target_idx
        # self.fc_input_size = self.hidden_size * 2
        self.hidden_size = hidden_dims[0]
        # print(self.hidden_size)

        self.lstm = nn.LSTM(
            n_feature,
            self.hidden_size,
            n_block,
            batch_first=True,
            bidirectional=options["bidirectional"],
        )
        self.fc = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        h0 = torch.zeros(self.n_block, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.n_block, x.size(0), self.hidden_size).to(x.device)
        # print(f"h0, c0 {h0.shape} {c0.shape}")
        x, _ = self.lstm(x, (h0, c0))
        if self.target_idx:
            x = x[:, :, self.target_idx]

        x = self.fc(x)
        # print("x:", x.shape)

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
    model = LstmModel(
        seq_len,
        n_feature,
        hidden_dims,
        dropout,
        n_block,
        pred_len,
        target_idx,
        options,
    )
    return model
