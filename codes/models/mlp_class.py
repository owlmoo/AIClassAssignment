from torch import nn
from torch.nn.functional import cross_entropy


class MlpClassifier(nn.Module):
    def __init__(self, seq_len, input_dim, device):
        # B*L*N -> B*L*20
        super(MlpClassifier, self).__init__()
        self.device = device
        self.layers1 = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 20),
            nn.ReLU(inplace=True),
        ).to(device)
        self.flatten_op = nn.Flatten(start_dim=1, end_dim=-1)
        self.layers2 = nn.Sequential(
            nn.Linear(seq_len * 20, seq_len * 20 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(seq_len * 20 // 4, seq_len * 20 // 16),
            nn.ReLU(inplace=True),
            nn.Linear(seq_len * 20 // 16, 2),
        ).to(device)
        self.init_param()

    def forward(self, X, return_loss=False):
        pred = self.layers1(X['x'].to(self.device))
        pred = self.flatten_op(pred)
        pred = self.layers2(pred)
        if return_loss:
            loss = cross_entropy(pred, X['y'].to(self.device))
        else:
            loss = None
        return pred, loss

    def init_param(self):
        for m in self.parameters():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
