import torch
from torch import nn
from torch.nn.functional import cross_entropy


class CNNClassifier(nn.Module):
    def __init__(self, device):
        super(CNNClassifier, self).__init__()
        self.device = device
        self.conv1 = nn.Sequential(
            nn.Conv1d(128, 64, (3,)),
            nn.ReLU()
        ).to(self.device)
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, (3,)),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.5)
        ).to(self.device)
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, (3,)),
            nn.ReLU()
        ).to(self.device)
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, (3,)),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.5)
        ).to(self.device)
        self.flatten_op = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc = nn.Sequential(
            nn.Linear(
                in_features=128 * 130,
                out_features=128
            ),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(
                in_features=128,
                out_features=2
            )
        ).to(self.device)

        self.init_param()

    def forward(self, X, return_loss=False):
        out = X['x'].to(self.device).permute(0, 2, 1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.flatten_op(out)
        result = self.fc(out)

        if return_loss:
            loss = cross_entropy(result, X['y'].to(self.device))
        else:
            loss = None
        return result, loss

    def init_param(self):
        for m in self.parameters():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)


if __name__ == '__main__':
    test_data = torch.rand((512, 532, 128))
    test_data = test_data.permute(0, 2, 1)
    print(test_data.shape)
    layers1 = nn.Sequential(
        nn.Conv1d(128, 64, (3,)),
        nn.Conv1d(64, 64, (3,)),
        nn.MaxPool1d(2),
        nn.Conv1d(64, 128, (3,)),
        nn.Conv1d(128, 128, (3,)),
        nn.MaxPool1d(2)
    )
    layers2 = nn.Sequential(
        nn.Conv1d(128, 64, (3,)),
        nn.Conv1d(64, 128, (3,)),
        nn.MaxPool1d(2),
    )
    layers3 = nn.Sequential(
        nn.Conv1d(128, 128, (3,)),
        nn.MaxPool1d(2),
    )
    after_data = layers1(test_data)
    print(after_data.shape)
