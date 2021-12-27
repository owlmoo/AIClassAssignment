from torch import nn
import torch
from torch.nn.functional import cross_entropy
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, device, lstm_layers=1, bidirectional=True):
        # B*L*N -> B*L*20
        self.device = device
        super(LSTMClassifier, self).__init__()
        self.layers1 = nn.LSTM(input_dim, input_dim, lstm_layers, batch_first=True, bidirectional=bidirectional).to(
            device)
        mlp_input_dim = input_dim * lstm_layers
        if bidirectional:
            mlp_input_dim = mlp_input_dim * 2
        self.layers2 = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_input_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_input_dim * 4, mlp_input_dim * 16),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_input_dim * 16, mlp_input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_input_dim, input_dim // 16),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 16, input_dim // 64),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 64, 2)
        ).to(device)
        self.init_param()

    def prepare_for_lstm(self, X):
        x = X['x']
        length = X['length']
        label = X['y']
        # torch.flip函数在给定的维度上将数组翻转
        # torch.argsort函数得出一个索引列表，按照这个索引列表取出元素可以得到排好序的列表
        # 降序变成升序
        index = torch.argsort(length, descending=True)
        # index = torch.flip(torch.argsort(length), [0])
        x = x[index]
        length = length[index]
        pack_x = pack_padded_sequence(x, length, batch_first=True, enforce_sorted=True)
        index_list = index.cpu().numpy().tolist()
        reverse_index = []
        for i in range(len(index_list)):
            reverse_index.append(index_list.index(i))
        return pack_x, label, reverse_index

    def forward(self, X, return_loss=False):
        x, label, reverse_index = self.prepare_for_lstm(X)
        pred, (h_n, c_n) = self.layers1(x.to(self.device))
        h_n = h_n.permute(1, 0, 2)
        h_n = torch.flatten(h_n, 1, -1)
        h_n = h_n[reverse_index, :]

        pred = self.layers2(h_n)
        if return_loss:
            loss = cross_entropy(pred, label.to(self.device))
        else:
            loss = None
        return pred, loss

    def init_param(self):
        for m in self.parameters():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
