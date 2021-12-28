from torch import nn
from my_dataset import MyDataLoader
import tqdm
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import Embedding, LSTM
from torch import LongTensor


def prepare_for_lstm(X):
    x = X['x']
    length = X['length']
    label = X['y']
    # 降序变成升序
    index = torch.flip(torch.argsort(length), [0])
    x = x[index]
    length = length[index]
    pack_x = pack_padded_sequence(x, length, batch_first=True, enforce_sorted=True)
    index_list = index.cpu().numpy().tolist()
    reverse_index = []
    for i in range(len(index_list)):
        reverse_index.append(index_list.index(i))
    return pack_x, label, reverse_index


if __name__ == '__main__':
    # train_loader = MyDataLoader('../file/pre_datas_train.npy', '../file/pre_datas_train_label.npy',
    #                             batch_size=128, drop_last=True)
    # valid_loader = MyDataLoader('../file/pre_datas_valid.npy', '../file/pre_datas_valid_label.npy',
    #                             batch_size=512, drop_last=False)
    #
    # device = torch.device('cuda:0')

    seqs = ['long_str', 'tiny', 'medium']
    vocab = ['<pad>'] + sorted(set([char for seq in seqs for char in seq]))
    vectorized_seqs = [[vocab.index(tok) for tok in seq] for seq in seqs]
    embed = Embedding(len(vocab), 4)
    lstm = LSTM(input_size=4, hidden_size=5, batch_first=True)
    seq_lengths = LongTensor(list(map(len, vectorized_seqs)))
    print(seq_lengths)
    seq_tensor = Variable(torch.zeros((len(vectorized_seqs), seq_lengths.max()))).long()
    for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seqlen] = LongTensor(seq)

    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]

    embedded_seq_tensor = embed(seq_tensor)
    packed_input = pack_padded_sequence(embedded_seq_tensor, seq_lengths.cpu().numpy(), batch_first=True)
    packed_output, (ht, ct) = lstm(packed_input)
    print(packed_output)
    output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
    print(output)
