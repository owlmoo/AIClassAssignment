import torch
import os
import sys
from time import localtime
from tqdm import tqdm
from tensorboardX import SummaryWriter
from my_dataset import MyDataLoader

from models.mlp_class import MlpClassifier
from models.cnn_class import CNNClassifier
# from models.cnn_class2 import CNNClassifier
# from models.cnn_class3 import CNNClassifier
from models.lstm_class import LSTMClassifier


def valid(model, test_loader):
    model.eval()
    total_acc = 0
    total_num = 0
    tf_num = 0
    for i, data_item in tqdm(enumerate(test_loader)):
        with torch.no_grad():
            pred, _ = model(data_item, return_loss=False)
            # 取得预测的标签
            pred = torch.argmax(pred, dim=-1)
            # 计算预测和标签相同的个数
            acc = pred.cpu() == data_item['y']
            total_acc += (torch.sum(acc))
            total_num += (pred.shape[0])
            # 找到标签中所有为正常的数据
            normal_index = torch.where(data_item['y'] == 1)
            # 取得对应的预测值
            normal_pred = pred[normal_index]
            normal_label = data_item['y'][normal_index]
            # 找到所有标签为正常,预测为异常的数据
            tf_num += (torch.sum(normal_pred.cpu() != normal_label))
    print("total_acc:{} total_tf_num:{} total:num:{}".format(total_acc, tf_num, total_num))
    print("acc_rate:{} tf_rate:{}".format(total_acc / total_num, tf_num / total_num))
    return total_acc / total_num, tf_num / total_num


def train(model, train_loader, opt, steps):
    model.train()
    for i, data_item in tqdm(enumerate(train_loader)):
        _, loss = model(data_item, return_loss=True)
        writer.add_scalar('train_loss', loss.data, steps+1)
        steps += 1
        opt.zero_grad()
        loss.backward()
        opt.step()
    return steps

# 对模型进行测试
def test_model(model, path, test_loader):
    if path is None:
        pass
    else:
        model.load_state_dict(torch.load(path))
    acc_rate, tf_rate = valid(model, test_loader)
    print('acc_rate:{}'.format(acc_rate))
    print('tf_rate:{}'.format(tf_rate))

# 对模型进行训练
def train_model(model, train_loader, valid_loader, total_epoch=20):
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    steps = 0
    checkpoint_path = '../checkpoint'
    ti = localtime()
    write_path = '/lstm_{}.tar'.format(ti.tm_mday)
    print('start to train:')
    best_acc = 0.0
    for epoch_ in range(total_epoch):
        steps = train(model, train_loader, opt, steps)
        valid_acc, valid_tf = valid(model, valid_loader)
        writer.add_scalar('valid_acc', valid_acc, epoch_+1)
        writer.add_scalar('valid_tf', valid_tf, epoch_+1)
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), checkpoint_path + write_path)
    print('finished train')


if __name__ == '__main__':
    train_loader = MyDataLoader('../file/pre_datas_train.npy', '../file/pre_datas_train_label.npy',
                                batch_size=128, drop_last=True)
    valid_loader = MyDataLoader('../file/pre_datas_valid.npy', '../file/pre_datas_valid_label.npy',
                                batch_size=512, drop_last=False)

    device = torch.device('cuda:0')
    # writer = SummaryWriter('../log')

    # LSTM的参数
    lstm_layer = 2
    bidirectional = True

    modelType = sys.argv[1]
    epochs = int(sys.argv[2])

    if modelType == 'LSTM':
        model = LSTMClassifier(input_dim=128, device=device, lstm_layers=lstm_layer, bidirectional=bidirectional)
    elif modelType == 'CNN':
        model = CNNClassifier(device=device)
    elif modelType == 'MLP':
        model = MlpClassifier(seq_len=532, input_dim=128, device=device)
    else:
        model = None
    # train_model(model, train_loader, valid_loader, total_epoch=epochs)
    # writer.close()
    test_loader = MyDataLoader('../file/pre_datas_train.npy', '../file/pre_datas_train_label.npy',
                               batch_size=512, drop_last=False)

    model_path = "../checkpoint/lstm_28_12h58m.tar"
    test_model(model, model_path, test_loader)
