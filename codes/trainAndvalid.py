import torch
import os
from time import localtime
from tqdm import tqdm
from my_dataset import MyDataLoader
from models.lstm_class import LSTMClassifier
from models.cnn_class import CNNClassifier
from models.mlp_class import MlpClassifier

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
            # 找到所有标签我为正常,预测为异常的数据
            tf_num += (torch.sum(normal_pred.cpu() != normal_label))
    print("total_acc:{} total_tf_num:{} total:num:{}".format(total_acc, tf_num, total_num))
    return total_acc / total_num, tf_num / total_num


def train(model, train_loader, opt):
    model.train()
    for i, data_item in tqdm(enumerate(train_loader)):
        _, loss = model(data_item, return_loss=True)
        opt.zero_grad()
        loss.backward()
        opt.step()

# 对模型进行测试
def test_model(model, path, test_loader):
    if path is None:
        pass
    else:
        model.load_state_dict(torch.load(path))
    acc_rate, tf_rate = valid(model, test_loader)
    print('acc_rate:{}'.format(acc_rate))
    print('误报率:{}'.format(tf_rate))

# 对模型进行训练
def train_model(model, train_loader, valid_loader, total_epoch=20):
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    checkpoint_path = '../checkpoint'
    if os.path.exists(checkpoint_path):
        pass
    else:
        os.mkdir(checkpoint_path)

    print('start to train:')
    for epoch_ in range(total_epoch):
        train(model, train_loader, opt)
        valid(model, valid_loader)
    print('finished train')
    ti = localtime()
    write_path = '/lstm_{}_{}h{}m.tar'.format(ti.tm_mday, ti.tm_hour, ti.tm_min)
    torch.save(model.state_dict(), checkpoint_path + write_path)


if __name__ == '__main__':
    train_loader = MyDataLoader('../file/pre_datas_train.npy', '../file/pre_datas_train_label.npy',
                                batch_size=128, drop_last=True)
    valid_loader = MyDataLoader('../file/pre_datas_valid.npy', '../file/pre_datas_valid_label.npy',
                                batch_size=512, drop_last=False)

    device = torch.device('cuda:0')

    # LSTM的参数
    input_dim = 128
    lstm_layer = 2
    bidirectional = True

    model = LSTMClassifier(input_dim=input_dim, device=device, lstm_layers=lstm_layer, bidirectional=False)
    # model = CNNClassifier(device=device)
    # model = MlpClassifier(seq_len=532, input_dim=128, device=device)
    train_model(model, train_loader, valid_loader, total_epoch=10)

    test_loader = MyDataLoader('../file/pre_datas_train.npy', '../file/pre_datas_train_label.npy',
                               batch_size=512, drop_last=False)

    model_path = " "
    test_model(model, model_path, test_loader)
