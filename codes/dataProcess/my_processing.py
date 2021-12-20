from utils import GeneSeg
import csv, pickle, random, json
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import torch

vec_dir = "../../file/word2vec.pickle"
pre_datas_train = "file/pre_datas_train.npy"
pre_datas_label_train = "file/pre_datas_train_label.npy"
pre_datas_test = "file/pre_datas_test.npy"
pre_datas_label_test = "file/pre_datas_test_label.npy"
pre_datas_valid = "file/pre_datas_valid.npy"
pre_datas_label_valid = "file/pre_datas_valid_label.npy"
process_datas_dir = "file/process_datas.pickle"

# 变成onehot形式
def to_categorical(labels):
    labels_ = [[1, 0] if _ == 1 else [0, 1] for _ in labels]
    labels_ = np.array(labels_)
    return labels_

def pre_process():
    with open(vec_dir, "rb") as f:
        word2vec = pickle.load(f)
        dictionary = word2vec["dictionary"]
    # 这一步是导入xss数据
    xssed_data = []
    normal_data = []
    with open("../../data/xssed.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, fieldnames=["payload"])
        for row in reader:
            payload = row["payload"]
            # word是一个处理过的词语构成的列表
            word = GeneSeg(payload)
            xssed_data.append(word)
    with open("../../data/normal_examples.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, fieldnames=["payload"])
        for row in reader:
            payload = row["payload"]
            word = GeneSeg(payload)
            normal_data.append(word)
    # 产生labels标签
    xssed_num = len(xssed_data)
    normal_num = len(normal_data)
    xssed_labels = [1] * xssed_num
    normal_labels = [0] * normal_num

    datas = xssed_data + normal_data
    labels = xssed_labels + normal_labels
    labels = to_categorical(labels)

    # 将单词列表转换成index
    def to_index(data):
        d_index = []
        for word in data:
            if word in dictionary.keys():
                d_index.append(dictionary[word])
            else:
                d_index.append(dictionary["UNK"])
        return d_index
    # 将单词列表转换成index列表，并对齐
    datas_index = [torch.tensor(to_index(data)) for data in datas]
    datas_index = pad_sequence(datas_index, batch_first=True, padding_value=-1)
    datas_index = datas_index.numpy()
    # 随机打乱
    rand = random.sample(range(len(datas_index)), len(datas_index))
    datas = [datas_index[index] for index in rand]
    labels = [labels[index] for index in rand]
    # 划分训练集、验证集、测试集
    train_datas, test_datas, train_labels, test_labels = train_test_split(datas, labels, test_size=0.4)
    vaild_data, test_datas, vaild_labels, test_labels = train_test_split(test_datas, test_labels, test_size=0.5)

    print("Write trian datas to:", pre_datas_train)
    np.save(pre_datas_train, train_datas)
    np.save(pre_datas_label_train, train_labels)

    print("Write test datas to:", pre_datas_test)
    np.save(pre_datas_test, test_datas)
    np.save(pre_datas_label_test, test_labels)

    print("Write test datas to:", pre_datas_valid)
    np.save(pre_datas_valid, vaild_data)
    np.save(pre_datas_label_valid, vaild_labels)

    print("Write datas over!")


if __name__ == "__main__":
    pre_process()
