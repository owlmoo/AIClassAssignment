import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data_path, label_path):
        super(MyDataset, self).__init__()
        with open('../file/word2vec.pickle', 'rb') as f:
            wordmodel = pickle.load(f)

        self.embeding = wordmodel['embeddings']
        self.reverse_dict = wordmodel['reverse_dictionary']
        self.x = np.load(data_path)
        self.y = np.load(label_path)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        # 获得padding后的句子的表示,单词被使用词表的索引表示
        data = self.x[index]
        embedding = []
        length = 0
        for d in data:
            if d != -1:
                # 根据索引取单词
                key = self.reverse_dict[d]
                # 根据单词获得单词的word2vec向量表示
                embedding.append(self.embeding[key])
                length += 1
            else:
                # 如果是padding的,那就添加0
                embedding.append(np.zeros(128, ))
        if length == 0:
            length = 1
        embedding = np.array(embedding, dtype=np.float32).copy()

        label_ = self.y[index]

        # 0 代表着有害的数据，1 代表着正常的数据
        if label_[0] == 1:
            label = 0
        else:
            label = 1

        return {
            'x': embedding,
            'y': label,
            'length': length
        }


def MyDataLoader(datas_dir, datas_label_dir, batch_size=512, drop_last=False):
    dataset = MyDataset(datas_dir, datas_label_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=drop_last)
    return dataloader

