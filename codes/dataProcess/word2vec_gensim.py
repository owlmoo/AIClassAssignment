import csv
import pickle
from collections import Counter
from codes.dataProcess.utils import GeneSeg
from gensim.models.word2vec import Word2Vec

vocabulary_size = 3000
embedding_size = 128
skip_window = 5
num_sampled = 64
num_iter = 5
vec_dir = "../../file/word2vec.pickle"

# 构建数据集
def build_dataset(datas, words):
    count = [["UNK", -1]]
    # 该函数构成了一个字典，用于统计和记录该数据集合中的词语及词频字典
    counter = Counter(words)
    count.extend(counter.most_common(vocabulary_size - 1))
    vocabulary = [c[0] for c in count]
    data_set = []
    for data in datas:
        d_set = []
        for word in data:
            if word in vocabulary:
                d_set.append(word)
            else:
                d_set.append("UNK")
                count[0][1] += 1
        data_set.append(d_set)
    return data_set

def save(embeddings):
    # string 到 index 的字典，其中 index 是按照词频进行排序的
    dictionary = dict([(embeddings.index_to_key[i], i) for i in range(len(embeddings.index_to_key))])
    # embeddings 是按照index顺序的vector列表
    # index 到 string 的字典，正好反过来
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    word2vec = {"dictionary": dictionary, "embeddings": embeddings, "reverse_dictionary": reverse_dictionary}
    with open(vec_dir, "wb") as f:
        pickle.dump(word2vec, f)

if __name__ == '__main__':
    words = []
    datas = []

    # words 形成了每个词语的列表，而datas形成了每个句子的列表，每个句子是词语的列表
    with open("../../data/xssed.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, fieldnames=["payload"])
        for row in reader:
            payload = row["payload"]
            word = GeneSeg(payload)
            datas.append(word)
            words += word
    data_set = build_dataset(datas, words)

    # Word2Vec参数解析：vector_size表示输出词向量的维度
    # window是句子中当前词与目标词之间的最大距离
    # negative如果>0，则会采用negative sampling, 用于设置多少个noise words
    model = Word2Vec(data_set, vector_size=embedding_size, window=skip_window, negative=num_sampled, epochs=num_iter)
    embeddings = model.wv

    save(embeddings)
    print("Saved words vec to", vec_dir)

