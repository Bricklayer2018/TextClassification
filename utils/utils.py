# -*- encoding:utf-8 -*-
import pickle
import os,bz2
import numpy as np

def read_id_data():
    data = {}
    def read(dataset):
        x,y = [],[]
        dataset_path = './data/THUCNewsSubset/seg_id_data/cnews.'+dataset+'.seg.id.txt'
        if not os.path.exists(dataset_path):
            print("the file {} is not found.".format(dataset_path))
            raise FileNotFoundError
        with open(dataset_path,'r',encoding='utf-8') as fr:
            for line in fr:
                if line[-1] == '\n':
                    line = line[:-1]
                # sentence 句子
                # x.append(line.split('\t')[1:]) # 完整的句子
                sent_id = list(map(int,''.join(line.split('\t')[1:]).split(' ')))
                x.append(sent_id) # 词组列表

                label_id = int(line.split('\t')[0])
                # label 标签
                y.append(label_id)
        if dataset == 'train':
            data['train_x'] = x[1:500]
            data['train_y'] = y[1:500]
        elif dataset == 'test':
            data['test_x'] = x
            data['test_y'] = y
        elif dataset == 'valid':
            data['valid_x'] = x
            data['valid_y'] = y

    # 读取不同的数据集
    read('train')
    read('test')
    read('valid')
    return data

def read_data():
    data = {}
    def read(dataset):
        x,y = [],[]
        dataset_path = './data/THUCNewsSubset/cnews.'+dataset+'.seg.txt'
        if not os.path.exists(dataset_path):
            print("the file {} is not found.".format(dataset_path))
            raise FileNotFoundError
        with open(dataset_path,'r',encoding='utf-8') as fr:
            for line in fr:
                if line[-1] == '\n':
                    line = line[:-1]
                # sentence 句子
                # x.append(line.split('\t')[1:]) # 完整的句子
                x.append(''.join(line.split('\t')[1:]).split(' ')) # 词组列表

                # label 标签
                y.append(line.split('\t')[0])
        if dataset == 'train':
            data['train_x'] = x
            data['train_y'] = y
        elif dataset == 'test':
            data['test_x'] = x
            data['test_y'] = y
        elif dataset == 'valid':
            data['valid_x'] = x
            data['valid_y'] = y

    # 读取不同的数据集
    read('train')
    read('test')
    read('valid')
    return data

def load_pretrain_word2vec(w2v_path):
    """
    加载预训练词向量文件，格式如下：
    {'a':array([0.1,0.2,...,0.9],
    'b':array([0.1,0.3,...,0.02])}
    """
    if not os.path.exists(w2v_path):
        print("the file {} is not found.".format(w2v_path))
        raise FileNotFoundError
    
    word2vec = {}
    words = []
    vectors = []
    file = bz2.open(w2v_path,'r')
    count = 1
    for line in file:
        if count == 1: # 过滤第1行的词总数和词向量维度
            count += 1
            continue
        line = line.decode() # 转换为utf-8编码
        w2v_list = line.split(' ')
        word = w2v_list[0]
        vector = np.array([float(item) for item in w2v_list[1:len(w2v_list)-1]])

        word2vec[word] = vector
        words.append(word)
        vectors.append(vector.tolist())
    file.close()

    # add ‘PAD’ and ‘UNK’ words
    word2vec["PAD"] = np.random.uniform(-0.01, 0.01, 300).astype(np.float32)
    word2vec["UNK"] = np.zeros(300).astype(np.float32)
    words.extend(["PAD","UNK"])
    vectors.extend([word2vec["PAD"].tolist(),word2vec["UNK"].tolist()])

    return word2vec,words,np.array(vectors)

def load_vocab(vocab_path):
    if not os.path.exists(vocab_path):
        print("the file {} is not found.".format(vocab_path))
        raise FileNotFoundError
    vocab = []
    with open(vocab_path,'r',encoding='utf-8') as fr:
        for word in fr:
            vocab.append(word.rstrip('\n'))
    return vocab

def load_label(label_path):
    if not os.path.exists(label_path):
        print("the file {} is not found.".format(label_path))
        raise FileNotFoundError
    label,idx = [],[]
    label_idx_dict = {}
    with open(label_path,'r',encoding='utf-8') as fr:
        for line in fr:
            line = line.rstrip('\n')
            key = line.split('\t')[0]
            value = line.split('\t')[-1]
            label.append(key)
            idx.append(value)
            label_idx_dict[key] = int(value)
    return label,idx,label_idx_dict

def load_word_to_id(path):
    if not os.path.exists(path):
        print("the file {} is not found.".format(path))
        raise FileNotFoundError
    word_to_id = {}
    with open(path,'r',encoding='utf-8') as fr:
        for line in fr:
            line = line.rstrip('\n')
            key = line.split('\t')[0]
            value = line.split('\t')[-1]
            word_to_id[key] = int(value)
    return word_to_id

def load_npy_vector(path):
    if not os.path.exists(path):
        print("the file {} is not found.".format(path))
        raise FileNotFoundError
    w2v = np.load(path)
    return w2v

def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
