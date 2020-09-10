# -*- encoding:utf-8 -*-
import pickle
import os,bz2

def read_data():
    data = {}
    def read(mode):
        x,y = [],[]
        # TODO:添加路径检测功能
        # if not os.path.exists(w2v_path):
        #     print("the file {} is not found.".format(w2v_path))
        #     raise FileNotFoundError
        with open('./data/THUCNews/ModelTestData/'+mode+'.seg.txt','r',encoding='utf-8') as fr:
            for line in fr:
                if line[-1] == '\n':
                    line = line[:-1]
                # sentence 句子
                # x.append(line.split('\t')[1:]) # 完整的句子
                x.append(''.join(line.split('\t')[1:]).split(' ')) # 词组列表

                # label 标签
                y.append(line.split('\t')[0])
        if mode == 'train':
            data['train_x'] = x
            data['train_y'] = y
        elif mode == 'test':
            data['test_x'] = x
            data['test_y'] = y
        elif mode == 'valid':
            data['valid_x'] = x
            data['valid_y'] = y

    # 读取不同的数据集
    read('train')
    read('test')
    read('valid')
    return data

def load_word2vec(w2v_path):
    if not os.path.exists(w2v_path):
        print("the file {} is not found.".format(w2v_path))
        raise FileNotFoundError
    
    word2vec = {}
    file = bz2.open(w2v_path,'r')
    count = 1
    for line in file:
        if count == 1: # 过滤第1行的词总数和词向量维度
            count += 1
            print(line.decode())
            continue
        line = line.decode() # 转换为utf-8编码
        w2v_list = line.split(' ')
        word = w2v_list[0]
        vector = [float(item) for item in w2v_list[1:len(w2v_list)-1]]
        word2vec[word] = vector
    file.close()
    return word2vec
