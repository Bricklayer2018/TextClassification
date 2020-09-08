# -*- encoding:utf-8 -*-
import pickle

def read_data():
    data = {}
    def read(mode):
        x,y = [],[]
        with open('./data/THUCNews/'+mode+'.20.txt','r',encoding='utf-8') as fr:
            for line in fr:
                if line[-1] == '\n':
                    line = line[:-1]
                # sentence 句子
                x.append(line.split('\t')[1:])
                # x.append(''.join(line.split('\t')[1:]).split(' ')) # 词组列表

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

read_data()