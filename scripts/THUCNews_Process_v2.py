# —*- coding: utf-8 -*-
"""
Description: Preprocessing THUCNews corpus, merge all txt files in the same category
"""
import os
import time
import re
import random

def get_current_time():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

def get_all_folders_name(path):
    """
    Args:
        path:

    Returns:

    """
    folderList = os.listdir(path)
    if not folderList:
        return []

    return folderList

def label_sentence(label,sentence):
    """
    Args:
        label:
        sentence:

    Returns:

    """
    if not sentence:
        return ''
    sentence = re.sub(u'[　]', u' ', sentence) # 替换句中全角Unicode空格（ascii码值是12288）
    return label + '\t' + sentence.lstrip('\t').lstrip(' ') + '\n'

def merge_all_files(path):
    folderNameList = get_all_folders_name(path)
    if not folderNameList:
        raise FileNotFoundError

    for name in folderNameList:
        folderPath = os.path.join(path,name)
        print("Now is processing the folder:",folderPath)
        fileList = os.listdir(folderPath)
        for file in fileList:
            filePath = os.path.join(folderPath,file)
            with open(filePath,'r',encoding='utf-8') as fr:
                for line in fr:
                    sentence = label_sentence(name,line.rstrip('\n').rstrip(' '))
                    if sentence == '':
                        continue
                    with open(os.path.join(folderPath,name+'.txt'),'a',encoding='utf-8') as fw:
                        fw.write(sentence)
    return folderNameList

def split_dataset(path,folderNameList,train_percent = 0.85,valid_percent = 0.10,test_percent = 0.05):
    """
    Args:
        path: thr root path
        folderNameList: list type
        train:
        valid:
        test:

    Returns:

    """

    all_train,all_valid,all_test = [],[],[]
    for folder in folderNameList:
        filename = folder + '.txt'
        filePath = os.path.join(path,folder,filename)
        
        lines = []
        with open(filePath,'r',encoding='utf-8') as fr:
            datas = fr.readlines()
            random.shuffle(datas)
            lines.extend(datas)

            # for line in fr:
            #    lines.append(line)

        length = len(lines)
        print("******Now processing '{}'******".format(folder))
        print("---the total data length is : {}".format(str(length)))
        
        train_data = lines[:int(length*train_percent)]
        valid_data = lines[int(length*train_percent) : int(length*(train_percent + valid_percent))]
        test_data = lines[int(length*(train_percent + valid_percent)):]
        print("---train data length is {},valid data length is {},test data length is {}".format(len(train_data),len(valid_data),len(test_data)))
       
        # 在当前路径下保存训练数据集、验证数据集和测试数据集
        train = os.path.join(os.path.join(path,folder),folder+'.train.txt')
        valid = os.path.join(os.path.join(path,folder),folder+'.valid.txt')
        test = os.path.join(os.path.join(path,folder),folder+'.test.txt')

        # 保存训练数据集
        if os.path.exists(train):
            print("---the file '{}' is exist, delete it.".format(train))
            os.remove(train)

        print("---saving train data to the path：'{}'".format(train))
        with open(train,'a',encoding='utf-8') as fw:
            for line in train_data:
                fw.write(line)

        # 保存验证数据集
        if os.path.exists(valid):
            print("---the file '{}' is exist, delete it.".format(valid))
            os.remove(valid)

        print("---saving valid data to the path：'{}'".format(valid))
        with open(valid,'a',encoding='utf-8') as fw:
            for line in valid_data:
                fw.write(line)

        # 保存测试数据集
        if os.path.exists(test):
            print("---the file '{}' is exist, delete it.".format(test))
            os.remove(test)

        print("---saving test data to the path：{}\n".format(test))
        with open(test,'a',encoding='utf-8') as fw:
            for line in test_data:
                fw.write(line)

        all_train.extend(train_data)
        all_valid.extend(valid_data)
        all_test.extend(test_data)

    print("******Saving all train,valid,test data.******")
    print("---all train data length is {}, all valid data length is {}, all test data length is {}".format(len(all_train),len(all_valid),len(all_test)))
    all_train_path = os.path.join(rootpath,'train2.txt')
    all_valid_path = os.path.join(rootpath,'valid2.txt')
    all_test_path = os.path.join(rootpath,'test2.txt')

    if os.path.exists(all_train_path):
        print("---the file '{}' is exist, delete it.".format(all_train_path))
        os.remove(all_train_path)
    
    print("---saving all train data to the path：'{}'".format(all_train_path))
    with open(all_train_path,'a',encoding='utf-8') as fw:
        for line in all_train:
            fw.write(line)

    if os.path.exists(all_valid_path):
        print("---the file '{}' is exist, delete it.".format(all_valid_path))
        os.remove(all_valid_path)

    print("---saving all valid data to the path：'{}'".format(all_valid_path))
    with open(all_valid_path,'a',encoding='utf-8') as fw:
        for line in all_valid:
            fw.write(line)

    if os.path.exists(all_test_path):
        print("---the file '{}' is exist, delete it.".format(all_test_path))
        os.remove(all_test_path)

    print("---saving all test data to the path：'{}'".format(all_test_path))
    with open(all_test_path,'a',encoding='utf-8') as fw:
        for line in all_test:
            fw.write(line)


if __name__ == '__main__':
    rootpath = "./THUCNews/THUCNews/"
    
    print("Start time:{}".format(get_current_time()))
    # folderNameList = merge_all_files(rootpath)
    # folderNameList = ['体育','娱乐','家居','彩票','房产','教育']
    folderNameList = ['时尚','时政','星座','游戏','社会','科技','股票','财经']
    split_dataset(rootpath,folderNameList)
    print("Finish time:{}".format(get_current_time()))
