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

def split_dataset(path,folderNameList,train_percent=0.85,valid_percent=0.10,test_percent=0.05):
    """
    Args:
        path: thr root path
        folderNameList: list type
        train:
        valid:
        test:

    Returns:

    """
    for folder in folderNameList:
        filePath = os.path.join(path,folder)
        fileList = os.listdir(filePath)
        lines = []
        for fileName in fileList:
            tmpPath = os.path.join(filePath,fileName)
            with open(tmpPath,'r',encoding='utf-8') as fr:
                datas = fr.readlines()
                random.shuffle(datas)
                lines.extend(datas)

        length = len(lines)
        train_data = lines[:,int(length*train_percent)]
        valid_data = lines[(int(length*train_percent) + 1):int(length*(train_percent+valid_percent))]
        test_data = lines[(int(length*(train_percent+valid_percent)) + 1),:]
        print("train data length is {}\n,valid data length is {}\n,test data length is {}\n".format(len(train_data),\
                                                                                                    len(valid_data),\
                                                                                                    len(test_data)))

        # 在当前路径下保存训练数据集、验证数据集和测试数据集
        train = os.path.join(filePath,folder+'.train.txt')
        valid = os.path.join(filePath,folder+'.valid.txt')
        test = os.path.join(filePath,folder+'.test.txt')

        # 保存训练数据集
        print("saving train data to the path：{}".format(train))
        with open(train,'a',encoding='utf-8') as fw:
            for line in train_data:
                fw.write(line)

        # 保存验证数据集
        print("saving valid data to the path：{}".format(valid))
        with open(valid,'a',encoding='utf-8') as fw:
            for line in valid_data:
                fw.write(line)

        # 保存测试数据集
        print("saving test data to the path：{}".format(test))
        with open(test,'a',encoding='utf-8') as fw:
            for line in test_data:
                fw.write(line)

if __name__ == '__main__':
    rootpath = "E:\\00_Gitworkspace\\00_Corpus\\02文本分类数据集\\01_THUCNews中文文本数据集\\MergeTHUCNews_Label_Sentence"
    if not os.path.exists(rootpath):
        os.mkdir(rootpath)

    print("Start time:{}".format(get_current_time()))
    # folderNameList = merge_all_files(rootpath)
    folderNameList = ['体育','娱乐','家居','彩票','房产','教育','时尚','时政','星座','游戏','社会','科技','股票','财经']
    split_dataset(rootpath,folderNameList)
    print("Finish time:{}".format(get_current_time()))