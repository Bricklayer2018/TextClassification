# —*- coding: utf-8 -*-
"""
Description: Preprocessing THUCNews corpus, merge all txt files in the same category
"""
import os
import time
import re

def get_current_time():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

def get_all_folders_name(path):
    """
    :param path:
    :return:
    """
    folderList = os.listdir(path)
    if not folderList:
        return []

    return folderList

def label_sentence(label,sentence):
    """
    :param label:
    :param sentence:
    :return:
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

if __name__ == '__main__':
    rootpath = "E:\\00_Gitworkspace\\00_Corpus\\02文本分类数据集\\01_THUCNews中文文本数据集\\THUCNews"
    print("Start time:{}".format(get_current_time()))
    merge_all_files(rootpath)
    print("Finish time:{}".format(get_current_time()))