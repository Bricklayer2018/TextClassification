# -*- encoding:utf-8 -*-
"""
处理THUCNews中文文本数据集
1）随机提取train.txt,valid.txt,test.txt 20%的数据作为模型的前期测试，用于程序流程的测试
2）使用HanLP工具对中文文本进行分词
THUCNews：http://thuctc.thunlp.org/
"""

from pyhanlp import *
from random import shuffle
import os

def sampleDataset(strDataPath,percent = 0.2):
    lines = []; length = 0
    # check the path is exist or not
    if not os.path.exists(strDataPath):
        FileNotFoundError(strDataPath + "is not exist!")

    # read data and filter sentences with less than 15 characters
    with open(strDataPath,'r',encoding='utf-8') as fr:
        for line in fr:
            length += 1
            segline = line.split('\t')
            label,sentence = segline[0],segline[1:]
            if len("".join(sentence)) <= 50: # 过滤字符数小于15的句子
                continue
            lines.append(line)
    shuffle(lines) # 乱序排列
    pickData = lines[:int(length*0.2)]
    return pickData

def saveSentence(path,data):
    with open(path,'a',encoding='utf-8') as fw:
        for line in data:
            fw.write(line)

# 对句子进行分词
def words_seg(sentence,tokenizer):
    wordlist = []
    for item in tokenizer.segment(sentence):
        wordlist.append(item.word)
    # print(" ".join(wordlist))
    return " ".join(wordlist)

# 切词并保存
def cutSentenceAndSave(path1,path2):
    with open(path1,'r',encoding='utf-8') as fr:
        for line in fr:
            label = line.split('\t')[0]
            sentence = "".join(line.split('\t')[1:])
            res = words_seg(sentence, NLPTokenizer)
            with open(path2,'a',encoding='utf-8') as fw:
                fw.write(label + '\t' + res)

# add api function
NLPTokenizer = JClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")

if __name__ == "__main__":
    # pickData = sampleDataset('./THUCNews/train.txt')
    # saveSentence('./THUCNews/train.20.txt',pickData)
    path1 = './THUCNews/train.20.txt'
    path2 = './THUCNews/train.20.seg.txt'
    cutSentenceAndSave(path1,path2)
    # print(words_seg("这是一个中文分词器",NLPTokenizer))
