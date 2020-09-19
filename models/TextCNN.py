# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import BasicModule as basic

# 卷积神经网络模型搭建，基类的继承
class TextCNN(basic.BasicModule):
    def __init__(self,**kwargs):
        super(TextCNN,self).__init__()  # 超函数

        # 模型参数解析
        self.MODEL = kwargs['MODEL'] # rand, static, non-static or multichannel
        self.BATCH_SIZE = kwargs['BATCH_SIZE']
        self.MAX_SENT_LEN = kwargs['MAX_SENT_LEN']
        self.WORD_DIM = kwargs['WORD_DIM']
        self.VOCAB_SIZE = kwargs['VOCAB_SIZE']
        self.CLASS_SIZE = kwargs['CLASS_SIZE']
        self.FILTERS = kwargs['FILTERS']  # n-gram
        self.FILTER_NUM = kwargs['FILTER_NUM']
        self.DROPOUT_PROB = kwargs['DROPOUT_PROB']
        self.NORM_LIMIT = kwargs['NORM_LIMIT']
        self.IN_CHANNEL = 1
        self.GPU = kwargs['GPU']
        self.WV_MATRIX = kwargs['WV_MATRIX']

        # 断言判断卷积核数量是否一致
        assert (len(self.FILTERS) == len(self.FILTER_NUM))

        # one for UNK and one for zero padding
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2,self.WORD_DIM,padding_idx=self.VOCAB_SIZE + 1)

        if self.MODEL == 'static' or self.MODEL == 'non-static' or self.MODEL == 'multichannel':
            self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
            if self.MODEL == 'static':
                self.embedding.weight.requires_grad = False
            elif self.MODEL == 'multichannel':
                self.embedding2 = nn.Embedding(self.VOCAB_SIZE + 2,self.WORD_DIM,padding_idx=self.VOCAB_SIZE + 1)
                self.embedding2.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
                self.embedding2.weight.requires_grad = False
                self.IN_CHANNEL = 2

        # 卷积层配置(一维卷积的输入通道和输出通道如何理解)
        # self.convs = nn.ModuleList()
        # for i in range(len(self.FILTERS)):
        #     '''
        #     输入通道为1
        #     输出通道为100，
        #     卷积核大小：3或4或5*300
        #     '''
        #     self.convs.append(nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], (self.FILTERS[i],self.WORD_DIM), stride=1))
        self.convs = nn.ModuleList(
                    [nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], (self.FILTERS[i], self.WORD_DIM)) \
                    for i in range(len(self.FILTERS))])

        self.dropout = nn.Dropout(self.DROPOUT_PROB)

        # 线性变换层[300,14]
        self.linear = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)
 

    def forward(self,x):
        # Step 1：对输入的训练样本进行embedding操作，转换为[batch_size,max_len,word_dim]
        x = self.embedding(x)

        if self.MODEL == 'multichannel':
            # 两个通道的embedding进行拼接
            x2 = self.embedding2(x)
            x = torch.cat((x,x2),1) # 纵向拼接max_len = 2*max_len
        
        # Step 2：经embedding后的训练数据，reshape为[batch_size,in_channel,max_len,word_dim]
        x = x.view(x.size(0),1,x.size(1),self.WORD_DIM)

        # Step 3：卷积操作，卷积核为3*word_dim，4*word_dim和5*word_dim大小的各100个，即输出100个通道
        # batch_size中的每个样本经单个卷积操作后转换为[batch_size,out_channel,w,h=1],其中，原有输入数据有padding的情况下，w=max_len
        x = [F.relu(conv(x)) for conv in self.convs]

        # Step 4：池化层(均值池化和最大值池化)
        # 经过最大池化层,维度变为(batch_size, out_chanel, w=1, h=1)
        x = [F.max_pool2d(x_item, kernel_size = (x_item.size(2), x_item.size(3))) for x_item in x]

        # Step 5: 将不同卷积核的运算结果维度[batch_size,out_channel,w=1,h=1]展平为[batch_size,out_channel*w*h]
        x = [x_item.view(x_item.size(0),-1) for x_item in x]

        # Step 6：将不同卷积核提取的特征组合起来,维度变为(batch, sum:outchanel*w*h)
        x = torch.cat(x,1)

        # Step 7：dropout层
        x = self.dropout(x)

        # Step 8：全连接层
        logits = self.linear(x)

        return logits

if __name__ == "__main__":
    # model parameter
    params = {
        "MODEL": 1,
        "SAVE_MODEL": 2,
        "EARLY_STOPPING": 3,
        "EPOCH": 4,
        "LEARNING_RATE": 5,
        "MAX_SENT_LEN": 6,
        "BATCH_SIZE": 32,
        "WORD_DIM": 300,
        "VOCAB_SIZE": 7,
        "CLASS_SIZE": 8,
        "FILTERS": [3, 4, 5],
        "FILTER_NUM": [100, 100, 100],
        "DROPOUT_PROB": 0.5,
        "NORM_LIMIT": 3,
        'WV_MATRIX':10,
        "GPU": 11
    }
    print("params:",params)
    TextCNN = TextCNN(**params)