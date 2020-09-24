# -*- coding:utf-8 -*-
# from pyhanlp import *
import argparse
import random
from utils import *
from models import TextCNN
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import copy,time

def train(data,params):
    # 词向量初始化
    # if params['MODEL'] != 'rand':
    #     # load pretrian word2vec
    #     # word2vec_dict = load_pretrain_word2vec(args.word2vec_data_path)
    #     # 语料中的词分配词向量，由于已经做了word_to_idx和idx_to_word，逐个遍历即可
    #     # wv_matrix的索引也可以通过idx_to_word进行索引
    #     for i in range(len(data['vocab'])):
    #         word = data['idx_to_word'][i]
    #         if word in word2vec_dict:
    #             wv_matrix.append(word2vec_dict[word])
    #         else:
    #             wv_matrix.append(np.random.uniform(-0.01,0.01,300).astype(np.float32))
    # else:
    #     wv_matrix.append(np.random.uniform(-0.01,0.01,size = (len(data['vocab']),300)))
    
    # # one for UNK and one for zero padding
    # wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype(np.float32))
    # wv_matrix.append(np.zeros(params['WORD_DIM']).astype(np.float32))
    if params['MODEL'] != 'rand':
        wv_matrix = load_npy_vector(params['PRETRAIN_W2V_PATH'])
    else:
        # TODO 随机初始化词向量
        pass

    params['WV_MATRIX'] = wv_matrix

    if torch.cuda.is_available():
        print("使用GPU进行训练")
        model = TextCNN(**params).cuda(params['GPU'])
    else:
        print("使用CPU进行训练")
        model = TextCNN(**params)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters,params["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss() # loss object

    pre_valid_acc = 0
    max_valid_acc = 0
    max_test_acc = 0
    for epoch in range(params["EPOCH"]):
        # data["train_x"], data["train_y"] = random.shuffle(data["train_x"], data["train_y"])
        count = 0
        # batch (可以使用yield生成器)
        for i in range(0,len(data["train_x"]),params["BATCH_SIZE"]):
            current_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            count += 1
            if count % 10 == 0:
                print("{} epoch = {}, 正在运行第{}个batch, 共有{}个batch".format(current_date,str(epoch+1),str(count),str(int(len(data["train_x"])/params["BATCH_SIZE"])+1)))
            
            batch_range = min(params["BATCH_SIZE"],len(data["train_x"]) - i)

            batch_x,batch_y = [],[]
            for sent in data["train_x"][i:i+batch_range]:
                if len(sent) < params["MAX_SENT_LEN"]:
                    sent += (params["MAX_SENT_LEN"] - len(sent))*[data["word_to_id"]['PAD']]
                else:
                    sent = sent[:params["MAX_SENT_LEN"]]
                batch_x.append(sent)

            batch_y = data["train_y"][i:i + batch_range]

            if torch.cuda.is_available():
                batch_x = torch.LongTensor(batch_x).cuda(params["GPU"])
                batch_y = torch.LongTensor(batch_y).cuda(params["GPU"])
            else:
                batch_x = torch.LongTensor(batch_x)
                batch_y = torch.LongTensor(batch_y)
            
            optimizer.zero_grad() # 梯度清零
            model.train() # Sets the module in training mode
            pred = model(batch_x) # 模型预测
            loss = criterion(pred,batch_y)
            loss.backward() # 反向传播求解梯度
            nn.utils.clip_grad_norm_(parameters, max_norm=params["NORM_LIMIT"]) # 梯度裁剪
            optimizer.step() # 更新权重参数

        # 每个epoch结束后对模型进行评估
        valid_acc = test(model,data,params,mode = "valid")
        test_acc = test(model,data,params,mode = "test")
        print("epoch:{}, valid_acc:{}, test_acc:{}".format(str(epoch+1),valid_acc,test_acc))

        if params["EARLY_STOPPING"] and valid_acc <= pre_valid_acc:
            print("early stopping by valid_acc!")
            break
        else:
            pre_valid_acc = valid_acc

        if valid_acc > max_valid_acc:
            max_valid_acc = valid_acc
            max_test_acc = test_acc
            best_model = copy.deepcopy(model)
    
    # 训练结束
    print("max valid acc:{}, test acc:{}".format(max_valid_acc,max_test_acc))
    return best_model

def test(model,data,params,mode = "test"):
    current_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    if mode == "valid":
        print("\t{} The model is validated on the validation set".format(current_date))
    elif mode == "test":
        print("\t{} The model is tested on the test set".format(current_date))
    model.eval() # Sets the module in evaluation mode
    
    if mode == "valid":
        x, y = data["valid_x"], data["valid_y"]
    elif mode == "test":
        x, y = data["test_x"], data["test_y"]    

    batch_x,batch_y = [],[]
    for sent in x:
        if len(sent) < params["MAX_SENT_LEN"]:
            sent += (params["MAX_SENT_LEN"] - len(sent))*[data["word_to_id"]['PAD']]
        else:
            sent = sent[:params["MAX_SENT_LEN"]]
        batch_x.append(sent)
    batch_y = y

    print("the length of batch_x is :{}".format(str(len(batch_x))))

    if torch.cuda.is_available():
        batch_x = torch.LongTensor(batch_x).cuda(params["GPU"])
    else:
        batch_x = torch.LongTensor(batch_x)

    model_pred = model(batch_x).cpu().data.numpy() # logits转换成numpy数组
    pred = np.argmax(model_pred, axis = 1)

    # acc = sum([1 if p == y else 0 for p, y in zip(pred, y)]) / len(pred)
    score = 0
    for p,y in zip(pred,batch_y):
        if p == y:
            score += 1
        else:
            pass
    acc = score / len(x)
    return acc

def main():
    # 参数解析
    parser = argparse.ArgumentParser(description="Description：TextCNN classifier",prog="TextCNN")

    # path parameter
    parser.add_argument('--train_data_path',default='./data/THUCNews/ModelTestData/train.seg.txt',help="Training data for training neural networks")
    parser.add_argument('--valid_data_path',default='./data/THUCNews/ModelTestData/valid.seg.txt',help="Valid data for valid neural networks")
    parser.add_argument('--test_data_path',default='./data/THUCNews/ModelTestData/test.seg.txt',help="Test data for test neural networks")
    parser.add_argument('--word2vec_data_path',default='./data/THUCNewsSubset/vectors.npy',help="Test data for test neural networks")

    # other paramter
    parser.add_argument('--mode',default='train',help="train: train (with test) a model / test: test saved models")
    parser.add_argument('--model',default='static',help="available models: rand, static, non-static and multichannel")
    parser.add_argument('--save_model',default=False,action='store_false',help="whether saving model or not")
    parser.add_argument("--early_stopping", default=False, action='store_true', help="whether to apply early stopping")
    parser.add_argument("--epoch", default=10, type=int, help="number of max epoch")
    parser.add_argument("--learning_rate", default=1.0, type=float, help="learning rate")
    parser.add_argument('--gpu',default=0, type=int, help="whether using gpu or not") # 分配GPU设备
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')


    # 参数解析
    args = parser.parse_args()

    # # 加载预训练词典，保存为vocab.txt，word_to_id.txt,id_to_word.txt,vectors.npy
    # word2vec_dict,words_list,vectors_array = load_pretrain_word2vec(args.word2vec_data_path)

    # vocab = words_list
    # word_to_id = {w: i for i, w in enumerate(words_list)}
    # id_to_word = {i: w for i, w in enumerate(words_list)}

    # # Step 1：保存vocab.txt
    # with open("./data/THUCNewsSubset/vocab.txt",'w',encoding='utf-8') as fw:
    #     fw.write('\n'.join(words_list))

    # # Step 2：保存vectors.npy
    # np.save('./data/THUCNewsSubset/vectors.npy',vectors_array)

    # # Step 3：保存word_to_id.txt, id_to_word.txt
    # for item in sorted(word_to_id.items(),key=lambda x: x[1],reverse=False): # 升序
    #     word,idx = item
    #     with open("./data/THUCNewsSubset/word_to_id.txt",'a',encoding='utf-8') as fw:
    #         fw.write(word  + '\t' + str(idx) + '\n')

    # for item in sorted(id_to_word.items(),key=lambda x: x[0],reverse=False): # 升序
    #     idx,word = item
    #     with open("./data/THUCNewsSubset/id_to_word.txt",'a',encoding='utf-8') as fw:
    #         fw.write(str(idx)  + '\t' + word + '\n')
 
    ## dataset process
    # data = read_data() # 读取预处理后的数据集，返回data字典
    
    # data["label"] = sorted(list(set(data["train_y"]))) # labels
    # class_to_id = {w: i for i, w in enumerate(data["class"])}
    # for item in sorted(class_to_id.items(),key=lambda x: x[1],reverse=False):
    #     label,idx = item
    #     with open("./data/THUCNewsSubset/class.txt",'a',encoding='utf-8') as fw:
    #         fw.write(label + '\t' + str(idx) + '\n')

    data = read_id_data()
    data["vocab"] = load_vocab("./data/THUCNewsSubset/vocab.txt")
    data["label"], _ , label_to_id = load_label("./data/THUCNewsSubset/label.txt")
    data["word_to_id"] = load_word_to_id("./data/THUCNewsSubset/word_to_id.txt")

    # model parameter
    params = {
        "PRETRAIN_W2V_PATH":args.word2vec_data_path,
        "MODEL": args.model,
        "SAVE_MODEL": args.save_model,
        "EARLY_STOPPING": args.early_stopping,
        "EPOCH": args.epoch,
        "LEARNING_RATE": args.learning_rate,
        "MAX_SENT_LEN": 150,
        "BATCH_SIZE": 32,
        "WORD_DIM": 300,
        "VOCAB_SIZE": len(data["vocab"]),
        "CLASS_SIZE": len(data["label"]),
        "FILTERS": [3, 4, 5],
        "FILTER_NUM": [100, 100, 100],
        "DROPOUT_PROB": 0.5,
        "NORM_LIMIT": 3,
        "GPU": args.gpu
    }

    if args.mode == 'train':
        print("=" * 20 + "TRAINING STARTED" + "=" * 20)
        model = train(data,params)
    else:
        pass

if __name__ == '__main__':
    print('Start.')
    main()
    print('Done.')