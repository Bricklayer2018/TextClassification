# -*- coding:utf-8 -*-
# from pyhanlp import *
import argparse
from utils import read_data

def main():
    # 参数解析
    parser = argparse.ArgumentParser(description="Description：TextCNN classifier",prog="TextCNN")

    # path parameter
    parser.add_argument('--train_data_path',default='./train.txt',help="Training data for training neural networks")
    parser.add_argument('--valid_data_path',default='./valid.txt',help="Valid data for valid neural networks")
    parser.add_argument('--test_data_path',default='./test.txt',help="Test data for test neural networks")
    parser.add_argument('--word2vec_data_path',default='word2vec',help="Test data for test neural networks")

    # other paramter
    parser.add_argument('--mode',default='train',help="train: train (with test) a model / test: test saved models")
    parser.add_argument('--model',default='rand',help="available models: rand, static, non-static and multichannel")
    parser.add_argument('--save_model',default=False,action='store_false',help="whether saving model or not")
    parser.add_argument("--early_stopping", default=False, action='store_true', help="whether to apply early stopping")
    parser.add_argument("--epoch", default=100, type=int, help="number of max epoch")
    parser.add_argument("--learning_rate", default=1.0, type=float, help="learning rate")
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    parser.add_argument('--gpu',default=False,action='store_false',help="whether using gpu or not")

    # 参数解析
    args = parser.parse_args()

    # dataset process
    data = read_data() # 读取预处理后的数据集，返回data字典

    data["vocab"] = sorted(list(set([w for sent in data["train_x"] + data["valid_x"] + data["test_x"] for w in sent])))
    data["classes"] = sorted(list(set(data["train_y"]))) # labels
    data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
    data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}

    # model parameter
    params = {
        "MODEL": args.model,
        # "DATASET": args.dataset,
        "SAVE_MODEL": args.save_model,
        "EARLY_STOPPING": args.early_stopping,
        "EPOCH": args.epoch,
        "LEARNING_RATE": args.learning_rate,
        "MAX_SENT_LEN": max([len(sent) for sent in data["train_x"] + data["dev_x"] + data["test_x"]]),
        "BATCH_SIZE": 32,
        "WORD_DIM": 300,
        "VOCAB_SIZE": len(data["vocab"]),
        "CLASS_SIZE": len(data["classes"]),
        "FILTERS": [3, 4, 5],
        "FILTER_NUM": [100, 100, 100],
        "DROPOUT_PROB": 0.5,
        "NORM_LIMIT": 3,
        "GPU": args.gpu
    }


    print(args)

if __name__ == '__main__':
    main()