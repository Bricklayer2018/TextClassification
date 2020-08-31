# -*- coding:utf-8 -*-
# from pyhanlp import *
import argparse

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
    args = parser.parse_args()
    print(args)

if __name__ == '__main__':
    main()