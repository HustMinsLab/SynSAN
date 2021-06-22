# -*- coding: utf-8 -*-
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='SynSAN for Social Emotion Classification')

    # word2vec
    parser.add_argument('--WORD2VEC_DIR', default='dataset/word2vec.txt')

    # dataset
    parser.add_argument('--data_size', default=1246, type=int, help='Dataset size')
    parser.add_argument('--train_size', default=1121, type=int, help='Train set size')
    parser.add_argument('--test_size', default=125, type=int, help='Test set size')
    parser.add_argument('--num_class', default=6, type=int, help='Number of classes')

    # model arguments
    parser.add_argument('--in_dim', default=300, type=int, help='Size of input word vector')
    parser.add_argument('--h_dim', default=300, type=int, help='Size of hidden state')

    # training arguments
    parser.add_argument('--num_epoch', default=30, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=50, type=int, help='batchsize for optimizer updates')
    parser.add_argument('--lr', default=0.005, type=float, metavar='LR', help='initial learning rate')

    args = parser.parse_args()
    return args
