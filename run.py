# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from model import SynSAN
import numpy as np
from func import train, test
from parameter import parse_args
from load_data import load_data


args = parse_args()
with open(args.WORD2VEC_DIR, 'r', encoding='utf-8') as f:
    w2v_model = eval(f.readline())

# load data
pair_data, label, lines, parse_result = load_data()
print('Data Loaded')

with open('dataset/data_index_1.txt', 'r', encoding='utf-8') as f:
    data_index = eval(f.readline())

# network
net = SynSAN(args)
optimizer = optim.Adam(net.parameters(), lr=args.lr)
criterion = nn.KLDivLoss()

# train and test
print('======h_dim:', args.h_dim, ' lr:', args.lr, ' batch_size:', args.batch_size,'======')
test_dis = []
for epoch in range(args.num_epoch):
    print('epoch:', epoch+1)
    train(args, net, optimizer, criterion, pair_data, label, data_index, lines, w2v_model)
    dis = test(args, net, criterion, pair_data, label, data_index, lines, w2v_model)
    test_dis.append(dis)
