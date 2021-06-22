import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SynSelfAtt(nn.Module):
    def __init__(self, in_dim, h_dim):
        super(SynSelfAtt, self).__init__()
        self.Wx = nn.Linear(in_dim, h_dim)
        # self-attention
        self.Wh = nn.Linear(h_dim, h_dim)
        self.Uh = nn.Linear(h_dim, h_dim)
        # gate
        self.Wg = nn.Linear(h_dim, h_dim)
        self.Ug = nn.Linear(h_dim, h_dim)
        self.dropout = nn.Dropout(0.5)

    def update(self, x):
        center_update = []
        for vecs in x:
            center_vec = torch.tanh(self.Wx(vecs[0:1]))
            neighbor_vecs = torch.tanh(self.Wx(vecs[1:]))
            # self-attention
            temp = torch.sigmoid(self.Wh(center_vec)+self.Uh(neighbor_vecs))
            a = F.softmax(temp, dim=0)
            con_vec = torch.sum(torch.mul(a, neighbor_vecs), dim=0, keepdim=True)
            # gate
            gate = torch.sigmoid(self.Wg(center_vec)+self.Ug(con_vec))
            center_update.append(torch.mul(gate, center_vec)+torch.mul(1-gate, con_vec))
        return self.dropout(torch.cat(center_update, dim=0))

    def forward(self, batch_x):
        return [self.update(x) for x in batch_x]


class Attention(nn.Module):
    def __init__(self, h_dim):
        super(Attention, self).__init__()
        self.Wh = nn.Linear(h_dim, h_dim)
        self.V = nn.Linear(h_dim, h_dim)

    def forward(self, x):
        a = self.V(torch.sigmoid(self.Wh(x)))
        a = F.softmax(a, dim=0)
        return torch.sum(torch.mul(a, x), dim=0, keepdim=True)

class SynSAN(nn.Module):
    def __init__(self, args):
        super(SynSAN, self).__init__()
        # syntax-aware self-attention
        self.synselfatt = SynSelfAtt(args.in_dim, args.h_dim)
        # attention
        self.attention = Attention(args.h_dim)
        self.drop = nn.Dropout(0.5)
        # classifier
        self.classifier = nn.Linear(args.h_dim, args.num_class)

    def forward(self, batch_x):
        # syntax-aware self-attention
        batch_z = self.synselfatt(batch_x)

        # attention
        batch_sent_vec = self.drop(torch.cat([self.attention(z) for z in batch_z], dim=0))

        return F.log_softmax(self.classifier(batch_sent_vec), dim=-1)