import torch
import numpy as np
from sklearn import metrics
from torch.autograd import Variable


def dis_result(pred, label):
    # Eu
    eu = np.mean(np.linalg.norm(pred-label, axis=1))
    # Sr
    sr=np.mean(np.sum(np.abs(pred-label),axis=1)/np.sum(pred+label,axis=1))
    # Sq
    sq=np.mean(np.sum(np.square(pred-label)/(pred+label),axis=1))
    # Co
    co=np.mean(np.sum(pred*label,axis=1)/(np.linalg.norm(pred,axis=1)*(np.linalg.norm(label,axis=1))))
    # In
    inters = 0.0
    for i in range(pred.shape[0]):
        r=np.sum(np.min(np.concatenate((pred[i:i+1],label[i:i+1]),axis=0),axis=0))
        inters+=r
    inters /= pred.shape[0]
    return eu, sr, sq, co, inters


def pair_2_vec(pairs, text, model):
    x_vecs = []
    for i, word in enumerate(text):
        try:
            center_vec = torch.FloatTensor(model[word]).view(1,-1)
        except KeyError:
            center_vec = torch.zeros(1, 300)
        neighbor_vecs = []
        for pair in pairs:
            if word in pair:
                neighbor_word = pair[0] if word==pair[1] else pair[1]
                try:
                    vec = torch.FloatTensor(model[neighbor_word]).view(1,-1)
                except KeyError:
                    vec = torch.zeros(1, 300)
                neighbor_vecs.append(vec)
        if neighbor_vecs == []:
            continue
        x_vecs.append(Variable(torch.cat([center_vec]+neighbor_vecs, dim=0)))
    return x_vecs


def get_batch(batch_index, pair_data, label, data_index, docs, model):
    batch_index = [data_index[idx] for idx in batch_index]
    batch_x = []
    for i, idx in enumerate(batch_index):
        pairs = pair_data[idx]
        if len(pairs) == 0:
            continue
        text = docs[idx]
        x_vecs = pair_2_vec(pairs, text, model)
        batch_x.append(x_vecs)
    batch_y = torch.cat([torch.from_numpy(label[idx]).view(1,-1) for idx in batch_index], dim=0)
    return batch_x, Variable(batch_y)


def train(args, net, optimizer, criterion, pair_data, label, data_index, docs, model):
    net.train()
    optimizer.zero_grad()
    indices = torch.randperm(args.train_size)
    running_loss = 0.
    running_acc = 0.
    for batch_id in range(args.train_size//args.batch_size+1):
        batch_index = indices[batch_id*args.batch_size:(batch_id+1)*args.batch_size]
        train_x, train_y = get_batch(batch_index, pair_data, label, data_index, docs, model)
        out = net(train_x)

        loss = criterion(out, train_y)
        running_loss += loss.data[0]
        # acc
        _, pred = torch.max(out, 1)
        _, y = torch.max(train_y, 1)
        num_correct = (pred == y).sum()
        running_acc += num_correct.data[0]

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print('Loss: {:.4f}, Acc: {:.4f}'.format(running_loss/(batch_id+1), running_acc/args.train_size))


def test(args, net, criterion, pair_data, label, data_index, docs, model):
    net.eval()
    y_true = []
    y_pred = []
    eval_loss = 0.

    batch_index = np.arange(args.train_size, args.data_size)
    test_x, test_y = get_batch(batch_index, pair_data, label, data_index, docs, model)
    out = net(test_x)
    # loss
    loss = criterion(out, test_y)
    eval_loss += loss.data[0]
    # distribution metrics
    dr = dis_result(np.power(np.e, out.data.numpy()), test_y.data.numpy())
    # classification metrics
    _, pred = torch.max(out, 1)
    _, y = torch.max(test_y, 1)
    y_pred.extend(pred.data.numpy().tolist())
    y_true.extend(y.data.numpy().tolist())
    macro_P = metrics.precision_score(y_true, y_pred, average='macro')
    macro_R = metrics.recall_score(y_true, y_pred, average='macro')
    micro_f1 = metrics.f1_score(y_true, y_pred, average='micro')
    macro_f1 = metrics.f1_score(y_true, y_pred, average='macro')

    print('TestLoss:{:.4f}, MicroF1:{:.4f}, MacroF1:{:.4f}, MacroP:{:.4f}, MacroR:{:.4f}'.format(
          eval_loss, micro_f1, macro_f1, macro_P, macro_R))
    print('eu:{:.4f},sr:{:.4f},sq:{:.4f},kl:{:.4f},co:{:.4f},in:{:.4f}'.format(
          dr[0],dr[1],dr[2],eval_loss*args.num_class,dr[3],dr[4]))
    return np.power(np.e, out.data.numpy()), test_y.data.numpy()
