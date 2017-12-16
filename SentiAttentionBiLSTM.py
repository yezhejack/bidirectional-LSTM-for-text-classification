#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import time
import json
import logging
import argparse
import os
import math
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils.weight_norm as weight_norm
from gensim.models.keyedvectors import KeyedVectors
import data_loader
from models.RNN import *
logging.basicConfig(level=logging.INFO)

def safe_div(x,y):
    if y == 0:
        return 0.0
    return float(x)/y

def calculate_metrics(gold,pred,labels):
    if gold.shape != pred.shape:
        logging.info('gold shape is not same as pred shape')
        return 0.0, 0.0
    num_correct = np.sum(gold==pred)
    matrix = np.zeros([len(labels), len(labels)], dtype=np.int32)
    for label_gold, label_pred in zip(gold, pred):
        matrix[label_pred, label_gold] += 1
    acc_pos = safe_div(matrix[1,1], np.sum(matrix[1,:]))
    recall_pos = safe_div(matrix[1,1], np.sum(matrix[:,1]))
    F1_pos = safe_div(2*recall_pos*acc_pos, recall_pos+acc_pos)

    acc_neg = safe_div(matrix[0,0], np.sum(matrix[0,:]))
    recall_neg = safe_div(matrix[0,0], np.sum(matrix[:,0]))
    F1_neg = safe_div(2*recall_neg*acc_neg, recall_neg+acc_neg)
    return safe_div(F1_pos+F1_neg, 2), safe_div(num_correct, gold.shape[0])

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--embedding_path",
                        help="the path to the embedding, word2vec format",
                        default='data/GoogleNews-vectors-negative300.align.txt')
    parser.add_argument("--isBinary", action="store_true")
    parser.add_argument("--gru", help="provide this for using gru layer instead of lstm layer", action="store_true")
    parser.add_argument("--embedding_freeze", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epoches", type=int, default=50)
    parser.add_argument("--max_len_rnn", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alias", default="bilstm")
    args=parser.parse_args()
    vocab_path = os.path.join("data", "%s.vocab" % (args.alias))
    checkpoint_path = os.path.join("checkpoint", "%s.ckp" % (args.alias))
    # set seed
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    # Load word embedding and build vocabulary
    wv = KeyedVectors.load_word2vec_format(args.embedding_path, binary=args.isBinary)
    index_to_word = [key for key in wv.vocab]
    word_to_index = {}
    for index, word in enumerate(index_to_word):
        word_to_index[word] = index
    with open(vocab_path, "w") as f:
        f.write(json.dumps(word_to_index))

    dataset = data_loader.Load_SemEval2016(word_to_index, max_len=args.max_len_rnn)
    embed_size = wv[index_to_word[0]].size
    vocab_size = len(index_to_word)
    embedding_matrix = np.zeros((vocab_size, embed_size), dtype=np.float)
    for i, word in enumerate(index_to_word):
        embedding_matrix[i] = np.array(wv[word])
    print("embed_size:%d" % (embed_size))
    print("vocab_size:%d" % (vocab_size))
    collate_fn = data_loader.my_collate_fn
    if torch.cuda.is_available():
        collate_fn = data_loader.my_collate_fn_cuda

    train_iter = DataLoader(data_loader.MyData(dataset['train_sentences'], dataset['train_labels']), args.batch_size, shuffle=True, collate_fn=collate_fn)
    weight = torch.FloatTensor([0.0, 0.0, 0.0])
    for batch in train_iter:
        for label in batch['labels']:
            weight[int(label)] += 1
    weight = 1 / weight
    weight = 3 / torch.sum(weight) * weight
    dev_iter = DataLoader(data_loader.MyData(dataset['dev_sentences'], dataset['dev_labels']), args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_iter = DataLoader(data_loader.MyData(dataset['test_sentences'], dataset['test_labels']), args.batch_size, shuffle=False, collate_fn=collate_fn)
    sen_list, label_list = data_loader.Load_SemEval2016_Test(word_to_index, max_len=args.max_len_rnn)
    sem_iter = DataLoader(data_loader.MyData(sen_list, label_list), args.batch_size, shuffle=False, collate_fn=collate_fn)
    model = ""
    if args.gru:
        model = BiGRU(embedding_matrix, hidden_size=args.hidden_size, embedding_freeze=args.embedding_freeze)
    else:
        model = SentiAttentionBiLSTM(embedding_matrix, hidden_size=args.hidden_size, embedding_freeze=args.embedding_freeze)
    del(embedding_matrix)
    del(index_to_word)
    del(word_to_index)
    if torch.cuda.is_available():
        model.cuda()
        weight = weight.cuda()

    optimizer = torch.optim.Adam(model.custom_params, lr=args.lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    criterion = nn.CrossEntropyLoss(weight=weight, size_average=False)
    eval_criterion = nn.CrossEntropyLoss(size_average=False)
    max_dev_F1 = 0.0
    final_test_F1 = 0.0
    for epoch in range(args.epoches):
        epoch_sum = 0.0
        # training
        model.train()
        gold = []
        pred = []
        for i, batch in enumerate(train_iter):
            model.hidden1 = model.init_hidden(batch_size = int(batch['labels'].data.size()[0]))
            model.hidden2 = model.init_hidden(batch_size = int(batch['labels'].data.size()[0]))
            optimizer.zero_grad()
            output = model(batch['sentence'])
            _, outputs_label = torch.max(output, 1)
            loss = criterion(output, batch['labels'])
            torch.nn.utils.clip_grad_norm(model.custom_params, 5)
            #print('loss=%f' % (loss.data[0]/int(batch['labels'].data.size()[0])))
            epoch_sum += loss.data[0]
            loss.backward()
            optimizer.step()
            for pred_label in outputs_label:
                pred.append(int(pred_label))
            for gold_label in batch['labels'].data:
                gold.append(int(gold_label))
        F1, Acc = calculate_metrics(np.array(gold), np.array(pred), [0,1,2])
        print('[#%d epoch] train avg loss = %f / F1 = %f / Acc = %f' % (epoch+1, epoch_sum/len(dataset['train_labels']), F1, Acc))

        # evaluate dev data
        model.eval()
        gold = []
        pred = []
        epoch_sum = 0.0
        for i, batch in enumerate(dev_iter):
            model.hidden1 = model.init_hidden(batch_size = int(batch['labels'].data.size()[0]))
            model.hidden2 = model.init_hidden(batch_size = int(batch['labels'].data.size()[0]))
            output = model(batch['sentence'])
            _, outputs_label = torch.max(output, 1)
            loss = eval_criterion(output, batch['labels'])
            epoch_sum += loss.data[0]
            for pred_label in outputs_label:
                pred.append(int(pred_label))
            for gold_label in batch['labels'].data:
                gold.append(int(gold_label))
        #scheduler.step(epoch_sum)
        F1, Acc = calculate_metrics(np.array(gold), np.array(pred), [0,1,2])
        print('[#%d epoch] dev avg loss = %f / F1 = %f / Acc = %f' % (epoch+1, epoch_sum/len(dataset['dev_labels']), F1, Acc))

        gold = []
        pred = []
        epoch_sum = 0.0
        for i, batch in enumerate(test_iter):
            model.hidden1 = model.init_hidden(batch_size = int(batch['labels'].data.size()[0]))
            model.hidden2 = model.init_hidden(batch_size = int(batch['labels'].data.size()[0]))
            output = model(batch['sentence'])
            _, outputs_label = torch.max(output, 1)
            loss = eval_criterion(output, batch['labels'])
            epoch_sum += loss.data[0]
            for pred_label in outputs_label:
                pred.append(int(pred_label))
            for gold_label in batch['labels'].data:
                gold.append(int(gold_label))
        test_F1, Acc = calculate_metrics(np.array(gold), np.array(pred), [0,1,2])
        print('\033[1;32m[#%d epoch] test avg loss = %f / F1 = %f / Acc = %f\033[0m' % (epoch+1, epoch_sum/len(dataset['test_labels']), test_F1, Acc))
        if F1 > max_dev_F1:
            max_dev_F1 = F1
            final_test_F1 = test_F1
            if os.path.exists(checkpoint_path)==False:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            os.system("rm %s" % (checkpoint_path))
            torch.save(model, checkpoint_path)

    print(sys.argv)
    print('Dev F1 = %f, Test F1 = %f' % (max_dev_F1, final_test_F1))
