#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import time
import json
import logging
import argparse
import os
from collections import namedtuple
import math
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils.weight_norm as weight_norm
import torch
from gensim.models.keyedvectors import KeyedVectors
import data_loader
import sys
logging.basicConfig(level=logging.INFO)

class MyData(Dataset):
    def __init__(self,x,y):
        self.x=x
        self.y=y

    def __len__(self):
        return len(self.y)

    def __getitem__(self,idx):
        return {'sentence':self.x[idx],'label':self.y[idx]}

def my_collate_fn(x):
    lengths = np.array([len(term['sentence']) for term in x])
    sorted_index = np.argsort(-lengths)
    lengths = lengths[sorted_index]
    # control the maximum length of LSTM
    max_len = lengths[0]
    batch_size = len(x)
    sentence_tensor = torch.LongTensor(batch_size, int(max_len)).zero_()
    for i, index in enumerate(sorted_index):
        sentence_tensor[i][:lengths[i]] = torch.LongTensor(x[index]['sentence'][:max_len])
    labels = Variable(torch.LongTensor([x[i]['label'] for i in sorted_index]))
    packed_sequences = torch.nn.utils.rnn.pack_padded_sequence(Variable(sentence_tensor.t()), lengths)
    return {'sentence':packed_sequences, 'labels':labels}

def my_collate_fn_cuda(x):
    lengths = np.array([len(term['sentence']) for term in x])
    sorted_index = np.argsort(-lengths)
    lengths = lengths[sorted_index]
    # control the maximum length of LSTM
    max_len = lengths[0]
    batch_size = len(x)
    sentence_tensor = torch.LongTensor(batch_size, int(max_len)).zero_()
    for i, index in enumerate(sorted_index):
        sentence_tensor[i][:lengths[i]] = torch.LongTensor(x[index]['sentence'])
    labels = Variable(torch.LongTensor([x[i]['label'] for i in sorted_index]))
    packed_sequences = torch.nn.utils.rnn.pack_padded_sequence(Variable(sentence_tensor.t()).cuda(), lengths)
    return {'sentence':packed_sequences, 'labels':labels.cuda()}

class BiLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size=100, num_layer=2):
        super(BiLSTM,self).__init__()
        # LSTM layer
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.lstm1 = nn.LSTM(embed_size, self.hidden_size, num_layer, bidirectional=True)
        for param in self.lstm1.parameters():
            if param.data.dim() > 1:
                nn.init.orthogonal(param)
            else:
                nn.init.normal(param)

        self.lstm2 = nn.LSTM(self.hidden_size, self.hidden_size, num_layer, bidirectional=True)

        for param in self.lstm2.parameters():
            if param.data.dim() > 1:
                nn.init.orthogonal(param)
            else:
                nn.init.normal(param)

        # Fully-connected layer
        self.fc = weight_norm(nn.Linear(self.hidden_size,3))
        for param in self.fc.parameters():
            if param.data.dim() > 1:
                nn.init.orthogonal(param)
            else:
                nn.init.normal(param)

        self.hidden1=self.init_hidden()
        self.hidden2=self.init_hidden()

    def init_hidden(self, batch_size=3):
        if torch.cuda.is_available():
            return (Variable(torch.zeros(self.num_layer*2, batch_size, self.hidden_size)).cuda(),
                    Variable(torch.zeros(self.num_layer*2, batch_size, self.hidden_size)).cuda())
        else:
            return (Variable(torch.zeros(self.num_layer*2, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.num_layer*2, batch_size, self.hidden_size)))

    def forward(self, embeds, lengths):
        # get embedding vectors of input
        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True)
        # self.hidden = num_layers*num_directions batch_size hidden_size
        out_lstm1, self.hidden1 = self.lstm1(packed_embeds, self.hidden1)
        padded_out_lstm1, lengths = torch.nn.utils.rnn.pad_packed_sequence(out_lstm1, padding_value=int(0))
        logging.debug("padded_out_lstm size:%s" % (str(padded_out_lstm1.size())))
        logging.debug("hidden_size=%d" % (self.hidden_size))
        sum_padded_out_lstm1 = 0
        for tensor in torch.split(padded_out_lstm1, self.hidden_size, dim=2):
            sum_padded_out_lstm1 += tensor
        packed_out_lstm1 = torch.nn.utils.rnn.pack_padded_sequence(sum_padded_out_lstm1, lengths)

        _, self.hidden2 = self.lstm2(packed_out_lstm1, self.hidden2)
        output = torch.squeeze(self.hidden2[0][-1] + self.hidden2[0][-2], 0)
        logging.debug("output size:%s" % str(output.size()))
        output = self.fc(output)
        return output

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
                        help = "the path to the embedding, word2vec format",
                        default = 'data/GoogleNews-vectors-negative300.align.txt')
    parser.add_argument("--isBinary", action = "store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoches", type=int, default=50)
    parser.add_argument("--max_len_rnn", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    args=parser.parse_args()

    # Load word embedding and build vocabulary
    wv = KeyedVectors.load_word2vec_format(args.embedding_path, binary=args.isBinary)
    index_to_word = [key for key in wv.vocab]
    word_to_index = {}
    for index, word in enumerate(index_to_word):
        word_to_index[word] = index

    # initialize embedding layer
    dataset = data_loader.Load_SemEval2016(word_to_index, max_len=args.max_len_rnn)
    embed_size = wv[index_to_word[0]].size
    vocab_size = len(index_to_word)
    embedding_matrix = np.zeros((vocab_size, embed_size), dtype=np.float)
    for i, word in enumerate(index_to_word):
        embedding_matrix[i] = np.array(wv[word])
    vocab_size = embedding_matrix.shape[0]
    embed_size = embedding_matrix.shape[1]
    embedding_matrix = torch.FloatTensor(embedding_matrix)
    embedding_layer = nn.Embedding(vocab_size, embed_size)
    embedding_layer.weight = nn.Parameter(embedding_matrix, requires_grad=False)
    embedding_layer.cuda()
    print("embed_size:%d" % (embed_size))
    print("vocab_size:%d" % (vocab_size))

    collate_fn = my_collate_fn
    if torch.cuda.is_available():
        collate_fn = my_collate_fn_cuda

    train_iter = DataLoader(MyData(dataset['train_sentences'], dataset['train_labels']), args.batch_size, shuffle=True, collate_fn=collate_fn)
    weight = torch.FloatTensor([0.0, 0.0, 0.0])
    for batch in train_iter:
        for label in batch['labels']:
            weight[int(label)] += 1
    weight = 1 / weight
    weight = 3 / torch.sum(weight) * weight
    dev_iter = DataLoader(MyData(dataset['dev_sentences'], dataset['dev_labels']), args.batch_size, collate_fn=collate_fn)
    test_iter = DataLoader(MyData(dataset['test_sentences'], dataset['test_labels']), args.batch_size, collate_fn=collate_fn)

    # build LSTM
    model = BiLSTM(embed_size, hidden_size=args.hidden_size)
    if torch.cuda.is_available():
        model.cuda()
        weight = weight.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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
            #print('batch size=%d' % (batch['labels'].size()[0]))
            #print(batch['labels'].data)
            model.hidden1 = model.init_hidden(batch_size = int(batch['labels'].data.size()[0]))
            model.hidden2 = model.init_hidden(batch_size = int(batch['labels'].data.size()[0]))
            optimizer.zero_grad()
            padded_sentences, lengths = torch.nn.utils.rnn.pad_packed_sequence(batch['sentence'], padding_value=int(0), batch_first=True)
            embeds = embedding_layer(padded_sentences)
            output = model(embeds, lengths)
            _, outputs_label = torch.max(output, 1)
            loss = criterion(output, batch['labels'])
            #print('loss=%f' % (loss.data[0]/int(batch['labels'].data.size()[0])))
            epoch_sum += loss.data[0]
            loss.backward()
            optimizer.step()
            for pred_label in outputs_label:
                pred.append(int(pred_label))
            for gold_label in batch['labels'].data:
                gold.append(int(gold_label))
        F1, Acc = calculate_metrics(np.array(gold), np.array(pred), [0,1,2])
        print('\033[1;32m[#%d epoch] train avg loss = %f / F1 = %f / Acc = %f.\033[0m' % (epoch+1, epoch_sum/len(dataset['train_labels']), F1, Acc))

        # evaluate dev data
        model.eval()
        gold = []
        pred = []
        epoch_sum = 0.0
        for i, batch in enumerate(dev_iter):
            model.hidden1 = model.init_hidden(batch_size = int(batch['labels'].data.size()[0]))
            model.hidden2 = model.init_hidden(batch_size = int(batch['labels'].data.size()[0]))
            padded_sentences, lengths = torch.nn.utils.rnn.pad_packed_sequence(batch['sentence'], padding_value=int(0), batch_first=True)
            embeds = embedding_layer(padded_sentences)
            output = model(embeds, lengths)
            _, outputs_label = torch.max(output, 1)
            loss = eval_criterion(output, batch['labels'])
            epoch_sum += loss.data[0]
            for pred_label in outputs_label:
                pred.append(int(pred_label))
            for gold_label in batch['labels'].data:
                gold.append(int(gold_label))
        scheduler.step(epoch_sum)
        F1, Acc = calculate_metrics(np.array(gold), np.array(pred), [0,1,2])
        print('\033[1;34m[#%d epoch] dev avg loss = %f / F1 = %f / Acc = %f.\033[0m' % (epoch+1, epoch_sum/len(dataset['dev_labels']), F1, Acc))

        gold = []
        pred = []
        epoch_sum = 0.0
        for i, batch in enumerate(test_iter):
            model.hidden1 = model.init_hidden(batch_size = int(batch['labels'].data.size()[0]))
            model.hidden2 = model.init_hidden(batch_size = int(batch['labels'].data.size()[0]))
            padded_sentences, lengths = torch.nn.utils.rnn.pad_packed_sequence(batch['sentence'], padding_value=int(0), batch_first=True)
            embeds = embedding_layer(padded_sentences)
            output = model(embeds, lengths)
            _, outputs_label = torch.max(output, 1)
            loss = eval_criterion(output, batch['labels'])
            epoch_sum += loss.data[0]
            for pred_label in outputs_label:
                pred.append(int(pred_label))
            for gold_label in batch['labels'].data:
                gold.append(int(gold_label))
        test_F1, Acc = calculate_metrics(np.array(gold), np.array(pred), [0,1,2])
        print('\033[1;31m[#%d epoch] test avg loss = %f / F1 = %f / Acc = %f.\033[0m' % (epoch+1, epoch_sum/len(dataset['test_labels']), test_F1, Acc))
        if F1 > max_dev_F1:
            max_dev_F1 = F1
            final_test_F1 = test_F1
    print(sys.argv)
    print('Dev F1 = %f, Test F1 = %f' % (max_dev_F1, final_test_F1))
    with open('result/tunning.txt', 'a') as f:
        for parameter in sys.argv[1:]:
            f.write(parameter+'\t')
        f.write('Dev F1 = %f, Test F1 = %f\n' % (max_dev_F1, final_test_F1))

