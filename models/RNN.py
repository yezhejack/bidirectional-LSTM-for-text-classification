#!/bin/bash python
import logging
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils.weight_norm as weight_norm
class BiLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_size=150, num_layer=2, embedding_freeze=False):
        super(BiLSTM,self).__init__()

        # embedding layer
        vocab_size = embedding_matrix.shape[0]
        embed_size = embedding_matrix.shape[1]
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.embed.weight = nn.Parameter(torch.from_numpy(embedding_matrix).type(torch.FloatTensor), requires_grad=not embedding_freeze)
        self.embed_dropout = nn.Dropout(p=0.3)
        self.custom_params = []
        if embedding_freeze == False:
            self.custom_params.append(self.embed.weight)

        # The first LSTM layer
        self.lstm1 = nn.LSTM(embed_size, self.hidden_size, num_layer, dropout=0.3, bidirectional=True)
        for param in self.lstm1.parameters():
            self.custom_params.append(param)
            if param.data.dim() > 1:
                nn.init.orthogonal(param)
            else:
                nn.init.normal(param)

        self.lstm1_dropout = nn.Dropout(p=0.3)

        # The second LSTM layer
        self.lstm2 = nn.LSTM(2*self.hidden_size, self.hidden_size, num_layer, dropout=0.3, bidirectional=True)
        for param in self.lstm2.parameters():
            self.custom_params.append(param)
            if param.data.dim() > 1:
                nn.init.orthogonal(param)
            else:
                nn.init.normal(param)
        self.lstm2_dropout = nn.Dropout(p=0.3)
        # Attention
        self.attention = nn.Linear(2*self.hidden_size,1)
        self.attention_dropout = nn.Dropout(p=0.5)

        # Fully-connected layer
        self.fc = weight_norm(nn.Linear(2*self.hidden_size,3))
        for param in self.fc.parameters():
            self.custom_params.append(param)
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

    def forward(self, sentences):
        # get embedding vectors of input
        padded_sentences, lengths = torch.nn.utils.rnn.pad_packed_sequence(sentences, padding_value=int(0), batch_first=True)
        embeds = self.embed(padded_sentences)
        noise = Variable(torch.zeros(embeds.shape).cuda())
        noise.data.normal_(std=0.3)
        embeds += noise
        embeds = self.embed_dropout(embeds)
        # add noise
        
        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True)
        
        # First LSTM layer
        # self.hidden = num_layers*num_directions batch_size hidden_size
        packed_out_lstm1, self.hidden1 = self.lstm1(packed_embeds, self.hidden1)
        padded_out_lstm1, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_out_lstm1, padding_value=int(0))
        padded_out_lstm1 = self.lstm1_dropout(padded_out_lstm1)
        packed_out_lstm1 = torch.nn.utils.rnn.pack_padded_sequence(padded_out_lstm1, lengths)
   
        # Second LSTM layer
        packed_out_lstm2, self.hidden2 = self.lstm2(packed_out_lstm1, self.hidden2)
        padded_out_lstm2, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_out_lstm2, padding_value=int(0), batch_first=True)
        padded_out_lstm2 = self.lstm2_dropout(padded_out_lstm2)

        # attention
        unnormalize_weight = F.tanh(torch.squeeze(self.attention(padded_out_lstm2), 2))
        unnormalize_weight = F.softmax(unnormalize_weight, dim=1)
        unnormalize_weight = torch.nn.utils.rnn.pack_padded_sequence(unnormalize_weight, lengths, batch_first=True)
        unnormalize_weight, lengths = torch.nn.utils.rnn.pad_packed_sequence(unnormalize_weight, padding_value=0.0, batch_first=True)
        logging.debug("unnormalize_weight size: %s" % (str(unnormalize_weight.size())))
        normalize_weight = torch.nn.functional.normalize(unnormalize_weight, p=1, dim=1)
        normalize_weight = normalize_weight.view(normalize_weight.size(0), 1, -1)
        weighted_sum = torch.squeeze(normalize_weight.bmm(padded_out_lstm2), 1)
        
        # fully connected layer
        output = self.fc(self.attention_dropout(weighted_sum))
        return output

class BiGRU(nn.Module):
    def __init__(self, embedding_matrix, hidden_size=100, num_layer=2, embedding_freeze=False):
        super(BiGRU, self).__init__()

        # embedding layer
        vocab_size = embedding_matrix.shape[0]
        embed_size = embedding_matrix.shape[1]
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.embed.weight = nn.Parameter(torch.from_numpy(embedding_matrix).type(torch.FloatTensor), requires_grad=not embedding_freeze)
        self.embed_dropout = nn.Dropout(p=0.3)
        self.custom_params = []
        if embedding_freeze == False:
            self.custom_params.append(self.embed.weight)

        # The first GRU layer
        self.gru1 = nn.GRU(embed_size, self.hidden_size, num_layer, dropout=0.5, bidirectional=True)
        for param in self.gru1.parameters():
            self.custom_params.append(param)
            if param.data.dim() > 1:
                nn.init.orthogonal(param)
            else:
                nn.init.normal(param)

        self.connection_dropout = nn.Dropout(p=0.25)

        # The second LSTM layer
        self.gru2 = nn.GRU(self.hidden_size, self.hidden_size, num_layer, dropout=0.5, bidirectional=True)
        for param in self.gru2.parameters():
            self.custom_params.append(param)
            if param.data.dim() > 1:
                nn.init.orthogonal(param)
            else:
                nn.init.normal(param)

        # Attention
        self.attention = nn.Linear(2*self.hidden_size,1)

        # Fully-connected layer
        self.fc = weight_norm(nn.Linear(2*self.hidden_size,3))
        for param in self.fc.parameters():
            self.custom_params.append(param)
            if param.data.dim() > 1:
                nn.init.orthogonal(param)
            else:
                nn.init.normal(param)

        self.hidden1=self.init_hidden()
        self.hidden2=self.init_hidden()

    def init_hidden(self, batch_size=3):
        if torch.cuda.is_available():
            return Variable(torch.zeros(self.num_layer*2, batch_size, self.hidden_size)).cuda()
        else:
            return Variable(torch.zeros(self.num_layer*2, batch_size, self.hidden_size))

    def forward(self, sentences):
        # get embedding vectors of input
        padded_sentences, lengths = torch.nn.utils.rnn.pad_packed_sequence(sentences, padding_value=int(0), batch_first=True)
        embeds = self.embed_dropout(self.embed(padded_sentences))
        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True)

        # self.hidden = num_layers*num_directions batch_size hidden_size
        out_gru1, self.hidden1 = self.gru1(packed_embeds, self.hidden1)
        padded_out_gru1, lengths = torch.nn.utils.rnn.pad_packed_sequence(out_gru1, padding_value=int(0))
        logging.debug("padded_out_gru size:%s" % (str(padded_out_gru1.size())))
        logging.debug("hidden_size=%d" % (self.hidden_size))
        sum_padded_out_gru1 = 0
        for tensor in torch.split(padded_out_gru1, self.hidden_size, dim=2):
            sum_padded_out_gru1 += tensor
        packed_out_gru1 = torch.nn.utils.rnn.pack_padded_sequence(self.connection_dropout(sum_padded_out_gru1), lengths)

        # gru2 and
        packed_out_gru2, self.hidden2 = self.gru2(packed_out_gru1, self.hidden2)

        # attention
        padded_out_gru2, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_out_gru2, padding_value=int(0), batch_first=True)
        unnormalize_weight = F.tanh(torch.squeeze(self.attention(padded_out_gru2), 2)) # seq_len x batch_size
        unnormalize_weight = F.softmax(unnormalize_weight, dim=1)
        unnormalize_weight = torch.nn.utils.rnn.pack_padded_sequence(unnormalize_weight, lengths, batch_first=True)
        unnormalize_weight, lengths = torch.nn.utils.rnn.pad_packed_sequence(unnormalize_weight, padding_value=0.0, batch_first=True)
        logging.debug("unnormalize_weight size: %s" % (str(unnormalize_weight.size())))
        normalize_weight = torch.nn.functional.normalize(unnormalize_weight, p=1, dim=1)
        normalize_weight = normalize_weight.view(normalize_weight.size(0), 1, -1)
        weighted_sum = torch.squeeze(normalize_weight.bmm(padded_out_gru2), 1)

        # fully connected layer
        output = self.fc(weighted_sum)
        return output

class NoAttentionBiLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_size=100, num_layer=2, embedding_freeze=False):
        super(NoAttentionBiLSTM, self).__init__()

        # embedding layer
        vocab_size = embedding_matrix.shape[0]
        embed_size = embedding_matrix.shape[1]
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.embed.weight = nn.Parameter(torch.from_numpy(embedding_matrix).type(torch.FloatTensor), requires_grad=not embedding_freeze)
        self.embed_dropout = nn.Dropout(p=0.3)
        self.custom_params = []
        if embedding_freeze == False:
            self.custom_params.append(self.embed.weight)

        # The first LSTM layer
        self.lstm1 = nn.LSTM(embed_size, self.hidden_size, num_layer, dropout=0.5, bidirectional=True)
        for param in self.lstm1.parameters():
            self.custom_params.append(param)
            if param.data.dim() > 1:
                nn.init.orthogonal(param)
            else:
                nn.init.normal(param)

        self.connection_dropout = nn.Dropout(p=0.25)

        # The second LSTM layer
        self.lstm2 = nn.LSTM(self.hidden_size, self.hidden_size, num_layer, dropout=0.5, bidirectional=True)
        for param in self.lstm2.parameters():
            self.custom_params.append(param)
            if param.data.dim() > 1:
                nn.init.orthogonal(param)
            else:
                nn.init.normal(param)

        # Fully-connected layer
        self.fc = weight_norm(nn.Linear(2*self.hidden_size,3))
        for param in self.fc.parameters():
            self.custom_params.append(param)
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

    def forward(self, sentences):
        # get embedding vectors of input
        padded_sentences, lengths = torch.nn.utils.rnn.pad_packed_sequence(sentences, padding_value=int(0), batch_first=True)
        embeds = self.embed_dropout(self.embed(padded_sentences))
        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True)

        # self.hidden = num_layers*num_directions batch_size hidden_size
        out_lstm1, self.hidden1 = self.lstm1(packed_embeds, self.hidden1)
        padded_out_lstm1, lengths = torch.nn.utils.rnn.pad_packed_sequence(out_lstm1, padding_value=int(0))
        logging.debug("padded_out_lstm size:%s" % (str(padded_out_lstm1.size())))
        logging.debug("hidden_size=%d" % (self.hidden_size))
        sum_padded_out_lstm1 = 0
        for tensor in torch.split(padded_out_lstm1, self.hidden_size, dim=2):
            sum_padded_out_lstm1 += tensor
        packed_out_lstm1 = torch.nn.utils.rnn.pack_padded_sequence(self.connection_dropout(sum_padded_out_lstm1), lengths)

        # lstm2
        packed_out_lstm2, self.hidden2 = self.lstm2(packed_out_lstm1, self.hidden2)

        # average sum of output of lstm
        padded_out_lstm2, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_out_lstm2, padding_value=int(0), batch_first=True)
        unnormalize_weight = Variable(torch.ones(padded_out_lstm2.shape[0], lengths[0]).cuda()) # batch_size x seq_len
        unnormalize_weight = torch.nn.utils.rnn.pack_padded_sequence(unnormalize_weight, lengths, batch_first=True)
        unnormalize_weight, lengths = torch.nn.utils.rnn.pad_packed_sequence(unnormalize_weight, padding_value=0.0, batch_first=True)
        logging.debug("unnormalize_weight size: %s" % (str(unnormalize_weight.size())))
        normalize_weight = torch.nn.functional.normalize(unnormalize_weight, p=1, dim=1)
        normalize_weight = normalize_weight.view(normalize_weight.size(0), 1, -1)
        weighted_sum = torch.squeeze(normalize_weight.bmm(padded_out_lstm2), 1)

        # fully connected layer
        output = self.fc(weighted_sum)
        return output

class SentiAttentionBiLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_size=100, num_layer=2, embedding_freeze=False):
        super(SentiAttentionBiLSTM, self).__init__()

        # embedding layer
        vocab_size = embedding_matrix.shape[0]
        embed_size = embedding_matrix.shape[1]
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.embed.weight = nn.Parameter(torch.from_numpy(embedding_matrix).type(torch.FloatTensor), requires_grad=not embedding_freeze)
        self.embed_dropout = nn.Dropout(p=0.3)
        self.custom_params = []
        if embedding_freeze == False:
            self.custom_params.append(self.embed.weight)

        # The first LSTM layer
        self.lstm1 = nn.LSTM(embed_size, self.hidden_size, num_layer, dropout=0.5, bidirectional=True)
        for param in self.lstm1.parameters():
            self.custom_params.append(param)
            if param.data.dim() > 1:
                nn.init.orthogonal(param)
            else:
                nn.init.normal(param)

        self.connection_dropout = nn.Dropout(p=0.25)

        # The second LSTM layer
        self.lstm2 = nn.LSTM(self.hidden_size, self.hidden_size, num_layer, dropout=0.5, bidirectional=True)
        for param in self.lstm2.parameters():
            self.custom_params.append(param)
            if param.data.dim() > 1:
                nn.init.orthogonal(param)
            else:
                nn.init.normal(param)

        # load attention layer
        self.attention = torch.load("../sentiment_softmax/checkpoint/attention.300.3.cpk")
        for param in self.attention.parameters():
            self.custom_params.append(param)
        self.mask = Variable(torch.cuda.FloatTensor([-1, 0, 1]).view(3,1))
        # Fully-connected layer
        self.fc = weight_norm(nn.Linear(2*self.hidden_size,3))
        for param in self.fc.parameters():
            self.custom_params.append(param)
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

    def forward(self, sentences):
        # get embedding vectors of input
        padded_sentences, lengths = torch.nn.utils.rnn.pad_packed_sequence(sentences, padding_value=int(0), batch_first=True)
        embeds = self.embed_dropout(self.embed(padded_sentences))
        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True)

        # self.hidden = num_layers*num_directions batch_size hidden_size
        out_lstm1, self.hidden1 = self.lstm1(packed_embeds, self.hidden1)
        padded_out_lstm1, lengths = torch.nn.utils.rnn.pad_packed_sequence(out_lstm1, padding_value=int(0))
        logging.debug("padded_out_lstm size:%s" % (str(padded_out_lstm1.size())))
        logging.debug("hidden_size=%d" % (self.hidden_size))
        sum_padded_out_lstm1 = 0
        for tensor in torch.split(padded_out_lstm1, self.hidden_size, dim=2):
            sum_padded_out_lstm1 += tensor
        packed_out_lstm1 = torch.nn.utils.rnn.pack_padded_sequence(self.connection_dropout(sum_padded_out_lstm1), lengths)

        # lstm2
        packed_out_lstm2, self.hidden2 = self.lstm2(packed_out_lstm1, self.hidden2)

        # attention
        padded_out_lstm2, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_out_lstm2, padding_value=int(0), batch_first=True)
        unnormalize_weight = torch.squeeze(self.attention(embeds).matmul(self.mask), dim=2) # batch_size x seq_len
        unnormalize_weight = torch.nn.utils.rnn.pack_padded_sequence(unnormalize_weight, lengths, batch_first=True)
        unnormalize_weight, lengths = torch.nn.utils.rnn.pad_packed_sequence(unnormalize_weight, padding_value=0.0, batch_first=True)
        logging.debug("unnormalize_weight size: %s" % (str(unnormalize_weight.size())))
        norm = Variable(torch.cuda.FloatTensor(lengths)).view(-1, 1)
        norm = 1 / norm
        normalize_weight = unnormalize_weight * norm
        normalize_weight = normalize_weight.view(normalize_weight.shape[0], 1, -1)
        logging.debug("normalize_weight size: %s" % (str(normalize_weight.size())))
        weighted_sum = torch.squeeze(normalize_weight.bmm(padded_out_lstm2), 1)

        # fully connected layer
        output = self.fc(weighted_sum)
        return output