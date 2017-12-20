import logging
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

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

    # build reverse index map to reconstruct the original order
    reverse_sorted_index = np.zeros(len(sorted_index), dtype=int)
    for i, j in enumerate(sorted_index):
        reverse_sorted_index[j]=i
    lengths = lengths[sorted_index]
    # control the maximum length of LSTM
    max_len = lengths[0]
    batch_size = len(x)
    sentence_tensor = torch.LongTensor(batch_size, int(max_len)).zero_()
    for i, index in enumerate(sorted_index):
        sentence_tensor[i][:lengths[i]] = torch.LongTensor(x[index]['sentence'])
    labels = Variable(torch.LongTensor([x[i]['label'] for i in sorted_index]))
    packed_sequences = torch.nn.utils.rnn.pack_padded_sequence(Variable(sentence_tensor.t()).cuda(), lengths)
    return {'sentence':packed_sequences, 'labels':labels.cuda(), 'reverse_sorted_index':reverse_sorted_index}

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def Load(word_to_index, path, max_len=20, label=0):
    sen_list = []
    label_list = []
    with open(path) as f:
        line = f.readline()
        while line != "":
            line = clean_str(line)
            sen = []
            for word in line.split()[:max_len]:
                if word in word_to_index:
                    sen.append(word_to_index[word])
                else:
                    sen.append(word_to_index['the'])
            if len(sen) == 0:
                sen = [word_to_index['the']]
            sen_list.append(sen)
            label_list.append(label)
            line = f.readline()

    return sen_list, label_list

def Load_SemEval2016(word_to_index, max_len=20, padding="</s>"):
    labels_map = {'positive':1, 'neutral':2, 'negative':0}
    dataset = {"train_sentences":[], 
               "train_labels":[],
               "dev_sentences":[],
               "dev_labels":[],
               "test_sentences":[],
               "test_labels":[]}    
    for mode in ['train', 'dev', 'test']:
        for label in ['positive', 'neutral', 'negative']:
            sen_list, label_list = Load(word_to_index, "data/SemEval2016/%s.%s.tokenize" % (mode, label), max_len=max_len, label=labels_map[label])
            dataset["%s_sentences" % mode] += sen_list
            dataset["%s_labels" % mode] += label_list

    logging.info("Data statistic")
    for key in dataset:
        logging.info(key+":"+str(len(dataset[key])))
    return dataset

def Load_SemEval2016_Test(word_to_index, max_len=20, padding="</s>"):
    labels_map = {'positive':1, 'neutral':2, 'negative':0}
    sen_list = []
    label_list = []
    label_file = "scorer/SemEval2016_task4_subtaskA_test_gold.txt"
    input_file = "data/SemEval2016/SemEval2016-task4-test.subtask-A.tokenize"
    f = open(label_file)
    g = open(input_file)
    line_f = f.readline()
    line_g = g.readline()
    while line_f != "" and line_g != "":
        line_f = line_f.split()
        label_list.append(labels_map[line_f[2]])
        line_g = clean_str(line_g)
        sen = []
        for word in line_g.split()[:max_len]:
            if word in word_to_index:
                sen.append(word_to_index[word])
            else:
                sen.append(word_to_index[padding])
        if len(sen) == 0:
            sen = [word_to_index[padding]]   
        sen_list.append(sen)
        line_f = f.readline()
        line_g = g.readline()

    return sen_list, label_list
