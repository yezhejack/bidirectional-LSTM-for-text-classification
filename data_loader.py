import logging
import re
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
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

def Load_SemEval2016(word_to_index, max_len=20):
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

def Load_SemEval2016_Test(word_to_index, max_len=20):
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
                sen.append(word_to_index['the'])
        if len(sen) == 0:
            sen = [word_to_index['the']]   
        sen_list.append(sen)
        line_f = f.readline()
        line_g = g.readline()

    return sen_list, label_list
