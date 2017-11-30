import logging
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
            with open("data/SemEval2016/%s.%s.tokenize" % (mode, label)) as f:
                line = f.readline()
                while line != "":
                    sen = []
                    for word in line.split()[:max_len]:
                        if word in word_to_index:
                            sen.append(word_to_index[word])
                        else:
                            sen.append(word_to_index['</s>'])
                    if len(sen) == 0:
                        sen = [word_to_index['</s>']]
                    dataset["%s_sentences" % mode].append(sen)
                    dataset["%s_labels" % mode].append(labels_map[label])
                    line = f.readline()
    logging.info("Data statistic")
    for key in dataset:
        logging.info(key+":"+str(len(dataset[key])))
    return dataset