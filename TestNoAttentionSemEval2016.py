# coding: utf-8
import torch
import json
import numpy as np
import argparse
import os
import logging
import subprocess
import os
from NoAttentionBiLSTM import *
import data_loader
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--semeval_file",help="the path of the test file",default = "data/SemEval2016/SemEval2016-task4-test.subtask-A.tokenize")
    parser.add_argument("--max_length",help='sentence size',type = int, default = 100)
    parser.add_argument("--batch_size",help='batch size',type = int, default = 32)
    parser.add_argument("--alias",help="the alias of the experiment", default="bilstm")
    args=parser.parse_args()

    vocab_path = os.path.join("data", "%s.vocab" % (args.alias))
    checkpoint_path = os.path.join("checkpoint", "%s.ckp" % (args.alias))
    result_path = os.path.join("result", "%s.csv" % (args.alias))

    f=open(vocab_path)
    word_to_index=json.loads(f.readline())
    f.close()
    
    # prepare test data
    sentence_list, fake_label_list = data_loader.Load(word_to_index, args.semeval_file, max_len=args.max_length)
    test_iter = DataLoader(MyData(sentence_list, fake_label_list), args.batch_size, collate_fn=my_collate_fn_cuda)
    
    # load checkpoint model
    model = torch.load(checkpoint_path)
    model.eval()
    # predict
    pred = []
    for i, batch in enumerate(test_iter):
        model.hidden1 = model.init_hidden(batch_size = int(batch['labels'].data.size()[0]))
        model.hidden2 = model.init_hidden(batch_size = int(batch['labels'].data.size()[0]))
        output = model(batch['sentence'])
        _, outputs_label = torch.max(output, 1)
        for index in batch['reverse_sorted_index']:
            pred.append(int(outputs_label.data[index]))
    
    with open('scorer/%s.pred' % (args.alias), 'w') as o:
        labels = ['negative','positive','neutral']
        for i,p in enumerate(pred):
            o.write('%d\t%s\n' % (i+1,labels[p]))
    
    os.chdir("scorer")
    out_bytes = subprocess.check_output(['perl','SemEval2016_task4_test_scorer_subtaskA.pl','%s.pred' % (args.alias)])
    output = out_bytes.decode('utf-8')
    os.chdir("..")
    
    print(output)
    output = output.split()

    # output to the csv result
    if os.path.exists(result_path):
        with open(result_path,'a') as o:
            o.write(args.alias)
            for i in range(7):
                o.write('\t%s' % (output[2+i*4]))
            o.write('\n')
    else:
        with open(result_path,'w') as o:
            o.write("Alias")
            for i in range(7):
                o.write('\t%s' %(output[1+i*4]))
            o.write('\n')
            o.write(args.alias)
            for i in range(7):
                o.write('\t%s' % (output[2+i*4]))
            o.write('\n')