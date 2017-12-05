#!/bin/bash
for ((i=1;i<=30;i++));
    do
        python BiLSTM.py --embedding_freeze --epoches 150 --batch_size 128 --alias embedding_freeze
        python TestSemEval2016.py --batch_size 128 --alias embedding_freeze
    done