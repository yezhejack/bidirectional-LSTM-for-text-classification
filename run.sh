#!/bin/bash
for ((i=1;i<=30;i++));
    do
        python BiLSTM.py --epoches 150 --batch_size 128 --gru --alias gru
        python TestSemEval2016.py --batch_size 128 --alias gru
    done
