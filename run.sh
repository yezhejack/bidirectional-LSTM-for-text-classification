#!/bin/bash
for ((i=1;i<=30;i++));
    do
        python MyAttentionBiLSTM.py --embedding_freeze --epoches 150 --batch_size 128  --alias SentiAttention
        python TestSemEval2016.py --batch_size 128 --alias SentiAttention
    done
