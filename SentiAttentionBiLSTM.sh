#!/bin/bash
for ((i=1;i<=30;i++));
    do
        python SentiAttentionBiLSTM.py --embedding_freeze --epoches 150 --batch_size 128  --alias SentiAttention
        python TestSentiAttentionSemEval2016.py --batch_size 128 --alias SentiAttention
    done

