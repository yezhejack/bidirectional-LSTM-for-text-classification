#!/bin/bash
for ((i=1;i<=30;i++));
    do
        python NoAttentionBiLSTM.py --embedding_freeze --epoches 150 --batch_size 128  --alias NoAttention
        python TestNoAttentionSemEval2016.py --batch_size 128 --alias NoAttention
    done
