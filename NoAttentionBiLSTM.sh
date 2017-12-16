#!/bin/bash
for ((i=1;i<=30;i++));
    do
        python NoAttentionBiLSTM.py --embedding_freeze --epoches 50 --batch_size 128  --alias NoAttentionBiLSTM
        python TestSemEval2016.py --batch_size 128  --alias NoAttentionBiLSTM
    done
