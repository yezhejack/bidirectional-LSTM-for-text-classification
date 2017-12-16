#!/bin/bash
start_time=`date`
echo $start_time
for ((i=1;i<=30;i++));
    do
        python BiLSTM.py --embedding_path ~/corpus/datastories.twitter.300d.txt --padding '<pad>' --embedding_freeze --epoches 50 --batch_size 128  --alias datastories
        python TestSemEval2016.py --batch_size 128 --alias datastories
    done
echo $start_time
echo `date`