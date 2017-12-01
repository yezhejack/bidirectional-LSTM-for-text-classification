#!/bin/bash
python BiLSTM.py --hidden_size 600 --embedding_path ~/data/GoogleNews-vectors-negative300.bin --isBinary --embedding_freeze --epoches 30
