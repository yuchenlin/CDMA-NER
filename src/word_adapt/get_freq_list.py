#!/usr/bin/env python
# encoding: utf-8

import sys
import os

vocab = dict()

def update_freq(file):
    txt = open(file).read()
    for w in txt.split():
        if w in vocab:
            vocab[w] += 1
        else:
            vocab[w] = 1
path = '/path/to/your/corpus/' # this is the path to lemmatized corpus (where each file is a single document)
files = os.listdir(path)
print('todo:', len(files))
todo = len(files)

for file in files:
    update_freq(path + file)

with open('/x.freqlist','w') as f: # [x] is twitter or newswire
    f.write("\n".join([w+"\t"+str(vocab[w]) for w in vocab]))



