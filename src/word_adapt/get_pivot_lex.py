# Stop Word List is from https://www.ranks.nl/stopwords
import re
import string
PATH_STOPWORDS = "../../resources/stop_words.txt"
PATH_DOMAIN_1 = "../../resources/nw_lemma.freqlist"
PATH_DOMAIN_2 = "../../resources/twitter_lemma_5.freqlist"

stop_words = set(open(PATH_STOPWORDS, encoding="utf8").read().split())
word_list_1 = []
word_list_2 = []
word_dict_1 = {}
word_dict_2 = {}

import string
alphabet = set(string.ascii_lowercase)

def load_list(path, word_list, word_dict):
    with open(path, encoding="utf8") as lf:
        for line in lf.readlines():
            ls = line.split()

            if ls[0] in stop_words or not alphabet >= set(ls[0].lower()):
                continue
            try:
                word_list.append((ls[0], int(ls[1])))
                word_dict[ls[0]] = int(ls[1])
            except Exception as e:
                continue

    word_list.sort(key=lambda x:x[1], reverse=True)


load_list(PATH_DOMAIN_1, word_list_1, word_dict_1)
load_list(PATH_DOMAIN_2, word_list_2, word_dict_2)

max_s = word_list_1[0][1] # max freq for norm
max_t = word_list_2[0][1]

set_s = set([w[0] for w in word_list_1[:8000]])
set_t = set([w[0] for w in word_list_2[:8000]])

pivot = set_s.intersection(set_t)

print(len(pivot))
print(pivot)

pivot_lexicon = {}
for p in pivot:
    f_s = word_dict_1[p]
    f_t = word_dict_2[p]
    f_s_bar = (f_s + 0.0) / max_s
    f_t_bar = (f_t + 0.0) / max_t
    c = 2 * f_s_bar * f_t_bar / (f_s_bar + f_t_bar)
    pivot_lexicon[p]=c

print(pivot_lexicon)

import json
json.dump(pivot_lexicon, open("../../resources/p1.json", 'w'))