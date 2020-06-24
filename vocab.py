import re
import json
import math
import random

import numpy as np


PAD = '<PAD>'
UNK = '<UNK>'
SEP = '<SEP>'
BOS = '<s>'
EOS = '</s>'
UNDER_BAR = '_'
SLASH = '-'
VERB = 'v'
RE_NUM = re.compile(r'[0-9]')


class Vocab(object):
  """Mapping between symbols and IDs"""
  def __init__(self):
    self.i2w = []
    self.w2i = {}

  def add_word(self, word):
    if word not in self.w2i:
      new_id = self.size()
      self.i2w.append(word)
      self.w2i[word] = new_id

  def get_id(self, word):
    if word not in self.w2i:
      return self.w2i[UNK]
    else:
      return self.w2i.get(word)

  def get_word(self, w_id):
    return self.i2w[w_id]

  def has_key(self, word):
    return word in self.w2i

  def size(self):
    return len(self.i2w)

  def __len__(self):
    return len(self.i2w)

  def save(self, path):
    with open(path, 'w', encoding='utf-8') as fout:
      for i, w in enumerate(self.i2w):
        fout.write(str(i) + '\t' + w)

  @classmethod
  def load(self, path):
    vocab = Vocab()
    with open(path, 'r', encoding='utf-8') as fin:
      for line in fin:
        # w = line.strip().split('\t')[1]
        w = line.strip()
        vocab.add_word(w)
    return vocab

  @classmethod
  def load_word(cls, path):
    vocab = Vocab()
    # 0 for PAD, index for words start form 1
    vocab.add_word(PAD)
    with open(path, 'r', encoding='utf-8') as fin:
      for line in fin:
        w = line.strip()
        vocab.add_word(w)
    return vocab

  @classmethod
  def load_word2vec(cls, path):
    vocab = Vocab()
    vocab.add_word(PAD)  # 0 PAD
    vocab.add_word(UNK)  # 1 UNK
    embedding = []

    with open(path, 'r', encoding='utf-8') as fin:
      word_num, dim = fin.readline().strip().split(' ')
      for line in fin:
        e = line.strip().split(' ')
        vocab.add_word(e[0])
        embedding.append(np.array(e[1:], dtype=np.float32))

    pad_emb = np.zeros(int(dim), dtype=np.float32)
    unk_emb = np.mean(embedding, axis=0)
    embedding = [pad_emb, unk_emb] + embedding
    return vocab, np.array(embedding, dtype=np.float32)
