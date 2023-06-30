#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-07-08
'''

import os
from collections import Counter
import numpy as np
import logging
import pickle
from tqdm import tqdm


class Vocabulary(object):

    def __init__(self, device,
                 pre_trained_word_embed_file = None,
                 do_lowercase=True,
                 special_tokens=None):
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.word_vocab = None
        self.char_vocab = None
        self.word2idx = None
        self.idx2word = None
        self.char2idx = None
        self.idx2char = None
        self.special_tokens = special_tokens
        self.do_lowercase = do_lowercase
        self.word_embd = None
        self.word_counter = None
        self.device = device
        self.pre_trained_word_embed_file = pre_trained_word_embed_file

    def build_vocab(self, instances, min_word_count=-1, min_char_count=-1):
        # PAD_TOKEN located at index 0 by default
        # UNK_TOKEN located at index 1 by default
        self.word_vocab = [self.PAD_TOKEN, self.UNK_TOKEN]
        self.char_vocab = [self.PAD_TOKEN, self.UNK_TOKEN]

        self.word_counter = Counter()
        self.char_counter = Counter()
        if self.special_tokens is not None and isinstance(self.special_tokens, list):
            self.word_vocab.extend(self.special_tokens)

        logging.info("Building vocabulary.")
        for instance in tqdm(instances):
            for token in instance['question_tokens']:
                token = token.lower() if self.do_lowercase else token
                for char in token:
                    self.char_counter[char] += 1
                self.word_counter[token] += 1
            for token in instance['answer_tokens']:
                token = token.lower() if self.do_lowercase else token
                for char in token:
                    self.char_counter[char] += 1
                self.word_counter[token] += 1
        for w, v in self.word_counter.most_common():
            if v >= min_word_count:
                self.word_vocab.append(w)
        for c, v in self.char_counter.most_common():
            if v >= min_char_count:
                self.char_vocab.append(c)

        if self.pre_trained_word_embed_file and os.path.exists(self.pre_trained_word_embed_file):
            self.make_word_embedding()
        else:
            self._build_index_mapper()

    def _build_index_mapper(self):
        self.word2idx = dict(zip(self.word_vocab, range(len(self.word_vocab))))
        self.idx2word = dict(zip(range(len(self.word_vocab)), self.word_vocab))
        self.char2idx = dict(zip(self.char_vocab, range(len(self.char_vocab))))
        self.idx2char = dict(zip(range(len(self.char_vocab)), self.char_vocab))

    def make_word_embedding(self, init_scale=0.02):

        logging.info("Making word embedding.")
        # 1. Parse pretrained embedding
        embedding_dict = dict()
        vocab_set = set(self.word_vocab)
        with open(self.pre_trained_word_embed_file, mode='r', encoding="utf8") as f:
            for line in tqdm(f):
                if len(line.rstrip().split(" ")) <= 2: continue
                word, vector = line.rstrip().split(" ", 1)
                if word in vocab_set:
                    embedding_dict[word] = np.fromstring(vector, dtype=np.float, sep=" ")

        # 2. Update word vocab according to pretrained word embedding
        newword_vocab = []
        special_tokens_set = set(self.special_tokens if self.special_tokens is not None else [])
        for word in self.word_vocab:
            if word in [self.PAD_TOKEN, self.UNK_TOKEN] or word in embedding_dict or word in special_tokens_set:
                newword_vocab.append(word)
        self.word_vocab = newword_vocab
        self._build_index_mapper()

        # 3. Make word embedding matrix
        embedding_size = embedding_dict[list(embedding_dict.keys())[0]].shape[0]
        embedding_list = []
        for word in self.word_vocab:
            if word == self.PAD_TOKEN:
                embedding_list.append(np.zeros([1, embedding_size], dtype=np.float))
            elif word == self.UNK_TOKEN or word in special_tokens_set:
                embedding_list.append(np.random.uniform(-init_scale, init_scale, [1, embedding_size]))
            else:
                embedding_list.append(np.reshape(embedding_dict[word], [1, embedding_size]))
        self.word_embd = np.concatenate(embedding_list, axis=0)

    def get_pad_idx(self):
        assert self.word2idx[self.PAD_TOKEN] == self.char2idx[self.PAD_TOKEN]
        return self.word2idx[self.PAD_TOKEN]

    def get_unk_idx(self):
        assert self.word2idx[self.UNK_TOKEN] == self.char2idx[self.UNK_TOKEN]
        return self.word2idx[self.UNK_TOKEN]

    def get_word_vocab_size(self):
        if self.word_vocab is None:
            return 0
        return len(self.word_vocab)

    def get_char_vocab_size(self):
        if self.char_vocab is None:
            return 0
        return len(self.char_vocab)

    def get_char_vocab(self):
        if self.char_vocab is None:
            return 0
        return self.char_vocab

    def save(self, file_path):
        logging.info("Saving vocabulary at {}".format(file_path))
        with open(file_path, mode = "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, file_path):
        logging.info("Loading vocabulary from {}".format(file_path))
        with open(file_path, mode = "rb") as f:
            vocab_data = pickle.load(f)
            self.__dict__.update(vocab_data)
