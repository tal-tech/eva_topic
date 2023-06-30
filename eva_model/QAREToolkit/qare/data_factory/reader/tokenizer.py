#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-07-08
'''

import os
import jieba
# import spacy
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk import word_tokenize

class BaseTokenizer(object):

    def __init__(self, *args, **kwargs):
        pass

    def word_tokenizer(self, *args, **kwargs):
        raise NotImplementedError


# class EnglisthTokenizer(BaseTokenizer):
#
#     def __init__(self):
#         self.nlp = spacy.load('en', disable=['parser','tagger','entity'])
#
#     def word_tokenizer(self, doc):
#         doc = self.nlp(doc)
#         tokens = [token.text for token in doc]
#         return tokens

class EnglisthTokenizer(BaseTokenizer):

    def __init__(self):
        self.word_lemma = WordNetLemmatizer()

    def word_tokenizer(self, sentence, need_segment = False):
        tokens = []
        sentence = sentence.replace("_", "")
        for word, tag in pos_tag(word_tokenize(sentence)):
            word = word.strip().lower()
            if word in ["i", "he", "she", "they", "them", "you", "alse",
                        "am", "is", "are", "was", "were", "be", "have", "had"] or (
                not word.isalpha()
            ):
                continue
            if tag.startswith('NN'):
                tokens.append(self.word_lemma.lemmatize(word, pos='n'))
            elif tag.startswith('VB'):
                tokens.append(self.word_lemma.lemmatize(word, pos='v'))
            elif tag.startswith('JJ'):
                tokens.append(self.word_lemma.lemmatize(word, pos='a'))
            # elif tag.startswith('R'):
            #     tokens.append(self.word_lemma.lemmatize(word, pos='r'))
            # else:
            #     tokens.append(word)
        return tokens


class ChineseTokenizer(BaseTokenizer):

    def __init__(self, user_dict_path = None):
        self.tokenizer = jieba

        if user_dict_path and os.path.exists(user_dict_path):
            self.tokenizer.load_userdict(user_dict_path)
        self.word_tokenizer("api acceleration")


    def word_tokenizer(self, sentence, need_segment = True):
        if need_segment:
            tokens = list(self.tokenizer.cut(sentence))
        else:
            tokens = list(sentence.strip())
        return tokens
