#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-07-29
'''

import torch
import torch.nn.functional as F
from eva_model.QAREToolkit.qare.models.base_model import BaseModel
from eva_model.QAREToolkit.qare.nn.layers import Linear
from eva_model.QAREToolkit.qare.nn.layers import WordEmbedding, RNNLayer, Conv2d, Highway


class BiDAF(BaseModel):
    """
    BiDAF model
    Args:
        vocab: The object of vocabulary
        config: The object of model config
    Inputs:
        - paragraph: (batch, paragraph_sentence_len)
        - question: (batch, question_sentence_len)

    Outputs:
        - output: (batch, n_classes)
    """

    def __init__(self, vocab, config):
        super(BiDAF, self).__init__(vocab, config)

        # 1. Character Embedding Layer
        self.char_emb = WordEmbedding(vocab.get_char_vocab_size(), self.config.char_embedding_size)
        torch.nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)

        self.char_conv = Conv2d(1, self.config.char_channel_size, (self.config.char_embedding_size, self.config.char_channel_width))

        # 2. Word Embedding Layer
        self.word_emb = WordEmbedding(vocab_size=vocab.get_word_vocab_size(),
                                      embd_size=self.config.word_embedding_size,
                                      pre_word_embd=vocab.word_embd)

        # Highway network
        assert self.config.hidden_size * 2 == (self.config.char_channel_size + self.config.word_embedding_size)
        self.highway_network = Highway(2, self.config.hidden_size * 2)

        # 3. Contextual Embedding Layer
        self.context_LSTM = RNNLayer(input_size=self.config.hidden_size * 2,
                                     hidden_size=self.config.hidden_size,
                                     dropout_p=self.config.dropout,
                                     bidirectional=True)

        # 4. Attention Flow Layer
        self.att_weight_p = Linear(self.config.hidden_size * 2, 1)
        self.att_weight_q = Linear(self.config.hidden_size * 2, 1)
        self.att_weight_pq = Linear(self.config.hidden_size * 2, 1)

        # 5. Modeling Layer
        self.modeling_LSTM1 = RNNLayer(input_size=self.config.hidden_size * 8,
                                       hidden_size=self.config.hidden_size,
                                       dropout_p=self.config.dropout,
                                       bidirectional=True)

        self.modeling_LSTM2 = RNNLayer(input_size=self.config.hidden_size * 2,
                                       hidden_size=self.config.hidden_size,
                                       dropout_p=self.config.dropout,
                                       bidirectional=True)

        # 6. Output Layer
        self.output_LSTM = RNNLayer(input_size=self.config.hidden_size * 2,
                                    hidden_size=self.config.hidden_size,
                                    dropout_p=self.config.dropout,
                                    bidirectional=True)
        self.dropout = torch.nn.Dropout(p=self.config.dropout)
        self.fc_last = torch.nn.Linear(self.config.hidden_size * 2, self.config.n_classes)

    def loss_function(self):
        return torch.nn.CrossEntropyLoss()

    def optimizer(self):
        optimizer_kwargs = {"lr": self.config.learning_rate,
                            "rho": 0.9, "eps": 1e-6, "weight_decay": 0}
        optimizer_ = torch.optim.Adadelta
        return optimizer_, optimizer_kwargs

    def forward(self, question, paragraph):
        # 1. Character Embedding Layer
        paragraph_char = self.char_emb_layer(paragraph["char"])
        question_char = self.char_emb_layer(question["char"])

        # 2. Word Embedding Layer
        paragraph_word, paragraph_mask = self.word_emb.forward(paragraph["word"])
        question_word, question_mask = self.word_emb.forward(question["word"])

        # Highway network

        p_mix_char_word = torch.cat((paragraph_char, paragraph_word), dim=-1)
        q_mix_char_word = torch.cat((question_char, question_word), dim=-1)
        p = self.highway_network.forward(p_mix_char_word)
        q = self.highway_network.forward(q_mix_char_word)

        # 3. Contextual Embedding Layer
        p, _ = self.context_LSTM.forward(p, paragraph_mask)
        q, _ = self.context_LSTM.forward(q, question_mask)

        # 4. Attention Flow Layer
        g = self.att_flow_layer(p, q)
        
        # 5. Modeling Layer
        _m1, _ = self.modeling_LSTM1.forward(g, paragraph_mask)
        m, _ = self.modeling_LSTM2.forward(_m1, paragraph_mask)
        
        # 6. Output Layer
        output = self.output_layer(m, paragraph_mask)

        return output

    def char_emb_layer(self, x):
        """
        :param x: (batch, seq_len, word_len)
        :return: (batch, seq_len, char_channel_size)
        """
        batch_size = x.size(0)

        # (batch, seq_len, word_len, char_dim)
        _char_emb, _ = self.char_emb.forward(x)
        x = self.dropout(_char_emb)
        x = x.transpose(2, 3)
        # (batch * seq_len, 1, char_dim, word_len)
        x = x.view(-1, x.size(2), x.size(3)).unsqueeze(1)

        # (batch * seq_len, char_channel_size, 1, conv_len) -> (batch * seq_len, char_channel_size, conv_len)
        x = self.char_conv.forward(x).squeeze(dim=2)

        # (batch * seq_len, char_channel_size, 1) -> (batch * seq_len, char_channel_size)
        x = F.max_pool1d(x, x.size(2)).squeeze(dim=2)

        # (batch, seq_len, char_channel_size)
        x = x.view(batch_size, -1, self.config.char_channel_size)

        return x

    def att_flow_layer(self, p, q):
        """
        :param p: (batch, p_len, hidden_size * 2)
        :param q: (batch, q_len, hidden_size * 2)
        :return: (batch, p_len, q_len)
        """
        batch_size = p.size(0)
        p_len = p.size(1)
        q_len = q.size(1)

        pq = []
        for i in range(q_len):
            # (batch, 1, hidden_size * 2)
            qi = q.select(1, i).unsqueeze(1)
            # (batch, p_len, 1)
            pi = self.att_weight_pq.forward(p * qi).squeeze(dim=2)
            pq.append(pi)
            
        # (batch, p_len, q_len)
        pq = torch.stack(pq, dim=-1)

        # (batch, p_len, q_len)
        s = self.att_weight_p.forward(p).expand(batch_size, -1, q_len) + \
            self.att_weight_q.forward(q).permute(0, 2, 1).expand(batch_size, p_len, -1) + pq

        # (batch, p_len, q_len)
        a = F.softmax(s, dim=2)
        # (batch, p_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, p_len, hidden_size * 2)
        p2q_att = torch.bmm(a, q)
        # (batch, 1, p_len)
        b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
        # (batch, 1, p_len) * (batch, p_len, hidden_size * 2) -> (batch, hidden_size * 2)
        q2p_att = torch.bmm(b, p).squeeze(dim=1)
        # (batch, p_len, hidden_size * 2) (tiled)
        q2p_att = q2p_att.unsqueeze(1).expand(batch_size, p_len, -1)

        # (batch, p_len, hidden_size * 8)
        x = torch.cat((p, p2q_att, p * p2q_att, p * q2p_att), dim=-1)
        return x

    def output_layer(self, m, l):
        """
        :param m: (batch, p_len ,hidden_size * 2)
        :param l: (batch, p_len)
        :return: o: (batch, n_classes)
        """
        _, m2 = self.output_LSTM.forward(m, l)
        o = self.fc_last(m2)
        o = F.softmax(o, dim=1)  # (batch, n_classes)
        return o
