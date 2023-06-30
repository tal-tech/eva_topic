#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-08-12
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from eva_model.QAREToolkit.qare.models.base_model import BaseModel
from eva_model.QAREToolkit.qare.nn.layers import WordEmbedding
from eva_model.QAREToolkit.qare.nn.layers import RNNLayer
from eva_model.QAREToolkit.qare.nn.layers import Linear
from eva_model.QAREToolkit.qare.nn.operations import masked_softmax

class RNet(BaseModel):
    """
    R-Net model
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
        super(RNet, self).__init__(vocab, config)

        # 1. Character Embedding Layer
        self.char_emb_layer = CharEmbedding(vocab, config)

        # 2. Word Embedding Layer
        self.word_emb = WordEmbedding(vocab_size=vocab.get_word_vocab_size(),
                                      embd_size=config.word_embedding_size,
                                      pre_word_embd=vocab.word_embd)
        char_direction_num = 2 if config.char_rnn_bidirectional else 1
        char_rnn_emb_size = char_direction_num * config.char_rnn_num_layers * config.hidden_size
        emb_size = char_rnn_emb_size + config.word_embedding_size

        # 3. Q P Encoder Layer
        self.encoder = RNNLayer(input_size=emb_size,
                                hidden_size=config.hidden_size,
                                bidirectional=True,
                                dropout_p=config.dropout,
                                enable_layer_norm=False,
                                network_mode="GRU")

        # 4. Q P Matcher Layer
        self.pqmatcher = PQMatcher(config, config.hidden_size * 2)
        # 5. Self Matcher Layer
        self.selfmatcher = SelfMatcher(config, config.hidden_size)

        # 6. Output Layer
        self.fc_last = Linear(config.hidden_size, self.config.n_classes)

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

        P = torch.cat((paragraph_word, paragraph_char), dim=2)
        Q = torch.cat((question_word, question_char), dim=2)

        Up, _ = self.encoder.forward(P, paragraph_mask)
        Uq, _ = self.encoder.forward(Q, question_mask)

        v = self.pqmatcher(Up, Uq, question_mask)
        _, h = self.selfmatcher(v, paragraph_mask)
        output = self.fc_last.forward(h)  # (batch, n_classes)
        output = F.softmax(output, dim=1)  # (batch, n_classes)
        return output

# Using bidirectional gru hidden state to represent char embedding for a word
class CharEmbedding(nn.Module):
    def __init__(self, vocab, config):
        super(CharEmbedding, self).__init__()
        self.char_emb = WordEmbedding(vocab.get_char_vocab_size(), config.char_embedding_size)
        torch.nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)

        self.char_rnn_emb = nn.GRU(
            batch_first=True,
            input_size=config.char_embedding_size,
            bidirectional=config.char_rnn_bidirectional,
            num_layers=config.char_rnn_num_layers,
            hidden_size=config.hidden_size)
        self.dropout = torch.nn.Dropout(p=config.char_dropout)

        char_direction_num = 2 if config.char_rnn_bidirectional else 1
        self.char_rnn_emb_size = char_direction_num * config.char_rnn_num_layers * config.hidden_size

        self.reset_parameters()

    def reset_parameters(self):
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, input):
        batch_size = input.size(0)
        seq_len = input.size(1)
        # input: (batch, seq_len, word_len)
        input = input.view(-1, input.size(2))
        c_emb, _ = self.char_emb.forward(input)
        _, h = self.char_rnn_emb(c_emb)
        h = h.transpose(0, 1)
        h = h.contiguous().view(batch_size, seq_len, self.char_rnn_emb_size)
        h = self.dropout(h)
        # o_char_emb: (batch, seq_len, char_embedding_size * char_rnn_num_layers * char_rnn_bdirectional_num)
        o_char_emb = h
        return o_char_emb

# Using passage and question to obtain question-aware passage representation by Co-attention
class PQMatcher(nn.Module):
    def __init__(self, config, in_size):
        super(PQMatcher, self).__init__()
        self.in_size = in_size
        self.hidden_size = config.hidden_size
        self.gru = nn.GRUCell(input_size=in_size * 2, hidden_size=config.hidden_size)
        self.Wp = Linear(in_size, config.hidden_size, bias=False)
        self.Wq = Linear(in_size, config.hidden_size, bias=False)
        self.Wv = Linear(config.hidden_size, config.hidden_size, bias=False)
        self.V = Linear(config.hidden_size, 1, bias=False)
        self.Wg = Linear(in_size * 2, in_size * 2, bias=False)
        self.dropout = nn.Dropout(p=config.dropout)
        
        
    def forward(self, Up, Uq, Uq_mask):
        # Up: (batch, p_len, encode_size)
        # Uq: (batch, q_len, encode_size)
        batch_size = Up.size(0)
        lp = Up.size(1)
        v = Up.new_zeros(batch_size, self.hidden_size)
        vs = [v]

        for i in range(lp):
            Upi = Up.transpose(0, 1)[i, ...]  # (batch, encode_size)
            Wup = self.Wp.forward(Upi)  # (batch, hidden_size)
            Wuq = self.Wq.forward(Uq).transpose(0, 1)  # Wuq: (batch, q_len, hidden_size)
            Wvv = self.Wv.forward(vs[i])  # (batch, hidden_size)
            x = torch.tanh(Wup + Wuq + Wvv).transpose(0, 1)  # Wuq: (batch, q_len, hidden_size)
            s = self.V.forward(x).squeeze(dim=2)  # s: (batch, q_len)
            a = masked_softmax(s, Uq_mask, dim=1).unsqueeze(1)  # s: (batch, 1, q_len)
            c = torch.bmm(a, Uq).squeeze(1)  # c: (batch, encode_size)
            r = torch.cat((Upi, c), dim=1)  # c: (batch, encode_size * 2)
            g = torch.sigmoid(self.Wg.forward(r))  # c: (batch, encode_size * 2)
            c_ = g * r  # c: (batch, encode_size * 2)
            vsi = self.gru(c_, vs[i])  # (batch, hidden_size)
            vs.append(vsi)
        vs = torch.stack(vs[1:], dim=1)  # (batch, p_len, hidden_size)
        vs = self.dropout(vs)  # vs: (batch, p_len, hidden_size)
        return vs


# Input is question-aware passage representation
# Output is self-attention question-aware passage representation
class SelfMatcher(nn.Module):
    def __init__(self, config, in_size):
        super(SelfMatcher, self).__init__()
        self.hidden_size = config.hidden_size
        self.in_size = in_size
        self.gru = nn.GRUCell(input_size=in_size * 2, hidden_size=config.hidden_size)
        self.Wp = Linear(self.in_size, config.hidden_size, bias=False)
        self.Wp_ = Linear(self.in_size, config.hidden_size, bias=False)
        self.V = Linear(config.hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, v, mask):
        # v: (batch, p_len, in_size)
        (batch_size, l, _) = v.size()  # (batch, p_len, in_size)
        
        h_0 = v.new_zeros(batch_size, self.hidden_size)
        hs = [h_0]
        for i in range(l):
            vi = v.transpose(0, 1)[i, ...]  # (batch, in_size)
            Wpv = self.Wp_.forward(v).transpose(0, 1)  # (p_len, batch, hidden_size)
            Wpv_ =self.Wp.forward(vi)
            x = torch.tanh(Wpv + Wpv_).transpose(0, 1)  # (batch, p_len, hidden_size)
            s = self.V.forward(x).squeeze(dim=2)  # (batch, p_len)
            a = masked_softmax(s, mask, dim=1).unsqueeze(1)  # (batch, 1, p_len)
            c = torch.bmm(a, v).squeeze(1)  # (batch, 1, in_size)
            r = torch.cat((vi, c), dim=1)  # c: (batch, in_size * 2)
            hsi = self.gru(r, hs[i])  # (batch, hidden_size)
            hs.append(hsi)
        hs = torch.stack(hs[1:], dim=1)  # (batch, p_len, hidden_size)
        hs = self.dropout(hs)  # (batch, p_len, hidden_size)
        h = hs.select(1, index=-1)
        return hs, h



