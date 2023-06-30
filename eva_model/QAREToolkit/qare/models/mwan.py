#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-07-24
'''

import torch
import torch.nn.functional as F
from eva_model.QAREToolkit.qare.models.base_model import BaseModel
from eva_model.QAREToolkit.qare.nn import layers


class MWAN(BaseModel):
    """
    MwAN model
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
        super(MWAN, self).__init__(vocab, config)

        # Set config
        embd_size = self.config.word_embedding_size
        self.hidden_size = self.config.hidden_size
        encoding_layer_dropout_p = self.config.encoding_layer_dropout_p
        enable_layer_norm = self.config.enable_layer_norm
        network_mode = self.config.network_mode

        word_embedding_size = self.config.word_embedding_size

        encoding_layer_bidirection = self.config.encoding_layer_bidirection
        encoding_layer_direction_num = 2 if encoding_layer_bidirection else 1

        # construct model
        self.embedding = layers.WordEmbedding(vocab_size=vocab.get_word_vocab_size(),
                                              embd_size=embd_size,
                                              pre_word_embd=vocab.word_embd)

        self.rnn_encoder = layers.RNNLayer(input_size=word_embedding_size,
                                           hidden_size=self.hidden_size,
                                           bidirectional=encoding_layer_bidirection,
                                           dropout_p=encoding_layer_dropout_p,
                                           enable_layer_norm=enable_layer_norm,
                                           network_mode=network_mode)

        encoding_layer_out_size = self.hidden_size * encoding_layer_direction_num
        self.multiway_matching = MultiwayMatching(encoding_layer_out_size,
                                                  encoding_layer_out_size,
                                                  self.hidden_size)
        self.inside_aggregation = InsideAggregation(encoding_layer_out_size,
                                                    self.hidden_size)

        batch_size = self.config.batch_size
        question_sentence_len = self.config.question_truncate_len
        answer_sentence_len = self.config.answer_truncate_len
        hc_shape = [batch_size, answer_sentence_len, encoding_layer_out_size]
        question_encode_shape = [batch_size, question_sentence_len, encoding_layer_out_size]
        ho_shape = [batch_size, answer_sentence_len, encoding_layer_out_size]
        self.mixed_aggregation = MixedAggregation(hc_shape, self.hidden_size)
        self.prediction = Prediction(question_encode_shape,
                                     ho_shape, self.hidden_size)
        self.fc_last = torch.nn.Linear(encoding_layer_out_size, self.config.n_classes)

    def loss_function(self):
        return torch.nn.CrossEntropyLoss()

    def optimizer(self):
        optimizer_kwargs = {"lr": self.config.learning_rate,
                            "rho": 0.9, "eps": 1e-6, "weight_decay": 0}
        optimizer_ = torch.optim.Adadelta
        return optimizer_, optimizer_kwargs

    def forward(self, question, paragraph):

        # get embedding
        paragraph_embd, paragraph_mask = self.embedding.forward(paragraph)   # (batch, paragraph_len, word_embedding_size)
        question_embd, question_mask = self.embedding.forward(question) # (batch, question_len, word_embedding_size)

        # preprocessing encode
        paragraph_encode, _ = self.rnn_encoder.forward(paragraph_embd, paragraph_mask)    # (batch, paragraph_len, hidden_size * direction_num)
        question_encode, _ = self.rnn_encoder.forward(question_embd, question_mask)    # (batch, question_len, hidden_size * direction_num)

        qc, qb, qd, qm = self.multiway_matching.forward(paragraph_encode, question_encode)
        hc = self.inside_aggregation.forward(paragraph_encode, qc)
        hb = self.inside_aggregation.forward(paragraph_encode, qb)
        hd = self.inside_aggregation.forward(paragraph_encode, qd)
        hm = self.inside_aggregation.forward(paragraph_encode, qm)

        ho = self.mixed_aggregation.forward(hc, hb, hd, hm)

        rp = self.prediction.forward(question_encode, ho)
        output = self.fc_last(rp)
        output = F.softmax(output, dim=1)  # (batch, n_classes)

        return output


class MultiwayMatching(torch.nn.Module):

    def __init__(self, hp_input_size, hq_input_size, hidden_size):
        super(MultiwayMatching, self).__init__()

        # Concat Attention
        self.Wc1 = torch.nn.Linear(hq_input_size, hidden_size, bias=False)
        self.Wc2 = torch.nn.Linear(hp_input_size, hidden_size, bias=False)
        self.vc = torch.nn.Linear(hidden_size, 1, bias=False)

        # Bilinear Attention
        self.Wb = torch.nn.Linear(hq_input_size, hp_input_size, bias=False)

        # Dot Attention
        self.Wd = torch.nn.Linear(hq_input_size, hidden_size, bias=False)
        self.vd = torch.nn.Linear(hidden_size, 1, bias=False)

        # Minus Attention
        self.Wm = torch.nn.Linear(hq_input_size, hidden_size, bias=False)
        self.vm = torch.nn.Linear(hidden_size, 1, bias=False)

    def forward(self, Hp, Hq):
        _s1 = self.Wc1(Hq).unsqueeze(1)  # (batch, 1, q_len, hidden_size)
        _s2 = self.Wc2(Hp).unsqueeze(2)  # (batch, p_len, 1, hidden_size)
        sjt = self.vc(torch.tanh(_s1 + _s2)).squeeze(dim=3)  # (batch, p_len, q_len)
        ait = F.softmax(sjt, dim=2)  # (batch, p_len, q_len)
        qc = ait @ Hq  # (batch, p_len, hq_size)

        _s1 = self.Wb(Hq).transpose(1, 2)  # (batch, hp_size, hq_len)
        sjt = Hp @ _s1  # (batch, p_len, q_len)
        ait = F.softmax(sjt, dim=2)  # (batch, p_len, q_len)
        qb = ait @ Hq  # (batch, p_len, hq_size)

        _s1 = Hq.unsqueeze(1)  # (batch, 1, q_len, hidden_size)
        _s2 = Hp.unsqueeze(2)  # (batch, p_len, 1, hidden_size)
        sjt = self.vd(torch.tanh(self.Wd(_s1 * _s2))).squeeze(dim=3)  # (batch, p_len, q_len)
        ait = F.softmax(sjt, dim=2)  # (batch, p_len, q_len)
        qd = ait @ Hq  # (batch, p_len, hq_size)

        _s1 = Hq.unsqueeze(1)  # (batch, 1, q_len, hidden_size)
        _s2 = Hp.unsqueeze(2)  # (batch, p_len, 1, hidden_size)
        sjt = self.vm(torch.tanh(self.Wm(_s1 - _s2))).squeeze(dim=3)  # (batch, p_len, q_len)

        ait = F.softmax(sjt, dim=2)  # (batch, p_len, q_len)
        qm = ait @ Hq  # (batch, p_len, hq_size)

        return [qc, qb, qd, qm]


class InsideAggregation(torch.nn.Module):

    def __init__(self, hp_input_size, hidden_size):
        super(InsideAggregation, self).__init__()

        self.Wg = torch.nn.Linear(hp_input_size * 2, 1, bias=False)
        self.inside_agg_encoder = torch.nn.GRU(input_size=hp_input_size * 2,
                                               hidden_size=hidden_size,
                                               num_layers=1,
                                               batch_first=True,
                                               bidirectional=True)

    def forward(self, Hp, qx):
        xc = torch.cat((qx, Hp), dim=2)  # (batch, p_len, hp_size * 2)
        gt = torch.sigmoid(self.Wg(xc))  # (batch, p_len, 1)
        xc_star = gt * xc  # (batch, p_len, hp_size * 2)
        o, _ = self.inside_agg_encoder(xc_star)  # o: (batch, p_len, hidden_size * 2)

        return o


class MixedAggregation(torch.nn.Module):

    def __init__(self, input_shape, hidden_size):
        super(MixedAggregation, self).__init__()

        self.W1 = torch.nn.Linear(input_shape[2], hidden_size, bias=False)
        self.W2 = torch.nn.Linear(hidden_size, input_shape[1], bias=False)
        initrange = 0.1
        self.va = torch.nn.Parameter(torch.Tensor(4, hidden_size).uniform_(-initrange, initrange))
        self.v = torch.nn.Linear(hidden_size, 1, bias=False)

        self.mixed_agg_encoder = torch.nn.GRU(input_size=hidden_size * 2,
                                              hidden_size=hidden_size,
                                              num_layers=1,
                                              batch_first=True,
                                              bidirectional=True)

    def forward(self, hc, hb, hd, hm):
        batch_size = hc.shape[0]
        paragraph_len = hc.shape[1]
        h_size = hc.shape[2]

        h_cat = torch.cat((hc, hb, hd, hm), dim=2)
        h_cat = h_cat.reshape(shape=(batch_size, paragraph_len, 4, h_size))  # (batch, p_len, 4, hp_size)

        _s1 = self.W1(h_cat)
        _s2 = self.W2(self.va)
        _s2 = _s2.unsqueeze(0).unsqueeze(3).transpose(1, 2)
        s_ = self.v(torch.tanh(_s1 + _s2))  # (batch, p_len, 4, 1)
        a_ = F.softmax(s_, dim=2)  # (batch, p_len, 4, 1)
        x_ = a_ * h_cat
        x = torch.sum(x_, dim=2, keepdim=False)  # (batch, p_len, hp_size)
        o, _ = self.mixed_agg_encoder(x)

        return o


class Prediction(torch.nn.Module):

    def __init__(self, hq_input_shape, ho_input_shape, hidden_size):
        super(Prediction, self).__init__()

        self.Wq1 = torch.nn.Linear(hq_input_shape[2], hidden_size, bias=False)
        self.Wq2 = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        initrange = 0.1
        self.vq = torch.nn.Parameter(torch.Tensor(1, hidden_size).uniform_(-initrange, initrange))
        self.v1 = torch.nn.Linear(hidden_size, 1, bias=False)

        self.Wp1 = torch.nn.Linear(ho_input_shape[2], hidden_size, bias=False)
        self.Wp2 = torch.nn.Linear(hq_input_shape[2], hidden_size, bias=False)
        self.v2 = torch.nn.Linear(hidden_size, 1, bias=False)

    def forward(self, Hq, Ho):
        _s1 = self.Wq1(Hq)  # (batch, q_len, hidden_size)
        _s2 = self.Wq2(self.vq).unsqueeze(0)  # (1, 1, hidden_size)
        s_ = self.v1(torch.tanh(_s1 + _s2))  # (batch, q_len, 1)
        a_ = F.softmax(s_, dim=1)
        rq = torch.sum(a_ * Hq, dim=1, keepdim=False)  # (batch, hq_size)

        _s1 = self.Wp1(Ho)  # (batch, p_len, hidden_size)
        _s2 = self.Wp2(rq).unsqueeze(1)  # (batch, 1, hidden_size)
        s_ = self.v2(torch.tanh(_s1 + _s2))  # (batch, p_len, 1)
        a_ = F.softmax(s_, dim=1)  # (batch, p_len, 1)
        rp = torch.sum(a_ * Ho, dim=1, keepdim=False)  # (batch, hp_size)
        return rp