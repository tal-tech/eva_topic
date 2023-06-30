#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-07-26
'''

import torch
from eva_model.QAREToolkit.qare.nn import layers
import torch.nn.functional as F
from eva_model.QAREToolkit.qare.models.base_model import BaseModel


class StructuredSelfAttentiveNet(BaseModel):
    """
    Structured-Self-Attentive-Net model
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
        super(StructuredSelfAttentiveNet, self).__init__(vocab, config)
        self.embd_size = self.config.word_embedding_size
        self.hidden_size = self.config.hidden_size # u
        self.mlp_hidden_size = self.config.mlp_hidden
        self.r = self.config.attention_hops
        d = self.config.mlp_d
        self.word_emb = layers.WordEmbedding(vocab_size=vocab.get_word_vocab_size(),
                                              embd_size=self.embd_size,
                                              pre_word_embd=vocab.word_embd)
        self.encoder = torch.nn.GRU(self.embd_size,
                                 self.hidden_size,
                                 batch_first=True,
                                 bidirectional=True)
        self.fc1 = torch.nn.Linear(self.r*2*self.hidden_size, self.mlp_hidden_size)
        self.fc1.bias.data.fill_(0)
        self.fc2 = torch.nn.Linear(self.mlp_hidden_size, self.config.n_classes)
        self.fc2.bias.data.fill_(0)
        initrange = 0.1
        self.Ws1 = torch.nn.Parameter(torch.Tensor(1, d, 2*self.hidden_size).uniform_(-initrange, initrange))
        self.Ws2 = torch.nn.Parameter(torch.Tensor(1, self.r, d).uniform_(-initrange, initrange))

    def loss_function(self):
        return torch.nn.CrossEntropyLoss()

    def optimizer(self):
        optimizer_kwargs = {"lr": self.config.learning_rate,
                            "rho": 0.9, "eps": 1e-6, "weight_decay": 0}
        optimizer_ = torch.optim.Adadelta
        return optimizer_, optimizer_kwargs

    def forward(self, question, paragraph):
        x = torch.cat((question, paragraph), dim=1)
        batch_size = x.size(0) # batch size
        x, _ = self.word_emb(x) # (batch_size, n, embd_size)
        H, _ = self.encoder(x) # (batch_size, n, 2u)
        H_T = torch.transpose(H, 2, 1).contiguous() # (batch_size, 2u, n)
        A = torch.tanh(torch.bmm(self.Ws1.repeat(batch_size, 1, 1), H_T)) # (batch_size, d, n)
        A = torch.bmm(self.Ws2.repeat(batch_size, 1, 1), A) # (batch_size, r, n)
        A = F.softmax(A, dim=2) # (batch_size, r, n)
        M = A @ H # (batch_size, r, 2u)
        out = F.relu(self.fc1(M.view(batch_size, -1))) # (batch_size, mlp_hidden)
        output = self.fc2(out) # (batch_size, n_classes)
        output = F.softmax(output, dim=1)  # (batch, n_classes)
        return output