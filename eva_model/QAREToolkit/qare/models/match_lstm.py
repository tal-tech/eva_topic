#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-07-09
'''

import torch
import torch.nn.functional as F
from eva_model.QAREToolkit.qare.nn import layers
from eva_model.QAREToolkit.qare.nn import operations
from eva_model.QAREToolkit.qare.models.base_model import BaseModel
from eva_model.QAREToolkit.qare.nn.layers import Linear


class MatchLSTM(BaseModel):
    """
    Match-lstm model
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
        super(MatchLSTM, self).__init__(vocab, config)

        embd_size = self.config.word_embedding_size
        hidden_size = self.config.hidden_size
        match_rnn_layer_dropout_p = self.config.match_rnn_layer_dropout_p
        preprocessing_layer_dropout_p = self.config.preprocessing_layer_dropout_p
        enable_layer_norm = self.config.enable_layer_norm
        network_mode = self.config.network_mode
        gated_attention = self.config.gated_attention

        word_embedding_size = self.config.word_embedding_size
        preprocessing_layer_bidirection = self.config.preprocessing_layer_bidirection
        preprocessing_layer_direction_num = 2 if preprocessing_layer_bidirection else 1

        match_lstm_layer_bidirection = self.config.match_lstm_layer_bidirection
        match_lstm_layer_bidirection = 2 if match_lstm_layer_bidirection else 1

        # Construct model
        self.embedding = layers.WordEmbedding(vocab_size=vocab.get_word_vocab_size(),
                                              embd_size=embd_size,
                                              pre_word_embd=vocab.word_embd)

        self.preprocessing_encoder = layers.RNNLayer(input_size=word_embedding_size,
                                                     hidden_size=hidden_size,
                                                     bidirectional=preprocessing_layer_bidirection,
                                                     dropout_p=preprocessing_layer_dropout_p,
                                                     enable_layer_norm=self.config.enable_layer_norm,
                                                     network_mode=self.config.network_mode)

        preprocessing_layer_out_size = hidden_size * preprocessing_layer_direction_num

        self.match_rnn = MatchRNN(network_mode=network_mode,
                                  hp_input_size=preprocessing_layer_out_size,
                                  hq_input_size=preprocessing_layer_out_size,
                                  hidden_size=hidden_size,
                                  bidirectional=match_lstm_layer_bidirection,
                                  gated_attention=gated_attention,
                                  dropout_p=match_rnn_layer_dropout_p,
                                  enable_layer_norm=enable_layer_norm)
        preprocessing_layer_out_size = hidden_size * match_lstm_layer_bidirection
        self.fc_last = torch.nn.Linear(preprocessing_layer_out_size, self.config.n_classes)

    def loss_function(self):
        return torch.nn.CrossEntropyLoss()

    def optimizer(self):
        optimizer_kwargs = {"lr": self.config.learning_rate,
                            "rho": 0.9, "eps": 1e-6, "weight_decay": 0}
        optimizer_ = torch.optim.Adadelta
        return optimizer_, optimizer_kwargs

    def forward(self, question, paragraph):
        # get embedding
        paragraph_embd, paragraph_mask = self.embedding(paragraph)   # (batch, paragraph_len, word_embedding_size)
        question_embd, question_mask = self.embedding(question) # (batch, question_len, word_embedding_size)

        # preprocessing encode
        paragraph_encode, _ = self.preprocessing_encoder.forward(paragraph_embd, paragraph_mask)    # (batch, paragraph_len, hidden_size * direction_num)
        question_encode, _ = self.preprocessing_encoder.forward(question_embd, question_mask)    # (batch, question_len, hidden_size * direction_num)

        # match lstm: (batch, sentence_len, hidden_size * direction_num)
        qt_aware_ct, qt_aware_last_hidden = self.match_rnn.forward(paragraph_encode, paragraph_mask,
                                                                   question_encode, question_mask)
        output = self.fc_last.forward(qt_aware_last_hidden)    # (batch, n_classes)
        output = F.softmax(output, dim=1)  # (batch, n_classes)

        return output


class MatchLSTMAttention(torch.nn.Module):
    """
    Attention mechanism in match-lstm
    Args:
        hp_input_size: The number of expected features in the input Hp
        hq_input_size: The number of expected features in the input Hq
        hidden_size: The number of hidden units

    Inputs:
        - Hpi(batch, input_size): An paragraph word encoded
        - Hq(question_len, batch, input_size): whole question encoded
        - Hr_last(batch, hidden_size): last lstm hidden output

    Outputs:
        - alpha(batch, question_len): attention vector
    """

    def __init__(self, hp_input_size, hq_input_size, hidden_size):
        super(MatchLSTMAttention, self).__init__()

        self.linear_wq = torch.nn.Linear(hq_input_size, hidden_size)
        self.linear_wp = torch.nn.Linear(hp_input_size, hidden_size)
        self.linear_wr = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_wg = torch.nn.Linear(hidden_size, 1)

    def forward(self, Hpi, Hq, Hr_last, Hq_mask):
        wq_hq = self.linear_wq(Hq)  # (batch, question_len, hidden_size)
        wp_hp = self.linear_wp(Hpi).unsqueeze(1)  # (batch, 1, hidden_size)
        wr_hr = self.linear_wr(Hr_last).unsqueeze(1)  # (batch, 1, hidden_size)
        G = torch.tanh(wq_hq + wp_hp + wr_hr)  # (batch, question_len, hidden_size), auto broadcast

        wg_g = self.linear_wg(G).squeeze(2)  # (batch, question_len)

        alpha = operations.masked_softmax(wg_g, mask=Hq_mask, dim=1)  # (batch, question_len)
        return alpha


class UniMatchRNN(torch.nn.Module):
    """
    Interaction paragraph and question with attention mechanism, one direction
    Args:
        network_mode: LSTM or GRU
        hp_input_size: The number of expected features in the input Hp
        hq_input_size: The number of expected features in the input Hq
        hidden_size: The number of hidden units
        gated_attention: Gated attention
        enable_layer_norm: Enable layer normalization

    Inputs:
        - Hp(batch, paragraph_len, input_size): paragraph encoded
        - Hq(batch, question_len, input_size): question encoded
        - Hq_mask(batch, question_len): mask of Hq

    Outputs:
        - Hr(batch, paragraph_len, hidden_size): question-aware paragraph representation
    """

    def __init__(self, network_mode, hp_input_size, hq_input_size, hidden_size, gated_attention, enable_layer_norm):
        super(UniMatchRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gated_attention = gated_attention
        self.enable_layer_norm = enable_layer_norm

        self.attention = MatchLSTMAttention(hp_input_size, hq_input_size, hidden_size)

        rnn_in_size = hp_input_size + hq_input_size
        if self.gated_attention:
            self.gated_linear = torch.nn.Linear(rnn_in_size, rnn_in_size)

        if self.enable_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(rnn_in_size)

        self.network_mode = network_mode
        if network_mode == 'LSTM':
            self.hidden_cell = torch.nn.LSTMCell(input_size=rnn_in_size, hidden_size=hidden_size)
        elif network_mode == 'GRU':
            self.hidden_cell = torch.nn.GRUCell(input_size=rnn_in_size, hidden_size=hidden_size)
        else:
            raise ValueError('Wrong network_mode select %s, only support for LSTM or GRU' % network_mode)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reproduce default initialization weights to initialize Embeddings/weights
        """
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, Hp, Hq, Hq_mask):
        batch_size = Hp.shape[0]
        paragraph_len = Hp.shape[1]

        # Init hidden with the same type of input data
        h_0 = Hp.new_zeros(batch_size, self.hidden_size)
        hidden = [(h_0, h_0)] if self.network_mode == 'LSTM' else [h_0]

        for t in range(paragraph_len):
            Hpi = Hp.transpose(0, 1)[t, ...]  # (batch, input_size)

            attention_input = hidden[t][0] if self.network_mode == 'LSTM' else hidden[t]

            alpha = self.attention.forward(Hpi, Hq, attention_input, Hq_mask)  # (batch, question_len)

            question_alpha = torch.bmm(alpha.unsqueeze(1), Hq) \
                .squeeze(1)  # (batch, input_size)

            zi = torch.cat((Hpi, question_alpha), dim=1)  # (batch, rnn_in_size)

            # Gated attetion
            if self.gated_attention:
                gate = torch.sigmoid(self.gated_linear.forward(zi))
                zi = gate * zi

            # Layer normalization
            if self.enable_layer_norm:
                zi = self.layer_norm(zi)  # (batch, rnn_in_size)

            hri = self.hidden_cell.forward(zi, hidden[t])  # (batch, hidden_size), when lstm output tuple

            hidden.append(hri)

        hidden_state = list(map(lambda x: x[0], hidden)) if self.network_mode == 'LSTM' else hidden

        Hr = torch.stack(hidden_state[1:], dim=1)  # (batch, paragraph_len, hidden_size)

        return Hr


class MatchRNN(torch.nn.Module):
    """
    Interaction paragraph and question with attention mechanism
    Args:
        network_mode: LSTM or GRU
        hp_input_size: The number of expected features in the input Hp
        hq_input_size: The number of expected features in the input Hq
        hidden_size: The number of hidden units
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``
        gated_attention: Gated attention
        dropout_p: Dropout probability to input data, and also dropout along hidden layers
        enable_layer_norm: Enable layer normalization
        
    Inputs:
        Hp(batch, paragraph_len, input_size): paragraph encoded
        Hq(batch, question_len, input_size): question encoded
        hp_mask(batch, paragraph_len): each paragraph valued length without padding values
        Hq_mask(batch, question_len): each question valued length without padding values

    Outputs:
        Hr(batch, paragraph_len, hidden_size * num_directions): question-aware paragraph representation
        last_hidden(batch, hidden_size * num_directions): last hidden representation
    """

    def __init__(self, network_mode, hp_input_size, hq_input_size, hidden_size, bidirectional, gated_attention,
                 dropout_p, enable_layer_norm):
        super(MatchRNN, self).__init__()
        self.bidirectional = bidirectional
        self.gated_attention = gated_attention

        self.left_match_rnn = UniMatchRNN(network_mode, hp_input_size, hq_input_size, hidden_size, gated_attention,
                                          enable_layer_norm)
        if bidirectional:
            self.right_match_rnn = UniMatchRNN(network_mode, hp_input_size, hq_input_size, hidden_size, gated_attention,
                                               enable_layer_norm)
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, Hp, hp_mask, Hq, Hq_mask):
        Hp = self.dropout(Hp)
        Hq = self.dropout(Hq)

        left_hidden = self.left_match_rnn.forward(Hp, Hq, Hq_mask)
        rtn_hidden = left_hidden

        if self.bidirectional:
            Hp_inv = operations.masked_flip(Hp, hp_mask, flip_dim=1)
            right_hidden_inv = self.right_match_rnn.forward(Hp_inv, Hq, Hq_mask)

            # flip back to normal sequence
            right_hidden = operations.masked_flip(right_hidden_inv, hp_mask, flip_dim=1)

            rtn_hidden = torch.cat((left_hidden, right_hidden), dim=2)

        Hr = hp_mask.unsqueeze(2) * rtn_hidden # (batch_size, paragraph_len, hidden_size * direction_num)
        last_hidden = rtn_hidden.transpose(0, 1)[-1, :] # (batch_size, hidden_size * direction_num)

        return Hr, last_hidden