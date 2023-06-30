#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-07-09
'''

import torch
import logging
import math
import torch.nn.functional as F
from eva_model.QAREToolkit.qare.nn.operations import masked_logits
from eva_model.QAREToolkit.qare.nn import operations


class Layer(torch.nn.Module):
    
    def __init__(self):
        super(Layer, self).__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class WordEmbedding(Layer):
    """
    Word embedding
    Args:
        vocab_size: The size of vocabulary
        embd_size: The size of word embedding
        pre_word_embd: Pretrained word embedding; None(Default)
    Inputs:
        - x: (batch, sentence_len)

    Outputs:
        - out_emb: (batch, sentence_len, embd_size)
        - mask: (batch, sentence_len)
    """

    def __init__(self, vocab_size, embd_size, pre_word_embd=None):
        super(WordEmbedding, self).__init__()

        self.embedding = torch.nn.Embedding(vocab_size, embd_size)
        if pre_word_embd is not None:
            logging.info('Set pretrained embedding weights')
            weights = torch.FloatTensor(pre_word_embd)
            assert weights.shape[1] == embd_size, "Pretrained embedding size not consistent with config!"
            self.embedding.from_pretrained(weights)
        self.weight = self.embedding.weight

    def forward(self, x):
        mask = operations.compute_mask(x)
        out_emb = self.embedding(x)

        return out_emb, mask


class Linear(Layer):
    """
    Linear with dropout
    Args:
        in_features: The number of input features
        out_features: The number of output features
        dropout_p: dropout probability to input data, and also dropout along hidden layers(default 0.0)

    Inputs:
        - x (..., in_features): input tensor.

    Outputs:
        - o (..., out_features): output tensor.
    """

    def __init__(self, in_features, out_features, bias=True, dropout=0.0, init_method = "kaiming_normal", init_param = {}):
        super(Linear, self).__init__()

        self.bias = bias
        self.linear = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        if dropout > 0:
            self.dropout = torch.nn.Dropout(p=dropout)
        self.init_method = init_method
        self.init_param = init_param
        self.reset_params()

    def reset_params(self):
        if self.init_method == "kaiming_normal":
            torch.nn.init.kaiming_normal_(self.linear.weight, **self.init_param)
        elif self.init_method == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(self.linear.weight, **self.init_param)
        elif self.init_method == "normal":
            torch.nn.init.normal_(self.linear.weight, **self.init_param)
        elif self.init_method == "normal":
            torch.nn.init.uniform_(self.linear.weight, **self.init_param)
        elif self.init_method == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.linear.weight, **self.init_param)
        elif self.init_method == "xavier_normal":
            torch.nn.init.xavier_normal_(self.linear.weight, **self.init_param)
        else:
            raise ValueError("init_method not supported")
        if self.bias:
            torch.nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        o = self.linear(x)
        return o

class RNNLayer(Layer):
    """
    RNN with packed sequence and dropout
    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        dropout_p: dropout probability to input data, and also dropout along hidden layers
        bidirectional: True-bidirectional RNN; False(Default)
        layer_num: Number of layers; 1(Default)
        enable_layer_norm: Layer normalization
        network_mode: LSTM of GRU

    Inputs:
        - input (batch, sentence_len, input_size): tensor containing the features
          of the input sequence.
        - mask (batch, sentence_len): tensor show whether a padding index for each element in the batch.

    Outputs: 
        - output (batch, sentence_len, hidden_size * num_directions): tensor
          containing the output features `(h_t)` from the last layer of the RNN,
          for each t.
        - o_last (batch, hidden_size * num_directions): the final hidden state of rnn
    """

    def __init__(self, input_size,
                 hidden_size,
                 dropout_p, 
                 bidirectional = False,
                 layer_num = 1, 
                 enable_layer_norm = False,
                 network_mode = "LSTM"):
        super(RNNLayer, self).__init__()
        self.enable_layer_norm = enable_layer_norm

        if network_mode.upper() == "LSTM":
            self.hidden = torch.nn.LSTM(input_size = input_size,
                                        hidden_size = hidden_size,
                                        num_layers = layer_num,
                                        batch_first = True,
                                        bidirectional = bidirectional)
        elif network_mode.upper() == "GRU":
            self.hidden = torch.nn.GRU(input_size = input_size,
                                       hidden_size = hidden_size,
                                       num_layers = layer_num,
                                       batch_first = True,
                                       bidirectional = bidirectional)
        else:
            raise ValueError('Error network_mode-%s, should be LSTM or GRU!' % network_mode)

        self.dropout = torch.nn.Dropout(p=dropout_p)

        if enable_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(input_size)
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

    def forward(self, v, mask):
        # layer normalization
        if self.enable_layer_norm:
            batch, sentence_len, input_size = v.shape
            v = v.view(-1, input_size)
            v = self.layer_norm(v)
            v = v.view(batch, sentence_len, input_size)
        # get sorted v
        max_sequence = v.shape[1]
        lengths = mask.eq(1).long().sum(1)

        lengths_sort, idx_sort = torch.sort(lengths, dim = 0, descending = True)

        _, idx_unsort = torch.sort(idx_sort, dim = 0)
        v_sort = v.index_select(0, idx_sort)
        v_pack = torch.nn.utils.rnn.pack_padded_sequence(v_sort, lengths_sort, batch_first = True)

        v_dropout = self.dropout.forward(v_pack.data)
        v_pack_dropout = torch.nn.utils.rnn.PackedSequence(v_dropout, v_pack.batch_sizes)

        o_pack_dropout, _ = self.hidden.forward(v_pack_dropout)
        o, _ = torch.nn.utils.rnn.pad_packed_sequence(o_pack_dropout, batch_first = True, total_length = max_sequence)

        # output
        output = o.index_select(0, idx_unsort)  # batch_size first

        # get the last time state
        len_idx = (lengths - 1).view(-1, 1).expand(-1, output.size(2)).unsqueeze(1)
        o_last = output.gather(1, len_idx)
        o_last = o_last.squeeze(1)

        return output, o_last


class Conv1d(Layer):
    '''
    Args:
        in_channels (int) – Number of channels in the input features
        out_channels (int) – Number of channels produced by the convolution
        kernel_size (int or tuple) – Size of the convolving kernel. Default: 1
        relu (bool) – Activation function after the convolving. Default: False
        stride (int or tuple, optional) – Stride of the convolution. Default: 1
        padding (int or tuple, optional) – Zero-padding added to both sides of the input. Default: 0
        padding_mode (string, optional) – zeros
        groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional) – If True, adds a learnable bias to the output. Default: True

    Inputs:
        - input (batch, channel_in, height, width): tensor

    Outputs:
        - output (batch, channel_out, h_out, w_out): tensor
    '''
    def __init__(self, in_channels, out_channels, kernel_size=1, relu=False, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups,
                             bias=bias)
        if relu is True:
            self.relu = True
            torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        else:
            self.relu = False
            torch.nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        if self.relu == True:
            return F.relu(self.conv(x))
        else:
            return self.conv(x)


class Conv2d(Layer):
    '''
    Args:
        in_channels (int) – Number of channels in the input features
        out_channels (int) – Number of channels produced by the convolution
        kernel_size (int or tuple) – Size of the convolving kernel
        stride (int or tuple, optional) – Stride of the convolution. Default: 1
        padding (int or tuple, optional) – Zero-padding added to both sides of the input. Default: 0
        padding_mode (string, optional) – zeros
        dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1
        groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional) – If True, adds a learnable bias to the output. Default: True

    Inputs:
        - input (batch, channel_in, height, width): tensor

    Outputs:
        - output (batch, channel_out, h_out, w_out): tensor
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__()
        self.bias = bias
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.conv.weight)
        if self.bias:
            torch.nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        o = self.conv(x)
        return o


class DepthwiseSeparableConv(Layer):
    '''
    Depthwise Separable Convolutions(Depthwise Conv & Pointwise Conv)
    Args:
        in_channels (int) – Number of channels in the input features
        out_channels (int) – Number of channels produced by the convolution
        kernel_size (int or tuple) – Size of the convolving kernel
        dim (int) – Dim of the convolution. Default: 1
        bias (bool) – Need bias. Default: True

    Inputs:
        - input
            if dim = 1: (batch, channel_in, width): tensor
            elif dim = 2: (batch, channel_in, height, width): tensor

    Outputs:
        - output
            if dim = 1: (batch, channel_out, width): tensor
            elif dim = 2: (batch, channel_out, height, width): tensor
    '''
    def __init__(self, in_channels, out_channels, kernel_size, dim=1, bias=True):
        super().__init__()
        if dim == 1:
            self.depthwise_conv = torch.nn.Conv1d(in_channels=in_channels,
                                                  out_channels=in_channels,
                                                  kernel_size=kernel_size,
                                                  groups=in_channels,
                                                  padding=kernel_size // 2,
                                                  bias=False)
            self.pointwise_conv = torch.nn.Conv1d(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=1,
                                                  padding=0,
                                                  bias=bias)
        elif dim == 2:
            self.depthwise_conv = torch.nn.Conv2d(in_channels=in_channels,
                                                  out_channels=in_channels,
                                                  kernel_size=kernel_size,
                                                  groups=in_channels,
                                                  padding=kernel_size // 2,
                                                  bias=False)
            self.pointwise_conv = torch.nn.Conv2d(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=1,
                                                  padding=0,
                                                  bias=bias)
        else:
            raise Exception("Wrong dimension for Depthwise Separable Convolution!")
        self.reset_parameters()

    def reset_parameters(self):

        torch.nn.init.kaiming_normal_(self.depthwise_conv.weight)
        torch.nn.init.kaiming_normal_(self.pointwise_conv.weight)
        torch.nn.init.constant_(self.pointwise_conv.bias, 0.0)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class Highway(Layer):
    '''
    Highway Network
    Args:
        layer_num (int) – Number of channels in the input features
        size (int) – Size of highway input and output neural units

    Inputs:
        - input (batch, seq_len, size): tensor

    Outputs:
        - output (batch, seq_len, size): tensor
    '''
    def __init__(self, layer_num, size):
        super().__init__()
        self.n = layer_num
        self.linear = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(size, size), torch.nn.ReLU())
            for _ in range(self.n)])
        self.gate = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(size, size))
            for _ in range(self.n)])

    def forward(self, x):
        for i in range(self.n):
            gate = self.gate[i](x)
            nonlinear = self.linear[i](x)
            x = gate * nonlinear + (1 - gate) * x
        return x


class PositionEncoder(Layer):
    '''
    Position Encoder Network
    Args:
        pos_length (int) – Sequence length
        d_model (int) – Size of the model

    Inputs:
        - input (batch, seq_len, size): tensor

    Outputs:
        - output (batch, seq_len, size): tensor
    '''
    def __init__(self, length, d_model, min_timescale=1.0, max_timescale=1.0e4):
        super().__init__()

        position = torch.arange(length).type(torch.float32)
        num_timescales = d_model // 2
        log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        signal = torch.cat((torch.sin(scaled_time), torch.cos(scaled_time)), dim=1)
        self.signal = signal.view(1, length, d_model)

    def forward(self, x):
        if x.is_cuda:
            self.signal = self.signal.cuda()
        x = x + self.signal
        return x


class MultiHeadAttention(Layer):
    '''
    Multi Head Attention Network
    Args:
        hidden_size (int) – Hidden units size
        num_heads (int) – Number of attetion heads
        dropout (fload) – Dropout probability

    Inputs:
        - input (batch, hidden_size, seq_len): tensor

    Outputs:
        - output (batch, hidden_size, seq_len): tensor
    '''
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.mem_conv = Conv1d(hidden_size, hidden_size * 2, kernel_size=1, relu=False, bias=False)
        self.query_conv = Conv1d(hidden_size, hidden_size, kernel_size=1, relu=False, bias=False)

        bias = torch.empty(1)
        torch.nn.init.constant_(bias, 0)
        self.bias = torch.nn.Parameter(bias)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout

    def forward(self, queries, mask):
        mem = queries
        # memory: (batch, hidden_size * 2, seq_len) -> (batch, seq_len, hidden_size * 2)
        memory = self.mem_conv(mem).transpose(1, 2)
        # query: (batch, hidden_size, seq_len) -> (batch, seq_len, hidden_size)
        query = self.query_conv(queries).transpose(1, 2)
        # Q, K, V: (batch, num_heads, seq_len, hidden_size/num_heads)
        Q = self.split_last_dim(query, self.num_heads)
        K, V = [self.split_last_dim(tensor, self.num_heads) for tensor in torch.split(memory, self.hidden_size, dim=2)]
        key_dim_per_head = self.hidden_size // self.num_heads
        Q *= key_dim_per_head ** -0.5
        # x: (batch, num_heads, length_q, dim_v)
        x = self.dot_product_attention(Q, K, V, mask=mask)
        o = self.combine_last_two_dim(x.permute(0, 2, 1, 3)).transpose(1, 2)
        return o

    def dot_product_attention(self, q, k, v, bias=False, mask=None):
        """dot-product attention.
        Args:
        q: a Tensor with shape [batch, heads, length_q, dim_q]
        k: a Tensor with shape [batch, heads, length_kv, dim_k]
        v: a Tensor with shape [batch, heads, length_kv, dim_v]
        bias: bias Tensor (see attention_bias())
        mask: an optional Tensor
        Returns: a Tensor with shape (batch, num_heads, length_q, dim_v)
        """
        # logits: (batch, num_heads, length_q, length_kv)
        logits = torch.matmul(q, k.permute(0, 1, 3, 2))
        if bias:
            logits += self.bias
        if mask is not None:
            shapes = list(logits.size())
            mask = mask.view(shapes[0], 1, 1, shapes[-1])
            logits = masked_logits(logits, mask)
        weights = F.softmax(logits, dim=-1)
        # dropping out the attention links for each of the heads
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        # o: (batch, num_heads, length_q, dim_v)
        o = torch.matmul(weights, v)
        return o

    def split_last_dim(self, x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
        x: a Tensor with shape [..., m]
        n: an integer.
        Returns:
        a Tensor with shape [..., n, m/n]
        """
        old_shape = list(x.size())
        last_dim = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last_dim // n if last_dim else None]
        ret = x.view(new_shape)
        return ret.permute(0, 2, 1, 3)

    def combine_last_two_dim(self, x):
        """Reshape x so that the last two dimension become one.
        Args:
        x: a Tensor with shape [..., a, b]
        Returns:
        a Tensor with shape [..., ab]
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)
        return ret