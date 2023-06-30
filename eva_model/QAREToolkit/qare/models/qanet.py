import torch
import torch.nn as nn
import torch.nn.functional as F
from eva_model.QAREToolkit.qare.nn.layers import WordEmbedding
from eva_model.QAREToolkit.qare.models.base_model import BaseModel
from eva_model.QAREToolkit.qare.nn.operations import masked_logits
from eva_model.QAREToolkit.qare.nn.layers import PositionEncoder
from eva_model.QAREToolkit.qare.nn.layers import MultiHeadAttention
from eva_model.QAREToolkit.qare.nn.layers import Conv1d


class QANet(BaseModel):
    """
    QANet model
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
        super().__init__(vocab, config)
        self.char_emb = WordEmbedding(vocab.get_char_vocab_size(), config.char_embedding_size)
        torch.nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)

        self.word_emb = WordEmbedding(vocab_size=vocab.get_word_vocab_size(),
                                      embd_size=config.word_embedding_size,
                                      pre_word_embd=vocab.word_embd)
        self.emb = Embedding(config)
        self.c_emb_enc = EncoderBlock(conv_num=4,
                                      ch_num=config.hidden_size,
                                      k=7,
                                      length=config.answer_truncate_len,
                                      config=config)
        self.q_emb_enc = EncoderBlock(conv_num=4,
                                      ch_num=config.hidden_size,
                                      k=7,
                                      length=config.question_truncate_len,
                                      config=config)
        self.cq_att = CQAttention(config)
        self.cq_resizer = Conv1d(config.hidden_size * 4, config.hidden_size)
        self.model_enc_blks = nn.ModuleList([EncoderBlock(conv_num=2,
                                                          ch_num=config.hidden_size,
                                                          k=5,
                                                          length=config.answer_truncate_len,
                                                          config=config) for _ in range(7)])
        # self.fc_layer = torch.nn.Linear(config.hidden_size * config.answer_truncate_len,
        #                                 config.n_classes)
        self.fc_layer = torch.nn.Linear(config.hidden_size, config.n_classes)
        self.dropout = config.dropout

    def loss_function(self):
        return torch.nn.CrossEntropyLoss()

    def optimizer(self):
        optimizer_kwargs = {"lr": self.config.learning_rate,
                            "rho": 0.9, "eps": 1e-6, "weight_decay": 0}
        optimizer_ = torch.optim.Adadelta
        return optimizer_, optimizer_kwargs

    def forward(self, question, paragraph):

        Cc, _ = self.char_emb.forward(paragraph["char"])
        # (batch, q_len, word_len, char_dim)
        Qc, _ = self.char_emb.forward(question["char"])

        # 2. Word Embedding Layer
        # (batch, p_len, word_dim)
        Cw, maskC = self.word_emb.forward(paragraph["word"])
        # (batch, q_len, word_dim)
        Qw, maskQ = self.word_emb.forward(question["word"])

        C, Q = self.emb(Cc, Cw), self.emb(Qc, Qw)
        Ce = self.c_emb_enc(C, maskC, 1, 1)
        Qe = self.q_emb_enc(Q, maskQ, 1, 1)


        X = self.cq_att(Ce, Qe, maskC, maskQ)
        M0 = self.cq_resizer.forward(X)
        M1 = F.dropout(M0, p=self.dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
            M1 = blk(M1, maskC, i * (2 + 2) + 1, 7)
        M2 = M1
        for i, blk in enumerate(self.model_enc_blks):
            M2 = blk(M2, maskC, i * (2 + 2) + 1, 7)
        M3 = F.dropout(M2, p=self.dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
            M3 = blk(M3, maskC, i * (2 + 2) + 1, 7)
        m3 = M3.select(2, -1)
        # M3 = M3.contiguous().view(M3.size(0), M3.size(1) * M3.size(2))
        o = self.fc_layer.forward(m3)
        # o = self.fc_layer.forward(M3)
        output = F.softmax(o, dim=1)  # (batch, n_classes)
        return output


class Highway(nn.Module):
    def __init__(self, layer_num, size, config):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([Conv1d(size, size, relu=True, bias=True) for _ in range(self.n)])
        self.gate = nn.ModuleList([Conv1d(size, size, bias=True) for _ in range(self.n)])
        self.dropout = config.dropout

    def forward(self, x):
        # x: (batch_size, hidden_size, length)
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = self.linear[i](x)
            nonlinear = F.dropout(nonlinear, p=self.dropout, training=self.training)
            x = gate * nonlinear + (1 - gate) * x
        return x


class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv2d = nn.Conv2d(config.char_embedding_size, config.hidden_size, kernel_size=(1, 5), padding=0, bias=True)
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')
        self.conv1d = Conv1d(config.word_embedding_size + config.hidden_size,
                                         config.hidden_size, bias=False)
        self.high = Highway(2, config.hidden_size, config)
        self.dropout_char = config.dropout_char
        self.dropout = config.dropout

    def forward(self, ch_emb, wd_emb):
        ch_emb = ch_emb.permute(0, 3, 1, 2)
        ch_emb = F.dropout(ch_emb, p=self.dropout_char, training=self.training)
        ch_emb = self.conv2d(ch_emb)
        ch_emb = F.relu(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=3)

        wd_emb = F.dropout(wd_emb, p=self.dropout, training=self.training)
        wd_emb = wd_emb.transpose(1, 2)
        emb = torch.cat((ch_emb, wd_emb), dim=1)
        emb = self.conv1d.forward(emb)
        emb = self.high(emb)
        return emb


class EncoderBlock(nn.Module):
    def __init__(self, conv_num, ch_num, k, length, config):
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(ch_num, ch_num, k) for _ in range(conv_num)])
        self.pos_encoder = PositionEncoder(length, config.hidden_size)
        self.self_att = MultiHeadAttention(config.hidden_size, config.num_heads, config.dropout)
        self.FFN_1 = Conv1d(ch_num, ch_num, relu=True, bias=True)
        self.FFN_2 = Conv1d(ch_num, ch_num, bias=True)
        self.norm_C = nn.ModuleList([nn.LayerNorm(config.hidden_size) for _ in range(conv_num)])
        self.norm_1 = nn.LayerNorm(config.hidden_size)
        self.norm_2 = nn.LayerNorm(config.hidden_size)
        self.conv_num = conv_num
        self.dropout = config.dropout

    def forward(self, x, mask, l, blks):
        # x: (batch_size, hidden_size, length)
        # mask: (batch_size, length)
        total_layers = (self.conv_num + 1) * blks
        x = x.transpose(1, 2)
        out = self.pos_encoder.forward(x)
        out = out.transpose(1, 2)
        for i, conv in enumerate(self.convs):
            res = out
            out = self.norm_C[i](out.transpose(1, 2)).transpose(1, 2)
            if (i) % 2 == 0:
                out = F.dropout(out, p=self.dropout, training=self.training)
            out = conv(out)
            out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
            l += 1
        res = out
        out = self.norm_1(out.transpose(1, 2)).transpose(1, 2)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.self_att.forward(out, mask)
        out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
        l += 1
        res = out

        out = self.norm_2(out.transpose(1, 2)).transpose(1, 2)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.FFN_1.forward(out)
        out = self.FFN_2.forward(out)
        # out: (batch_size, hidden_size, length)
        out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
        return out

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0, 1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                        padding=k // 2, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))


class CQAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        w4C = torch.empty(config.hidden_size, 1)
        w4Q = torch.empty(config.hidden_size, 1)
        w4mlu = torch.empty(1, 1, config.hidden_size)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)
        self.dropout = config.dropout
        self.Lc = config.answer_truncate_len
        self.Lq = config.question_truncate_len

    def forward(self, C, Q, Cmask, Qmask):
        C = C.transpose(1, 2)
        Q = Q.transpose(1, 2)
        batch_size_c = C.size()[0]
        # C: (batch, p_len, hidden_size)
        # Q: (batch, q_len, hidden_size)
        S = self.trilinear_for_attention(C, Q)
        Cmask = Cmask.view(batch_size_c, self.Lc, 1)
        Qmask = Qmask.view(batch_size_c, 1, self.Lq)
        # S1: (batch, p_len, q_len)
        S1 = F.softmax(masked_logits(S, Qmask), dim=2)
        # S2: (batch, p_len, q_len)
        S2 = F.softmax(masked_logits(S, Cmask), dim=1)
        # A: (batch, p_len, hidden_size)
        A = torch.bmm(S1, Q)
        # B: (batch, p_len, hidden_size)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        # o: (batch, p_len, hidden_size * 4)
        o = torch.cat((C, A, torch.mul(C, A), torch.mul(C, B)), dim=2)
        # o: (batch, hidden_size * 4, p_len)
        o = o.transpose(1, 2)
        return o

    def trilinear_for_attention(self, C, Q):
        C = F.dropout(C, p=self.dropout, training=self.training)
        Q = F.dropout(Q, p=self.dropout, training=self.training)
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, self.Lq])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, self.Lc, -1])
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1, 2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res
