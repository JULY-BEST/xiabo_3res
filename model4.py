#without MCP
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse


def position_embedding(input, d_model):
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out


def sinusoid_encoding_table(max_len, d_model):
    pos = torch.arange(max_len, dtype=torch.float32)
    out = position_embedding(pos, d_model)
    return out


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention
    """

    def __init__(self, d_model, d_k, d_v, h):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights(gain=1.0)

    def init_weights(self, gain=1.0):
        nn.init.xavier_normal_(self.fc_q.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_k.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_v.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_o.weight, gain=gain)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values):
        """
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :return:
        """
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        att = torch.softmax(att, -1)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer with Dropout and Layer Normalization.
    """

    def __init__(self, d_model, d_k, d_v, h, dff=2048, dropout=.1):
        super(MultiHeadAttention, self).__init__()

        # self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        # self.layer_norm2 = nn.LayerNorm(d_model)

        self.fc = nn.Sequential(*[nn.Linear(d_model, dff), nn.ReLU(inplace=True), nn.Dropout(p=dropout),
                                  nn.Linear(dff, d_model)])

    def forward(self, queries, keys, values):
        att = self.attention(queries, keys, values)
        att = self.dropout(att)
        # att = self.layer_norm(queries + att)
        att = self.fc(att)
        att = self.dropout(att)
        return self.layer_norm(queries + att)


class ScaledDotProductAttention_(nn.Module):
    """
    Scaled dot-product attention
    """

    def __init__(self, d_model, d_k, d_v, h):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(ScaledDotProductAttention_, self).__init__()
        # print(d_model)
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights(gain=1.0)

    def init_weights(self, gain=1.0):
        nn.init.xavier_normal_(self.fc_q.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_k.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_v.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_o.weight, gain=gain)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values):
        """
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :return:
        """
        b_s, nq = queries.shape[:2]
        # nk = keys.shape[1]
        # print(queries.shape)
        # print(b_s)
        q = self.fc_q(queries).view(b_s, self.h, self.d_k).permute(0, 1, 2)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, self.h, self.d_k).permute(0, 2, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, self.h, self.d_v).permute(0, 1, 2)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        att = torch.softmax(att, -1)

        out = torch.matmul(att, v).permute(0, 1, 2).contiguous().view(b_s, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool1d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x)


class SSL(nn.Module):
    def __init__(self, channels):
        super(SSL, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels,
                               bias=False)  # conv_block_my(channels, channels, kernel_size = 3, stride=1, padding = 1, dilation=1)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels,
                               bias=False)  # conv_block_my(channels, channels, kernel_size = 3, stride=1, padding = 5, dilation=5)
        self.conv7 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels,
                               bias=False)  # conv_block_my(channels, channels, kernel_size = 3, stride=1, padding = 7, dilation=7)
        self.conv9 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels,
                               bias=False)  # conv_block_my(channels, channels, kernel_size = 3, stride=1, padding = 9, dilation=9)

        self.conv_cat = nn.Conv2d(channels * 4, channels, kernel_size=3, padding=1, groups=channels,
                                  bias=False)  # conv_block_my(channels*4, channels, kernel_size = 3, stride = 1, padding = 1, dilation=1)

    def forward(self, x):
        aa = DWTForward(J=1, mode='zero', wave='db3').cuda(device=0)
        yl, yh = aa(x)

        yh_out = yh[0]
        ylh = yh_out[:, :, 0, :, :]
        yhl = yh_out[:, :, 1, :, :]
        yhh = yh_out[:, :, 2, :, :]

        conv_rec1 = self.conv5(yl)
        conv_rec5 = self.conv5(ylh)
        conv_rec7 = self.conv7(yhl)
        conv_rec9 = self.conv9(yhh)

        cat_all = torch.stack((conv_rec5, conv_rec7, conv_rec9), dim=2)
        rec_yh = []
        rec_yh.append(cat_all)

        ifm = DWTInverse(wave='db3', mode='zero').cuda(device=0)
        Y = ifm((conv_rec1, rec_yh))

        return Y


class MultiHeadAttention_(nn.Module):
    """
    Multi-head attention layer with Dropout and Layer Normalization.
    """

    def __init__(self, d_model, d_k, d_v, h, dff=2048, dropout=.1):
        super(MultiHeadAttention_, self).__init__()

        # self.attention = ScaledDotProductAttention_(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.s = SSL(1)
        # self.token_mixer = Pooling(pool_size=3)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        # self.layer_norm2 = nn.LayerNorm(d_model)
        self.token_mixer1 = Pooling(pool_size=3)
        self.token_mixer2 = Pooling(pool_size=5)
        self.token_mixer3 = Pooling(pool_size=7)

        hidden_features = int(d_model * 2.66)

        self.project_in = nn.Conv1d(d_model, hidden_features * 2, kernel_size=1)

        self.dwconv = nn.Conv1d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2)

        self.project_out = nn.Conv1d(hidden_features, d_model, kernel_size=1)

        # self.fc = nn.Sequential(*[nn.Linear(d_model, dff), nn.ReLU(inplace=True), nn.Dropout(p=dropout),
        #                           nn.Linear(dff, d_model)])

    # def forward(self, queries, keys, values):
    def forward(self, x):
        '''删掉MCP'''
        # x = x.unsqueeze(1)
        # x = self.s(x)
        # # print(x.shape)
        # x = x.squeeze(1)
        att1 = self.token_mixer1(x)
        att2 = self.token_mixer2(x)
        att3 = self.token_mixer3(x)

        att = (att1 + att2 + att3) / 3
        # att = self.token_mixer(x)
        # print(att.shape)
        # att = self.attention(queries, keys, values)
        att = self.dropout(att)
        # att = self.layer_norm(queries + att)
        g = self.project_in(att.permute(0, 2, 1))
        x1, x2 = self.dwconv(g).chunk(2, dim=1)
        g = F.gelu(x1) * x2
        g = self.project_out(g)

        att = self.dropout(g.permute(0, 2, 1))

        return self.layer_norm(x + att)


class EncoderSelfAttention(nn.Module):
    def __init__(self, input_dim, d_model, d_k, d_v, n_head, dff=2048, dropout_transformer=0.1, n_module=6):
        """
        :param input_dim: Input embedding dimension (e.g., 128 for your case)
        :param d_model: Model embedding dimension (can match input_dim or differ)
        :param d_k: Dimension of query/key
        :param d_v: Dimension of value
        :param n_head: Number of attention heads
        :param dff: Feedforward network dimension
        :param dropout_transformer: Dropout rate
        :param n_module: Number of encoder layers
        """
        super(EncoderSelfAttention, self).__init__()

        # Project input features to d_model if necessary
        self.input_projection = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()

        # Positional encoding
        self.positional_encoding = sinusoid_encoding_table(50, d_model)  # Adjust max_len as needed
        self.dropout = nn.Dropout(p=dropout_transformer)

        # Multi-head attention encoder layers
        self.encoder_layers = nn.ModuleList([
            MultiHeadAttention_(d_model, d_k, d_v, n_head, dff, dropout_transformer)
            for _ in range(n_module)
        ])

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        :param x: Input tensor of shape [batch_size, seq_len, input_dim]
        """
        # Input projection if needed
        x = self.input_projection(x)

        # Add positional encoding
        pos_enc = self.positional_encoding[:x.size(1), :].unsqueeze(0).to(x.device)  # Shape: [1, seq_len, d_model]
        x = x + pos_enc
        x = self.dropout(x)

        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)  # Multi-head attention + token mixer

        # Final layer normalization
        x = self.layer_norm(x)
        return x

