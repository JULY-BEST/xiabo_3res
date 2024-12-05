import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.fft
import fairseq
import os
from torch import Tensor
from jaxtyping import  Float32
from typing import Literal
from .rmsnorm import RMSNorm
from typing import Union
from .ACRNN import acrnn

class ASR_model(nn.Module):
    def __init__(self):
        super(ASR_model, self).__init__()
        cp_path = os.path.join(
            '/home/wangrui/AMSDF-main/pretrained_models/xlsr2_300m.pt')  # Change the pre-trained XLSR model path.
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0].cuda()
        self.linear = nn.Linear(1024, 128)
        self.bn = nn.BatchNorm1d(num_features=50)
        self.selu = nn.SELU(inplace=True)

    def forward(self, x):
        emb = self.model(x, mask=False, features_only=True)['x']
        emb = self.linear(emb)
        emb = F.max_pool2d(emb, (4, 2))
        emb = self.bn(emb)
        emb = self.selu(emb)
        return emb  # (bs,frame_number,feat_out_dim)


class Gated_MLP_block(nn.Module):
    def __init__(self, input_dim: int, expansion_factor: int = 3) -> None:
        super().__init__()
        self.D = input_dim
        self.M = expansion_factor
        self.gelu = nn.GELU()  # 移除 approximate 参数
        self.p1 = nn.Linear(in_features=self.D, out_features=self.D * self.M)
        self.p2 = nn.Linear(in_features=self.D, out_features=self.D * self.M)
        self.p3 = nn.Linear(in_features=self.D * self.M, out_features=self.D)

    def forward(self, x: Tensor) -> Tensor:
        # left branch
        x1 = self.p1(x)
        x1 = self.gelu(x1)

        # right branch
        x2 = self.p2(x)

        y = x1 * x2
        y = self.p3(y)

        return y


class Temporal_Conv1D(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int, stride: int = 1, padding: int = 0):

        super(Temporal_Conv1D, self).__init__()

        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=output_dim,
                                kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()  # 激活函数，可以根据需要替换为其他激活函数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: 输入张量，形状为 [batch_size, seq_len, input_dim]
        :return: 输出张量，形状为 [batch_size, seq_len, output_dim]
        """
        # Conv1d 输入需要是 [batch_size, input_dim, seq_len]
        x = x.transpose(1, 2)  # 将输入的形状从 [batch_size, seq_len, input_dim] 转换为 [batch_size, input_dim, seq_len]

        x = self.conv1d(x)  # 应用卷积操作，结果形状为 [batch_size, output_dim, seq_len]

        x = self.relu(x)  # 激活函数处理

        x = x.transpose(1, 2)  # 将输出的形状从 [batch_size, output_dim, seq_len] 转换为 [batch_size, seq_len, output_dim]

        return x


class Real_Gated_Linear_Recurrent_Unit(nn.Module):
    c = 8.0

    def __init__(self, dim: int, expansion_factor: Union[int, float] = 1,
                 device="cuda", dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.dtype = dtype
        self.device = device
        self.input_dim = dim
        self.hidden_dim = int(round(dim * expansion_factor))

        self.Wa = nn.Parameter(torch.empty(self.hidden_dim, dim, **factory_kwargs))
        self.Wx = nn.Parameter(torch.empty(self.hidden_dim, dim, **factory_kwargs))
        self.ba = nn.Parameter(torch.empty(self.hidden_dim, **factory_kwargs))
        self.bx = nn.Parameter(torch.empty(self.hidden_dim, **factory_kwargs))

        # 修改 Lambda 初始化为 nn.Parameter
        self.Lambda = nn.Parameter(torch.empty(self.hidden_dim, **factory_kwargs))  # 正确的方式

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # https://tinyurl.com/lecuninit
        nn.init.normal_(self.Wa, mean=0, std=1 / (self.input_dim ** 0.5))
        nn.init.normal_(self.Wx, mean=0, std=1 / (self.input_dim ** 0.5))

        # init bias
        nn.init.zeros_(self.ba)  # 可以根据需要初始化偏置
        nn.init.zeros_(self.bx)

        # init Λ
        nn.init.uniform_(self.Lambda, a=0.9, b=0.999)
        self.Lambda.data = - torch.log(
            (self.Lambda.data ** (-1. / self.c)) - 1.
        )

    def foresee(self, x: Tensor) -> Tensor:
        # print("x.shape",x.shape)
        batch_size, sequence_length = x.shape[:2]
        ht = torch.zeros(batch_size, self.hidden_dim, dtype=self.dtype, device=self.device)  # 确保 ht 在正确的设备上
        y = torch.empty(batch_size, sequence_length, self.hidden_dim, dtype=self.dtype,
                        device=self.device)  # 确保 y 在正确的设备上

        for t in range(sequence_length):
            xt = x[:, t, :].to(self.device)  # 确保 xt 在正确的设备上
            rt = torch.sigmoid(F.linear(xt, self.Wa, self.ba)).to(self.device)  # (1)
            it = torch.sigmoid(F.linear(xt, self.Wx, self.bx)).to(self.device)  # (2)

            log_at = - self.c * F.softplus(-self.Lambda, beta=1, threshold=20) * rt  # (6)
            at = torch.exp(log_at).to(self.device)  # 确保 at 在正确的设备上
            # print(f"t: {t}, xt shape: {xt.shape}, rt shape: {rt.shape}, it shape: {it.shape}, at shape: {at.shape}, ht shape: {ht.shape}")
            ht = at * ht + torch.sqrt(1 - at ** 2) * (it * xt)  # (4)

            y[:, t, :] = ht

        return y

    def forward(self, x: Tensor) -> Tensor:
        batch_size, sequence_length = x.shape[:2]
        ht = torch.zeros(batch_size, self.hidden_dim, dtype=self.dtype, device=self.device)  # 确保 ht 在正确的设备上
        y = []
        for t in range(sequence_length):
            xt = x[:, t, :].to(self.device)  # 确保 xt 在正确的设备上
            rt = torch.sigmoid(F.linear(xt, self.Wa, self.ba)).to(self.device)  # (1)
            it = torch.sigmoid(F.linear(xt, self.Wx, self.bx)).to(self.device)  # (2)

            log_at = - self.c * F.softplus(-self.Lambda, beta=1, threshold=20) * rt  # (6)
            at = torch.exp(log_at).to(self.device)  # 确保 at 在正确的设备上

            ht = at * ht + torch.sqrt(1 - at ** 2) * (it * xt)  # (4)

            y.append(ht.unsqueeze(1))

        y = torch.cat(y, dim=1)

        return y


class Recurrent_block(nn.Module):
    def __init__(self, input_dim: int, D_rnn: int, temporal_conv_output_dim: int, kernel_size: int = 1):
        super().__init__()

        self.D = input_dim
        self.D_rnn = D_rnn
        self.temporal_conv_output_dim = temporal_conv_output_dim

        self.gelu = nn.GELU()
        self.p1 = nn.Linear(in_features=self.D, out_features=D_rnn)  # 左支路的线性层
        self.p2 = nn.Linear(in_features=self.D, out_features=D_rnn)  # 右支路的线性层

        # 右支路的 Temporal Conv1D 和 Real Gated Linear Recurrent Unit
        self.temporal_conv1d = Temporal_Conv1D(input_dim=D_rnn, output_dim=temporal_conv_output_dim, kernel_size=kernel_size)
        self.rglru = Real_Gated_Linear_Recurrent_Unit(dim=temporal_conv_output_dim)

        # 融合后使用的线性层
        self.p3 = nn.Linear(in_features=D_rnn, out_features=self.D)

    def forward(self, x: Tensor) -> Tensor:
        if x is None:
            raise ValueError("Input to Recurrent_block is None")

        # 左支路
        x1 = self.p1(x)
        # print("x1.shape",x1.shape)
        x1 = self.gelu(x1)
        # print("x1.shape",x1.shape)

        # 右支路
        x2 = self.p2(x)
        # print("x2.shape",x2.shape)
        x2 = self.temporal_conv1d(x2)
        # print("x2.shape",x2.shape)
        x2 = self.rglru.foresee(x2)
        # print("x2.shape",x2.shape)

        # 与左支路相乘
        x2 = x1 * x2

        # 最终通过线性层输出
        out = self.p3(x2)

        return out


class Residual_block(nn.Module):
    def __init__(self, input_dim:int):
        super().__init__()
        self.mlp = Gated_MLP_block(input_dim, expansion_factor=3)
        temporal_conv_output_dim = 128  # Temporal Conv1D 的输出特征维度
        self.tmb = Recurrent_block(input_dim, D_rnn=input_dim, temporal_conv_output_dim=temporal_conv_output_dim)
        self.rmsnorm = RMSNorm(d=input_dim)  # RMSNorm 输入的维度是 input_dim

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.rmsnorm(x)  # RMSNorm 层接收的输入是 x，维度是 [batch_size, seq_len, input_dim]
        # print(f"x1 after RMSNorm: {x1.shape}")  # Debugging print statement

        x1 = self.tmb(x1)  # RNN 层接收的输入是 x1，维度是 [batch_size, seq_len, input_dim]
        # print(f"x1 after Recurrent block: {x1}")  # Debugging print statement

        if x1 is None:
            raise ValueError("x1 is None, check the output of Recurrent_block")

        x = x + x1  # 残差连接，x 和 x1 都是 [batch_size, seq_len, input_dim]
        # print(f"Shape after residual addition: {x.shape}")  # Debugging print statement

        x2 = self.rmsnorm(x)  # RMSNorm 层接收的输入是 x，维度是 [batch_size, seq_len, input_dim]
        x2 = self.mlp(x2)  # MLP 层接收的输入是 x2，维度是 [batch_size, seq_len, input_dim]

        x = x + x2  # 残差连接，x 和 x2 都是 [batch_size, seq_len, input_dim]

        return x
        
class WaveletAttention(nn.Module):
    def __init__(self, embed_dim, wavelet='db3', J=1, mode='zero'):
        super(WaveletAttention, self).__init__()
        self.dwt = DWTForward(J=J, wave=wavelet, mode=mode)
        self.idwt = DWTInverse(wave=wavelet, mode=mode)
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)

    def forward(self, x):
        # 输入 x 是 [batch_size, seq_len, embed_dim]
        batch_size, seq_len, embed_dim = x.size()

        # 小波变换
        x_dwt, _ = self.dwt(x)  # 高频和低频部分
        x_dwt = x_dwt.squeeze(1)  # [batch_size, seq_len, embed_dim]

        # 卷积提取特征
        x = self.conv(x_dwt)
        x = F.relu(x)

        # 小波逆变换
        x_reconstructed = self.idwt(x)  # 逆变换重建特征

        return x_reconstructed

class Module(nn.Module):
    def __init__(self, embed_dim=128, depth=4, device="cuda", bidirectional=True):
        super(TransformerWithWaveletAttention, self).__init__()

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 特征提取部分
        self.text_view_extract = ASR_model().to(self.device)  # 用于提取特征
        self.residual_block1 = Residual_block(embed_dim).to(self.device)
        self.residual_block2 = Residual_block(embed_dim).to(self.device)
        self.residual_block3 = Residual_block(embed_dim).to(self.device)

        # 小波注意力模块
        self.wavelet_attention = WaveletAttention(embed_dim=embed_dim, wavelet='db3', J=1, mode='zero').to(self.device)

        # 自定义 TransformerEncoderLayer
        self.encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=8, dropout=0.1)
        self.encoder_layer.self_attn = self.wavelet_attention  # 替换注意力机制

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=depth).to(self.device)

        # GRU层
        self.gru1 = nn.GRU(input_size=embed_dim, hidden_size=embed_dim, num_layers=1,
                           bidirectional=bidirectional, batch_first=True).to(self.device)
        self.gru2 = nn.GRU(input_size=embed_dim * (2 if bidirectional else 1), hidden_size=embed_dim, num_layers=1,
                           bidirectional=bidirectional, batch_first=True).to(self.device)
        self.gru3 = nn.GRU(input_size=embed_dim * (2 if bidirectional else 1), hidden_size=embed_dim, num_layers=1,
                           bidirectional=bidirectional, batch_first=True).to(self.device)

        # 全连接层
        self.fc = nn.Linear(embed_dim, 2).to(self.device)  # 确保全连接层也在正确的设备上

    def forward(self, inputs, inputs2=None, Freq_aug=None):
        x = inputs
        """ multi-view features """
        text_view = self.text_view_extract(x)  # 10 50 64
        concat_view = torch.cat([text_view, text_view], dim=-1)  # 10 50 128 (拼接后的特征)

        # 输入到残差块
        x = self.residual_block1(concat_view)
        x = self.residual_block2(x)
        x = self.residual_block3(x)

        # 转换为 Transformer 输入格式
        x = x.transpose(0, 1)  # [seq_len, batch_size, embed_dim]

        # 经过 Transformer 层（小波注意力替代自注意力机制）
        x = self.transformer_encoder(x)  # [seq_len, batch_size, embed_dim]

        # 转换回原来的维度 [batch_size, seq_len, embed_dim]
        x = x.transpose(0, 1)  # [batch_size, seq_len, embed_dim]

        # 取最后一个时间步的特征
        x = x[:, -1, :]  # (bs, input_dim)

        # 通过全连接层进行二分类输出
        x = self.fc(x)

        return x  # (bs, 2) - 二分类输出




