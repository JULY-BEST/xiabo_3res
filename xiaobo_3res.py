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

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
 return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias, stride=stride)

def dwt_init(x):
    # print(f"Input to DWT: {x.shape}")
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    output = torch.cat((x_LL, x_HL, x_LH, x_HH), 1)
    # print(f"Output from DWT: {output.shape}")
    return output

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_height, out_width])

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()

    def forward(self, x):
        return iwt_init(x)

class SALayer(nn.Module):
    def __init__(self, kernel_size=5, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, 1, keepdim=True)
        channel_pool = torch.cat([max_pool, avg_pool], dim=1)
        y = self.conv_du(channel_pool)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class HWAB(nn.Module):
    def __init__(self, n_feat, o_feat, kernel_size=1, reduction=16, bias=False, act=nn.PReLU()):
        super(HWAB, self).__init__()
        self.dwt = DWT()
        self.iwt = IWT()

        # 添加一个扩展通道数的卷积层
        self.expand_conv = nn.Conv2d(4, 256, kernel_size=1, stride=1, padding=0)  # 输入通道 4 -> 输出通道 256

        modules_body = [
            conv(n_feat * 2, n_feat, kernel_size, bias=bias),
            act,
            conv(n_feat, n_feat * 2, kernel_size, bias=bias)
        ]
        self.body = nn.Sequential(*modules_body)
        self.WSA = SALayer()
        self.WCA = CALayer(n_feat * 2, reduction, bias=bias)
        self.conv1x1 = nn.Conv2d(n_feat * 4, n_feat * 2, kernel_size=1, bias=bias)
        self.conv3x3 = nn.Conv2d(n_feat, o_feat, kernel_size=3, padding=1, bias=bias)
        self.activate = act
        self.conv1x1_final = nn.Conv2d(n_feat, o_feat, kernel_size=1, bias=bias)
        self.conv_change_channels = nn.Conv2d(65, 128, kernel_size=1, bias=False)

        # 用于调整 residual 通道数的卷积层，保证与 out 的通道数匹配
        self.residual_conv = nn.Conv2d(2, 128, kernel_size=1, bias=False)  # 将 residual 的通道数调整为 128

    def forward(self, x):
        residual = x
        # print("residual",residual.shape)

        # 假设输入 x 是 (batch_size, channels, height, width)，然后分割成两部分
        wavelet_path_in, identity_path = torch.chunk(x, 2, dim=1)

        # DWT变换
        x_dwt = self.dwt(wavelet_path_in)

        # 扩展通道数
        x_dwt = self.expand_conv(x_dwt)  # 现在 x_dwt 的通道数是 256

        # 确保所有张量都在相同的设备上
        device = x.device  # 获取输入 x 所在的设备
        wavelet_path_in = wavelet_path_in.to(device)
        identity_path = identity_path.to(device)

        res = self.body(x_dwt)
        branch_sa = self.WSA(res)
        branch_ca = self.WCA(res)
        res = torch.cat([branch_sa, branch_ca], dim=1)
        res = self.conv1x1(res) + x_dwt

        # 逆小波变换
        wavelet_path = self.iwt(res)

        # 确保 wavelet_path 和 identity_path 在相同的设备上
        wavelet_path = wavelet_path.to(device)
        identity_path = identity_path.to(device)
        # 在拼接之前检查 wavelet_path 和 identity_path 的通道数
        # print(f"wavelet_path.shape: {wavelet_path.shape}")
        # print(f"identity_path.shape: {identity_path.shape}")

        # 将 wavelet_path 和 identity_path 拼接并进行卷积
        out = torch.cat([wavelet_path, identity_path], dim=1).to(device)
        # print(f"cat.shape: {out.shape}")
        out = self.conv_change_channels(out)
        out = self.activate(self.conv3x3(out))
        # print(f"out.shape: {out.shape}")

        # 调整 residual 的通道数，确保与 out 的通道数一致
        residual = self.residual_conv(residual)  # 调整 residual 的通道数

        out += self.conv1x1_final(residual)
        # print(f"out_final.shape: {out.shape}")

        return out




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


class Module(nn.Module):
    def __init__(self, embed_dim=128, depth=4, device="cuda"):
        super(Module, self).__init__()

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        """ feature extractor """
        self.text_view_extract = ASR_model().to(self.device)  # 用于提取特征
        self.residual_block1 = Residual_block(embed_dim).to(self.device)
        self.residual_block2 = Residual_block(embed_dim).to(self.device)
        self.residual_block3 = Residual_block(embed_dim).to(self.device)

        # 添加 Half Wavelet Attention Block (HWAB)
        self.hwab = HWAB(n_feat=embed_dim, o_feat=embed_dim).to(self.device)  # 使用给定的HWAB模块
        self.conv_reduce = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)

        self.fc = nn.Linear(embed_dim, 2).to(self.device)  # 确保全连接层也在正确的设备上

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def forward(self, inputs, inputs2=None, Freq_aug=None):
        x = inputs
        """ multi-view features """
        text_view = self.text_view_extract(x)  # 10 50 64
        concat_view = torch.cat([text_view, text_view], dim=-1)  # 10 50 128 (拼接后的特征)

        # 重塑 concat_view 的维度为 (batch_size, channels, height, width)
        # 假设 frame_number 作为 H，feat_out_dim 作为 W
        batch_size, seq_len, feature_dim = concat_view.size()

        # 假设 feature_dim 可以作为 channel, seq_len 作为 height，frame_number 作为 width
        concat_view_reshaped = concat_view.view(batch_size,1, feature_dim, seq_len)  # 调整为 (bs, channels, height, width)
        # print("concat_view_reshaped1",concat_view_reshaped.shape)
        concat_view_reshaped = torch.cat([concat_view_reshaped, concat_view_reshaped],
                                         dim=1)  # 结果形状: (bs, 2, feature_dim, seq_len)
        # print("concat_view_reshaped2", concat_view_reshaped.shape)

        # 通过 HWAB 进行处理
        x_hwab = self.hwab(concat_view_reshaped)  # [batch_size, channels, height, width]
        # print("x_hwab",x_hwab.shape)
        x_hwab = self.conv_reduce(x_hwab)

        x_hwab = x_hwab.squeeze(1)  # 去掉维度 128 -> torch.Size([1, 128, 50])
        # print("x_hwab", x_hwab.shape)
        # 2. 重新排列维度顺序，变为 [1, 50, 128]
        x_hwab = x_hwab.permute(0, 2, 1)

        # print("x_hwab", x_hwab.shape)

        # 输入到三个残差块

        x = self.residual_block1(x_hwab)
        x = self.residual_block2(x)
        x = self.residual_block3(x)

        # 对每个序列的输出取最后一个时间步的特征
        x = x[:, -1, :]  # (bs, input_dim)

        # 通过全连接层进行二分类输出
        output = self.fc(x)  # (bs, 2)

        return output  # (bs, 2) - 二分类输出



