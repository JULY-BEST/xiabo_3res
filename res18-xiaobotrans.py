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
from pytorch_wavelets import DWTForward, DWTInverse
from models.attention import EncoderSelfAttention
from torchvision.models import resnet18
from torch.nn.functional import interpolate
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



class Module(nn.Module):
    def __init__(self, embed_dim=128, depth=4, device="cuda", bidirectional=True):
        super(Module, self).__init__()

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_view_extract = ASR_model().to(self.device)  # 用于提取特征
        # 加载 ResNet18 并移除分类头
        self.resnet18 = resnet18(pretrained=True)
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-2])  # 保留特征提取部分
        # 输入通道调整（从单通道到 3 通道）
        self.input_conv = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)

        # 降维：将 ResNet 输出的 512 维映射到 128 维
        self.feature_reduction = nn.Linear(512, embed_dim)
        # 确保初始化时传递所有所需的参数
        d_k = 64  # Dimension of query/key
        d_v = 64  # Dimension of value
        n_head = 8  # Number of attention heads
        d_model = 128  # Model embedding dimension
        n_module = 6  # Number of encoder layers (default)
        dropout_transformer = 0.1  # Dropout rate
        self.self_attention = EncoderSelfAttention(input_dim=embed_dim, d_model=d_model,
                                                   d_k=d_k, d_v=d_v, n_head=n_head,
                                                   n_module=n_module, dropout_transformer=dropout_transformer).to(self.device)

        # 全连接层
        self.fc = nn.Linear(embed_dim, 2).to(self.device)  # 二分类输出

    def forward(self, inputs, inputs2=None, Freq_aug=None):
        x = inputs

        """多视角特征提取"""
        # 提取特征
        text_view = self.text_view_extract(x)  # 假设形状为 (batch_size, seq_len, feature_dim)
        # 拼接特征
        concat_view = torch.cat([text_view, text_view], dim=-1)  # 拼接后形状为 (batch_size, seq_len, 2 * feature_dim)


        """通过 ResNet18 提取高级特征"""
        # 调整 concat_view 的形状为 ResNet 输入 (batch_size, channels, height, width)
        concat_view = concat_view.unsqueeze(1)  # 新增通道维度，形状为 (batch_size, 1, seq_len, feature_dim)

        # 插值调整到 ResNet18 的输入大小 (batch_size, 1, 224, 224)
        concat_view = interpolate(concat_view, size=(224, 224), mode='bilinear', align_corners=False)

        # 转换单通道到 3 通道
        concat_view = self.input_conv(concat_view)

        # 使用 ResNet18 提取特征
        resnet_features = self.resnet18(concat_view)  # 输出形状为 (batch_size, 512, 7, 7)

        # 展平 spatial 维度并将通道数降维到 128
        resnet_features = resnet_features.mean(dim=[-2, -1])  # 先对 7x7 进行全局平均池化，形状变为 (batch_size, 512)
        resnet_features = self.feature_reduction(resnet_features)  # 降维，形状变为 (batch_size, 128)

        # 还原到序列形状，假设 seq_len 为 50
        seq_len = 50  # 根据实际情况设定 seq_len
        resnet_features = resnet_features.unsqueeze(1).expand(-1, seq_len, -1)  # 形状为 (batch_size, seq_len, 128)

        # 自注意力模块
        x = self.self_attention(resnet_features)  # 输出形状 (batch_size, seq_len, embed_dim)

        # 获取最后一个时间步的特征
        x = x[:, -1, :]  # (batch_size, embed_dim)

        # 全连接层进行二分类
        x = self.fc(x)  # 输出形状 (batch_size, 2)

        return x
