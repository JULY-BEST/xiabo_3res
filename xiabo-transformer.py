import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.fft
import fairseq
import os
from torch import Tensor
from jaxtyping import Float32
from typing import Literal
from .rmsnorm import RMSNorm
from typing import Union
from .ACRNN import acrnn
from pytorch_wavelets import DWTForward, DWTInverse
from models.attention import EncoderSelfAttention

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



        # 自注意力模块
        x = self.self_attention(x)  # 输出形状 (batch_size, seq_len, embed_dim)

        # 获取最后一个时间步的特征
        x = x[:, -1, :]  # (batch_size, embed_dim)

        # 全连接层进行二分类
        x = self.fc(x)  # 输出形状 (batch_size, 2)

        return x


