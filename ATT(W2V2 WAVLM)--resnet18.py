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
from .WavLM import WavLM, WavLMConfig
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


class WavLM_Model(nn.Module):
    def __init__(self):
        super(WavLM_Model, self).__init__()
        # 加载预训练的WavLM模型
        checkpoint = torch.load("/home/wangrui/WavLM-Large.pt")  # 替换为你的WavLM模型路径
        cfg = WavLMConfig(checkpoint['cfg'])  # 获取WavLM的配置
        self.model = WavLM(cfg)  # 创建WavLM模型
        self.model.load_state_dict(checkpoint['model'])  # 加载预训练模型的权重
        self.model.eval()  # 设置为评估模式
        # 线性层和处理层
        self.linear = nn.Linear(1024, 128)  # 将特征维度从1024降到128
        self.bn = nn.BatchNorm1d(50)  # 批量归一化
        self.selu = nn.SELU(inplace=True)  # SELU激活函数

    def forward(self, x):
        # 如果配置文件中要求进行归一化，进行归一化
        if self.model.cfg.normalize:
            x = torch.nn.functional.layer_norm(x, x.shape)

        # 提取特征，返回最后一层的特征
        rep = self.model.extract_features(x)[0]  # [0] 表示取出最后一层的特征
        emb = self.linear(rep)  # 降维到128维
        emb = F.max_pool2d(emb, (4, 2))
        emb = self.bn(emb)
        emb = self.selu(emb)

        return emb  # 输出的形状为 (batch_size, frame_number, 128)


def ScaledDotProductAttention(q, k, v, temperature):
    attn = torch.matmul(q / temperature, k.transpose(1, 2))
    dropout = nn.Dropout(0.1)
    attn = dropout(F.softmax(attn, dim=-1))
    output = torch.matmul(attn, v)
    return output, attn


class Module(nn.Module):
    def __init__(self, embed_dim=128, depth=4, device="cuda", bidirectional=True):
        super(Module, self).__init__()

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载 ResNet18 并移除分类头
        self.resnet18 = resnet18(pretrained=True)
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-2])  # 保留特征提取部分
        # 输入通道调整（从单通道到 3 通道）
        self.input_conv = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)

        # 降维：将 ResNet 输出的 512 维映射到 128 维
        self.feature_reduction = nn.Linear(512, embed_dim)
        
        # 特征提取部分
        self.text_view_extract = ASR_model().to(self.device)  # 用于提取特征
        self.wav_view_extract = WavLM_Model().to(self.device)

        # 全连接层
        self.fc = nn.Linear(embed_dim, 2).to(self.device)  # 二分类输出

    def forward(self, inputs, inputs2=None, Freq_aug=None):
        x = inputs

        """multi-view features"""
        text_view = self.text_view_extract(x)  # 10 50 64
        x1 = text_view
        # print(text_view.shape)
        wav_view = self.wav_view_extract(x)  # 10 50 64
        x2 = wav_view
        # print(wav_view.shape)
        q1 = x1
        q2 = x2

        k1 = x2
        v1 = x2

        k2 = x1
        v2 = x1

        d = 128
        out1, attn1 = ScaledDotProductAttention(q1, k1, v1, d ** 0.5)
        out2, attn2 = ScaledDotProductAttention(q2, k2, v2, d ** 0.5)

        x1 = out2
        x2 = out1

        x = torch.cat([x1, x2], dim=2)
 """通过 ResNet18 提取高级特征"""
        # 调整 concat_view 的形状为 ResNet 输入 (batch_size, channels, height, width)
        x = x.unsqueeze(1)  # 新增通道维度，形状为 (batch_size, 1, seq_len, feature_dim)

        # 插值调整到 ResNet18 的输入大小 (batch_size, 1, 224, 224)
        x = interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # 转换单通道到 3 通道
        x = self.input_conv(x)

        # 使用 ResNet18 提取特征
        resnet_features = self.resnet18(x)  # 输出形状为 (batch_size, 512, 7, 7)

        # 展平 spatial 维度并将通道数降维到 128
        resnet_features = resnet_features.mean(dim=[-2, -1])  # 先对 7x7 进行全局平均池化，形状变为 (batch_size, 512)
        resnet_features = self.feature_reduction(resnet_features)  # 降维，形状变为 (batch_size, 128)

        # 还原到序列形状，假设 seq_len 为 50
        seq_len = 50  # 根据实际情况设定 seq_len
        resnet_features = resnet_features.unsqueeze(1).expand(-1, seq_len, -1)  # 形状为 (batch_size, seq_len, 128)
        x = resnet_features

        # 取最后一个时间步的特征
        x = x[:, -1, :]  # (bs, input_dim)

        # 通过全连接层进行二分类输出
        x = self.fc(x)

        return x
