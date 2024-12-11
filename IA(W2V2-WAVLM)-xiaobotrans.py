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
from typing import Union
from .ACRNN import acrnn
from pytorch_wavelets import DWTForward, DWTInverse
from models.attention import EncoderSelfAttention
from .WavLM import WavLM, WavLMConfig

class ASR_model(nn.Module):
    def __init__(self):
        super(ASR_model, self).__init__()
        cp_path = os.path.join(
           "/home/wr/baseline_wr_ssl/xlsr2_300m.pt")  # Change the pre-trained XLSR model path.
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
        checkpoint = torch.load("/home/wr/WavLM-Large.pt")  # 替换为你的WavLM模型路径
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



class Module(nn.Module):
    def __init__(self, embed_dim=128, depth=4, device="cuda", bidirectional=True):
        super(Module, self).__init__()

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.text_view_extract=ASR_model()
        self.wav_view_extract=WavLM_Model()
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
        x=inputs
        """multi-view features"""
        text_view=self.text_view_extract(x) # 10 50 64
        x1=text_view
        x1 = torch.flatten(x1, 1)
        x1 = F.relu(self.fc2(x1), inplace=False)
        # print(text_view.shape)
        wav_view=self.wav_view_extract(x) # 10 50 64
        x2=wav_view
        x2 = torch.flatten(x2, 1)
        x2 = F.relu(self.fc2(x2), inplace=False)
        # print(wav_view.shape)
        x = torch.cat([x1, x2], dim=-1)  # [2,256]
        x = self.fc(x)  # [2,128]
        x11 = x * x1
        attn1 = F.softmax(x11, dim=-1)
        # output1 = torch.matmul(attn1, x)
        output1 = attn1 * x
        #output1 = output1 + x1
        # dropout = nn.Dropout(0.1)
        x1 = self.x_dropout(output1)

        # x12 = torch.matmul(x,x2)
        x12 = x * x2
        attn2 = F.softmax(x12, dim=-1)
        # output2 = torch.matmul(attn2, x)
        output2 = attn2 * x
        #output2 = output2 + x2
        # dropout = nn.Dropout(0.1)
        x2 = self.x_dropout(output2)

        x = x1 + x2
        print(x.shape)
        #concat_view = torch.cat([text_view,wav_view], dim=-1) # 10 50 128

        # 自注意力模块
        x = self.self_attention(x)  # 输出形状 (batch_size, seq_len, embed_dim)

        # 获取最后一个时间步的特征
        x = x[:, -1, :]  # (batch_size, embed_dim)

        # 全连接层进行二分类
        x = self.fc(x)  # 输出形状 (batch_size, 2)

        return x
