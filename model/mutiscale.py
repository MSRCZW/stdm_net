# -*-coding:utf-8-*-
import torch.nn as nn
import torch
import numpy as np
import math
def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)
class Attention(nn.Module):

    def __init__(self,
                 dim,   # 输入token的dim
                 inter_channels,
                 token,
                 num_heads=3,
                 s_0 = True,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or inter_channels ** -0.5
        self.qkv = nn.Conv2d(dim, inter_channels*2*3,1, bias=True)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Conv2d(dim*3, dim,1, bias=True)#nn.linear 张量的最后一个维度且相当于乘以一个dim*dim的矩阵
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.norm = nn.BatchNorm2d(dim)
        self.tanh = nn.Tanh()
        self.inter_channels = inter_channels
        self.dim = dim
        self.s_0 = s_0
        self.alphas = nn.Parameter(torch.ones(1, num_heads, 1, 1), requires_grad=True)
        self.attention0s = nn.Parameter(torch.ones(1, 3,token, token) / token,
                                        requires_grad=True)
    def forward(self, x):

        B, C, F, V= x.shape


        q, k = torch.chunk(self.qkv(x).view(B, 2 * 3, self.inter_channels, F, V), 2,
                           dim=1)  # nctv -> n num_subset c'tv

        attn = torch.einsum('nsctu,nsctv->nsuv', [q, k]) / (self.inter_channels * F)

        attn = self.tanh(attn) * self.alphas + self.attention0s.repeat(B, 1, 1, 1)

        x = torch.einsum('nctu,nsuv->nsctv', [x, attn]).contiguous() \
            .view(B, 3 * self.dim,F, V)

        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Conv2d(in_features, out_features, 1, 1, padding=0, bias=True)
        self.norm1 = nn.BatchNorm2d(out_features)

    def forward(self, x):

        x = self.conv1(x)
        x = self.norm1(x)
        return x
class Block(nn.Module):
    def __init__(self,in_features,out_features,inter_features,token):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.inter_features = inter_features
        self.mlp = Mlp(self.in_features, 2 * self.out_features, self.out_features)
        self.atten = Attention(self.in_features,self.inter_features,token)
        self.downs2 = nn.Sequential(
            nn.Conv2d(in_features, out_features, 1, bias=True),
            nn.BatchNorm2d(out_features),
        )
        self.relu = nn.LeakyReLU(0.1)
    def forward(self,x):
        y=x
        x = self.relu(x + self.atten(x))
        y = self.downs2(y)
        x = y + self.mlp(x)
        x = self.relu(x)
        return x
class PositionalEncoding(nn.Module):

    def __init__(self, channel, joint_num, time_len, domain):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len

        self.domain = domain

        if domain == "temporal":
            # temporal embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(t)
        elif domain == "spatial":
            # spatial embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(j_id)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()

        pe = torch.zeros(self.time_len * self.joint_num, channel)

        div_term = torch.exp(torch.arange(0, channel, 2).float() *
                             -(math.log(10000.0) / channel))  # channel//2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # nctv
        x = x + self.pe[:, :, :x.size(2)]
        return x
class TIT1AttentionBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=128, num_subset=3, num_node=25, num_frame=128,class_num= 60,F_token = 128,V_token=25):
        super(TIT1AttentionBlock, self).__init__()

        self.configs = [[64, 64, 16, 5], [64, 128, 32, 5],
                        [128, 128, 32, 25], [128, 128, 32, 25],
                        [128, 256, 64, 25], [256, 256, 64, 32],
                        [256, 256, 64, 32], [256, 256, 64, 16],
                        ]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subset = num_subset
        self.num_node = num_node
        self.num_frame = num_frame
        self.class_num = class_num
        self.F_token = F_token
        self.V_token = V_token
        self.embed0 = nn.Sequential(
            nn.Conv2d(3, 64, 1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
        )

        self.embed1 = nn.Sequential(
            nn.Conv2d(64, 128, (2, 1), (2, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(128),
        )
        self.embed2 = nn.Sequential(
            nn.Conv2d(128, 256, (2, 1), (2, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(256),
        )
        self.embed3 = nn.Sequential(
            nn.Conv2d(256, 256, (2, 1), (2, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(out_channels),
        )


        self.graph_layers = nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels,token) in enumerate(self.configs):

            self.graph_layers.append(
                Block(in_channels, out_channels, inter_channels,token))


        self.F_pos_embed = nn.Parameter(torch.zeros(1, self.configs[0][0],self.F_token))
        self.V_pos_embed = nn.Parameter(torch.zeros(1, self.configs[0][0], self.V_token))
        self.position1 = PositionalEncoding(64,5,128*5,'spatial')
        self.position2 = PositionalEncoding(128, 25, 64, 'spatial')
        self.position3 = PositionalEncoding(256, 32, 25, 'spatial')
        self.position4 = PositionalEncoding(256, 16, 1, 'spatial')
        # self.V_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.V_token, self.configs[0][0]))
        self.norm2s = nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, _) in enumerate(self.configs):

            self.norm2s.append(
                nn.BatchNorm1d(out_channels))
        #self.norm2 = nn.BatchNorm1d(self.out_channels)
        self.norm3 = nn.BatchNorm1d(256)
        self.head = nn.Linear(256,self.class_num)
        nn.init.trunc_normal_(self.F_pos_embed, std=0.02)
        self.data_bn = nn.BatchNorm1d(in_channels * 25)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)
    def forward(self,x):
        N,C,F,V,M = x.shape
        index = [2, 3, 8, 20, 4, 24, 23, 11, 10, 9, 22, 21, 7, 6, 5, 19, 18, 17, 16, 0, 15, 14, 13, 12, 1]
        x = x[:, :, :, index, :]
        t = (x[:, :, :, -6, :] + x[:, :, :, -1, :]) / 2
        x[:, :, :, -6, :] = x[:, :, :, -1, :] = t
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.view(N * M, C, F,V)
        x = self.embed0(x)
        x = x.view(N * M, 64, F, 5, 5).view(N * M, 64, F * 5, 5)
        x = self.position1(x)
        x = self.graph_layers[0](x)
        x = self.position1(x)
        x = self.graph_layers[0](x)
        x.view(N*M,64,F,5,5).view(N*M,64,F,25)
        x = self.embed1(x)
        x = x.view(N * M, 128, F//2 ,25)
        x = self.position2(x)
        x = self.graph_layers[2](x)
        x = self.position2(x)
        x = self.graph_layers[3](x)
        x = self.position2(x)
        x = self.graph_layers[3](x)
        x = self.position2(x)
        x = self.graph_layers[3](x)
        x = self.embed2(x)
        x = x.view(N * M, 256, F // 4, 25).permute(0,1,3,2).contiguous()
        x = self.position3(x)
        x = self.graph_layers[5](x)
        x = self.position3(x)
        x = self.graph_layers[5](x)
        x = self.position3(x)
        x = self.graph_layers[5](x)
        x = self.position3(x)
        x = self.graph_layers[5](x)
        x = x.view(N, M, 256, -1)
        x = x.permute(0, 1, 3, 2).contiguous().view(N, -1, 256,1)  # whole channels of one spatial
        x = x.mean(3).mean(1)
        x = self.head(x)

        return x