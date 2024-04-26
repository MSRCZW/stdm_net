
import torch
import torch.nn as nn
import math
import numpy as np


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)

# 划分元组，N*M, C, T, V -> N*M, P, C, T, V，P为元组维
def tuple_diviced(use_tuple, x, num_frame=300, tuple_frame=6): 

    N, C, T, V = x.shape

    # 无重复划分， 分为五组，每组六个节点，不足的元组用0补齐（空节点）
    if use_tuple == 0 : 
        x = torch.cat((x, torch.zeros((N, C, T, 5))), 3)
        shape0 = list(range(30))
        shape1 = [23, 24, 11, 10, 9, 8,
                  21, 22, 7, 6, 5, 4,
                  3, 2, 20, 1, 0, 25,
                  16, 17, 18, 19, 26, 27,
                  12, 13, 14, 15, 28, 29]
        x[:, :, :, shape0] =  x[:, :, :, shape1]
        y =  x[:, :, :, 0:6].unsqueeze(1)

        for i in range(4):
            y = torch.cat((y, x[:, :, :, 6*i:6*(i+1)].unsqueeze(1)), 1)
        
        z = y[:, :, :, 0:6, :]
        for i in range(1, int(num_frame/tuple_frame)):
            z = torch.cat((z, y[:, :, :, 6*i:6*(i+1), :]), dim=1)
        

    else : # 有重复划分， 分为九组， 每组三个节点
        x = torch.cat((x, x[:, :, :, 20]), 3)
        x = torch.cat((x, x[:, :, :, 0]), 3)
        shape0 = list(range(27))
        shape1 = [23, 24, 11, 21, 22, 7,
                  10, 9, 8, 6, 5, 4,
                  3, 2, 20, 19, 18, 17,
                  15, 14, 13, 16, 0, 12,
                  25, 1, 26]
        x[:, :, :, shape0] =  x[:, :, :, shape1]
        y =  x[:, :, :, 0:3].unsqueeze(1)
        for i in range(8):
            y = torch.cat((y, x[:, :, :, 3*i:3*(i+1)].unsqueeze(1)), 1)

        z = y[:, :, :, 0:6, :]
        for i in range(1, int(num_frame/tuple_frame)):
            z = torch.cat((z, y[:, :, :, 6*i:6*(i+1), :]), dim=1)

    return z

# 将元组化数据恢复原来结构
def un_tuple(use_tuple, x, tuple_node, tuple_frame=6): # N*M, P, C, T, V -> N*M, C, T, V


    if use_tuple == 1 :
        y = x[:, 0:9 :, :, :]
        for i in range(1, int(300/tuple_frame)):
            y = torch.cat((y, x[:, i*9:9*(i+1), :, :, :]), dim=3)
        z = y[:, 0, :, :, :]
        for i in range(1, 9):
            z = torch.cat((z, y[:, i, :, :, :]), dim=4)
        N, P, C, T, V = z.shape
        z.view(N, C, T, P*V)

        shape0 = list(range(27))
        shape1 = [23, 24, 11, 21, 22, 7,
                  10, 9, 8, 6, 5, 4,
                  3, 2, 20, 19, 18, 17,
                  15, 14, 13, 16, 0, 12,
                  25, 1, 26]
        y[:, :, :, shape1] = y[:, :, :, shape0]
        y = y[:, :, :, 0:25]
    else:
        y = x[:, 0:5 :, :, :]
        for i in range(1, int(300/tuple_frame)):
            y = torch.cat((y, x[:, i*5:5*(i+1), :, :, :]), dim=3)
        z = y[:, 0, :, :, :]
        for i in range(1, 5):
            z = torch.cat((z, y[:, i, :, :, :]), dim=4)
        N, P, C, T, V = z.shape
        z.view(N, C, T, P*V)

        shape0 = list(range(30))
        shape1 = [23, 24, 11, 10, 9, 8,
                  21, 22, 7, 6, 5, 4,
                  3, 2, 20, 1, 0, 25,
                  16, 17, 18, 19, 26, 27,
                  12, 13, 14, 15, 28, 29]
        y[:, :, :, shape1] = y[:, :, :, shape0]
        y = y[:, :, :, 0:25]
    return y


# 位置编码
class Pos_Embed(nn.Module):
    def __init__(self, channels, num_tuple, num_frames, num_joints):
        super().__init__()

        pos_list = []
        for tp in range(num_tuple):
            for tk in range(num_frames):
                for st in range(num_joints):
                    pos_list.append(st)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()

        pe = torch.zeros(num_tuple, num_frames * num_joints, channels)

        div_term = torch.exp(torch.arange(0, channels, 2).float() * -(math.log(10000.0) / channels)) 
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(num_tuple, num_frames, num_joints, channels).permute(0, 3, 1, 2).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # npctv
        x = self.pe[:, :, :x.size(2)]
        return x
# 模型a时空注意力模块的空间部分
class STAtt_Block_a_S(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels, use_tuple, num_tuple, num_subset=3, num_node=25,
                num_frame=32, kernel_size=1, stride=1, attentiondrop=0):
        super(STAtt_Block_a_S, self).__init__()
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset = num_subset
        self.use_tuple = use_tuple

        mask = [[1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0]]
        self.mask = torch.Tensor(mask).unsqueeze(1).unsqueeze(1).unsqueeze(0)
    
        pad = int((kernel_size - 1) / 2)
        
        self.atts_s1 = torch.zeros((1, num_subset, 1, num_node, num_node))
        self.atts_s2 = torch.zeros((1, num_subset, 1, num_node, num_node))
        self.atts_tp = torch.zeros((1, num_subset, num_tuple, num_tuple))

        self.pes = Pos_Embed(in_channels, num_node, num_frame)
        self.ff_nets = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels),
        )
        
        self.in_nets = nn.Conv2d(in_channels, 2 * num_subset * inter_channels, 1, bias=True)
        self.alphas = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
        # global nomalization
        self.attention0s = nn.Parameter(torch.ones(1, num_subset, num_node, num_node) / num_node,
                                        requires_grad=True)

        self.out_nets = nn.Sequential(
            nn.Conv2d(in_channels * num_subset, out_channels, 1, bias=True),
            nn.BatchNorm2d(out_channels),
            )


        if in_channels != out_channels or stride != 1:
            
            self.downs1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            self.downs2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
           
        else:   
            self.downs1 = lambda x: x
            self.downs2 = lambda x: x

        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)

    def forward(self, x):

        N, P, C, T, V= x.size()

        # 空间注意模块1
        attention = self.atts_s1
        y = self.pes(x)

        q, k = torch.chunk(self.in_nets(y).view(N, 2 * self.num_subset, P, self.inter_channels, T, V), 2,
                            dim=1)  # npctv -> np num_subset pc'tv
        attention = attention + self.tan(
            torch.einsum('nspctu,nspctv->nspuv', [q, k]) / (self.inter_channels * T)) * self.alphas

        attention = attention + self.attention0s.repeat(N, 1, 1, 1, 1)
        attention = self.drop(attention)
        y = torch.einsum('npctu,nspuv->nsctv', [x, attention]).contiguous() \
            .view(N, P, self.num_subset * self.in_channels, T, V)
        y = self.out_nets(y)  # npctv
        y = self.relu(self.downs1(x) + y)
        y = self.ff_nets(y)
        y = self.relu(self.downs2(x) + y)

        # 空间注意模块2
        attention = self.atts_s2
        q, k = torch.chunk(self.in_nets(y).view(N, 2 * self.num_subset, P, self.inter_channels, T, V), 2,
                            dim=1)  # nctv -> n num_subset pc'tv
        attention = attention + self.tan(
            torch.einsum('nspctu,nspctv->nspuv', [q, k]) / (self.inter_channels * T)) * self.alphas
        attention = attention + self.attention0s.repeat(N, 1, 1, 1, 1)
        attention = self.drop(attention)
        z = torch.einsum('npctu,nspuv->nspctv', [y, attention]).contiguous() \
            .view(N, P, self.num_subset * self.in_channels, T, V)
        z = self.out_nets(z)  # npctv
        z = self.relu(self.downs1(y) + z)
        z = self.ff_nets(z)
        z = self.relu(self.downs2(y) + z)

        # 元组注意模块
        attention = self.atts_tp
        q, k = torch.chunk(self.in_nets(y).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2,
                            dim=1)  # npctv -> n num_subset pc'tv
        attention = attention + self.tan(
            torch.einsum('nspctv,nsqctv->nspq', [q, k]) / (self.inter_channels * T)) * self.alphas
        attention = self.drop(attention)
        v = torch.einsum('nqctv,nspq->nspctv', [z, attention]).contiguous() \
            .view(N, P, self.num_subset * self.in_channels, T, V)
        v = self.out_nets(v)  # npctv
        v = self.relu(self.downs1(z) + v)
        v = self.ff_nets(v)
        v = self.relu(self.downs2(z) + v)

        # 将空节点的参数清零
        if self.use_tuple == 0 :
            v = v*self.mask

        return v

# 模型a时空注意力模块的时间部分
class STAtt_Block_a_T(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels, use_tuple, num_tuple, num_subset=3, num_node=25, 
                num_frame=32, kernel_size=1, stride=1,  attentiondrop=0):
        super(STAtt_Block_a_T, self).__init__()
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset = num_subset
        self.use_tuple = use_tuple

        mask = [[1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0]]
        self.mask = torch.Tensor(mask).unsqueeze(1).unsqueeze(1).unsqueeze(0)
       

        self.att_t1 = torch.zeros((1, num_subset, 1, num_frame, num_frame))
        self.att_t2 = torch.zeros((1, num_subset, 1, num_frame, num_frame))
        self.att_tp = torch.zeros((1, num_subset, num_tuple, num_tuple))

        self.pet = Pos_Embed(out_channels, num_tuple, num_node, num_frame)
        self.ff_nett = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1, bias=True),
            nn.BatchNorm2d(out_channels),
        )
        
        self.in_nett = nn.Conv2d(out_channels, 2 * num_subset * inter_channels, 1, bias=True)
        self.alphat = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)  
        self.out_nett = nn.Sequential(
            nn.Conv2d(out_channels * num_subset, out_channels, 1, bias=True),
            nn.BatchNorm2d(out_channels),
            )

        if in_channels != out_channels or stride != 1:
            
            self.downt1 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            self.downt2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            
            self.downs1 = lambda x: x
            self.downs2 = lambda x: x
            
        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)

    def forward(self, x):

        N, P, C, T, V = x.size()

        # 时间注意模块1
        attention = self.att_t1
        y = self.pet(x)
        q, k = torch.chunk(self.in_nett(z).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2,
                            dim=1)  # npctv -> n num_subset pc'tv
        attention = attention + self.tan(
            torch.einsum('nspctv,nspcqv->nsptq', [q, k]) / (self.inter_channels * V)) * self.alphat
        attention = self.drop(attention)
        y = torch.einsum('npctv,nsptq->nspcqv', [x, attention]).contiguous() \
            .view(N, P, self.num_subset * self.out_channels, T, V)
        y = self.out_nett(y)  # nctv
        y = self.relu(self.downt1(x) + y)
        y = self.ff_nett(z)
        y = self.relu(self.downt2(x) + y)

        # 空间注意模块2
        attention = self.att_t2
        q, k = torch.chunk(self.in_nett(z).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2,
                           dim=1)  # npctv -> n num_subset pc'tv
        attention = attention + self.tan(
            torch.einsum('nspctv,nspcqv->nsptq', [q, k]) / (self.inter_channels * V)) * self.alphat
        attention = self.drop(attention)
        z = torch.einsum('npctv,nsptq->nspcqv', [y, attention]).contiguous() \
            .view(N, P, self.num_subset * self.out_channels, T, V)
        z = self.out_nett(z)  # nctv
        z = self.relu(self.downt1(y) + z)
        z = self.ff_nett(z)
        z = self.relu(self.downt2(y) + z)

        # 元组注意模块
        attention = self.atts_tp
        q, k = torch.chunk(self.in_nets(y).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2,
                            dim=1)  # npctv -> n num_subset pc'tv
        attention = attention + self.tan(
            torch.einsum('nspctv,nsqctv->nspq', [q, k]) / (self.inter_channels * T)) * self.alphas
        attention = self.drop(attention)
        v = torch.einsum('nqctv,nspq->nspctv', [z, attention]).contiguous() \
            .view(N, P, self.num_subset * self.in_channels, T, V)
        v = self.out_nets(v)  # npctv
        v = self.relu(self.downs1(z) + v)
        v = self.ff_nets(v)
        v = self.relu(self.downs2(z) + v)

        # 将空节点的参数清零
        if self.use_tuple == 0 :
            v = v*self.mask

        return z

class TBSTANet_a(nn.Module):
    def __init__(self,  use_pooling=True, use_tuple=0, num_tuple=36, num_class=60, num_point=25, num_frame=32, num_subset=3, dropout=0.,
                config_T1=None, config_T2=None, config_S=None, num_person=2, num_channel=3, attentiondrop=0, dropout2d=0, ):
        super(TBSTANet_a, self).__init__()

        self.out_channels = config_S[-1][1]
        in_channels = config_T2[0][0]
        self.use_tuple = use_tuple
        self.use_pooling = use_pooling
        self.input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )

        param = {
            'use_tuple': use_tuple,
            'num_node': num_point,
            'num_subset': num_subset,
            'attentiondrop': attentiondrop
        }
        self.graph_layers_S= nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, stride) in enumerate(config_S):
            self.graph_layers_S.append(
                STAtt_Block_a_S(in_channels, out_channels, inter_channels, stride=stride, num_frame=num_frame,
                                 num_tuple= num_tuple, **param))

        self.graph_layers_T1= nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, stride) in enumerate(config_T1):
            self.graph_layers_S.append(
                STAtt_Block_a_S(in_channels, out_channels, inter_channels, stride=stride, num_frame=num_frame,
                                 num_tuple= num_tuple, **param))

        self.graph_layers_T2= nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, stride) in enumerate(config_T2):
            self.graph_layers_S.append(
                STAtt_Block_a_S(in_channels, out_channels, inter_channels, stride=stride, num_frame=num_frame,
                                 num_tuple= num_tuple, **param))

        self.graph_layers_T3= nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, stride) in enumerate(config_T2):
            self.graph_layers_S.append(

                STAtt_Block_a_S(in_channels, out_channels, inter_channels, stride=stride, num_frame=int(num_frame/2),
                                 num_tuple= num_tuple, **param))

        self.timepooling =  nn.Conv2d(config_T1[-1][0], config_T1[-1][0], (2, 1),  (2, 1))
        self.fc = nn.Linear(self.out_channels, num_class)

        self.drop_out = nn.Dropout(dropout)
        self.drop_out2d = nn.Dropout2d(dropout2d)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)

    def forward(self, x):
        """

        :param x: N M C T V
        :return: classes scores
        """
        N, C, T, V, M = x.shape

        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        x = tuple_diviced(self.use_tuple, x)
        N, P, C, T, V = x.shape

        x = self.input_map(x)

        for i, m in enumerate(self.graph_layers_S):
            x = m(x)
        for i, m in enumerate(self.graph_layers_T1):
            x = m(x) 

         # 网络分为两路，一路高速（池化），一路低速    
        if self.use_pooling :
            y = self.timepooling(x)
        else : y = x

        for i, m in enumerate(self.graph_layers_T2): 
            y = m(y)
        for i, m in enumerate(self.graph_layers_T3):
            x = m(x)   

        x = un_tuple(x) 
        y = un_tuple(y)
        # 高速流上采样， 与低速流融合
        if self.use_pooling :
            y = torch.nn.functional.interpolate(y, y.shape, (1, 1, 2, 1))
        x = torch.cat((x, y), dim=1)

        # NM, C, T, V
        x = x.view(N, M, P, 2*self.out_channels, -1)
        x = x.permute(0, 1, 3, 2).contiguous().view(N, -1, 2*self.out_channels, 1)  # whole channels of one spatial
        x = self.drop_out2d(x)
        x = x.mean(3).mean(1)

        x = self.drop_out(x)  # whole spatial of one channel

        return self.fc(x)


if __name__ == '__main__':
    config_S = [[64, 64, 16, 1], [64, 64, 16, 1],
              [64, 128, 32, 1], [128, 128, 32, 1] ]
    config_T1 = [[128, 256, 64, 1], [256, 256, 64, 1] ]
    config_T2 = [[256, 256, 64, 1], [256, 256, 64, 1] ]
    net = TBSTANet_a(config_S=config_S, config_T1=config_T1, config_T2=config_T2)  # .cuda()
    ske = torch.rand([2, 3, 300, 25, 2])  # .cuda()
    print(net(ske).shape)
