
#dstanet 27_2
import torch
import torch.nn as nn
import math
import numpy as np
from model.position_encoding import knowledge_embedded_PE


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)
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


class STAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels, num_subset=3, num_node=25, num_frame=32,
                 kernel_size=1, stride=1, glo_reg_s=True, att_s=True, glo_reg_t=True, att_t=True,
                 use_temporal_att=True, use_spatial_att=True, attentiondrop=0, use_pes=True, use_pet=True,use_kepe = False):
        super(STAttentionBlock, self).__init__()
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset = num_subset
        self.glo_reg_s = glo_reg_s
        self.att_s = att_s
        self.glo_reg_t = glo_reg_t
        self.att_t = att_t
        self.use_pes = use_pes
        self.use_pet = use_pet
        self.node_num = num_node
        pad = int((kernel_size - 1) / 2)
        self.use_spatial_att = use_spatial_att
        self.use_kepe = use_kepe
        if use_spatial_att:

            atts = torch.zeros((1, num_subset, num_node, num_node))
            self.register_buffer('atts', atts)
            if use_kepe:
                A = torch.tensor(knowledge_embedded_PE(), dtype=torch.float32, requires_grad=False)
                self.AM = nn.Parameter(torch.ones(60, 60), requires_grad=True)

                self.register_buffer('A', A)
                self.A = A
            self.pes = PositionalEncoding(in_channels, num_node, num_frame, 'spatial')
            self.ff_nets = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1, 1, padding=0, bias=True),
                nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(0.1),

            )
            if att_s:
                self.in_nets = nn.Conv2d(in_channels, 2 * num_subset * inter_channels, 1, bias=True)
                self.alphas = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
            if glo_reg_s:
                self.attention0s = nn.Parameter(torch.ones(1, num_subset, num_node, num_node) / num_node,
                                                requires_grad=True)

            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels * num_subset, in_channels, 1, bias=True),
                nn.BatchNorm2d(in_channels),
            )

        self.downs1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=True),
            nn.BatchNorm2d(in_channels),
        )
        self.downs2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=True),
            nn.BatchNorm2d(in_channels),
        )
        self.tcn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (7, 1), padding=(3, 0), bias=True, stride=(1, 1)),
            nn.BatchNorm2d(in_channels),
        )

        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)

    def forward(self, x):

        N, C, T, V = x.size()

        if self.use_spatial_att:
            attention = self.atts

            if self.use_pes:
                y = self.pes(x)
            else:
                y = x
            if self.att_s:
                q, k = torch.chunk(self.in_nets(y).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2,
                                   dim=1)  # nctv -> n num_subset c'tv
                attention = attention + self.tan(
                    torch.einsum('nsctu,nsctv->nsuv', [q, k]) / (self.inter_channels * T)) * self.alphas
            if self.glo_reg_s:
                attention = attention + self.attention0s.repeat(N, 1, 1, 1)
            attention = self.drop(attention)
            y = torch.einsum('nctu,nsuv->nsctv', [x, attention]).contiguous() \
                .view(N, self.num_subset * self.in_channels, T, V)
            y = self.out_nets(y)  # nctv

            y = self.relu(self.downs1(x) + y)

            y = self.ff_nets(y)

            y = self.relu(self.downs2(x)  + y)


        return y

class tcn_block(nn.Module):
    def __init__(self, in_channels,out_channels,stride):
        pad = 2
        super(tcn_block, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, (7, 1), padding=(3, 0), bias=True, stride=(stride, 1)),
            nn.BatchNorm2d(out_channels//4),
        )
        self.downt2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, (5, 1), (stride, 1), padding=(pad, 0), bias=True),
            nn.BatchNorm2d(out_channels//4),
        )
        self.downt3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, (3, 1), (stride, 1), padding=(1, 0), bias=True),
            nn.BatchNorm2d(out_channels // 2),
        )
        self.downt1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//1, (1, 1), (stride, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(out_channels//1),
        )
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x1 = self.tcn(x)
        x2 = self.downt2(x)
        x3 = self.downt1(x)
        x4 = self.downt3(x)
        x = torch.cat([x1,x2,x4], dim=1)
        x = x+x3
        x = self.relu(x)
        return x
class DSTANet(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_frame=32, num_subset=3, dropout=0., config=None, num_person=2,
                 num_channel=3, glo_reg_s=True, att_s=True, glo_reg_t=False, att_t=True,
                 use_temporal_att=True, use_spatial_att=True, attentiondrop=0, dropout2d=0, use_pet=True, use_pes=True):
        super(DSTANet, self).__init__()

        self.out_channels = config[-1][1]
        in_channels = config[0][0]

        self.input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )
        #第二scale到sclae3中间映射
        self.input_map11 = nn.Sequential(
            nn.Conv2d(256*5*4, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
        )
        self.num_frame = num_frame
        #时间段
        clip = [16,4]
        self.clip = clip
        #空间部位
        p_number = [6,2]
        self.p_number = p_number
        param = {
            'num_node': (num_point+5),
            'num_subset': num_subset,
            'glo_reg_s': glo_reg_s,
            'att_s': att_s,
            'glo_reg_t': glo_reg_t,
            'att_t': att_t,
            'use_spatial_att': use_spatial_att,
            'use_temporal_att': use_temporal_att,
            'use_pet': use_pet,
            'use_pes': use_pes,
            'attentiondrop': attentiondrop
        }
        param1 = {
            'num_node': 30*num_frame//2//self.clip[0]//self.p_number[0],
            'num_subset': num_subset,
            'glo_reg_s': glo_reg_s,
            'att_s': att_s,
            'glo_reg_t': glo_reg_t,
            'att_t': att_t,
            'use_spatial_att': use_spatial_att,
            'use_temporal_att': use_temporal_att,
            'use_pet': use_pet,
            'use_pes': use_pes,
            'attentiondrop': attentiondrop
        }
        param2 = {
            'num_node': self.clip[0]*self.p_number[0],
            'num_subset': num_subset,
            'glo_reg_s': glo_reg_s,
            'att_s': att_s,
            'glo_reg_t': glo_reg_t,
            'att_t': att_t,
            'use_spatial_att': use_spatial_att,
            'use_temporal_att': use_temporal_att,
            'use_pet': use_pet,
            'use_pes': use_pes,
            'attentiondrop': attentiondrop
        }
        param3 = {
            'num_node': self.clip[0]//2*self.p_number[0]//self.clip[1]//self.p_number[1],
            'num_subset': num_subset,
            'glo_reg_s': glo_reg_s,
            'att_s': att_s,
            'glo_reg_t': glo_reg_t,
            'att_t': att_t,
            'use_spatial_att': use_spatial_att,
            'use_temporal_att': use_temporal_att,
            'use_pet': use_pet,
            'use_pes': use_pes,
            'attentiondrop': attentiondrop
        }
        param4 = {
            'num_node': self.clip[1]*self.p_number[1],
            'num_subset': num_subset,
            'glo_reg_s': glo_reg_s,
            'att_s': att_s,
            'glo_reg_t': glo_reg_t,
            'att_t': att_t,
            'use_spatial_att': use_spatial_att,
            'use_temporal_att': use_temporal_att,
            'use_pet': use_pet,
            'use_pes': use_pes,
            'attentiondrop': attentiondrop
        }
        #补零分部位

        self.graph_layers = nn.ModuleList()
        self.graph_layers1 = nn.ModuleList()
        self.tcns = nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, stride) in enumerate(config):
            if index < 3:
                #scale 1 only joint each frame
                self.graph_layers.append(
                    STAttentionBlock(in_channels, out_channels, inter_channels, stride=stride, num_frame=num_frame,
                                     **param,use_kepe=True))
            if index>=3 and index <5:
                # scale 2 part-motion intra-inter
                self.graph_layers.append(
                    STAttentionBlock(in_channels, out_channels, inter_channels, stride=stride, num_frame=self.clip[0]*self.p_number[0],
                                     **param1))
                self.graph_layers1.append(
                    STAttentionBlock(in_channels*30*num_frame//2//self.clip[0]//self.p_number[0], in_channels*30*num_frame//2//self.clip[0]//self.p_number[0], in_channels*30*num_frame//2//self.clip[0]//self.p_number[0]//4, stride=stride, num_frame=1,
                                     **param2))

            if index >=5:
                # scale 3 part-motion intra-inter
                self.graph_layers.append(
                    STAttentionBlock(in_channels, out_channels, inter_channels, stride=stride,
                                     num_frame=self.clip[1] * self.p_number[1],
                                     **param3))
                self.graph_layers1.append(
                    STAttentionBlock(in_channels * self.clip[0]//2*self.p_number[0]//self.clip[1]//self.p_number[1], in_channels*self.clip[0]//2*self.p_number[0]//self.clip[0]//self.p_number[0], in_channels*self.clip[0]//2*self.p_number[0]//self.clip[0]//self.p_number[0]//4,
                                     stride=stride, num_frame=1,
                                     **param4))
            if index>=0 and index<=7:
                # tcn
                if index == 2 or index == 4:
                    self.tcns.append(tcn_block(in_channels,out_channels,2))
                else:
                    self.tcns.append(tcn_block(in_channels, out_channels, 1))


        self.fc = nn.Linear(640, num_class)
        self.BN1=nn.BatchNorm2d(128)
        self.BN2 = nn.BatchNorm2d(256)
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
        x = self.input_map(x)
        #print(x.shape)
        for i, m in enumerate(self.graph_layers):

            if i>=1 and i<=3:
                x = self.tcns[i-1](x)
            if i==3:
                x1 = x
                N1, C1, T1, V1 = x.shape

                x = x.view(N1, C1, self.clip[0], T1 // self.clip[0], self.p_number[0], V1 // self.p_number[0]).permute(
                    0, 1, 2, 4,
                    3,
                    5).contiguous().view(
                    N1, C1, self.clip[0] * self.p_number[0], -1)
            if i==5:
                x2 = x
                N1, C1, T1, V1 = x.shape
                x = x.view(N1,C1,self.clip[1],self.clip[0]//2//self.clip[1],self.p_number[1],self.p_number[0]//self.p_number[1],V1).permute(0,1,6,2,4,3,5).contiguous()
                x = x.view(N1,C1*V1,self.clip[1]*self.p_number[1],-1)
                x= self.input_map11(x)

            if i<3:

                x = m(x)
            if i>=3 and i<5:

                N1, C1, T1, V1 = x.shape
                y = x.permute(0, 1, 3, 2).contiguous().view(N1, -1, 1, self.clip[0] * self.p_number[0])
                x = m(x)

                y = self.graph_layers1[i-3](y)
                y = y.view(N1, C1, V1, T1 ).permute(0,1,3,2).contiguous()
                x = x+y
                #变为原来形式，用tcn聚合时间特征
                x = x.view(N1, C1, self.clip[0], self.p_number[0], -1, 30 // self.p_number[0]).permute(0,1,2,4,3,5).contiguous().view(
                    N1, C1, -1, 30)

                x = self.tcns[i](x)
                if i==4:
                    x = x.view(N1, C1*2, self.clip[0]//2, -1, self.p_number[0], 30 // self.p_number[0]).permute(0, 1, 2, 4,
                                                                                                           3,
                                                                                                           5).contiguous().view(
                        N1, C1*2, T1//2, V1)
                else:
                    x = x.view(N1, C1, self.clip[0], -1, self.p_number[0], 30 // self.p_number[0]).permute(0,1,2,4,3,5).contiguous().view(
                    N1, C1, T1, V1)

            if i>=5:

                N1, C1, T1, V1 = x.shape

                y = x.permute(0, 1, 3, 2).contiguous().view(N1,  -1, 1, self.clip[1] * self.p_number[1])
                #print(x.shape)
                x = m(x)
                y = self.graph_layers1[i - 3](y)
                y = y.view(N1, C1, V1, T1).permute(0, 1, 3, 2).contiguous()
                x = x + y
                x = x.view(N1, C1, self.clip[1], self.p_number[1], -1, self.p_number[0]// self.p_number[1]).permute(0, 1, 2, 4, 3,
                                                                                                       5).contiguous().view(
                    N1, C1, -1, self.p_number[0])
                x = self.tcns[i](x)
                x = x.view(N1, C1, self.clip[1], -1, self.p_number[1], self.p_number[0]// self.p_number[1]).permute(0, 1, 2, 4, 3,
                                                                                                       5).contiguous().view(
                    N1, C1, T1, V1)
        # NM, C, T, V
        x = x.view(N, M, self.out_channels, -1)

        x1 =self.BN1(x1)
        x2 = self.BN2(x2)
        x1 = x1.view(N, M, 128, -1)
        x2 = x2.view(N, M, 256, -1)
        x1 = x1.mean(-1).mean(1)
        x2 = x2.mean(-1).mean(1)
        x = x.mean(-1).mean(1)
        x = torch.cat([x,x1,x2],dim=1)
        x = self.drop_out(x)  # whole spatial of one channel

        return self.fc(x)


if __name__ == '__main__':
    config = [[64, 64, 16, 1], [64, 64, 16, 1],
              [64, 128, 32, 2], [128, 128, 32, 1],
              [128, 256, 64, 2], [256, 256, 64, 1],
              [256, 256, 64, 1], [256, 256, 64, 1],
              ]
    net = DSTANet(config=config)  # .cuda()
    ske = torch.rand([2, 3, 32, 25, 2])  # .cuda()
    print(net(ske).shape)
