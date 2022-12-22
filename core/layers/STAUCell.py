import typing_extensions
import torch
import torch.nn as nn
import math



class STAUCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, tau, cell_mode):
        super(STAUCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = (filter_size[0] // 2, filter_size[1] // 2)
        self.cell_mode = cell_mode
        self.d = num_hidden * height * width
        self.tau = tau
        self.states = ['residual', 'normal']
        if not self.cell_mode in self.states:
            raise AssertionError
        self.conv_t = nn.Sequential(
            nn.Conv2d(in_channel, 3 * num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([3 * num_hidden, height, width])
        )
        self.conv_t_next = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_s = nn.Sequential(
            nn.Conv2d(num_hidden, 3 * num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([3 * num_hidden, height, width])
        )
        self.conv_s_next = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.softmax = nn.Softmax(dim=0)

    def forward(self, T_t, S_t, t_att, s_att, t_spatial, s_spatial):
        s_next = self.conv_s_next(S_t)
        t_next = self.conv_t_next(T_t)
        weights_list = []
        weights_list_t = []
        # gates = {}
        # attention = {}
        # features = {}
        # features['t_att'] = t_att[:, 0, :].detach().cpu().numpy()
        # features['s_att'] = s_att[:, 0, :].detach().cpu().numpy()
        # features['t_spatial'] = t_spatial[:, 0, :].detach().cpu().numpy()
        # features['s_spatial'] = s_spatial[:, 0, :].detach().cpu().numpy()
        for i in range(self.tau):
            weights_list.append(
                (s_att[i] * s_next).sum(dim=(1, 2, 3)) / math.sqrt(self.d))
            # weights_list_t.append((t_att[i] * t_next).sum(dim=(1, 2, 3)) / math.sqrt(self.d))
        for j in range(t_spatial.shape[0]):
            weights_list_t.append(
                (t_spatial[j]*t_next).sum(dim=(1, 2, 3)) / math.sqrt(self.d))
        weights_list = torch.stack(weights_list, dim=0)
        weights_list = torch.reshape(
            weights_list, (*weights_list.shape, 1, 1, 1))
        weights_list = self.softmax(weights_list)
        weights_list_t = torch.stack(weights_list_t, dim=0)
        weights_list_t = torch.reshape(
            weights_list_t, (*weights_list_t.shape, 1, 1, 1))
        weights_list_t = self.softmax(weights_list_t)

        T_trend = t_att * weights_list
        T_trend = T_trend.sum(dim=0)

        S_trend = s_spatial * weights_list_t
        S_trend = S_trend.sum(dim=0)

        t_att_gate = torch.sigmoid(t_next)
        T_fusion = T_t * t_att_gate + (1 - t_att_gate) * T_trend

        s_att_gate = torch.sigmoid(s_next)
        S_fusion = S_t * s_att_gate + (1 - s_att_gate) * S_trend

        T_concat = self.conv_t(T_fusion)
        S_concat = self.conv_s(S_fusion)

        # S_concat = self.conv_s(S_t)
        t_g, t_t, t_s = torch.split(T_concat, self.num_hidden, dim=1)
        s_g, s_t, s_s = torch.split(S_concat, self.num_hidden, dim=1)
        T_gate = torch.sigmoid(t_g)
        S_gate = torch.sigmoid(s_g)
        T_new = T_gate * t_t + (1 - T_gate) * s_t
        S_new = S_gate * s_s + (1 - S_gate) * t_s
        if self.cell_mode == 'residual':
            S_new = S_new + S_t
        # print(t_att_gate.shape)
        # gates['t_att_gate'] = t_att_gate.detach().cpu().numpy()
        
        # gates['s_att_gate'] = s_att_gate.detach().cpu().numpy()
        # gates['T_gate'] = T_gate.detach().cpu().numpy()
        # gates['S_gate'] = S_gate.detach().cpu().numpy()
        # attention['weights_list_s'] = weights_list[:,
        #                                            0, :].detach().cpu().numpy()
        # attention['weights_list_t'] = weights_list_t[:,
                                                    #  0, :].detach().cpu().numpy()
        # visual_results = {}
        # visual_results['gates'] = gates
        # visual_results['features'] = features
        # visual_results['attention'] = attention
        return T_new, S_new
