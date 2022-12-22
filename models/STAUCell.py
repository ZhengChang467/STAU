import torch
import torch.nn as nn
import math

class STAUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(STAUCell, self).__init__()
        self.d = hidden_size
        # self.tau = tau
        # self.theta = theta
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fl_t = nn.Linear(input_size, 3 * hidden_size)
        self.fl_t_next = nn.Linear(input_size, hidden_size)
        self.fl_s = nn.Linear(input_size, 3 * hidden_size)
        self.fl_s_next = nn.Linear(input_size, hidden_size)
        self.softmax = nn.Softmax(dim=0)
        # self.t_att = torch.zeros(self.)
        # self.output_size = output_size
        # self.hidden_size = hidden_size
        # self.batch_size = batch_size
        # self.n_layers = n_layers
        # self.embed = nn.Linear(input_size, hidden_size)
        # self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        # self.output = nn.Sequential(
        #         nn.Linear(hidden_size, output_size),
        #         #nn.BatchNorm1d(output_size),
        #         nn.Tanh())
        # self.hidden = self.init_hidden()

    # def init_hidden(self):
    #     hidden = []
    #     for i in range(self.n_layers):
    #         hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()),
    #                        Variable(torch.zeros(self.batch_size, self.hidden_size).cuda())))
    #     return hidden

    def forward(self, T_t, S_t, t_att, s_att, t_spatial, s_spatial):
        t_att = torch.stack(t_att,dim=0)
        s_att = torch.stack(s_att,dim=0)
        t_spatial = torch.stack(t_spatial,dim=0)
        s_spatial = torch.stack(s_spatial,dim=0)
        # print(t_att.mean(dim=[1,2]),s_att.mean(dim=[1,2]),
        # t_spatial.mean(dim=[1,2]),s_spatial.mean(dim=[1,2]))
        s_next = self.fl_s_next(S_t)
        t_next = self.fl_t_next(T_t)
        weights_list = []
        weights_list_t = []
        # gates = {}
        # attention = {}
        # features = {}
        # features['t_att'] = t_att[:, 0, :].detach().cpu().numpy()
        # features['s_att'] = s_att[:, 0, :].detach().cpu().numpy()
        # features['t_spatial'] = t_spatial[:, 0, :].detach().cpu().numpy()
        # features['s_spatial'] = s_spatial[:, 0, :].detach().cpu().numpy()
        for i in range(t_att.shape[0]):
            weights_list.append(
                (s_att[i] * s_next).sum(dim=(1)) / math.sqrt(self.d))
            # weights_list_t.append((t_att[i] * t_next).sum(dim=(1, 2, 3)) / math.sqrt(self.d))
        for j in range(s_spatial.shape[0]):
            weights_list_t.append(
                (t_spatial[j]*t_next).sum(dim=(1)) / math.sqrt(self.d))

        weights_list = torch.stack(weights_list, dim=0)
        weights_list = torch.reshape(
            weights_list, (*weights_list.shape, 1))
        weights_list = self.softmax(weights_list)
        weights_list_t = torch.stack(weights_list_t, dim=0)
        weights_list_t = torch.reshape(
            weights_list_t, (*weights_list_t.shape, 1))
        weights_list_t = self.softmax(weights_list_t)

        T_trend = t_att * weights_list
        T_trend = T_trend.sum(dim=0)

        S_trend = s_spatial * weights_list_t
        S_trend = S_trend.sum(dim=0)

        t_att_gate = torch.sigmoid(t_next)
        T_fusion = T_t * t_att_gate + (1 - t_att_gate) * T_trend

        s_att_gate = torch.sigmoid(s_next)
        S_fusion = S_t * s_att_gate + (1 - s_att_gate) * S_trend

        T_concat = self.fl_t(T_fusion)
        S_concat = self.fl_s(S_fusion)

        # S_concat = self.conv_s(S_t)
        t_g, t_t, t_s = torch.split(T_concat, self.hidden_size, dim=1)
        s_g, s_t, s_s = torch.split(S_concat, self.hidden_size, dim=1)
        T_gate = torch.sigmoid(t_g)
        S_gate = torch.sigmoid(s_g)
        T_new = T_gate * t_t + (1 - T_gate) * s_t
        S_new = S_gate * s_s + (1 - S_gate) * t_s
        # if self.cell_mode == 'residual':
        #     S_new = S_new + S_t
        # gates['t_att_gate'] = t_att_gate[0].detach().cpu().numpy()
        # gates['s_att_gate'] = s_att_gate[0].detach().cpu().numpy()
        # gates['T_gate'] = T_gate[0].detach().cpu().numpy()
        # gates['S_gate'] = S_gate[0].detach().cpu().numpy()
        # attention['weights_list_s'] = weights_list[:,
        #                                            0, :].detach().cpu().numpy()
        # attention['weights_list_t'] = weights_list_t[:,
        #                                              0, :].detach().cpu().numpy()
        # visual_results = {}
        # visual_results['gates'] = gates
        # visual_results['features'] = features
        # visual_results['attention'] = attention
        return T_new, S_new
