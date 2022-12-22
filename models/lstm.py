import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from models.STAUCell import STAUCell

class lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.output = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                #nn.BatchNorm1d(output_size),
                nn.Tanh())
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(self.batch_size, self.hidden_size).cuda())))
        return hidden

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]

        return self.output(h_in)




class stau(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size, tau, theta):
        super(stau, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tau = tau
        self.theta = theta
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([STAUCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.output = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                #nn.BatchNorm1d(output_size),
                nn.Tanh())
        self.t_att, self.s_att = self.init_hidden()

    def init_hidden(self):
        t_att = []
        s_att = []
        # t_spatial = []
        # s_spatial = []
        for i in range(self.n_layers):
            t_att.append([])
            s_att.append([])
            for _ in range(self.tau):
                t_att[i].append(torch.zeros(self.batch_size, self.hidden_size).cuda())
                s_att[i].append(torch.zeros(self.batch_size, self.hidden_size).cuda())
        # for _ in range(self.theta):
        #     t_spatial.append(torch.zeros(self.batch_size, self.hidden_size).cuda())
        #     s_spatial.append(torch.zeros(self.batch_size, self.hidden_size).cuda())
        return t_att, s_att

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        s_spatial = []
        t_spatial = []
        for _ in range(self.theta):
            s_spatial.append(torch.zeros(self.batch_size, self.hidden_size).cuda())
            t_spatial.append(torch.zeros(self.batch_size, self.hidden_size).cuda())
        S_t = embedded
        # self.s_spatial[0] = torch.cat([self.s_spatial[0][1:,:],S_t.unsqueeze(dim=0)],dim=0)
        
        # self.s_spatial[0] = self.s_spatial[0][-self.theta:]

        for i in range(self.n_layers):
            # print(i)
            # self.s_att[i] = torch.cat([self.s_att[i][1:,:],S_t.unsqueeze(dim=0)],dim=0)
            s_spatial.append(S_t)
            t_spatial.append(self.t_att[i][-1])
            self.s_att[i].append(S_t)
            # self.s_att[i] = self.s_att[i][-self.tau:]

            S_t, T_t = self.lstm[i](self.t_att[i][-1], S_t, self.t_att[i][-self.tau:], self.s_att[i][-self.tau:], t_spatial[-self.theta:], s_spatial[-self.theta:])
            # updating spatial states
            # if i+1 < self.n_layers:
            #     # self.s_spatial[i+1] = torch.cat([self.s_spatial[i][1:,], S_t.unsqueeze(dim=0)],dim=0)
            #     self.s_spatial[i+1].append(S_t)
            #     self.s_spatial[i+1] = self.s_spatial[i+1][-self.theta:]
            # if i==0:
            #     # self.t_spatial[i][-1,:] = T_t
            #     self.t_spatial[i][-1] = T_t
            # else:
            #     # self.t_spatial[i] = torch.cat([self.t_spatial[i-1][1:,], T_t.unsqueeze(dim=0)],dim=0)
            #     self.t_spatial[i].append(T_t)
            #     self.t_spatial[i] = self.t_spatial[i][-self.theta:]

            # self.t_att[i] = torch.cat([self.t_att[i][1:,:], T_t.unsqueeze(dim=0)],dim=0)
            self.t_att[i].append(T_t)
            # self.t_att[i] = self.t_att[i][-self.tau:] 
        return self.output(S_t)

class gaussian_lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(gaussian_lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(self.batch_size, self.hidden_size).cuda())))
        return hidden

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
            

# model = stau(128,32,64,4,4,5,4).cuda()
# for t in range(5):
#     print(t)
#     input = torch.rand(size=(4,128)).cuda()
#     out = model(input)
