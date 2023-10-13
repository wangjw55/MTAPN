import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

class ARNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(ARNNAgent, self).__init__()
        self.args = args
        self.pre_bs = None

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # 输出改为1维，每次输出一个action对应的Q值
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
        # self.fc2 = nn.Sequential(
        #     nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim//2),
        #     nn.ReLU(),
        #     nn.Linear(args.rnn_hidden_dim//2, 1)
        #     )


        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)
        
        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, hidden_index=None):
        # (batch_size, agent_num, action_num, embedding_size)
        b, a, actions_num, e = inputs.size()

        # inputs = inputs.view(-1, actions_num, e)
        # h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        # hh_ = []
        # for k in range(actions_num):
        #     inputs_ = inputs[:,k,:].squeeze()
        #     x = F.relu(self.fc1(inputs_), inplace=True)
        #     hh = self.rnn(x, h_in)
        #     hh_.append(hh)
        # hh = th.stack(hh_, dim=2).reshape(-1, self.args.rnn_hidden_dim)

        inputs = inputs.reshape(-1, e)
        hidden_state = hidden_state.unsqueeze(2).expand(-1, -1, actions_num, -1)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        x = F.relu(self.fc1(inputs), inplace=True)
        hh = self.rnn(x, h_in)
        hh = hh.reshape(-1, self.args.rnn_hidden_dim)

        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(hh))
        else:
            q = self.fc2(hh)

        if hidden_index is not None:
            hh = hh.reshape(b, a, actions_num, -1)
            hh = hh[th.arange(b)[:,None,None],th.arange(a)[None,:,None],hidden_index].squeeze(2)
            return q.view(b, a, -1), hh
        else:
            return q.view(b, a, -1), hh.view(b, a, actions_num, -1)