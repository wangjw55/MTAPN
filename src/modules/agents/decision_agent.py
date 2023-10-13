import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

class DecisionAgent(nn.Module):
    def __init__(self, args):
        super(DecisionAgent, self).__init__()
        self.args = args

        # self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
        self.fc2 = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim//2),
            nn.ReLU(),
            nn.Linear(args.rnn_hidden_dim//2, 1)
            )

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)
        
        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc2, gain=args.gain)
 
    def forward(self, hh, b, a, actions_num, hidden_index=None):

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