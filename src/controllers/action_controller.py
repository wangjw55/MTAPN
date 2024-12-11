from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .action_basic_controller import ABasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np

# This multi-agent controller shares parameters between agents
class AMAC(ABasicMAC):
    def __init__(self, scheme, groups, args):
        super(AMAC, self).__init__(scheme, groups, args)
        
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals = self.forward(ep_batch, t_ep, hidden_index=None, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        
        if len(bs) != self.hidden_states.size(0):
            # hidden_states -> (batch_size, agent_num, action_num, rnn_dim)
            first_dim_size = self.hidden_states.size(0)
            second_dim_size = self.hidden_states.size(1)
            fourth_dim_size = self.hidden_states.size(3)
            new_hidden_states = th.zeros([first_dim_size,second_dim_size,fourth_dim_size], device=self.args.device)
            hidden_states_alive = self.hidden_states[bs]
            first_dim_size_alive = hidden_states_alive.size(0)

            hidden_states_keep = hidden_states_alive[th.arange(first_dim_size_alive)[:,None,None],th.arange(second_dim_size)[None,:,None],chosen_actions.unsqueeze(-1)]
            hidden_states_keep = hidden_states_keep.squeeze(2)

            new_hidden_states[bs] = hidden_states_keep
            self.hidden_states = new_hidden_states
        else:
            # hidden_states -> (batch_size, agent_num, action_num, rnn_dim)
            first_dim_size = self.hidden_states.size(0)
            second_dim_size = self.hidden_states.size(1)
            hidden_states_keep = self.hidden_states[th.arange(first_dim_size)[:,None,None],th.arange(second_dim_size)[None,:,None],chosen_actions.unsqueeze(-1)]
            self.hidden_states = hidden_states_keep.squeeze(2)
        return chosen_actions

    def forward(self, ep_batch, t, hidden_index=None, test_mode=False):
        if test_mode:
            self.agent.eval()

        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, hidden_index)

        return agent_outs