from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .action_basic_controller import ABasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np

# This multi-agent controller shares parameters between agents
class AMAC(ABasicMAC):
    def __init__(self, scheme_dict, groups_dict, args_dict, total_map_num):
        super(AMAC, self).__init__(scheme_dict, groups_dict, args_dict, total_map_num)
        
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, map_id=None):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals = self.forward(ep_batch, t_ep, test_mode=test_mode, map_id=map_id)
        chosen_actions = self.action_selector_dict[map_id].select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)

        if len(bs) != self.hidden_states_dict[map_id].size(0):
            # hidden_states -> (batch_size, agent_num, action_num, rnn_dim)
            first_dim_size = self.hidden_states_dict[map_id].size(0)
            second_dim_size = self.hidden_states_dict[map_id].size(1)
            fourth_dim_size = self.hidden_states_dict[map_id].size(3)
            new_hidden_states = th.zeros([first_dim_size,second_dim_size,fourth_dim_size], device=self.args_dict[0].device)
            hidden_states_alive = self.hidden_states_dict[map_id][bs]
            first_dim_size_alive = hidden_states_alive.size(0)

            hidden_states_keep = hidden_states_alive[th.arange(first_dim_size_alive)[:,None,None],th.arange(second_dim_size)[None,:,None],chosen_actions.unsqueeze(-1)]
            hidden_states_keep = hidden_states_keep.squeeze(2)

            new_hidden_states[bs] = hidden_states_keep
            self.hidden_states_dict[map_id] = new_hidden_states
        else:
            # hidden_states -> (batch_size, agent_num, action_num, rnn_dim)
            first_dim_size = self.hidden_states_dict[map_id].size(0)
            second_dim_size = self.hidden_states_dict[map_id].size(1)
            hidden_states_keep = self.hidden_states_dict[map_id][th.arange(first_dim_size)[:,None,None],th.arange(second_dim_size)[None,:,None],chosen_actions.unsqueeze(-1)]
            self.hidden_states_dict[map_id] = hidden_states_keep.squeeze(2)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False, map_id=None, hidden_index=None):
        if test_mode:
            self.perceive_agent_dict[map_id].eval()
            self.decision_agent.eval()

        agent_inputs = self._build_inputs(ep_batch, t, map_id)
        avail_actions = ep_batch["avail_actions"][:, t]
        hh, b, a, actions_num = self.perceive_agent_dict[map_id](agent_inputs, self.hidden_states_dict[map_id])
        agent_outs, self.hidden_states_dict[map_id] = self.decision_agent(hh, b, a, actions_num, hidden_index)

        return agent_outs