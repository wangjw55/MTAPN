from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from torch.nn.parallel import DataParallel
from d2l import torch as d2l

# This multi-agent controller shares parameters between agents
class ABasicMAC:
    def __init__(self, scheme_dict, groups_dict, args_dict, total_map_num):
        self.n_agents_dict = {}
        self.n_actions_dict = {}
        self.args_dict = args_dict
        self.total_map_num = total_map_num
        input_shape_dict = {}
        self.agent_output_type_dict = {}
        self.action_selector_dict = {}
        self.save_probs_dict = {}
        self.hidden_states_dict = {}

        for id in range(total_map_num):
            self.n_agents_dict[id] = self.args_dict[id].n_agents
            self.n_actions_dict[id] = self.args_dict[id].n_actions
            input_shape_dict[id] = self._get_input_shape(scheme_dict[id], id)
            self.agent_output_type_dict[id] = self.args_dict[id].agent_output_type
            self.action_selector_dict[id] = action_REGISTRY[self.args_dict[id].action_selector](self.args_dict[id])
            self.save_probs_dict[id] = getattr(self.args_dict[id], 'save_probs', False)
            self.hidden_states_dict[id] = None
        
        self._build_agents(input_shape_dict)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        if test_mode:
            self.agent.eval()
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                agent_outs = agent_outs.reshape(ep_batch.batch_size * self.n_agents, -1)
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e5

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        for id in range(self.total_map_num):
            self.hidden_states_dict[id] = self.perceive_agent_dict[id].module.init_hidden()
            if self.hidden_states_dict[id] is not None:
                # hidden_states (batchsize,agent nums,hidden dim)
                self.hidden_states_dict[id] = self.hidden_states_dict[id].unsqueeze(0).expand(batch_size, self.n_agents_dict[id], -1) 

    def parameters(self):
        perceive_params = []
        for id in range(self.total_map_num):
            perceive_params += self.perceive_agent_dict[id].parameters()
        return perceive_params

    def load_state(self, other_mac):
        for id in range(self.total_map_num):
            self.perceive_agent_dict[id].load_state_dict(other_mac.perceive_agent_dict[id].state_dict())
        self.decision_agent.load_state_dict(other_mac.decision_agent.state_dict())

    def cuda(self):
        for id in range(self.total_map_num):
            self.perceive_agent_dict[id].cuda()
        self.decision_agent.cuda()

    def save_models(self, path, map_id):
        th.save(self.perceive_agent_dict[map_id].state_dict(), ("{}/perceive_"+str(map_id)+".th").format(path))
        th.save(self.decision_agent.state_dict(), "{}/decision.th".format(path))

    def load_models(self, path):       
        self.decision_agent.load_state_dict(th.load("{}/decision.th".format(path), map_location=lambda storage, loc: storage))
        if self.args.fix_policy_para:
            for _,v in self.decision_agent.named_parameters():
                v.requires_grad = False

    def _build_agents(self, input_shape_dict):
        self.devices = d2l.try_all_gpus()
        self.perceive_agent_dict = {}
        for id in range(self.total_map_num):
            self.perceive_agent_dict[id] = DataParallel(agent_REGISTRY[self.args_dict[id].perceive_agent](input_shape_dict[id], self.args_dict[id]), device_ids=self.devices)
        self.decision_agent = DataParallel(agent_REGISTRY[self.args_dict[0].decision_agent](self.args_dict[0]), device_ids=self.devices)

    def _build_inputs(self, batch, t, map_id):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        # obs shape -> (batch_size/env_num, max_episode_length, agent_num, obs_size)
        inputs.append(batch["obs"][:, t].unsqueeze(2).expand(-1, -1, self.n_actions_dict[map_id], -1))
        if self.args_dict[map_id].obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:,t]).unsqueeze(2).expand(-1, -1, self.n_actions_dict[map_id], -1))
            else:
                inputs.append(batch["actions_onehot"][:,t-1].unsqueeze(2).expand(-1, -1, self.n_actions_dict[map_id], -1))
            
        if self.args_dict[map_id].obs_agent_id:
            inputs.append(th.eye(self.n_agents_dict[map_id], device=batch.device).unsqueeze(0).expand(bs, -1, -1).unsqueeze(2).expand(-1, -1, self.n_actions_dict[map_id], -1))
        
        all_actions = th.eye(self.n_actions_dict[map_id], device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, self.n_agents_dict[map_id], -1, -1)
        inputs.append(all_actions)

        inputs = th.cat([x.reshape(bs, self.n_agents_dict[map_id], self.n_actions_dict[map_id], -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme, map_id):
        input_shape = scheme["obs"]["vshape"]
        if self.args_dict[map_id].obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * 2
        if self.args_dict[map_id].obs_agent_id:
            input_shape += self.n_agents_dict[map_id]

        return input_shape
