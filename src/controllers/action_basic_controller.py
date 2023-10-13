from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from torch.nn.parallel import DataParallel
from d2l import torch as d2l

# This multi-agent controller shares parameters between agents
class ABasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        # 加入actions num
        self.n_actions = args.n_actions
        self.args = args
        input_shape = self._get_input_shape(scheme)
        # 初始化agent
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.save_probs = getattr(self.args, 'save_probs', False)

        self.hidden_states = None

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
        # hidden_states (1,hidden dim)
        self.hidden_states = self.agent.module.init_hidden()
        if self.hidden_states is not None:
            # hidden_states (batchsize,agent nums,hidden dim)
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def divide_parameters(self):
        new_para = []
        old_para = []
        for para in self.agent.named_parameters():
            if para[0] in ['fc2.weight', 'fc2.bias']:
                old_para.append(para[1])
            else:
                new_para.append(para[1])
        return new_para, old_para

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        # self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        
        loaded_param = th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage)
        model_dict=self.agent.state_dict()
        pretrained_dict = {k: v for k, v in loaded_param.items() if k in ['fc2.weight', 'fc2.bias']}
        model_dict.update(pretrained_dict)
        self.agent.load_state_dict(model_dict)
        
        # 将决策层设置为不进行梯度更新，如果微调决策层需要删除
        if self.args.fix_policy_para:
            for k,v in self.agent.named_parameters():
                if k in ['fc2.weight', 'fc2.bias']:
                    v.requires_grad = False

    def _build_agents(self, input_shape):
        # self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        self.devices = d2l.try_all_gpus()
        self.agent = DataParallel(agent_REGISTRY[self.args.agent](input_shape, self.args), device_ids=self.devices)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        # obs shape -> (batch_size/env_num, max_episode_length, agent_num, obs_size)
        inputs.append(batch["obs"][:, t].unsqueeze(2).expand(-1, -1, self.n_actions, -1))
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:,t]).unsqueeze(2).expand(-1, -1, self.n_actions, -1))
            else:
                inputs.append(batch["actions_onehot"][:,t-1].unsqueeze(2).expand(-1, -1, self.n_actions, -1))
            
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1).unsqueeze(2).expand(-1, -1, self.n_actions, -1))
        
        all_actions = th.eye(self.n_actions, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, self.n_agents, -1, -1)
        inputs.append(all_actions)
        
        inputs = th.cat([x.reshape(bs, self.n_agents, self.n_actions, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * 2
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
