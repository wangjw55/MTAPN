import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qatten import QattenMixer
from modules.agents import REGISTRY as agent_REGISTRY
from envs.matrix_game import print_matrix_status
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np
from utils.th_utils import get_parameters_num

class ANQLearner:
    def __init__(self, mac, logger_dict, args_dict, total_map_num):
        self.args_dict = args_dict
        self.mac = mac
        self.logger_dict = logger_dict
        self.total_map_num = total_map_num
        
        self.last_target_update_episode = 0 
        self.device = th.device('cuda' if args_dict[0].use_cuda  else 'cpu')
        self.mixer_dict = {}
        self.target_mixer_dict = {}
        self.optimiser_dict = {}

        self.decision_params = list(self.mac.decision_agent.parameters())
        if args_dict[0].use_small_lr_for_decision:
            decision_lr = args_dict[0].lr / self.total_map_num
        else:
            decision_lr = args_dict[0].lr
        if self.args_dict[0].optimizer == 'adam':
            self.optimiser_decision = Adam(params=self.decision_params, lr=decision_lr, weight_decay=getattr(args_dict[0], "weight_decay", 0))
        else:
            self.optimiser_decision = RMSprop(params=self.decision_params, lr=decision_lr, alpha=args_dict[0].optim_alpha, eps=args_dict[0].optim_eps)
        
        self.params = mac.parameters()
        for i in range(self.total_map_num):
            if args_dict[i].mixer == "qatten":
                self.mixer_dict[i] = QattenMixer(args_dict[i])
            elif args_dict[i].mixer == "vdn":
                self.mixer_dict[i] = VDNMixer()
            elif args_dict[i].mixer == "qmix":
                self.mixer_dict[i] = Mixer(args_dict[i])
            else:
                raise "mixer error"
            self.target_mixer_dict[i] = copy.deepcopy(self.mixer_dict[i])
            self.params += list(self.mixer_dict[i].parameters())

            print('Mixer Size: ')
            print(get_parameters_num(self.mixer_dict[i].parameters()))

        if self.args_dict[0].optimizer == 'adam':
            self.optimiser = Adam(params=self.params,  lr=args_dict[0].lr, weight_decay=getattr(args_dict[0], "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args_dict[0].lr, alpha=args_dict[0].optim_alpha, eps=args_dict[0].optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        
        self.log_stats_t_list = [(-self.args_dict[0].learner_log_interval - 1) for _ in range(total_map_num)]
        
        self.train_t = 0

        # priority replay
        self.use_per = getattr(self.args_dict[0], 'use_per', False)
        self.return_priority = getattr(self.args_dict[0], "return_priority", False)
        if self.use_per:
            self.priority_max = float('-inf')
            self.priority_min = float('inf')
        
    def train(self, batch_dict: EpisodeBatch, t_env_dict: int, battle_won_dict, episode_num: int, per_weight=None):
        # Get the relevant quantities
        rewards_dict = {}
        actions_dict = {}
        terminated_dict = {}
        mask_dict = {}
        avail_actions_dict = {}

        loss_list = []
        L_td_list = []
        masked_td_error_list = []
        chosen_action_qvals_list= []
        mask_list = []
        targets_list = []
        for id in range(self.total_map_num):
            rewards_dict[id] = batch_dict[id]["reward"][:, :-1]
            actions_dict[id] = batch_dict[id]["actions"][:, :-1]
            terminated_dict[id] = batch_dict[id]["terminated"][:, :-1].float()
            mask_dict[id] = batch_dict[id]["filled"][:, :-1].float()
            mask_dict[id][:, 1:] = mask_dict[id][:, 1:] * (1 - terminated_dict[id][:, :-1])
            avail_actions_dict[id] = batch_dict[id]["avail_actions"]
        
            # Calculate estimated Q-Values
            self.mac.perceive_agent_dict[id].train()
            self.mac.decision_agent.train()
            mac_out = []
            self.mac.init_hidden(batch_dict[id].batch_size)

            self.hidden_index = actions_dict[id]

            for t in range(batch_dict[id].max_seq_length):
                if t != batch_dict[id].max_seq_length-1:
                    agent_outs = self.mac.forward(batch_dict[id], t=t, test_mode=False, map_id=id, hidden_index=self.hidden_index[:,t])
                else:
                    agent_outs = self.mac.forward(batch_dict[id], t=t, test_mode=False, map_id=id)
                mac_out.append(agent_outs)
            # mac out -> (batch_size, agent_num, action_num)
            mac_out = th.stack(mac_out, dim=1)  # Concat over time

            # Pick the Q-Values for the actions taken by each agent
            chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions_dict[id]).squeeze(3)  # Remove the last dim
            chosen_action_qvals_ = chosen_action_qvals

            # Calculate the Q-Values necessary for the target
            with th.no_grad():
                self.target_mac.perceive_agent_dict[id].train()
                self.target_mac.decision_agent.train()
                target_mac_out = []
                self.target_mac.init_hidden(batch_dict[id].batch_size)
                for t in range(batch_dict[id].max_seq_length):
                    if t != batch_dict[id].max_seq_length-1:
                        target_agent_outs = self.target_mac.forward(batch_dict[id], t=t, test_mode=False, map_id=id, hidden_index=self.hidden_index[:,t])
                    else:
                        target_agent_outs = self.target_mac.forward(batch_dict[id], t=t, test_mode=False, map_id=id)
                    target_mac_out.append(target_agent_outs)

                # We don't need the first timesteps Q-Value estimate for calculating targets
                target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

                # Max over target Q-Values/ Double q learning
                mac_out_detach = mac_out.clone().detach()
                mac_out_detach[avail_actions_dict[id] == 0] = -9999999
                cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
                target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
                
                # Calculate n-step Q-Learning targets
                target_max_qvals = self.target_mixer_dict[id](target_max_qvals, batch_dict[id]["state"])

                if getattr(self.args_dict[id], 'q_lambda', False):
                    qvals = th.gather(target_mac_out, 3, batch_dict[id]["actions"]).squeeze(3)
                    qvals = self.target_mixer_dict[id](qvals, batch_dict[id]["state"])

                    targets = build_q_lambda_targets(rewards_dict[id], terminated_dict[id], mask_dict[id], target_max_qvals, qvals,
                                        self.args_dict[id].gamma, self.args_dict[id].td_lambda)
                else:
                    targets = build_td_lambda_targets(rewards_dict[id], terminated_dict[id], mask_dict[id], target_max_qvals, 
                                                        self.args_dict[id].n_agents, self.args_dict[id].gamma, self.args_dict[id].td_lambda)

            # Mixer
            chosen_action_qvals = self.mixer_dict[id](chosen_action_qvals, batch_dict[id]["state"][:, :-1])

            td_error = (chosen_action_qvals - targets.detach())
            td_error2 = 0.5 * td_error.pow(2)

            mask = mask_dict[id].expand_as(td_error2)
            masked_td_error = td_error2 * mask

            # important sampling for PER
            if self.use_per:
                per_weight = th.from_numpy(per_weight).unsqueeze(-1).to(device=self.device)
                masked_td_error = masked_td_error.sum(1) * per_weight

            loss = L_td = masked_td_error.sum() / mask.sum()
            loss_list.append(loss)

            L_td_list.append(L_td)
            masked_td_error_list.append(masked_td_error)
            chosen_action_qvals_list.append(chosen_action_qvals)
            mask_list.append(mask)
            targets_list.append(targets)

        # Optimise
        self.optimiser.zero_grad()
        self.optimiser_decision.zero_grad()
        
        if self.args_dict[0].use_weighted_sum_loss:
            loss_weight = [0.01]*self.total_map_num
            battle_lost_total = self.total_map_num-sum(battle_won_dict)
            for i in range(self.total_map_num):
                if battle_won_dict[i] != 1.0:
                    loss_weight[i] = (1-battle_won_dict[i])/battle_lost_total
            # print('battle_lost_dict:', [1-battle_won_dict[i] for i in range(len(battle_won_dict))])
            # print('loss_weight:', loss_weight)
            total_loss = sum([loss_list[i] * loss_weight[i] for i in range(len(loss_list))])
        else:
            total_loss = sum(loss_list)
        
        total_loss.backward()

        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args_dict[0].grad_norm_clip)
        self.optimiser.step()
        grad_norm_dec = th.nn.utils.clip_grad_norm_(self.decision_params, self.args_dict[0].grad_norm_clip)
        self.optimiser_decision.step()

        if (episode_num - self.last_target_update_episode) / self.args_dict[0].target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        for i in range(self.total_map_num):
            if t_env_dict[i] - self.log_stats_t_list[i] >= self.args_dict[i].learner_log_interval:
                self.logger_dict[i].log_stat("loss_td", L_td_list[i].item(), t_env_dict[i])
                self.logger_dict[i].log_stat("grad_norm", grad_norm, t_env_dict[i])
                mask_elems = mask_list[i].sum().item()
                self.logger_dict[i].log_stat("td_error_abs", (masked_td_error_list[i].abs().sum().item()/mask_elems), t_env_dict[i])
                self.logger_dict[i].log_stat("q_taken_mean", (chosen_action_qvals_list[i] * mask_list[i]).sum().item()/(mask_elems * self.args_dict[i].n_agents), t_env_dict[i])
                self.logger_dict[i].log_stat("target_mean", (targets_list[i] * mask_list[i]).sum().item()/(mask_elems * self.args_dict[i].n_agents), t_env_dict[i])
                self.log_stats_t_list[i] = t_env_dict[i]

        return 0

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        for id in range(self.total_map_num): 
            if self.mixer_dict[id] is not None:
                self.target_mixer_dict[id].load_state_dict(self.mixer_dict[id].state_dict())
            self.logger_dict[0].console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        for i in range(self.total_map_num): 
            if self.mixer_dict[i] is not None:
                self.mixer_dict[i].cuda()
                self.target_mixer_dict[i].cuda()
            
    def save_models(self, path, map_id):
        self.mac.save_models(path, map_id)
        if self.mixer_dict[map_id] is not None:
            th.save(self.mixer_dict[map_id].state_dict(), ("{}/mixer"+str(map_id)+".th").format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        th.save(self.optimiser_decision.state_dict(), "{}/dec_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        # if self.mixer is not None:
        #     self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        # self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
