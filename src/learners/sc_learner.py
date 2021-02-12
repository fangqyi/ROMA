import copy

import torch
import torch.nn.functional as F
from torch.optim import RMSprop
import copy
import numpy as np

from modules.critics.sc import SCControlCritic, SCExecutionCritic
from modules.mixers.qmix import QMixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import DirMixer


class SCLearner:
    def __init__(self, mac, scheme, logger, args):
        torch.autograd.set_detect_anomaly(True)
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.control_critic = SCControlCritic(scheme, args)
        self.execution_critic = SCExecutionCritic(scheme, args)
        self.target_control_critic = copy.deepcopy(self.control_critic)
        self.target_execution_critic = copy.deepcopy(self.execution_critic)

        self.control_actor_params = list(self.mac.agent_dlstm_parameters()) + list(self.mac.latent_state_encoder_parameters())
        self.execution_actor_params = list(self.mac.agent_lstm_parameters())
        self.control_critic_params = list(self.control_critic.parameters())
        self.execution_critic_params = list(self.execution_critic.parameters())

        self.control_mixer = None
        if args.control_mixer is not None:
            if args.control_mixer == "vdn":
                self.control_mixer = VDNMixer()
            elif args.control_mixer == "qmix":
                self.control_mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.control_mixer))
            self.target_control_mixer = copy.deepcopy(self.control_mixer)
            self.control_critic_params += list(self.control_mixer.parameters())

        self.execution_mixer = None
        if args.execution_mixer is not None:
            if args.execution_mixer == "vdn":
                self.execution_mixer = VDNMixer()
            elif args.execution_mixer == "qmix":
                self.execution_mixer = DirMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.execution_mixer))
            self.target_execution_mixer = copy.deepcopy(self.execution_mixer)
            self.execution_critic_params += list(self.execution_mixer.parameters())     

        self.control_actor_optimiser = RMSprop(params=self.control_actor_params, lr=args.control_actor_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.control_critic_optimiser = RMSprop(params=self.control_critic_params, lr=args.control_critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.execution_actor_optimiser = RMSprop(params=self.execution_actor_params, lr=args.execution_actor_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.execution_critic_optimiser = RMSprop(params=self.execution_critic_params, lr=args.execution_critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # self.target_mac = copy.deepcopy(mac)
        # FIXME: implement double mac

    # control:
    #   - decentralized actor pi(|):
    #   - decentralized critic q(|) with mixer: estimate rewards

    # execution:
    #   - decentralized actor pi(|):
    #   - decentralized critic q(|) with mixer: estimate directional contributions

    def train(self, batch, t_env, epsidoe_num):
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]

        dirs_vals, execution_critic_train_stats = self._train_execution_critic(batch, terminated, mask)
        # [bs, seq_len, n_agents, latent_state_dim]
        q_vals, control_critic_train_stats = self._train_control_critic(batch, rewards, terminated, mask)
        # [bs, seq_len, n_agents]
        self.critic_training_steps += 1

        actions = actions[:, :-1]

        lstm_out = []
        dlstm_query_out = []
        dlstm_key_out = []
        dlstm_rule_out = []
        lstm_r = []
        self.mac.init_hidden(bs)
        self.mac.init_goals(bs)
        for t in range(batch["reward"].shape[1] - 1):
            # skip inferring latent state
            lstm_output, dlstm_output = self.mac.forward(batch, t)
            lstm_out.append(lstm_output)  # [batch_num][n_agents][n_actions]
            dlstm_query_out.append(dlstm_output[0])
            dlstm_key_out.append(dlstm_output[1])
            dlstm_rule_out.append(dlstm_output[2])
            intr_r = self._get_instrinsic_r(dirs_vals, t)
            lstm_r.append(self.args.instr_r_rate * intr_r + q_vals[:, t])
        lstm_out = torch.stack(lstm_out, dim=1)  # [batch_num][seq][n_agents][n_actions]
        dlstm_query_out = torch.stack(dlstm_query_out, dim=1)
        dlstm_key_out = torch.stack(dlstm_key_out, dim=1)
        dlstm_rule_out = torch.stack(dlstm_rule_out, dim=1)
        lstm_r = torch.stack(lstm_r, dim=1)

        dlstm_loss_partial = []
        for t in range(batch["reward"].shape[1] - self.args.horizon-1):  # FIXEME: can implement slice instead of t iterations
            dlstm_loss_partial_t = self._get_dlistm_partial(dirs_vals, dlstm_query_out, dlstm_key_out, dlstm_rule_out,
                                                              t)
            dlstm_loss_partial.append(dlstm_loss_partial_t)
        dlstm_loss_partial = torch.stack(dlstm_loss_partial, dim=1)  # [bs, seq_len-c, n_agents]

        # Mask out unavailable actions, renormalise (as in action selection)
        lstm_out[avail_actions == 0] = 0  # [batch_num][seq][n_agents][n_actions]
        lstm_out = lstm_out/lstm_out.sum(dim=-1, keepdim=True)
        lstm_out[avail_actions == 0] = 0

        # FIXME: implement q baseline
        q_vals = q_vals[:, :-self.args.horizon-1]
        q_vals = q_vals.reshape(-1, 1).squeeze(1)
        lstm_r = lstm_r.reshape(-1, 1).squeeze(1)
        pi = lstm_out.reshape(-1, self.n_actions)

        mask_dlstm = mask.clone().repeat(1, 1, self.n_agents)[:, :-self.args.horizon].reshape(-1, 1).squeeze(1)
        mask = mask.repeat(1, 1, self.n_agents).view(-1)
        # print("mask_dlstm shape:{}".format(mask_dlstm.shape))

        pi_taken = torch.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = torch.log(pi_taken)

        dlstm_loss_partial = dlstm_loss_partial.reshape(-1, 1).squeeze(1)
        dlstm_loss_partial[mask_dlstm == 0] = 0.0

        dlstm_loss = ((dlstm_loss_partial * q_vals.detach()) * mask_dlstm).sum() / mask_dlstm.sum()
        dlstm_loss += self.mac.compute_lat_state_kl_div()
        lstm_loss = ((log_pi_taken * lstm_r) * mask).sum() / mask.sum()

        self.control_actor_optimiser.zero_grad()
        dlstm_loss.backward(retain_graph=True)
        dlstm_grad_norm = torch.nn.utils.clip_grad_norm_(self.control_actor_params, self.args.grad_norm_clip)
        self.control_actor_optimiser.step()

        self.execution_actor_optimiser.zero_grad()
        lstm_loss.backward()
        lstm_grad_norm = torch.nn.utils.clip_grad_norm_(self.execution_actor_params, self.args.grad_norm_clip)
        self.execution_actor_optimiser.step()

        if (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.critic_training_steps

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(control_critic_train_stats["control_critic_td_loss"])
            for key in ["control_critic_td_loss", "control_critic_grad_norm"]:
                self.logger.log_stat(key, sum(control_critic_train_stats[key])/ts_logged, t_env)

            ts_logged = len(execution_critic_train_stats["execution_critic_td_loss"])
            for key in ["execution_critic_td_loss", "execution_critic_grad_norm"]:
                self.logger.log_stat(key, sum(execution_critic_train_stats[key]) / ts_logged, t_env)

            self.logger.log_stat("control_actor_loss", dlstm_loss.item(), t_env)
            self.logger.log_stat("execution_actor_loss", lstm_loss.item(), t_env)
            self.logger.log_stat("control_grad_norm", dlstm_grad_norm, t_env)
            self.logger.log_stat("execution_grad_norm", lstm_grad_norm, t_env)
            self.log_stats_t = t_env

    def _get_instrinsic_r(self, dirs_val, t_ep):
        r = 0
        for t in range(0, min(self.args.horizon, t_ep)):
            idx = t+1
            dec_lat_states = dirs_val[:, t_ep-idx]  # [bs][n_agents][lat_dim]
            goals = self.mac.get_prev_goal(idx)  # [batch_num][n_agents][rule_dim]
            # calculate the cosine similarity between agent's estimated contribution and its goal
            r += F.cosine_similarity(dec_lat_states, goals, dim=2)
            # penalize if agent has no goal
            r -= self.args.cum_goal_zeros_penalty_rate * F.cosine_similarity(torch.zeros(goals.shape), goals, dim=2)
        return r/self.args.horizon # [bs][n_agents]

    def _get_dlistm_partial(self, dirs_vals, queries, keys, rules, t):
        # calculate dcos(zt+c−zt,gt(θ)) in multi-agent context

        # print("dirs_vals shape: {}".format(dirs_vals.shape))
        dir_vals_t = dirs_vals[:, t]
        dir_vals_tc = dirs_vals[:, t+self.args.horizon]
        query = queries[:, t]
        key = keys[:, t]
        rule = rules[:, t]

        dlstm_partial = []
        for i in range(self.n_agents):
            # calc decomposed latent's projection on rule at t
            p_i = [torch.einsum('ij,ij->i', dir_vals_t[:, j], rule[:, i]) for j in range(self.n_agents)]
            # rule is already normalized
            p_i = torch.stack(p_i, dim=1)  # [bs, n_agents]
            p_t = [torch.einsum('i,ij->ij', p_i[:, j], rule[:, i]) for j in range(self.n_agents)]
            p_t = torch.stack(p_t, dim=1)  # [bs, n_agents, latent_state_dim]

            # calc decomposed latent's projection on rule at t+c
            p_i_c = [torch.einsum('ij,ij->i', dir_vals_tc[:, j], rule[:, i]) for j in range(self.n_agents)]
            # rule is already normalized
            p_i_c = torch.stack(p_i_c, dim=1)  # [bs, n_agents]
            p_tc = [torch.einsum('i,ij->ij', p_i_c[:, j], rule[:, i]) for j in range(self.n_agents)]
            p_tc = torch.stack(p_tc, dim=1)  # [bs, n_agents, latent_state_dim]

            p_diff = p_t - p_tc   # [bs, n_agents, latent_state_dim]

            qk_i = [torch.einsum('ij,ij->i', query[:, i], key[:, j])  # 2d torch.dot
                    / self.args.attention_noramlization_squared_dk for j in range(self.n_agents)]
            qk_i_t = torch.stack(qk_i, dim=1)  # [bs, n_agents]
            a_i = torch.nn.functional.softmax(qk_i_t, dim=1)
            # eq 2 in TarMac  # FIXME: only save attention and use attention directly

            dlstm_partial_i = [torch.einsum("i,ij->ij", 1/a_i[:, j], p_diff[:, j]) for j in range(self.n_agents)]
            dlstm_partial_i = torch.stack(dlstm_partial_i, dim=1).sum(dim=1)  # [bs, latent_state_dim]
            dlstm_partial_i = F.cosine_similarity(dlstm_partial_i, rule[:, i], dim=1)

            dlstm_partial.append(dlstm_partial_i)

        return torch.stack(dlstm_partial, dim=1)  # [bs, n_agents]

    def _train_control_critic(self, batch, rewards, terminated, mask):  # FIXME: terminated?
        bs = batch.batch_size
        qs_vals = torch.zeros(bs, batch["reward"].shape[1], self.n_agents)

        running_log = {
            "control_critic_td_loss": [],
            "control_critic_grad_norm": [],
        }

        for t in range(batch["reward"].shape[1]-1):
            # print(mask.shape)
            mask_t = mask[:, t]
            if mask_t.sum() == 0:
                continue

            qs_t = self.control_critic(batch, t)  # [bs, n_agents]
            qs_tot = self.control_mixer(qs_t.unsqueeze(1), batch["latent_state"][:, t].unsqueeze(1))
            qs_vals[:, t] = qs_t
            # [bs_size][t][lat_dim]
            
            target_qs_t = self.target_control_critic(batch, t+1)
            target_qs_tot = self.target_control_mixer(target_qs_t.unsqueeze(1), batch["latent_state"][:, t+1].unsqueeze(1))  

            td_loss = qs_tot - (rewards[:, t] + self.args.control_discount*(1 - terminated[:,t])*target_qs_tot.detach())
            td_loss = (td_loss**2).sum()

            self.control_critic_optimiser.zero_grad()
            td_loss.backward(retain_graph=True)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.control_critic_params, self.args.grad_norm_clip)
            self.control_critic_optimiser.step()

            running_log["control_critic_td_loss"].append(td_loss.item())
            running_log["control_critic_grad_norm"].append(grad_norm)
        
        return qs_vals, running_log

    def _train_execution_critic(self, batch, terminated, mask):
        bs = batch.batch_size
        dirs_tot_vals = torch.zeros(bs, batch["reward"].shape[1], self.n_agents, self.args.latent_state_dim)

        running_log = {
            "execution_critic_td_loss": [],
            "execution_critic_grad_norm": [],
        }

        for t in range(batch["reward"].shape[1]-1):
            # print(mask.shape)
            mask_t = mask[:, t]
            if mask_t.sum() == 0:
                continue

            # distance between latent states
            lat_state_target_dis = torch.sub(batch["latent_state"][:, t+1], batch["latent_state"][:, t])
            # [bs_size, latent_state_dim]

            dirs_t = self.execution_critic(batch, t)  # [bs, n_agents, latent_state_dim]
            dirs_tot = self.execution_mixer(dirs_t.unsqueeze(1), batch["latent_state"][:, t].unsqueeze(1))  
            dirs_tot_vals[:, t] = dirs_t
            # [bs_size][t][lat_dim]
            
            target_dirs_t = self.target_execution_critic(batch, t+1)
            target_dirs_tot = self.target_execution_mixer(target_dirs_t.unsqueeze(1), batch["latent_state"][:, t+1].unsqueeze(1))  

            td_loss = dirs_tot - (lat_state_target_dis + self.args.execution_discount*(1 - terminated[:, t])*target_dirs_tot.detach())
            td_loss = (td_loss ** 2).sum()

            self.execution_critic_optimiser.zero_grad()
            td_loss.backward(retain_graph=True)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.control_critic_params, self.args.grad_norm_clip)
            self.execution_critic_optimiser.step()

            running_log["execution_critic_td_loss"].append(td_loss.item())
            running_log["execution_critic_grad_norm"].append(grad_norm)
        
        return dirs_tot_vals, running_log

    def _update_targets(self):
        self.target_control_critic.load_state_dict(self.control_critic.state_dict())
        self.target_control_mixer.load_state_dict(self.control_mixer.state_dict())
        self.target_execution_critic.load_state_dict(self.execution_critic.state_dict())
        self.target_execution_mixer.load_state_dict(self.execution_mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.control_critic.cuda()
        self.execution_critic.cuda()
        self.target_control_critic.cuda()
        self.target_execution_critic.cuda()
        self.control_mixer.cuda()
        self.execution_mixer.cuda()
        self.target_control_mixer()
        self.target_execution_mixer()

    def save_models(self, path):
        self.mac.save_models(path)
        torch.save(self.control_critic.state_dict(), "{}/control_critic.th".format(path))
        torch.save(self.execution_critic.state_dict(), "{}/execution_critic.th".format(path))
        torch.save(self.control_mixer.state_dict(), "{}/control_mixer.th".format(path))
        torch.save(self.execution_mixer.state_dict(), "{}/execution_mixer.th".format(path))
        torch.save(self.control_critic_optimiser.state_dict(), "{}/control_critic_opt.th".format(path))
        torch.save(self.control_actor_optimiser.state_dict(), "{}/control_actor_opt.th".format(path))
        torch.save(self.execution_critic_optimiser.state_dict(), "{}/execution_critic_opt.th".format(path))
        torch.save(self.execution_actor_optimiser.state_dict(), "{}/execution_actor_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.control_critic.load_state_dict(torch.load(
            "{}/control_critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.execution_critic.load_state_dict(torch.load(
            "{}/execution_critic.th".format(path), map_location=lambda storage, loc: storage))
        self.control_mixer.load_state_dict(
            torch.load("{}/control_mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.execution_mixer.load_state_dict(
            torch.load("{}/execution_mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.control_actor_optimiser.load_state_dict(
            torch.load("{}/control_actor_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.execution_actor_optimiser.load_state_dict(
            torch.load("{}/execution_actor_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.control_critic_optimiser.load_state_dict(
            torch.load("{}/control_critic_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.execution_critic_optimiser.load_state_dict(
            torch.load("{}/execution_critic_opt.th".format(path), map_location=lambda storage, loc: storage))