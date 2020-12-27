import copy

import torch
from torch.optim import RMSprop
import torch.nn.functional as F

from src.modules.critics.sc import SCCritic
from src.modules.mixers.qmix import QMixer
from src.modules.mixers.vdn import VDNMixer
from src.utils.rl_utils import build_td_lambda_targets


class SCLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.critic = SCCritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())
        self.params = self.agent_params + self.critic_params

        self.lat_state_mixer = None
        if args.lat_state_mixer is not None:
            if args.lat_state_mixer == "vdn":
                self.lat_state_mixer = VDNMixer()
            elif args.lat_state_mixer == "qmix":
                self.lat_state_mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.lat_state_mixer))
            self.mixer_params = self.lat_state_mixer.parameters() + self.mac.dec_lac_state_func_parameters()
            self.params += [self.lat_state_mixer.parameters(), self.mac.dec_lac_state_func_parameters()]
            self.target_lat_state_mixer = copy.deepcopy(self.lat_state_mixer)
        self.agent_lstm_params = self.mac.agent_lstm_parameters()
        self.agent_dlstm_params = self.mac.agent_dlstm_parameters()

        self.mixer_optimiser = RMSprop(params=self.mixer_params, lr=args.mixer_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.agent_lstm_optimiser = RMSprop(params=self.agent_lstm_params, lr=args.actor_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.agent_dlstm_optimiser = RMSprop(params=self.agent_dlstm_params, lr=args.actor_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha,
                                        eps=args.optim_eps)

    def train(self, batch, t_env, epsidoe_num):
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]

        critic_mask = mask.clone()

        mask = mask.repeat(1, 1, self.n_agents).view(-1)

        q_vals, critic_train_stats = self._train_critic(batch, rewards, terminated, actions)

        actions = actions[:, :-1]

        mac_out = []
        lstm_r = []
        for t in range(batch["rewards"].shape[1] - 1):
            # skip inferring latent state
            agent_outputs = self.mac.forward(batch, t)
            mac_out.append(agent_outputs)  # [batch_num][n_agents][n_actions]
            intr_goals = self.mac.get_current_goal()  # [batch_num][n_agents][rule_dim]
            intr_r = self._get_instrinsic_r(batch, t, intr_goals)
            lstm_r.append(self.args.instr_r_rate * intr_r + q_vals[:, t])
        mac_out = torch.stack(mac_out, dim=1)
        lstm_r = torch.stack(lstm_r, dim=1)

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0

        # FIXME: implement q baseline
        q_vals = q_vals.reshape(-1, 1)
        lstm_r = lstm_r.reshape(-1, 1)

        pi_taken = torch.gather(mac_out, dim=2, index=actions.reshape(bs, -1, 1)).squeeze(2)
        pi_taken[mask == 0] = 1.0
        pi_taken = torch.prod(pi_taken, dim=1).squeeze(1)
        log_pi_taken = torch.log(pi_taken)

        dlstm_loss = ((log_pi_taken * q_vals.detach()) * mask).sum() / mask.sum()
        lstm_loss = ((log_pi_taken * lstm_r) * mask).sum() / mask.sum()

        self.agent_dlstm_optimiser.zero_grad()
        dlstm_loss.backward()
        dlstm_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent_dlstm_params, self.args.grad_norm_clip)
        self.agent_dlstm_optimiser.step()
        self.agent_lstm_optimiser.zero_grad()
        lstm_loss.backward()
        lstm_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent_lstm_params, self.args.grad_norm_clip)
        self.agent_lstm_optimiser.step()

        if (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.critic_training_steps

    def _get_instrinsic_r(self, batch, t_ep, goals):
        r = 0
        for t in range(0, min(self.args.horizon, t_ep)):
            dec_lat_states = batch["dec_lat_state"][:, -(t+1)]  #[bs][n_agents][lat_dim]
            r += F.cosine_similarity(dec_lat_states, goals, dim=2)
            r -= self.args.cum_goal_zeros_penalty_rate * F.cosine_similarity(torch.zeros(goals.shape, goals))
        return r/self.args.horizon

    def _train_mixer(self, batch, bs):
        for t in range(batch["rewards"].shape[1]):
            lat_state_target_dis = torch.sub(batch["latent_state"][:, t+1], batch["latent_state"][:, t]) # [bs_size][lat_dim]
            lat_state_mixer_dis = self.lat_state_mixer(batch["dec_lat_state"][:, t].unsqueeze(1), batch["lat"]).squeeze(1).squeeze(1)  # [bs_size][t][lat_dim]
            lat_state_loss = (torch.sub(lat_state_target_dis, lat_state_mixer_dis)**2).sum(1).sum(0)/(bs*self.args.lat_state_dim)

            self.mixer_optimiser.zero_grad()
            lat_state_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.mixer_params, self.args.grad_norm_clip)
            self.mixer_optimiser.step()
            # FIXME: Add training logs

    def _train_critic(self, batch, rewards, terminated, mask):
        # Optimise critic
        target_q_vals = self.target_critic(batch)[:, :]
        #targets_taken = torch.gather(target_q_vals, dim=3, index=actions).squeeze(3)

        # Calculate td-lambda targets
        targets = build_td_lambda_targets(rewards, terminated, mask, target_q_vals , self.n_agents, self.args.gamma,
                                          self.args.td_lambda)

        q_vals = torch.zeros_like(target_q_vals)[:, :-1]

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        for t in reversed(range(rewards.size(1))):
            mask_t = mask[:, t].expand(-1, self.n_agents)
            if mask_t.sum() == 0:
                continue

            q_t = self.critic(batch, t)
            q_vals[:, t] = q_t
            targets_t = targets[:, t]

            td_error = (q_t - targets_t.detach())

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask_t

            # Normal L2 loss, take mean over actual data
            loss = (masked_td_error ** 2).sum() / mask_t.sum()
            self.critic_optimiser.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()
            self.critic_training_steps += 1

            running_log["critic_loss"].append(loss.item())
            running_log["critic_grad_norm"].append(grad_norm)
            mask_elems = mask_t.sum().item()
            running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
            running_log["q_taken_mean"].append((q_t * mask_t).sum().item() / mask_elems)
            running_log["target_mean"].append((targets_t * mask_t).sum().item() / mask_elems)

        return q_vals, running_log

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        torch.save(self.critic.state_dict(), "{}/critic.th".format(path))
        torch.save(self.agent_lstm_optimiser.state_dict(), "{}/agent_lstm_opt.th".format(path))
        torch.save(self.agent_dlstm_optimiser.state_dict(), "{}/agent_dlstm_opt.th".format(path))
        torch.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(torch.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_lstm_optimiser.load_state_dict(
            torch.load("{}/agent_lstm_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.agent_dlstm_optimiser.load_state_dict(
            torch.load("{}/agent_dlstm_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(
            torch.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
