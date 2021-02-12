import torch
import torch.nn.functional as F
import numpy as np
from components.action_selectors import REGISTRY as action_REGISTRY
from modules.agents import REGISTRY as agent_REGISTRY
from modules.utils import REGISTRY as utils_REGISTRY

class SCMAC():
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shapes = self._get_input_shapes(scheme)
        output_shapes = self._get_output_shapes(scheme)
        self._build_agents(input_shapes, output_shapes, args)
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.comm_size = self.args.communication_query_and_signature_size
        self.rule_size = self.args.latent_state_dim
        self.agent_hidden_states = None
        self.goals = []
        self.cur_dec_state = None
        self.latent_state_encoder = utils_REGISTRY[args.latent_state_encoder_class](
            input_size=scheme["state"]["vshape"],
            output_size=self.args.latent_state_dim,
            mlp_hidden_sizes=args.latent_state_encoder_hidden_sizes)


    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs, control_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                            test_mode=test_mode)
        return chosen_actions, control_outputs

    def compute_lat_state_kl_div(self):
        if self.args.use_lat_state_information_bottleneck:
            return self.latent_state_encoder.compute_kl_div()
        else:
            raise Exception("Current decomposed latent state estimator does not support information bottleneck")

    def init_goals(self, batch_size):
        """
        must call before running an episode, self.goals = [timestep 0: [agent_id][batch_id][rule_dim], ...]
        """
        goal_init = [torch.zeros([batch_size, self.rule_size])] * self.n_agents
        goal_init = torch.stack(goal_init, dim=1)  # [batch_num][n_agent][rule_dim]
        self.goals = [goal_init]

    def infer_latent_state(self, state):
        """

        Args:
            state:

        Returns:

        """
        return self.latent_state_encoder.infer_posterior(state)

    def get_cumulative_goal(self):
        """
        get current goal: mean cumulative goals over the horizon length
        Returns: torch.tensor [curr_episode_num][n_agents][batch_num][rule_dim]

        """
        if len(self.goals) < self.args.horizon:
            goals_sum = torch.stack(self.goals, dim=0)  # [curr_episode_num][batch_num][n_agents][rule_dim]
        else:
            goals_sum = torch.stack([self.goals[-i] for i in range(1, self.args.horizon + 1)], dim=0)
        cur_goal = torch.mean(goals_sum, dim=0)
        return cur_goal  # [batch_num][n_agents][rule_dim]

    def get_current_goal(self):
        return self.goals[-1]

    def get_prev_goal(self, t):
        return self.goals[-t]

    def forward(self, ep_batch, t, test_mode=False):
        """
        returns actions from timestep t at the episode batch, also updates internal goals

        Args:
            ep_batch:
            t:
            test_mode:

        Returns:
            actions: torch.tensor [batch_num][n_agents][action_dim]

        """
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        goals = self.get_cumulative_goal().view(ep_batch.batch_size * self.n_agents, -1)
        agent_outs, self.agent_hidden_states = self.agent(agent_inputs, self.agent_hidden_states, goals)
        local_outs, horizon_outs = agent_outs

        # agent actions
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimize their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                local_outs[reshaped_avail_actions == 0] = -1e10

            local_outs = torch.nn.functional.softmax(local_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = local_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                # With probability epsilon, we will pick an available action uniformly
                local_outs = ((1 - self.action_selector.epsilon) * local_outs
                              + torch.ones_like(local_outs) * self.action_selector.epsilon / epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    local_outs[reshaped_avail_actions == 0] = 0.0

        actions = local_outs.view(ep_batch.batch_size, self.n_agents, -1)

        # agent communications and update goals
        queries, keys, rules = horizon_outs.view(ep_batch.batch_size, self.n_agents, -1).split([self.comm_size,
                                                                                                self.comm_size,
                                                                                                self.rule_size], dim=-1)

        # [bs, n_agents, comm_size], [bs, n_agents, comm_size], [bs, n_agents, rule_size]
        rules = F.normalize(rules, dim=1)  # gt= ˆgt/||ˆgt||; in eq 3 from Feudal Net, no grad

        rules_no_grad = rules.detach()
        goals_t = []
        for i in range(self.n_agents):
            qi = queries[:, i]  # [bs, comm_size]
            qk_i = [torch.einsum('ij,ij->i', qi, keys[:, j]) / self.args.attention_noramlization_squared_dk
                    for j in range(self.n_agents)] # 2d torch.dot
            qk_i_t = torch.stack(qk_i, dim=1)  # [bs, n_agents]
            a_i = torch.nn.functional.softmax(qk_i_t, dim=1)
            # eq 2 in TarMac

            goals_a_i = [torch.einsum("i,ij->ij", a_i[:, j], rules_no_grad[:, j]) for j in range(self.n_agents)]
            goal_i = torch.stack(goals_a_i, dim=1).sum(dim=1)  # [bs, g_size]
            # eq 3 in TarMac

            goals_t.append(goal_i)
        goals_t = torch.stack(goals_t, dim=1)  # [batch][n_agents][rule_dim]
        self.goals.append(goals_t)

        return actions, (queries, keys, rules)

    def init_hidden(self, batch_size):
        self.agent_hidden_states = self.agent.init_hidden(batch_size)

    def parameters(self):
        return list(self.agent.parameters()) + list(self.latent_state_encoder.parameters())  # assume the check is recursive

    def agent_lstm_parameters(self):
        return self.agent.lstm_parameters()

    def agent_dlstm_parameters(self):
        return self.agent.dlstm_parameters()

    def latent_state_encoder_parameters(self):
        return self.latent_state_encoder.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.latent_state_encoder.load_state_dict(other_mac.latent_state_encoder.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.latent_state_encoder.cuda()

    def save_models(self, path):  # FIXME: add state encoder
        torch.save(self.agent.state_dict(), "{}/agent.th".format(path))
        torch.save(self.latent_state_encoder.state_dict(), "{}/state_encoder.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(torch.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.latent_state_encoder.load_state_dict(
            torch.load("{}/state_encoder.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shapes, output_shapes, args):
        self.agent = agent_REGISTRY[self.args.agent](input_shapes, output_shapes, args)

    def _build_inputs(self, batch, t):  # FIXME: consider execution modules need to take goal signals in inputs
        # control: last_action(opt) + obs + latent_state(opt) + agent_id(opt)
        # execution: last_action(opt) + obs + latent_state(opt) + agent_id(opt)

        bs = batch.batch_size
        lstm_inputs = []
        dlstm_inputs = []

        if self.args.obs_last_action:
            if t == 0:
                lstm_inputs.append(torch.zeros_like(batch["actions_onehot"][:, t]))
                dlstm_inputs.append(torch.zeros_like(batch["keys"][:, t]))
                dlstm_inputs.append(torch.zeros_like(batch["queries"][:, t]))
                dlstm_inputs.append(torch.zeros_like(batch["rules"][:, t]))
            else:
                lstm_inputs.append(batch["actions_onehot"][:, t - 1])
                dlstm_inputs.append(batch["keys"][:, t - 1])
                dlstm_inputs.append(batch["queries"][:, t - 1])
                dlstm_inputs.append(batch["rules"][:, t - 1])

        lstm_inputs.append(batch["obs"][:, t])
        dlstm_inputs.append(batch["obs"][:, t])

        if self.args.obs_latent_state:
            lstm_inputs.append(batch["latent_state"][:, t].unsqueeze(1).repeat(1, self.n_agents, 1))
            dlstm_inputs.append(batch["latent_state"][:, t].unsqueeze(1).repeat(1, self.n_agents, 1))

        if self.args.obs_agent_id:
            lstm_inputs.append(torch.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
            dlstm_inputs.append(torch.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        lstm_inputs = torch.cat([x.reshape(bs * self.n_agents, -1) for x in lstm_inputs], dim=1)
        dlstm_inputs = torch.cat([x.reshape(bs * self.n_agents, -1) for x in dlstm_inputs], dim=1)

        return lstm_inputs, dlstm_inputs

    def _get_input_shapes(self, scheme):
        # print(scheme)
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        if self.args.obs_latent_state:
            input_shape += scheme["latent_state"]["vshape"]
        lstm_input_shape = input_shape
        dlstm_input_shape = input_shape
        if self.args.obs_last_action:
            lstm_input_shape += scheme["actions_onehot"]["vshape"][0]
            dlstm_input_shape += (scheme["keys"]["vshape"] * 2 + scheme["latent_state"][
                "vshape"])  # key_size + query_size + rule_size
        # if self.args.obs_other_hidden_state:  # FIXME: implement other hidden state as input
        #     lstm_input_shape += self.args.dilated_lstm_hidden_dim
        #     dlstm_input_shape += self.args.lstm_hidden_dim

        return lstm_input_shape, dlstm_input_shape

    def _get_output_shapes(self, scheme):
        lstm_output_shapes = scheme["actions_onehot"]["vshape"][0]
        dlstm_output_shapes = scheme["queries"]["vshape"] + scheme["keys"]["vshape"] + scheme["rules"]["vshape"]
        return lstm_output_shapes, dlstm_output_shapes
