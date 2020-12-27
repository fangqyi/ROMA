import torch
import numpy as np
import torch.nn.functional as F

from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY

from src.utils.utils import identity, fanin_init


class SCMAC():
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shapes, dec_latent_state_input_shape = self._get_input_shapes(scheme)
        output_shapes, dec_latent_state_output_shape = self._get_output_shapes(scheme)
        self._build_agents(input_shapes, output_shapes)
        self.agent_local_output_type = args.agent_local_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.query_size = scheme["query"]["vshape"]
        self.key_size = scheme["key"]["vshape"]
        self.rule_size = scheme["rule"]["vshape"]
        self.agent_hidden_states = None
        self.goals = []
        self.cur_dec_state = None
        self.latent_state_encoder = args.latent_state_encoder_class(
                 input_size=scheme["state"]["vshape"],
                 output_size=scheme["latent_state"]["vshape"],
                 mlp_hidden_sizes=args.latent_state_encoder_hidden_sizes)
        self.dec_lat_state_func = args.decomposed_state_estimator_class(
            hidden_sizes=args.decomposed_latent_state_estimator_hidden_sizes,
            input_size=dec_latent_state_input_shape,
            output_size=dec_latent_state_output_shape,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_params=None,
        )


    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                            test_mode=test_mode)
        self.update_dec_lat_state(ep_batch, t_ep, chosen_actions)
        return chosen_actions

    def compute_dec_lat_state_kl_div(self):
        if self.args.use_dec_lat_state_information_bottleneck:
            return self.dec_lat_state_func.compute_kl_div()
        else:
            raise Exception("Current decomposed latent state estimator does not support information bottleneck")

    def compute_lat_state_kl_div(self):
        if self.args.use_lat_state_information_bottleneck:
            return self.latent_state_encoder.compute_kl_div()
        else:
            raise Exception("Current decomposed latent state estimator does not support information bottleneck")

    def get_dec_lat_state(self, bs=slice(None)):
        return self.cur_dec_state[bs]

    def init_goals(self, ep_batch):
        """
        must call before running an episode, self.goals = [timestep 0: [agent_id][batch_id][rule_dim], ...]
        Args:
            ep_batch:
        """
        goal_init = [torch.zeros([ep_batch.batch_size, self.rule_size])] * self.n_agents
        goal_init = torch.stack(goal_init, dim=0)  # [n_agent][batch_num][rule_dim]
        self.goals = [goal_init]

    def infer_latent_state(self, state):
        """

        Args:
            state:

        Returns:

        """
        return self.latent_state_encoder(state)

    def update_dec_lat_state(self, batch, t, actions):
        inputs = self._build_lat_state_inputs(batch, t, actions)
        self.cur_dec_state = self.dec_lat_state_func(inputs).view(batch.batch_size, self.n_agents, -1)

    def get_current_goal(self):
        """
        get current goal: mean cumulative goals over the horizon length
        Returns: torch.tensor [curr_episode_num][n_agents][batch_num][rule_dim]

        """
        if len(self.goals) < self.args.horizon:
            goals_sum = torch.stack(self.goals, dim=0)  # [curr_episode_num][batch_num][n_agents][rule_dim]
        else:
            goals_sum = torch.stack([self.goals[-i] for i in range(1, self.args.horizon+1)], dim=0)
        cur_goal = torch.mean(goals_sum, dim=0)
        return cur_goal  # [batch_num][n_agents][rule_dim]

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
        goals = self.get_current_goal().view(ep_batch.batch_size*self.n_agents, -1)
        agent_outs, self.agent_hidden_states = self.agent(agent_inputs, self.agent_hidden_states, goals)
        local_outs, horizon_outs = agent_outs

        if self.agent_local_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                local_outs[reshaped_avail_actions == 0] = -1e10

            local_outs = torch.nn.functional.softmax(local_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                local_outs = ((1 - self.action_selector.epsilon) * local_outs
                               + torch.ones_like(local_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    local_outs[reshaped_avail_actions == 0] = 0.0

        actions = local_outs.view(ep_batch.batch_size, self.n_agents, -1)

        queries, keys, rules = horizon_outs.view(ep_batch.batch_size, self.n_agents, -1).split([self.query_size,
                                                                                                self.key_size,
                                                                                                self.rule_size], dim=2)
        rules = torch.norm(rules, dim=1)

        goals_t = []
        for i in range(self.n_agents):
            query_iT = torch.transpose(queries[:, i].squeeze(1), dim0=0, dim1=1)
            qk = torch.tensor([torch.dot(query_iT, keys[:, j].squeeze(1))/np.sqrt(self.args.attention_noramlization_dk)
                  for j in range(self.n_agents)])
            a_i = torch.nn.functional.softmax(qk, dim=1)
            goal_i = torch.stack([a_i[:, j].squeeze(1) * rules[:, j].squeeze(1) for j in range(self.n_agents)], dim=0).sum(dim=0)
            goals_t.append(goal_i)
        goals_t = torch.stack(goals_t, dim=1)  # [batch][n_agents][rule_dim]
        self.goals.append(goals_t)

        return actions

    def init_hidden(self, batch_size):
        self.agent_hidden_states = self.agent.init_hidden(batch_size)

    def parameters(self):
        return self.agent.parameters() + self.latent_state_encoder.parameters() # assume the check is recursive

    def agent_lstm_parameters(self):
        return self.agent.lstm_parameters()

    def agent_dlstm_parameters(self):
        return self.agent.dlstm_parameters() + self.latent_state_encoder.parameters()

    def dec_lac_state_func_parameters(self):
        return self.dec_lat_state_func.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        torch.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(torch.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shapes, output_shapes):
        if self.args.agent is not "sc":
            raise Exception("Agent type not compatible: must be SCAgent")
        self.agent = agent_REGISTRY[self.args.agent](input_shapes, output_shapes)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        lstm_inputs = []
        dlstm_inputs = []

        if self.args.obs_last_action:
            if t == 0:
                lstm_inputs.append(torch.zeros_like(batch["actions_onehot"][:, t]))
                dlstm_inputs.append(torch.zeros_like(batch["actions_onehot"][:, t]))
            else:
                lstm_inputs.append(batch["actions_onehot"][:, t-1])
                dlstm_inputs.append(batch["actions_onehot"][:, t-1])
        lstm_inputs.append(batch["obs"][:, t])
        dlstm_inputs.append(batch["obs"][:, t])

        if self.args.obs_latent_state:
            lstm_inputs.append(batch["latent_state"][:, t])
            dlstm_inputs.append(batch["latent_state"][:, t])
        if self.args.obs_agent_id:
            lstm_inputs.append(torch.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
            dlstm_inputs.append(torch.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        lstm_inputs = torch.cat([x.reshape(bs * self.n_agents, -1) for x in lstm_inputs], dim=1)
        dlstm_inputs = torch.cat([x.reshape(bs * self.n_agents, -1) for x in dlstm_inputs], dim=1)

        return lstm_inputs, dlstm_inputs

    def _build_lat_state_inputs(self, batch, t, actions):
        bs = batch.batch_size
        inputs = [batch["obs"][:, t], batch["latent_state"][:, t], actions]
        if self.args.dec_lat_state_last_action:
            inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(torch.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        inputs = torch.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)

        return inputs

    def _get_input_shapes(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        if self.args.obs_latent_state:
            input_shape += scheme["latent_state"]["vshape"]
        lstm_input_shape = input_shape
        dlstm_input_shape = input_shape
        if self.args.obs_other_hidden_state:
            lstm_input_shape += lstm_input_shape + self.args.dilated_lstm_hidden_dim
            dlstm_input_shape += dlstm_input_shape + self.args.lstm_hidden_dim

        dec_latent_input_shape = scheme["obs"]["vshape"] + scheme["latent_state"]["vshape"] + scheme["actions_onehot"][
            "vshape"]
        if self.args.dec_lat_state_last_action:
            dec_latent_input_shape += scheme["actions_onehot"]["vshape"]
        if self.args.obs_agent_id:
            dec_latent_input_shape += self.n_agents

        return (lstm_input_shape, dlstm_input_shape), dec_latent_input_shape

    def _get_output_shapes(self, scheme):
        lstm_output_shapes = scheme["actions_onehot"]["vshape"]
        dlstm_output_shapes = scheme["query"]["vshape"] + scheme["key"]["vshape"] + scheme["rule"]["vshape"]
        dec_state_output_shapes = scheme["latent_state"]["vshape"]
        return (lstm_output_shapes, dlstm_output_shapes), dec_state_output_shapes