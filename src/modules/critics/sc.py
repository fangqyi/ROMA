import torch
import torch.nn.functional as F
import torch.nn as nn

from utils.utils import identity, fanin_init, LayerNorm, product_of_gaussians, zeros, ones


class SCExecutionCritic(nn.Module):  # FIXME: Normalization across directional dims
    def __init__(self, scheme, args):
        super(SCExecutionCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_action
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        output_shape = self.args.latent_state_dim
        self.output_type = "dir"

        self.critic = args.critic_class(
            hidden_sizes=args.critic_hidden_sizes,
            input_size=input_shape,
            output_size=output_shape,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_params=None,
        )

    def forward(self, batch, t):
        inputs = self._build_inputs(batch, t)
        bs = batch.batch_size
        return self.critic(inputs).reshape(bs, self.n_agents, -1)

    def _build_inputs(self, batch, t):  # FIXME: t=None to speed up training
        # assume latent_state: [bs, latent_state_size]
        # obs: [bs, seq_len, n_agents, obs_size]
        bs = batch.batch_size
        ts = slice(t, t+1)
        inputs = []

        # latent_state
        latent_state = batch["latent_state"][:, ts].repeat(1, self.n_agents, 1)  # [bs, n_agents, lat_state_size]
        inputs.append(latent_state)

        # local_observations
        observation = batch["obs"][:, ts].squeeze(1)
        inputs.append(observation)

        # actions
        actions = batch["actions_onehot"][:, ts].squeeze(1) # [bs, n_agents, action_size]
        inputs.append(actions)

        # last actions
        if t == 0:
            last_actions = torch.zeros_like(actions)
        else:
            last_actions = batch["actions_onehot"][:, slice(t - 1, t)].squeeze(1)
        inputs.append(last_actions)

        # agent_id
        agent_id = torch.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1)
        inputs.append(agent_id)

        inputs = torch.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        return scheme["latent_state"]["vshape"] + scheme["actions_onehot"]["vshape"][0] * 2 + scheme["obs"]["vshape"] + self.n_agents


class SCControlCritic(nn.Module):
    def __init__(self, scheme, args):
        super(SCControlCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_action
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        output_shape = 1
        self.output_type = "q"

        self.critic = args.critic_class(
            hidden_sizes=args.critic_hidden_sizes,
            input_size=input_shape,
            output_size=output_shape,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_params=None,
        )

    def forward(self, batch, t):
        inputs = self._build_inputs(batch, t)
        bs = batch.batch_size
        return self.critic(inputs).reshape(bs, self.n_agents)

    def _build_inputs(self, batch, t):
        # assume latent_state: [bs, latent_state_size]
        # obs: [bs, seq_len, n_agents, obs_size]
        bs = batch.batch_size
        ts = slice(t, t + 1)
        inputs = []

        # latent_state
        latent_state = batch["latent_state"][:, ts].repeat(1, self.n_agents, 1)  # [bs, n_agents, lat_state_size]
        inputs.append(latent_state)

        # local_observations
        observation = batch["obs"][:, ts].squeeze(1)
        inputs.append(observation)

        # keys, queries and rules
        keys = batch["keys"][:, ts].squeeze(1)  # [bs, n_agents, comm_size]
        queries = batch["queries"][:, ts].squeeze(1)
        rules = batch["rules"][:, ts].squeeze(1)
        inputs.append(keys)
        inputs.append(queries)
        inputs.append(rules)

        # last actions
        if t == 0:
            last_keys = torch.zeros_like(keys)
            last_queries = torch.zeros_like(queries)
            last_rules = torch.zeros_like(rules)
        else:
            last_keys = batch["keys"][:, slice(t - 1, t)].squeeze(1)
            last_queries = batch["queries"][:, slice(t - 1, t)].squeeze(1)
            last_rules = batch["rules"][:, slice(t - 1, t)].squeeze(1)
        inputs.append(last_keys)
        inputs.append(last_queries)
        inputs.append(last_rules)

        # agent_id
        agent_id = torch.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1)
        inputs.append(agent_id)

        inputs = torch.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        return scheme["latent_state"]["vshape"]*2 + self.args.communication_query_and_signature_size * 2 + scheme["obs"][
            "vshape"] + self.n_agents







