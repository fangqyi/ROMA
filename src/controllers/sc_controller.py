import torch
import numpy as np

from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY

class SCMAC():
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shapes = self._get_input_shapes(scheme)
        output_shapes = self._get_output_shapes(scheme)
        self._build_agents(input_shapes, output_shapes)
        self.agent_local_output_type = args.agent_local_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.query_size = scheme["query"]["vshape"]
        self.key_size = scheme["key"]["vshape"]
        self.rule_size = scheme["rule"]["vshape"]
        self.hidden_states = None
        self.goals = None
        self.latent_state_encoder = args.latent_state_encoder_class(
                 input_size=scheme["state"]["vshape"],
                 output_size=scheme["latent_state"]["vshape"],
                 mlp_hidden_sizes=args.latent_state_encoder_hidden_sizes)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                            test_mode=test_mode)
        return chosen_actions

    def init_goals(self, ep_batch):
        self.goals = [torch.zeros([ep_batch.batch_size, self.rule_size])] * self.n_agents

    def infer_latent_state(self, state):
        return self.latent_state_encoder(state)

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
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
        self.goals = []
        for i in range(self.n_agents):
            query_iT = torch.transpose(queries[:, i].squeeze(1), dim0=0, dim1=1)
            qk = torch.tensor([torch.dot(query_iT, keys[:, j].squeeze(1))/np.sqrt(self.args.attention_noramlization_dk)
                  for j in range(self.n_agents)])
            a_i = torch.nn.functional.softmax(qk, dim=1)
            goal_i = torch.stack([a_i[:,j].squeeze(1) * qk[:, j].squeeze(1) for j in range(self.n_agents)], dim=0).sum(dim=0)
            self.goals.append(goal_i)

        return actions

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden(batch_size)

    def parameters(self):
        return self.agent.parameters()  # assume the check is recursive

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

        if self.args.obs_agent_id:
            lstm_inputs.append(torch.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
            dlstm_inputs.append(torch.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        lstm_inputs = torch.cat([x.reshape(bs * self.n_agents, -1) for x in lstm_inputs], dim=1)
        dlstm_inputs = torch.cat([x.reshape(bs * self.n_agents, -1) for x in dlstm_inputs], dim=1)

        return lstm_inputs, dlstm_inputs

    def _get_input_shapes(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        if self.args.obs_latent_space:
            input_shape += scheme["latent_state"]["vshape"]
        lstm_input_shape = input_shape
        dlstm_input_shape = input_shape
        if self.args.obs_other_hidden_state:
            lstm_input_shape += lstm_input_shape + self.args.dilated_lstm_hidden_dim
            dlstm_input_shape += dlstm_input_shape + self.args.lstm_hidden_dim
        return lstm_input_shape, dlstm_input_shape

    def _get_output_shapes(self, scheme):
        lstm_output_shapes = scheme["actions_onehot"]["vshape"]
        dlstm_output_shapes = scheme["query"]["vshape"] + scheme["key"]["vshape"] + scheme["rule"]["vshape"]
        return lstm_output_shapes, dlstm_output_shapes