import torch
import torch.nn.functional as F
import torch.nn as nn

from src.utils.utils import identity, fanin_init, LayerNorm, product_of_gaussians, zeros, ones


class SCCritic(nn.Module):
    def __init__(self, scheme, args):
        super(SCCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_action
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "q"

        self.critic = args.critic_class(
            hidden_sizes=args.critic_hidden_sizes,
            input_size=input_shape,
            output_size=1,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_params=None,
        )

    def forward(self, batch, latent_state, t):
        inputs = self._build_inputs(batch, latent_state)
        return self.critic(inputs)

    def _build_inputs(self, batch, latent_state, t):
        bs = batch.batch_size
        ts = slice(t, t+1)
        inputs = [latent_state]
        # inputs.append(batch["latent_state"][:, ts].squeeze(1))

        # actions a
        actions = batch["actions_onehot"][:, ts].view(bs, 1, 1, -1).squeeze(1).squeeze(1)
        inputs.append(actions)

        # last actions
        if t == 0:
            last_actions = torch.zeros_like(actions)
        else:
            last_actions = batch["actions_onehot"][:, slice(t - 1, t)].view(bs, 1, 1, -1).squeeze(1).squeeze(1)
        inputs.append(last_actions)

        inputs = torch.cat([x.reshape(bs, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        return scheme["latent_state"]["vshape"] + scheme["actions_onehot"]["vshape"][0] * self.n_agents * 2








