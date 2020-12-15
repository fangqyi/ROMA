from src.controllers import BasicMAC


class CommMAC(BasicMAC):
    def __init__(self):

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        chosen_actions = None
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):

    def init_hidden(self, batch_size):

    def parameters(self):

    def load_state(self, other_mac):

    def cuda(self):

    def save_models(self, path):

    def load_models(self, path):

    def _build_agents(self, input_shape):

    def _build_inputs(self, batch, t):

    def _get_input_shape(self, scheme):

