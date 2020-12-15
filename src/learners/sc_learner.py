class SCLearner:
    def __init__(self, mac, scheme, logger, args):

    def train(self, batch, t_env, epsidoe_num):

    def _train_agent(self, batch, rewards, terminated, actions, avail_actions, mask, bs, max_t):

    def _train_critic(self, batch, rewards, terminated, actions, avail_actions, mask, bs, max_t):

    def _update_targets(self):

    def cuda(self):

    def save_models(self, path):

    def load_models(self, path):
