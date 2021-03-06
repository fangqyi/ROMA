import torch

from runners import EpisodeRunner


class SCEpisodeRunner(EpisodeRunner):

    def __init__(self, args, logger):
        super(SCEpisodeRunner, self).__init__(args, logger)

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        self.mac.init_goals(batch_size=self.batch_size)

        while not terminated:
            cur_state = self.env.get_state()
            latent_state = self.mac.infer_latent_state(torch.tensor(cur_state).unsqueeze(0))
            latent_state_div = self.mac.compute_lat_state_kl_div()

            pre_transition_data = {
                "state": [cur_state],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
                "latent_state": latent_state,
                "latent_state_kl_div": [(latent_state_div,)]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size
            actions, control_outputs = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            queries, keys, rules = control_outputs

            reward, terminated, env_info = self.env.step(actions[0])  # to remove [batch_size]
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "queries": queries,
                "keys": keys,
                "rules": rules,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_state = self.env.get_state()
        latent_state = self.mac.infer_latent_state(torch.tensor(last_state).unsqueeze(0))
        latent_state_div = self.mac.compute_lat_state_kl_div()

        last_data = {
            "state": [last_state],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
            "latent_state": latent_state,
            "latent_state_kl_div": [(latent_state_div,)],
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions, control_outputs = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        queries, keys, rules = control_outputs

        self.batch.update({"actions": actions,
                           "queries": queries,
                           "keys": keys,
                           "rules": rules,
                           }, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch
