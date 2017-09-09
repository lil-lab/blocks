import numpy as np


class ReplayMemory:
    """ An item for performing reinforcement learning consisting of (list of previous states, action, reward, state').
        Additional target is used in different ways by different algorithm. For Q-learning the target represents
        r + gamma * max_a' Q(s',a'); for MLE this is generally 1 whenever defined and for policy gradient the value
       of target is the multiplicative factor the log-likelihood in logp(a|s) * target.
    """

    def __init__(self, instruction_word_indices, instruction_mask, history_state, action,
                 reward, end_env, target, action_prob=None, previous_action_id=None):
        self.instruction_word_indices = instruction_word_indices
        self.instruction_mask = instruction_mask
        self.history_state = np.concatenate(list(history_state), 2) # Copy the history to prevent overwriting
        self.action = action
        self.reward = reward
        self.end_env = end_env
        self.target = target
        self.action_prob = action_prob
        self.previous_action_id = previous_action_id

    def get_instruction_word_indices(self):
        return self.instruction_word_indices

    def get_instruction_mask(self):
        return self.instruction_mask

    def get_history_of_states(self):
        return self.history_state

    def get_action(self):
        return self.action

    def get_reward(self):
        return self.reward

    def get_end_env(self):
        return self.end_env

    def get_target(self):
        return self.target

    def get_previous_action_id(self):
        return self.previous_action_id

    def set_target_retroactively(self, target):
        if self.target is not None:
            print "Setting a target that is not none. Bug. Exiting"
            exit(0)
        self.target = target
