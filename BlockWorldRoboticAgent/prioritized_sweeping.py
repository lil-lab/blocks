import replay_memory
import random
import logger 


class PrioritizedSweeping:
    """ A class for sampling from a replay memory using prioritized sweeping algorithm. """

    def __init__(self, min_reward, rho):

        # Minimum value of reward required to set p = 1
        # for that datapoint. p = 0 for all other datapoints.
        self.min_reward = min_reward

        # Fraction of datapoints with p = 1 condition in each minibatch
        self.rho = rho

    def sample(self, replay_items, batch_size):
        """ Select a sample of size at most batch_size from replay_items using prioritized sweeping.
         The batch size will be batch_size or size of replay memory, whichever is smaller. Not repetition occurs
        during sampling. """

        pos_size = int(self.rho * batch_size)

        p = []
        for replay_item in replay_items:
            reward = replay_memory.ReplayMemory.get_reward(replay_item)
            if reward >= self.min_reward:
                p.append(replay_item)

        # pick v samples from p
        sample = random.sample(p, min(pos_size, len(p)))
        pos = len(sample)

        # pick the remaining from the p
        sample = sample + random.sample(replay_items, min(batch_size - len(sample), len(replay_items)))
        logger.Log.debug("Prioritized sweeping: fraction of positive reward: " + str(pos/float(len(sample))))

        return sample
