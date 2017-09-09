import random as rnd


class EpsilonGreedyPolicy:
    """ Given a Q-value defines an epsilon greedy policy over actions A.
    For all actions B with maximum Q-value the policy assigns a probability
    of 1/|B|(1 - epsilon/|C|) and remaining actions C = A - B get
    probability of epsilon/|C|. The epsilon keeps decaying using
    epsilon = e0/(1 + alpha * t) where t represents the t^th iteration.
    A minimum value of epsilon is used to maintain some level of exploration
    """

    def __init__(self, epsilon, min_epsilon):
        self.epsilon = epsilon
        self.rnd = rnd.Random()
        self.step = 0
        self.min_epsilon = min_epsilon
        self.current_epsilon = epsilon
        if self.current_epsilon < min_epsilon:
            self.below_min_epsilon = True
        else:
            self.below_min_epsilon = False

        # alpha for decaying epsilon as epsilon_0/(1 + alpha * step)
        # The value below decreases epsilon from 1.0 to ~0.1 in 1M steps.
        reach_min_epsilon_steps = 500000
        if min_epsilon > 0.0:
            self.alpha = (epsilon/min_epsilon - 1.0)/float(reach_min_epsilon_steps)
        else:
            self.alpha = (epsilon/0.01 - 1.0)/float(reach_min_epsilon_steps)

    def decay_epsilon(self):
        if self.below_min_epsilon:
            print "Reached min. epsilon"
            return self.min_epsilon

        self.step += 1
        self.current_epsilon = self.epsilon / float(1 + self.alpha * self.step)
        if self.current_epsilon < self.min_epsilon:
            self.current_epsilon = self.min_epsilon
            self.below_min_epsilon = True

        print "Current epsilon " + str(self.current_epsilon) + " step " + str(self.step)

    def get_action(self, q_val):

        num_actions = len(q_val)

        if num_actions == 0:
            raise AssertionError("There must be atleast one action.")

        ix_max = [0]
        for i in range(1, num_actions):
            if q_val[i] > q_val[ix_max[0]]:
                ix_max[:] = [i]
            elif q_val[i] == q_val[ix_max[0]]:
                ix_max.append(i)

        epsilon = self.current_epsilon

        min_action_prob = epsilon/float(num_actions)
        max_action_prob = min_action_prob + (1 - epsilon)/float(len(ix_max))

        prob = [0.0] * num_actions

        for i in range(0, num_actions):
            if i in ix_max:
                prob[i] = max_action_prob
            else:
                prob[i] = min_action_prob

        v = self.rnd.random()

        for i in range(0, num_actions):
            v -= prob[i]
            if v <= 0:
                return i

        return num_actions - 1

    def get_action_subset(self, q_val, action_subset):
        """ Returns an action from the subset (S) using epsilon greedy over the subset """
        num_actions = len(action_subset)

        if num_actions == 0:
            raise AssertionError("There must be atleast one action.")

        ix_max = [action_subset[0]]
        for i in action_subset[1:]:
            if q_val[i] > q_val[ix_max[0]]:
                ix_max[:] = [i]
            elif q_val[i] == q_val[ix_max[0]]:
                ix_max.append(i)

        epsilon = self.current_epsilon

        min_action_prob = epsilon / float(num_actions)
        max_action_prob = min_action_prob + (1 - epsilon) / float(len(ix_max))

        prob = [0.0] * num_actions

        for i in range(0, num_actions):
            if i in ix_max:
                prob[i] = max_action_prob
            else:
                prob[i] = min_action_prob

        v = self.rnd.random()

        for i in range(0, num_actions):
            v -= prob[i]
            if v <= 0:
                return action_subset[i]

        return action_subset[-1]

    def get_argmax_action(self, q_val):
        """ Returns argmax_a qVal(a, s) """

        num_actions = len(q_val)

        if num_actions == 0:
            raise AssertionError("There must be atleast one action.")

        ix_max = [0]
        for i in range(1, num_actions):
            if q_val[i] > q_val[ix_max[0]]:
                ix_max[:] = [i]
            elif q_val[i] == q_val[ix_max[0]]:
                ix_max.append(i)

        return ix_max[self.rnd.randint(0, len(ix_max) - 1)]

    def get_argmax_action_subset(self, q_val, subset_actions):
        """ Returns argmax_a qVal(a, s) over subset (S) """

        num_actions = len(subset_actions)

        if num_actions == 0:
            raise AssertionError("There must be atleast one action.")

        ix_max = [subset_actions[0]]
        for i in subset_actions[1:]:
            if q_val[i] > q_val[ix_max[0]]:
                ix_max[:] = [i]
            elif q_val[i] == q_val[ix_max[0]]:
                ix_max.append(i)

        return ix_max[self.rnd.randint(0, len(ix_max) - 1)]
