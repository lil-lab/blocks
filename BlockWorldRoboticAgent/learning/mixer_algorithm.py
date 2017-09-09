import logger
import time
import collections
import tensorflow as tf
import numpy as np
from abstract_learning import AbstractLearning
import replay_memory as rm


class MixerAlgorithm(AbstractLearning):

    def __init__(self, agent):
        self.agent = agent
        self.replay_memory = None
        self.batch_size = None
        self.null_previous_action = None
        self.ps = None
        AbstractLearning.__init__(agent)

    def train(self):
        """ Performs MIXER (Mixed Incremental Cross-Entropy Reinforce) learning of Ranzato et al. 2016.
        Maintains a curriculum schedule T' which is gradually annealed. The agent follows oracle trajectory
        for T' steps and does MLE and for remaining T-T' steps it does REINFORCE algorithm. Another hyperparameter,
        decides the number of epochs until which only MLE has to be done. """

        start = time.time()

        max_epoch = 1000
        dataset_size = 667
        tuning_size = int(0.05 * dataset_size)
        train_size = dataset_size - tuning_size
        logger.Log.info("MIXER: Max Epoch: " + str(max_epoch) + " Train/Tuning: "
                        + str(train_size) + "/" + str(tuning_size))

        # Saver for logging the model
        saver = tf.train.Saver(max_to_keep=25)

        # Iteration is the number of parameter update steps performed in the training
        iteration = 0

        # Reinforce baseline
        baseline = 0

        # Validation metric
        avg_bisk_metric = self.agent.test(tuning_size)
        min_avg_bisk_metric = avg_bisk_metric
        patience = 0
        max_patience = 20
        follow_mle_first = 5
        t_schedule = 40
        logger.Log.info("Tuning Data: (Before Training) Avg. Bisk Metric: " + str(avg_bisk_metric))

        for epoch in range(1, max_epoch + 1):
            logger.Log.info("=================\n Starting Epoch: " + str(epoch) + ". Curriculum\n=================")

            # Get the curriculum schedule
            if epoch > follow_mle_first:
                t_schedule = max(0, t_schedule - 2)

            for data_point in range(1, train_size + 1):

                # Create a queue to handle history of states
                state = collections.deque([], 5)
                # Add the dummy images
                dummy_images = self.image_embedder.get_dummy_images()
                [state.append(v) for v in dummy_images]

                # Receive the instruction and the environment
                (_, _, current_env, instruction, trajectory) = self.receive_instruction_and_image()
                state.append(current_env)
                (text_input_word_indices, text_mask) = self.text_embedder.get_word_indices_and_mask(instruction)
                logger.Log.info("=================\n " + str(data_point) + ": Instruction: "
                                + str(instruction) + "\n=================")

                block_id = int(trajectory[0] / 4.0)
                total_reward_episode = 0
                steps = 0

                # Reinforce requires sampling from Q-function for the future.
                # So we cannot directly add entries to the global replay memory.
                replay_memory_items = []
                rewards = []

                # Perform a roll out
                while True:
                    if steps < t_schedule: # Follow the gold trajectory
                        follow_gold = True
                        action_id = trajectory[steps]
                        if action_id == self.num_actions - 1:
                            action_id = 4
                        else:
                            action_id %= 4
                    else: # Do exploration
                        follow_gold = False
                        # Compute the probability of the current state
                        prob = self.evaluate_qfunction(state, text_input_word_indices, text_mask)

                        # Sample from the prob. distribution
                        action_id = gp.GenericPolicy.sample_action_from_prob(prob)

                    action_str = self.message_protocol_kit.encode_action_from_pair(block_id, action_id)
                    logger.Log.debug("Sending Message: " + action_str)
                    self.connection.send_message(action_str)

                    # receive reward and a new environment as a response on the completion of action
                    (_, reward, new_env, is_reset) = self.receive_response_and_image()
                    logger.Log.debug("Received reward: " + str(reward))

                    # add to replay memory
                    if follow_gold:
                        replay_memory_item = rm.ReplayMemory(text_input_word_indices, text_mask,
                                                             state, action_id, 1.0, None, 1.0)
                    else:
                        replay_memory_item = rm.ReplayMemory(text_input_word_indices, text_mask,
                                                             state, action_id, 1.0, None, None)
                    replay_memory_items.append(replay_memory_item)
                    rewards.append(reward)

                    state.append(new_env)  ##### CHECK if state is being overwritten

                    # Update metric
                    total_reward_episode += reward
                    steps += 1

                    # Reset episode
                    if self.message_protocol_kit.is_reset_message(is_reset):
                        logger.Log.debug("Resetting the episode")
                        self.connection.send_message("Ok-Reset")
                        logger.Log.debug("Now waiting for response")

                        # Compute monte carlo q values
                        monte_carlo_q_val = [0] * steps     ###STOP and terminal state, not accounted for ever
                        for i in range(steps - 1, -1, -1):
                            if i == steps - 1:
                                monte_carlo_q_val[i] = rewards[i]
                            else:
                                monte_carlo_q_val[i] = monte_carlo_q_val[i + 1] + rewards[i]

                        #baseline = min(monte_carlo_q_val)
                        logger.Log.info("Reward: " + " ".join([str(v) for v in rewards]) + " steps: " + str(steps))
                        logger.Log.info("Cummulative Reward: " + " ".join([str(v) for v in monte_carlo_q_val]) +
                                        "Baseline: " + str(baseline))

                        # Define the targets
                        for replay_memory_item, cumm_reward in zip(replay_memory_items, monte_carlo_q_val):
                            # If the target has not been set then it came from REINFORCE
                            if rm.ReplayMemory.get_target(replay_memory_item) is None:
                                replay_memory_item.set_target_retroactively(cumm_reward - baseline)

                        #self.replay_memory.clear() ######## HACK
                        for replay_memory_item in replay_memory_items:
                            self.replay_memory.appendleft(replay_memory_item)

                        # Perform minibatch SGD
                        # Pick a sample using prioritized sweeping and perform backpropagation
                        sample = self.ps.sample(self.replay_memory, self.batch_size)
                        loss = self.min_loss(sample)
                        if np.isnan(loss):
                            logger.Log.error("NaN found. Exiting")
                            exit(0)
                        iteration += 1
                        logger.Log.info("Number of sample " + str(len(sample)) + " size of replay memory " \
                              + str(len(self.replay_memory)) + " loss = " + str(loss))

                        logger.Log.info("Total reward:" + str(total_reward_episode) + " Steps: " + str(steps))

                        # Save the model after every few iterations
                        if iteration > 0 and iteration % 2000 == 0:
                            save_path = saver.save(self.sess, "./saved/model_iteration_" + str(iteration) + "_.ckpt")
                            logger.Log.info("Model saved in file: " + str(save_path))

                        # Print time statistics
                        total_time = time.time() - start
                        logger.Log.info("Total time: " + str(total_time))

                        logger.Log.flush()
                        break #stop the rollout

            # Compute validation accuracy
            avg_bisk_metric = self.test(tuning_size)
            logger.Log.info("Tuning Data: (end of epoch " + str(epoch) + ") Avg. Bisk Metric: " +
                            str(avg_bisk_metric) + "Min was " + str(min_avg_bisk_metric))
            # Save the model
            save_path = saver.save(self.sess, "./saved/model_epoch_" + str(epoch) + ".ckpt")
            logger.Log.info("Model saved in file: " + str(save_path))

            if avg_bisk_metric >= min_avg_bisk_metric:
                if patience == max_patience:
                    logger.Log.info("Max patience reached. Terminating learning after " + str(epoch) +
                                    " epochs and " + str(iteration) + " iterations.")
                    break
                else:
                    logger.Log.info(
                        "Tuning accuracy did not improve. Increasing patience to " + str(patience + 1))
                    patience += 1
            else:
                logger.Log.info("Resetting patience to 0")
                patience = 0
            min_avg_bisk_metric = min(min_avg_bisk_metric, avg_bisk_metric)

        logger.Log.close()