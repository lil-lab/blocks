import logger
import time
import collections
import tensorflow as tf
import numpy as np
from abstract_learning import AbstractLearning
import replay_memory as rm
import generic_policy as gp


class PolicyGradient(AbstractLearning):

    def __init__(self, agent):
        self.agent = agent
        self.replay_memory = None
        self.batch_size = None
        self.null_previous_action = None
        self.ps = None

        # Compute loss function

        AbstractLearning.__init__(agent)

    def get_reinforce_self_critical_baseline(self):
        """ Returns baseline for self critical reinforce """

        # Create a queue to handle history of states
        state = collections.deque([], 5)
        # Add the dummy images
        dummy_images = self.image_embedder.get_dummy_images()
        [state.append(v) for v in dummy_images]

        # Receive the instruction and the environment
        (_, _, current_env, instruction, trajectory) = self.receive_instruction_and_image()
        state.append(current_env)
        (text_input_word_indices, text_mask) = self.text_embedder.get_word_indices_and_mask(instruction)
        logger.Log.info("=================\n " + ": Instruction: " + str(instruction) + "\n=================")

        block_id = int(trajectory[0] / 4.0)
        total_reward_episode = 0

        # Perform a roll out
        while True:
            # Compute the probability of the current state
            prob = self.evaluate_qfunction(state, text_input_word_indices, text_mask)

            # Sample from the prob. distribution
            action_id = self.test_policy(prob)

            action_str = self.message_protocol_kit.encode_action_from_pair(block_id, action_id)
            logger.Log.debug("Sending Message: " + action_str + " with probability " + str(prob[action_id]))
            self.connection.send_message(action_str)

            # receive reward and a new environment as a response on the completion of action
            (_, reward, new_env, is_reset) = self.receive_response_and_image()
            logger.Log.debug("Received reward: " + str(reward))

            state.append(new_env)  ##### CHECK if state is being overwritten

            # Update metric
            total_reward_episode += reward

            # Reset episode
            if self.message_protocol_kit.is_reset_message(is_reset):
                logger.Log.debug("Resetting the episode")
                self.connection.send_message("Ok-Reset")
                logger.Log.debug("Now waiting for response")

                logger.Log.flush()
                break  # stop the rollout

        return total_reward_episode

    def do_reinforce_learning_self_critical(self):
        """ Performs policy gradient learning using Reinforce on the Block World Task. The agent interacts with the
         simulator and performs roll-out followed by REINFORCE updates. """

        start = time.time()

        max_epoch = 1000
        dataset_size = 667
        tuning_size = int(0.05 * dataset_size)
        train_size = dataset_size - tuning_size
        logger.Log.info("REINFORCE: Max Epoch: " + str(max_epoch) + " Train/Tuning: "
                        + str(train_size) + "/" + str(tuning_size))

        # Saver for logging the model
        saver = tf.train.Saver(max_to_keep=120)

        # Iteration is the number of parameter update steps performed in the training
        iteration = 0

        # Reinforce baseline
        baseline = 0

        # Validation metric
        avg_bisk_metric = self.test(tuning_size)
        min_avg_bisk_metric = avg_bisk_metric
        patience = 0
        max_patience = 1000
        logger.Log.info("Tuning Data: (Before Training) Avg. Bisk Metric: " + str(avg_bisk_metric))

        for epoch in range(1, max_epoch + 1):
            logger.Log.info("=================\n Starting Epoch: " + str(epoch) + "\n=================")
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
                    # Compute the probability of the current state
                    prob = self.evaluate_qfunction(state, text_input_word_indices, text_mask)

                    # Sample from the prob. distribution
                    action_id = gp.GenericPolicy.sample_action_from_prob(prob)

                    action_str = self.message_protocol_kit.encode_action_from_pair(block_id, action_id)
                    logger.Log.debug("Sending Message: " + action_str + " with probability " + str(prob[action_id]))
                    self.connection.send_message(action_str)

                    # receive reward and a new environment as a response on the completion of action
                    (_, reward, new_env, is_reset) = self.receive_response_and_image()
                    logger.Log.debug("Received reward: " + str(reward))

                    # add to replay memory
                    replay_memory_item = rm.ReplayMemory(text_input_word_indices, text_mask,
                                                         state, action_id, reward, None, None, prob[action_id])
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

                        baseline = self.get_reinforce_self_critical_baseline()
                        logger.Log.info("Reward: " + " ".join([str(v) for v in rewards]) + " steps: " + str(steps))
                        logger.Log.info(" Total Reward: " + str(total_reward_episode) +
                                        ", Self Critical Baseline: " + str(baseline))

                        # Define the targets
                        for replay_memory_item in replay_memory_items:
                            replay_memory_item.set_target_retroactively(total_reward_episode - baseline)

                        self.replay_memory.clear()
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
                        logger.Log.info("Number of sample " + str(len(sample)) + " size of replay memory "
                                        + str(len(self.replay_memory)) + " loss = " + str(loss))

                        logger.Log.info("Total reward:" + str(total_reward_episode) + " Steps: " + str(steps))

                        # Print time statistics
                        total_time = time.time() - start
                        logger.Log.info("Total time: " + str(total_time))
                        logger.Log.flush()
                        break

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
