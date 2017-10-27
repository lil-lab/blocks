import logger
import time
import collections
import tensorflow as tf
import numpy as np
from abstract_learning import AbstractLearning
import replay_memory as rm
import prioritized_sweeping


class MaximumLikelihoodEstimation(AbstractLearning):
    """ Performs maximum likelhood estimation which is the following optimization:
                max_theta log policy(gold-action | agent-state; theta)
    """

    def __init__(self, agent, policy_model):
        self.agent = agent
        self.policy_model = policy_model

        # Replay memory
        max_replay_memory_size = 2000
        self.replay_memory = collections.deque(maxlen=max_replay_memory_size)
        rho = 0.5
        self.ps = prioritized_sweeping.PrioritizedSweeping(0, rho)

        optimizer = tf.train.AdamOptimizer(self.mle_learning_rate)
        loss = MaximumLikelihoodEstimation.calc_loss(
            self.policy_model.model_output, self.policy_model.model_output_indices)

        using_grad_clip = True
        grad_clip_val = 5.0
        if not using_grad_clip:
            train_step = optimizer.minimize(loss)
        else:
            gvs = optimizer.compute_gradients(loss)
            capped_gvs = [(tf.clip_by_norm(grad, grad_clip_val), var)
                          if grad is not None else (grad, var) for grad, var in gvs]
            train_step = optimizer.apply_gradients(capped_gvs)

        # Create summaries for training
        summary_loss = tf.scalar_summary("Loss", loss)
        update_summaries = [summary_loss]

        AbstractLearning.__init__(self, policy_model, loss, train_step, update_summaries)

    @staticmethod
    def calc_loss(action_output, action_indices):
        block_prob, direction_prob = action_output
        block_indices, direction_indices = action_indices

        # Loss is negative log-likelihood.
        block_indices_flattened = tf.range(0, tf.shape(block_prob)[0]) * tf.shape(block_prob)[1] + block_indices
        block_prob_action = tf.gather(tf.reshape(block_prob, [-1]), block_indices_flattened)

        direction_indices_flattened = \
            tf.range(0, tf.shape(direction_prob)[0]) * tf.shape(direction_prob)[1] + direction_indices
        direction_prob_action = tf.gather(tf.reshape(direction_prob, [-1]), direction_indices_flattened)

        neg_log_prob = -tf.add(tf.log(block_prob_action + 0.000001),
                               tf.log(direction_prob_action + 0.000001))

        loss = tf.reduce_mean(neg_log_prob)
        return loss

    def train(self, sess, train_writer, max_epoch=AbstractLearning.max_epochs, model_name="./model", terminate=True):
        """ Performs supervised learning on the Block World Task. The agent interacts with the
         simulator and performs roll-out followed by supervised learning. """

        start = time.time()

        dataset_size = AbstractLearning.dataset_size
        tuning_size = AbstractLearning.validation_datasize
        train_size = dataset_size - tuning_size
        logger.Log.info("Maximum Likelihood: Max Epoch: " + str(max_epoch) + " Train/Tuning: "
                        + str(train_size) + "/" + str(tuning_size))

        # Saver for logging the model
        saver = tf.train.Saver(max_to_keep=AbstractLearning.models_to_keep)

        # Iteration is the number of parameter update steps performed in the training
        iteration = 0

        # Validation metric
        avg_bisk_metric = self.agent.test(tuning_size, oracle=True)
        min_avg_bisk_metric = avg_bisk_metric
        patience = 0
        max_patience = AbstractLearning.max_patience
        logger.Log.info("Tuning Data: (Before Training) Avg. Bisk Metric: " + str(avg_bisk_metric))

        for epoch in range(1, max_epoch + 1):
            logger.Log.info("=================\n Starting Epoch: "
                            + str(epoch) + "\n=================")

            for data_point in range(1, train_size + 1):

                # Create a queue to handle history of states
                state = collections.deque([], 5)
                # Add the dummy images
                dummy_images = self.policy_model.image_embedder.get_padding_images()
                [state.append(v) for v in dummy_images]

                # Receive the instruction and the environment
                (_, _, current_env, instruction, trajectory) = self.agent.receive_instruction_and_image()
                state.append(current_env)
                (text_input_word_indices, text_mask) = \
                    self.policy_model.text_embedder.get_word_indices_and_mask(instruction)
                logger.Log.info("=================\n " + str(data_point) + ": Instruction: "
                                + str(instruction) + "\n=================")

                traj_ix = 0
                total_reward_episode = 0
                steps = 0
                previous_action = self.policy_model.null_previous_action
                block_id = int(trajectory[0] / 4.0)

                # Perform a roll out
                while True:
                    # Sample from the prob. distribution
                    action_id = trajectory[traj_ix]
                    traj_ix += 1
                    action_str = self.agent.message_protocol_kit.encode_action(action_id)
                    logger.Log.debug("Sending Message: " + action_str)
                    self.agent.connection.send_message(action_str)

                    # receive reward and a new environment as a response on the completion of action
                    (status_code, reward, new_env, is_reset) = self.agent.receive_response_and_image()
                    logger.Log.debug("Received reward: " + str(reward))

                    # add to replay memory
                    if action_id == 80:
                        direction_id = 4
                    else:
                        direction_id = action_id % 4
                    replay_memory_item = rm.ReplayMemory(text_input_word_indices, text_mask,
                                                         state, (block_id, direction_id), 1.0, new_env, None,
                                                         previous_action_id=previous_action)
                    self.replay_memory.appendleft(replay_memory_item)
                    state.append(new_env)

                    # Update metric
                    total_reward_episode += reward
                    steps += 1
                    previous_action = (direction_id, block_id)

                    # Reset episode
                    if self.agent.message_protocol_kit.is_reset_message(is_reset):
                        logger.Log.debug("Resetting the episode")
                        self.agent.connection.send_message("Ok-Reset")
                        logger.Log.debug("Now waiting for response")

                        # Perform minibatch SGD
                        # Pick a sample using prioritized sweeping and perform backpropagation
                        sample = self.ps.sample(self.replay_memory, self.batch_size)
                        loss = self.min_loss(sample, sess, train_writer)
                        if np.isnan(loss):
                            logger.Log.info("NaN found. Exiting")
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

            # Save the model
            save_path = saver.save(sess, "./saved/" + str(model_name) + "_epoch_" + str(epoch) + ".ckpt")
            logger.Log.info("Model saved in file: " + str(save_path))

            if epoch < max_epoch or not terminate:
                # Compute validation accuracy
                avg_bisk_metric = self.agent.test(tuning_size)
                logger.Log.info("Tuning Data: (end of epoch " + str(epoch) + ") Avg. Bisk Metric: "
                                + str(avg_bisk_metric) + "Min was " + str(min_avg_bisk_metric))

                if avg_bisk_metric >= min_avg_bisk_metric:
                    if patience == max_patience:
                        logger.Log.info("Max patience reached. Terminating learning after " + str(epoch) +
                                        " epochs and " + str(iteration) + " iterations.")
                        break
                    else:
                        logger.Log.info("Tuning accuracy did not improve. Increasing patience to " + str(patience + 1))
                        patience += 1
                else:
                    logger.Log.info("Resetting patience to 0")
                    patience = 0
                min_avg_bisk_metric = min(min_avg_bisk_metric, avg_bisk_metric)

        if terminate:
            logger.Log.close()
