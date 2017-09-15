import logger
import time
import collections
import tensorflow as tf
import numpy as np
from abstract_learning import AbstractLearning
import replay_memory as rm
import generic_policy as gp
from learning.ml_estimation import MaximumLikelihoodEstimation


class PolicyGradient(AbstractLearning):
    """ Policy gradient method with contextual bandit setting """

    def __init__(self, agent, policy_model, total_reward):
        self.agent = agent
        self.policy_model = policy_model
        self.total_reward = total_reward

        # Compute MLE loss function. MLE is used to initialize parameters for policy gradient
        self.mle_policy_gradient = MaximumLikelihoodEstimation(agent, policy_model)

        # Compute loss function
        loss, entropy_penalty = self.calc_loss(
            self.policy_model.model_output, self.policy_model.model_output_indices, self.policy_model.target)

        optimizer = tf.train.AdamOptimizer(AbstractLearning.rl_learning_rate)

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
        summary_target_min = tf.scalar_summary("Target Min", tf.reduce_min(self.policy_model.target))
        summary_target_max = tf.scalar_summary("Target Max", tf.reduce_max(self.policy_model.target))
        summary_target_mean = tf.scalar_summary("Target Mean", tf.reduce_mean(self.policy_model.target))
        summary_entropy_penalty = tf.scalar_summary("Entropy Penalty", entropy_penalty)
        update_summaries = [summary_loss, summary_target_min,
                            summary_target_max, summary_target_mean, summary_entropy_penalty]

        AbstractLearning.__init__(self, policy_model, loss, train_step, update_summaries)

    @staticmethod
    def calc_loss(action_output, action_indices, target):
        """ Policy Gradient creates a MLE loss and REINFORCE loss """

        block_prob, direction_prob = action_output
        block_indices, direction_indices = action_indices

        # Create MLE loss and train step
        block_indices_flattened = tf.range(0, tf.shape(block_prob)[0]) * tf.shape(block_prob)[1] + block_indices
        block_prob_action = tf.gather(tf.reshape(block_prob, [-1]), block_indices_flattened)

        direction_indices_flattened = tf.range(0, tf.shape(direction_prob)[0]) * tf.shape(direction_prob)[
            1] + direction_indices
        direction_prob_action = tf.gather(tf.reshape(direction_prob, [-1]), direction_indices_flattened)

        neg_log_prob = -tf.add(tf.log(block_prob_action + 0.000001),
                               tf.log(direction_prob_action + 0.000001))

        # Create Reinforce loss and train step
        # multiply by the weights
        wt_neg_log_prob = tf.mul(neg_log_prob, target)
        loss = tf.reduce_mean(wt_neg_log_prob)
        # Add entropy regularization
        lentropy = 0.1
        block_entropy = -tf.reduce_sum(tf.mul(block_prob, tf.log(block_prob + 0.000001)), 1)
        direction_entropy = -tf.reduce_sum(tf.mul(direction_prob, tf.log(direction_prob + 0.000001)), 1)
        entropy = tf.add(block_entropy, direction_entropy)
        entropy_penalty = tf.reduce_mean(entropy, 0)
        loss -= lentropy * entropy_penalty

        return loss, entropy_penalty

    def train(self, sess, train_writer, max_epoch=AbstractLearning.max_epochs, model_name="./model"):
        """ Performs policy gradient learning using Reinforce on the Block World Task. The agent interacts with the
         simulator and performs roll-out followed by REINFORCE updates. """

        start = time.time()

        # Initialization using 2 epochs of MLE
        self.mle_policy_gradient.train(sess, train_writer, max_epoch=2, model_name="./model_mle")
        # Reinitialize the direction parameters
        w1, b1 = self.policy_model.mix_and_gen_prob.get_direction_weights()
        sess.run(tf.initialize_variables([w1, b1]))

        dataset_size = AbstractLearning.dataset_size
        tuning_size = AbstractLearning.validation_datasize
        train_size = dataset_size - tuning_size
        logger.Log.info("REINFORCE: Max Epoch: " + str(max_epoch) + " Train/Tuning: "
                        + str(train_size) + "/" + str(tuning_size))

        # Saver for logging the model
        saver = tf.train.Saver(max_to_keep=AbstractLearning.models_to_keep)

        # Iteration is the number of parameter update steps performed in the training
        iteration = 0

        # Validation metric
        avg_bisk_metric = self.agent.test(tuning_size)
        min_avg_bisk_metric = avg_bisk_metric
        patience = 0
        max_patience = AbstractLearning.max_patience
        logger.Log.info("Tuning Data: (Before Training) Avg. Bisk Metric: " + str(avg_bisk_metric))

        for epoch in range(1, max_epoch + 1):
            logger.Log.info("=================\n Starting Epoch: " + str(epoch) + "\n=================")
            for data_point in range(1, train_size + 1):

                # Create a queue to handle history of states
                state = collections.deque([], 5)
                # Add the dummy images
                dummy_images = self.policy_model.image_embedder.get_dummy_images()
                [state.append(v) for v in dummy_images]

                # Receive the instruction and the environment
                (_, _, current_env, instruction, trajectory) = self.agent.receive_instruction_and_image()
                state.append(current_env)

                ########################
                text_indices = self.policy_model.text_embedder.convert_text_to_indices(instruction)
                _, text_embedder_bucket = self.policy_model.get_bucket_network(len(text_indices))
                (text_input_word_indices_bucket, text_mask_bucket) = text_embedder_bucket.pad_and_return_mask(
                    text_indices)
                (text_input_word_indices, text_mask) = self.policy_model.text_embedder.pad_and_return_mask(text_indices)
                ########################

                logger.Log.info("=================\n " + str(data_point) + ": Instruction: "
                                + str(instruction) + "\n=================")

                total_reward_episode = 0
                steps = 0

                # Reinforce requires sampling from Q-function for the future.
                # So we cannot directly add entries to the global replay memory.
                replay_memory_items = []
                rewards = []
                previous_status_code = self.policy_model.null_previous_action

                # Perform a roll out
                while True:
                    # Compute the probability of the current state
                    block_prob, direction_prob = self.policy_model.evaluate_qfunction(
                        state, text_input_word_indices_bucket, text_mask_bucket,
                        previous_action=previous_status_code, sess=sess)

                    # Sample from the prob. distribution
                    block_id = gp.GenericPolicy.sample_action_from_prob(block_prob)
                    direction_id = gp.GenericPolicy.sample_action_from_prob(direction_prob)

                    action_str = self.agent.message_protocol_kit.encode_action_from_pair(block_id, direction_id)
                    prob_action = block_prob[block_id] * direction_prob[direction_id]
                    logger.Log.debug("Sending Message: " + action_str + " with probability " + str(prob_action))
                    self.agent.connection.send_message(action_str)

                    # receive reward and a new environment as a response on the completion of action
                    (status_code, reward, new_env, is_reset) = self.agent.receive_response_and_image()
                    logger.Log.debug("Received reward: " + str(reward))

                    # add to replay memory
                    replay_memory_item = rm.ReplayMemory(text_input_word_indices, text_mask, state,
                                                         (block_id, direction_id), reward, None, None, prob_action,
                                                         previous_action_id=previous_status_code)
                    replay_memory_items.append(replay_memory_item)
                    rewards.append(reward)
                    state.append(new_env)

                    # Update metric
                    total_reward_episode += reward
                    steps += 1

                    previous_status_code = (direction_id, block_id)

                    # Reset episode
                    if self.agent.message_protocol_kit.is_reset_message(is_reset):
                        logger.Log.debug("Resetting the episode")
                        self.agent.connection.send_message("Ok-Reset")
                        logger.Log.debug("Now waiting for response")

                        if self.total_reward:
                            # Compute monte carlo q values
                            reward_multiplier = [0] * steps
                            for i in range(0, steps):
                                # Q-value approximated by roll-out
                                reward_multiplier[i] = sum(rewards[i:])
                        else:
                            # Use immediate reward only
                            reward_multiplier = rewards

                        # Define the targets
                        for replay_memory_item, cumm_reward in zip(replay_memory_items, reward_multiplier):
                            replay_memory_item.set_target_retroactively(cumm_reward)

                        # Perform 1 iteration of minibatch SGD using backpropagation
                        loss = self.min_loss(replay_memory_items, sess, train_writer)
                        if np.isnan(loss):
                            logger.Log.error("NaN found. Exiting")
                            exit(0)
                        iteration += 1
                        logger.Log.info("Number of sample " + str(len(replay_memory_items)) + " loss = " + str(loss))
                        logger.Log.info("Total reward:" + str(total_reward_episode) + " Steps: " + str(steps))

                        # Print time statistics
                        total_time = time.time() - start
                        logger.Log.info("Total time: " + str(total_time))
                        logger.Log.flush()
                        break

            # Compute validation accuracy
            avg_bisk_metric = self.agent.test(tuning_size)
            logger.Log.info("Tuning Data: (end of epoch " + str(epoch) + ") Avg. Bisk Metric: " +
                            str(avg_bisk_metric) + "Min was " + str(min_avg_bisk_metric))
            # Save the model
            save_path = saver.save(sess, "./saved/" + str(model_name) + "_epoch_" + str(epoch) + ".ckpt")
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
