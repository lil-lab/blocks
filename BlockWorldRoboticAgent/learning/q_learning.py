import logger
import tensorflow as tf
import time
import collections
import replay_memory as rm
from abstract_learning import AbstractLearning
import prioritized_sweeping
import epsilon_greedy_policy as egp


class QLearning(AbstractLearning):
    """ Deep Q-Learning algorithm of Mnih et al. 2015. Performs following optimization:
                 min_theta ( Q(s,a; theta) - y)^2
        where y = r | r + gamma * max_a Q(s', a; theta)

        Uses a target network with hard updates for modeling y.
        Uses epsilon-greedy behaviour policy and replay memory with prioritized sweeping.
    """

    def __init__(self, agent, q_network, target_q_network):
        """ Creates constructor for an abstract learning setup """

        self.agent = agent
        self.loss = None
        self.q_network = q_network
        self.target_q_network = target_q_network

        # Define epsilon greedy behaviour policy
        epsilon = 1.0
        min_epsilon = 0.1
        self.behaviour_policy = egp.EpsilonGreedyPolicy(epsilon, min_epsilon)

        # Replay memory and prioritized sweeping for sampling from the replay memory
        max_replay_memory_size = 2000
        self.replay_memory = collections.deque(maxlen=max_replay_memory_size)
        rho = 0.5
        self.ps = prioritized_sweeping.PrioritizedSweeping(0, rho)

        optimizer = tf.train.AdamOptimizer(self.rl_learning_rate)
        loss = self.calc_loss(self.q_network.model_output, self.q_network.model_output_indices, self.q_network.target)

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

        AbstractLearning.__init__(self, q_network, loss, train_step, update_summaries)

    @staticmethod
    def calc_loss(q_value, action_indices, target):

        # Loss is mean squared error.
        indices_flattened = tf.range(0, tf.shape(q_value)[0]) * tf.shape(q_value)[1] + action_indices
        q_val_action = tf.gather(tf.reshape(q_value, [-1]), indices_flattened)
        loss = tf.reduce_mean(tf.square(tf.sub(q_val_action, target)))
        return loss

    def copy_variables_to_target_network(self, sess):
        """ Copies variable values from the main network to the target network """

        ops = []

        # Copy embed_image variables
        ops += self.q_network.image_embedder.copy_variables_to(self.target_q_network.image_embedder)

        # Copy embed_token_seq variables (including RNN)
        ops += self.q_network.text_embedder.copy_variables_to(self.target_q_network.text_embedder)

        # Copy embed_previous_action variables
        ops += self.q_network.previous_action_embedder.copy_variables_to(self.target_q_network.previous_action_embedder)

        # Copy mix_and_gen_q_val variables
        ops += self.q_network.mix_and_gen_q_val.copy_variables_to(self.target_q_network.mix_and_gen_q_val)

        sess.run(ops)
        logger.Log.info("Hard synchronized target Q network with the main Q network network.")

    def train(self, sess, train_writer):
        """ Performs Q-learning on the Block World Task. The agent interacts with the
         simulator and performs roll-out followed by MSE updates. """

        start = time.time()

        max_epoch = AbstractLearning.max_epochs
        dataset_size = AbstractLearning.dataset_size
        tuning_size = AbstractLearning.validation_datasize
        train_size = dataset_size - tuning_size
        logger.Log.info("Deep Q-Learning: Max Epoch: " + str(max_epoch) + " Train/Tuning: "
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
            logger.Log.info("=================\n Starting Epoch: "
                            + str(epoch) + "\n=================")
            for data_point in range(1, train_size + 1):

                # Create a queue to handle history of states
                state = collections.deque([], 5)
                # Add the dummy images
                dummy_images = self.model.image_embedder.get_dummy_images()
                [state.append(v) for v in dummy_images]

                # Receive the instruction and the environment
                (_, bisk_metric, current_env, instruction, trajectory) = self.agent.receive_instruction_and_image()
                logger.Log.info("Train Bisk Metric " + str(bisk_metric))
                state.append(current_env)

                ########################
                text_indices = self.q_network.text_embedder.convert_text_to_indices(instruction)
                _, text_embedder_bucket = self.q_network.get_bucket_network(len(text_indices))
                (text_input_word_indices_bucket, text_mask_bucket) = text_embedder_bucket.pad_and_return_mask(
                    text_indices)
                (text_input_word_indices, text_mask) = self.q_network.text_embedder.pad_and_return_mask(text_indices)
                ########################

                logger.Log.info("=================\n " + str(data_point) + ": Instruction: "
                                + str(instruction) + "\n=================")

                total_reward_episode = 0
                steps = 0
                previous_action = self.q_network.null_previous_action

                # Perform a roll out
                while True:
                    # Compute the qVal of the current state
                    q_val = self.q_network.evaluate_qfunction(
                        state, text_input_word_indices_bucket, text_mask_bucket, previous_action, sess)

                    # take an action using a behaviour policy
                    action_id = self.behaviour_policy.get_action(q_val)
                    action_str = self.agent.message_protocol_kit.encode_action(action_id)
                    logger.Log.debug("Sending Message: " + action_str)
                    self.agent.connection.send_message(action_str)

                    # receive reward and a new environment as response on the completion of action
                    (_, reward, new_env, is_reset) = self.agent.receive_response_and_image()
                    logger.Log.debug("Received reward: " + str(reward))

                    # compute target y = r + gamma * max_a' Q(s', a')
                    copy_state = collections.deque(state, 5)
                    copy_state.append(new_env)
                    q_val_new = self.target_q_network.evaluate_qfunction(
                        copy_state, text_input_word_indices_bucket, text_mask_bucket, previous_action, sess)
                    if self.agent.message_protocol_kit.is_reset_message(is_reset):
                        # Terminal condition
                        y = reward
                    else:
                        y = reward + self.agent.gamma * q_val_new.max()
                    logger.Log.debug("Reward " + str(reward) + " Target " + str(y) + " max is " + str(q_val_new.max())
                                     + " current " + str(q_val[action_id]) + " diff " + str(y - q_val[action_id]))

                    # add to replay memory
                    replay_memory_item = rm.ReplayMemory(text_input_word_indices, text_mask,
                                                         state, action_id, reward, new_env, y,
                                                         previous_action_id=previous_action)
                    self.replay_memory.appendleft(replay_memory_item)
                    state.append(new_env)

                    # Update metric
                    total_reward_episode += reward
                    steps += 1
                    block_id = int(action_id / 4)
                    if action_id == 80:
                        direction_id = 4
                    else:
                        direction_id = action_id % 4
                    previous_action = (direction_id, block_id)

                    # Reset episode
                    if self.agent.message_protocol_kit.is_reset_message(is_reset):
                        logger.Log.debug("Resetting the episode")
                        self.agent.connection.send_message("Ok-Reset")
                        logger.Log.debug("Now waiting for response")

                        # Perform minibatch SGD
                        # Pick a sample using prioritized sweeping and perform backpropagation
                        sample = self.ps.sample(self.replay_memory, self.batch_size)
                        loss = self.min_loss(sample, sess, train_writer, factorized_actions=False)
                        iteration += 1
                        logger.Log.info("Number of sample " + str(len(sample)) + " size of replay memory "
                                        + str(len(self.replay_memory)) + " loss = " + str(loss))

                        # Decay the epsilon
                        self.behaviour_policy.decay_epsilon()
                        logger.Log.info("Total reward in this episode: " + str(total_reward_episode))

                        # Print time statistics
                        total_time = time.time() - start
                        logger.Log.info("Total time: " + str(total_time))
                        logger.Log.flush()
                        break

            # Synchronize the target network and main network
            self.copy_variables_to_target_network(sess)

            # Compute validation accuracy
            avg_bisk_metric = self.agent.test(tuning_size)
            logger.Log.info("Tuning Data: (end of epoch " + str(epoch) + ") Avg. Bisk Metric: "
                            + str(avg_bisk_metric) + "Min was " + str(min_avg_bisk_metric))
            # Save the model
            save_path = saver.save(sess, "./saved/model_epoch_" + str(epoch) + ".ckpt")
            logger.Log.info("Model saved in file: " + str(save_path))

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

        logger.Log.close()
