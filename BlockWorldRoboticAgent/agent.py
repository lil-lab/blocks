import collections
import sys
import tensorflow as tf
import logger
import message_protocol_util as mpu
import reliable_connect as rc
from learning.ml_estimation import MaximumLikelihoodEstimation
from learning.policy_gradient import PolicyGradient
from learning.q_learning import QLearning
from learning.policy_gradient_with_advantage import PolicyGradientWithAdvantage
from model.policy_network import PolicyNetwork
from model.q_network import ActionValueFunctionNetwork
from model.v_network import StateValueFunctionModel
import generic_policy as gp


# The different kind of training algorithm that are used to train the agent.
# SUPERVISEDMLE: Supervised learning that maximizes log-likelihood of next action.
# SIMPLEQLEARNING: Deep Q-learning with an epsilon-greedy behaviour policy.
# REINFORCE: Performs policy gradient using reinforce algorithm with entropy regularization but no baseline.
# PGADVANTAGE: Advantage Actor Critic (A2C) algorithm using V function as a baseline in REINFORCE (not asynchronous).
# CONTEXTUALBANDIT: Contextual Bandit algorithm that maximizes the immediate reward.
SUPERVISEDMLE, SIMPLEQLEARNING, REINFORCE, PGADVANTAGE, CONTEXTUALBANDIT = range(5)


class Agent:
    """" The block world agent that takes action and moves block around in a toy domain. """

    def __init__(self, train_alg, config, constants):

        # Initialize logger
        logger.Log.open("./log.txt")

        self.config = config

        # Connect to simulator
        if len(sys.argv) < 2:
            logger.Log.info("IP not given. Using localhost i.e. 0.0.0.0")
            self.unity_ip = "0.0.0.0"
        else:
            self.unity_ip = sys.argv[1]

        if len(sys.argv) < 3:
            logger.Log.info("PORT not given. Using 11000")
            self.PORT = 11000
        else:
            self.PORT = int(sys.argv[2])

        # Size of image
        image_dim = self.config.screen_size
        self.connection = rc.ReliableConnect(self.unity_ip, self.PORT, image_dim)
        self.connection.connect()

        # Dataset specific parameters
        self.num_block = 20
        self.num_direction = 4
        use_stop = True
        if use_stop:
            self.num_actions = self.num_block * self.num_direction + 1  # 1 for stopping
        else:
            self.num_actions = self.num_block * self.num_direction

        # Create toolkit of message protocol between simulator and agent
        self.message_protocol_kit = mpu.MessageProtocolUtil(self.num_direction, self.num_actions, use_stop)

        # Test policy
        self.test_policy = gp.GenericPolicy.get_argmax_action

        # MDP details
        self.gamma = 1.0

        # Training algorithm behaviour
        self.train_alg = train_alg

        # Define model and learning algorithm
        if self.train_alg == SUPERVISEDMLE:
            self.model = PolicyNetwork(image_dim, self.num_actions, constants)
            self.learning_alg = MaximumLikelihoodEstimation(self, self.model)
        elif self.train_alg == REINFORCE:
            self.model = PolicyNetwork(image_dim, self.num_actions, constants)
            self.learning_alg = PolicyGradient(self, self.model, total_reward=True)
        elif self.train_alg == CONTEXTUALBANDIT:
            self.model = PolicyNetwork(image_dim, self.num_actions, constants)
            self.learning_alg = PolicyGradient(self, self.model, total_reward=False)
        elif self.train_alg == PGADVANTAGE:
            self.model = PolicyNetwork(image_dim, self.num_actions, constants)
            self.state_value_model = StateValueFunctionModel(250, image_dim, 200, 24, 32)
            self.learning_alg = PolicyGradientWithAdvantage(self, self.model, self.state_value_model, total_reward=True)
        elif self.train_alg == SIMPLEQLEARNING:
            self.model = ActionValueFunctionNetwork(250, image_dim, 200, 24, 32)
            self.target_q_network = ActionValueFunctionNetwork(
                250, image_dim, 200, 24, 32, scope_name="Target_Q_Network")
            self.learning_alg = QLearning(self, self.model, self.target_q_network)
        else:
            raise AssertionError("Training algorithm " + str(self.train_alg) + " not found or implemented.")

        self.sess = None
        self.train_writer = None
        self.config.log_flag()
        logger.Log.info("Training Algorithm: " + str(self.train_alg) + ", Gamma: " + str(self.gamma))
        logger.Log.info("Created Agent.")

    def init_session(self, model_file=None, gpu_memory_fraction=0.5):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        if model_file is None:
            self.sess.run(tf.initialize_all_variables())
            logger.Log.info("Initialized all variables ")
            saver = tf.train.Saver()
            saver.save(self.sess, "./saved/init.ckpt")
        else:
            saver = tf.train.Saver()
            saver.restore(self.sess, model_file)
            logger.Log.info("Loaded model from the file " + str(model_file))

        self.train_writer = tf.train.SummaryWriter('./train_summaries/', self.sess.graph)

    def receive_instruction_and_image(self):
        """ Receives image and then reset message. Returns decoded
        message with image. """

        img = self.connection.receive_image()
        response = self.connection.receive_message()
        (status_code, bisk_metric, _, instruction, trajectory) = \
            self.message_protocol_kit.decode_reset_message(response)
        return status_code, bisk_metric, img, instruction, trajectory

    def receive_response_and_image(self):
        """ Receives image and then response message. Returns decoded
        message with image. """

        img = self.connection.receive_image()
        response = self.connection.receive_message()
        (status_code, reward, _, reset_file_name) = self.message_protocol_kit.decode_message(response)
        return status_code, reward, img, reset_file_name

    def test(self, dataset_size, oracle=False):
        """ Performs testing on the Block World Task. The agent interacts with the simulator
         which will iterate over the entire dataset and perform roll-out using test policy. """

        dummy_images = self.model.image_embedder.get_dummy_images()
        previous_state = collections.deque([], 5)

        sum_bisk_metric = 0
        sum_reward = 0
        sum_steps = 0
        right_block = 0
        first_right = 0

        for i in range(0, dataset_size):
            (status_code, bisk_metric, current_env, instruction, trajectory) = self.receive_instruction_and_image()
            sum_bisk_metric = sum_bisk_metric + bisk_metric
            logger.Log.info("Bisk Metric " + str(bisk_metric))
            logger.Log.info("Instruction: " + str(instruction))
            text_indices = self.model.text_embedder.convert_text_to_indices(instruction)
            _, text_embedder_bucket = self.model.get_bucket_network(len(text_indices))
            (text_input_word_indices, text_mask) = text_embedder_bucket.pad_and_return_mask(text_indices)

            previous_state.clear()
            [previous_state.append(v) for v in dummy_images]
            previous_state.append(current_env)

            gold_block_id = int(trajectory[0] / 4.0)

            steps = 0
            sample_expected_reward = 0
            traj_sum_prob = 0
            running_gamma = 1.0
            previous_action = self.model.null_previous_action
            blocks_moved = []
            first = True
            traj_ix = 0

            while True:
                # sample action from the likelihood distribution
                action_values = self.model.get_action_values(previous_state, text_input_word_indices,
                                                             text_mask, previous_action, self.sess)
                inferred_action = self.test_policy(action_values)
                if oracle:
                    inferred_action = trajectory[traj_ix]
                    traj_ix += 1
                action_str = self.message_protocol_kit.encode_action(inferred_action)
                block_id = int(inferred_action/4.0)
                direction_id = inferred_action % 4
                if inferred_action != 80:
                    blocks_moved.append(block_id)
                if first:
                    first = False
                    if block_id == gold_block_id:
                        first_right += 1

                # Find probability of this action
                logger.Log.debug(action_values)
                prob_action = action_values[inferred_action]
                logger.Log.info("Action probability " + str(prob_action))
                print "Action probability " + str(prob_action)

                print "Sending Message: " + action_str
                logger.Log.info(action_str + "\n")
                self.connection.send_message(action_str)

                # receive confirmation on the completion of action
                (status_code, reward, current_env, is_reset) = self.receive_response_and_image()
                print "Received reward " + str(reward)
                previous_state.append(current_env)

                # Update and print metric
                sample_expected_reward += running_gamma * reward
                traj_sum_prob += prob_action
                running_gamma *= self.gamma
                steps += 1
                previous_action = (direction_id, block_id)

                # Reset to a new task
                if self.message_protocol_kit.is_reset_message(is_reset):
                    print "Resetting the episode"
                    self.connection.send_message("Ok-Reset")

                    sum_reward += sample_expected_reward
                    sum_steps += steps

                    blocks_moved = set(blocks_moved)
                    if len(blocks_moved) == 1 and list(blocks_moved)[0] == gold_block_id:
                        right_block += 1

                    logger.Log.info("Example: " + str(i) + " Instruction: " + instruction + " Steps: " + str(steps))
                    logger.Log.info("\t Total expected reward: " + str(sample_expected_reward))
                    logger.Log.info("\t Avg. expected reward: " + str(sample_expected_reward/float(steps)))
                    logger.Log.info("\t Sum of trajectory action values: " + str(traj_sum_prob))
                    logger.Log.info("\t Avg. total action values: " + str(traj_sum_prob/float(steps)))
                    logger.Log.info("\n============================================")
                    logger.Log.flush()
                    break

        avg_bisk_metric = sum_bisk_metric/float(max(dataset_size, 1))
        avg_reward = sum_reward/float(max(dataset_size, 1))
        avg_steps = sum_steps/float(max(dataset_size, 1))
        block_accuracy = right_block/float(max(dataset_size, 1))

        logger.Log.info("Avg. Bisk Metric " + str(avg_bisk_metric))
        logger.Log.info("Block accuracy " + str(block_accuracy))
        logger.Log.info("First Block accuracy  " + str(first_right/float(max(dataset_size, 1))))
        logger.Log.info("Avg. reward " + str(avg_reward) + " Steps " + str(avg_steps))
        logger.Log.info("Testing finished.")
        logger.Log.flush()

        return avg_bisk_metric

    def test_oracle(self, dataset_size):
        """ Runs oracle algorithm on the dataset """

        sum_bisk_metric = 0

        for i in range(0, dataset_size):
            (_, bisk_metric, current_env, instruction, trajectory) = self.receive_instruction_and_image()
            sum_bisk_metric = sum_bisk_metric + bisk_metric
            logger.Log.info("Bisk Metric " + str(bisk_metric))
            logger.Log.info("Instruction: " + str(instruction))

            steps = 0
            sample_expected_reward = 0
            running_gamma = 1.0

            while True:
                # sample action from the likelihood distribution
                action_id = trajectory[steps]
                action_str = self.message_protocol_kit.encode_action(action_id)
                print "Sending Message: " + action_str
                logger.Log.info(action_str + "\n")
                self.connection.send_message(action_str)

                # receive confirmation on the completion of action
                (_, reward, _, is_reset) = self.receive_response_and_image()
                print "Received reward " + str(reward)

                # Update and print metric
                sample_expected_reward += running_gamma * reward
                running_gamma *= self.gamma
                steps += 1

                # Reset to a new task
                if self.message_protocol_kit.is_reset_message(is_reset):
                    print "Resetting the episode"
                    self.connection.send_message("Ok-Reset")

                    logger.Log.info("Example: " + str(i) + " Instruction: " + instruction + " Steps: " + str(steps))
                    logger.Log.info("\t Total expected reward: " + str(sample_expected_reward))
                    logger.Log.info("\t Avg. expected reward: " + str(sample_expected_reward/float(steps)))
                    logger.Log.info("\n============================================")
                    logger.Log.flush()
                    break

        avg_bisk_metric = sum_bisk_metric/float(dataset_size)
        logger.Log.info("Avg. Bisk Metric " + str(avg_bisk_metric))
        logger.Log.info("Testing finished.")
        logger.Log.flush()

        return avg_bisk_metric

    def test_range(self, dataset_size, folder_name, epoch_start, epoch_end, epoch_step=1):
        """ Tests a range of model saved at different epochs. Be careful to run it
        with only on a dev or validation set and not on the held-out test set. """
        for epoch in range(epoch_start, epoch_end, epoch_step):
            self.init_session(folder_name + "/model_epoch_" + str(epoch) + ".ckpt")
            res = self.test(dataset_size)
            logger.Log.info("Dev results for epoch = " + str(epoch) + " is " + str(res))

    def train(self):
        """ Perform training """
        self.learning_alg.train(self.sess, self.train_writer)
