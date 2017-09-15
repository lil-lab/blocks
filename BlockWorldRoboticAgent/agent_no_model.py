import sys
import logger
import message_protocol_util as mpu
import reliable_connect as rc
import generic_policy as gp
import random

ORACLE, RANDOM_WALK, STOP = range(3)


class AgentModelLess:
    """" The block world agent that implements oracle, random walk and the stop baseline.
    Requires no parameters to be tuned. This is implemented in a separate file to remove
    dependencies on tensorflow allowing it to run on systems without those dependencies. """

    def __init__(self, agent_type, config, constants):

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

        self.agent_type = agent_type

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
        self.config.log_flag()
        logger.Log.info("Created Agent.")

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

    def test(self, dataset_size):
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
                if self.agent_type == ORACLE:
                    action_id = trajectory[steps]
                elif self.agent_type == RANDOM_WALK:
                    action_id = random.randint(0, 81)
                elif self.agent_type == STOP:
                    action_id = 80
                else:
                    raise AssertionError("Unknown agent type. Found " + str(self.agent_type))

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
