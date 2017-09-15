import logger


class Config:

    TRAIN, DEV, TEST = range(3)

    def __init__(self, dataset_size, shuffle_before_select, dataset_file, simplified, horizon,
                 reward_function_type, use_localhost, stop_action_reward, screen_size):

        self.dataset_size = dataset_size
        self.shuffle_before_select = shuffle_before_select
        self.dataset_file = dataset_file
        self.simplified = simplified
        self.horizon = horizon
        self.reward_function_type = reward_function_type
        self.use_localhost = use_localhost
        self.stop_action_reward = stop_action_reward
        self.screen_size = screen_size

        if dataset_file == "trainset.json":
            self.data_mode = Config.TRAIN
        elif dataset_file == "devset.json":
            self.data_mode = Config.DEV
        elif dataset_file == "testset.json":
            self.data_mode = Config.TEST
        else:
            raise AssertionError("Unknown dataset file " + str(self.data_mode))

    @staticmethod
    def parse(file_name):
        """ Parse config object from a file. File consists of ordered key value pair
         in the format key:value. """
        lines = open(file_name).readlines()

        dataset_size = int(lines[0][lines[0].index(':') + 1:])
        shuffle_before_select = True if lines[1][lines[1].index(':') + 1:] == "true" else False
        dataset_file = lines[2][lines[2].index(':') + 1:]
        simplified = True if lines[3][lines[3].index(':') + 1:] == "true" else False
        horizon = int(lines[4][lines[4].index(':') + 1:])
        reward_function_type = int(lines[5][lines[5].index(':') + 1:])
        use_localhost = True if lines[6][lines[6].index(':') + 1:] == "true" else False
        stop_action_reward = True if lines[7][lines[7].index(':') + 1:] == "true" else False
        screen_size = int(lines[8][lines[8].index(':') + 1:])

        return Config(dataset_size, shuffle_before_select, dataset_file, simplified, horizon,
                      reward_function_type, use_localhost, stop_action_reward, screen_size)

    def log_flag(self):
        """ Logs the config variables. """
        logger.Log.info("Dataset size: " + str(self.dataset_size))
        logger.Log.info("Shuffle before select: " + str(self.shuffle_before_select))
        logger.Log.info("Dataset file: " + str(self.dataset_file))
        logger.Log.info("Simplified: " + str(self.simplified))
        logger.Log.info("Horizon: " + str(self.horizon))
        logger.Log.info("Reward function type: " + str(self.reward_function_type))
        logger.Log.info("Use localhost: " + str(self.use_localhost))
        logger.Log.info("Stop action reward: " + str(self.stop_action_reward))
        logger.Log.info("Screen size: " + str(self.screen_size))
