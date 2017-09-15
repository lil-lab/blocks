import constants
from agent_no_model import *
from config import Config

# Read the constants file
constants_hyperparam = constants.constants

# Read the configuration file that connects agent and simulator
config = Config.parse("../BlockWorldSimulator/Assets/config.txt")

# Create the agent
agent = AgentModelLess(ORACLE, config)

dataset_size = 0

if config.data_mode == Config.TRAIN:
    dataset_size = constants_hyperparam["train_size"]
elif config.data_mode == Config.DEV:
    dataset_size = constants_hyperparam["dev_size"]
elif config.data_mode == Config.TEST:
    dataset_size = constants_hyperparam["test_size"]
else:
    raise AssertionError("Unknown or unhandled data_mode. Found " + str(config.data_mode))

# Test the agent
agent.test(dataset_size)
