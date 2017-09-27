import constants
from agent import *
from config import Config
import tensorflow as tf

# Read the constants file
constants_hyperparam = constants.constants

# Read the configuration file that connects agent and simulator
config = Config.parse("../BlockWorldSimulator/Assets/config.txt")

# Create the agent
agent = Agent(CONTEXTUALBANDIT, config, constants_hyperparam)

dataset_size = 0

if config.data_mode == Config.TRAIN:
    dataset_size = constants_hyperparam["train_size"]
elif config.data_mode == Config.DEV:
    dataset_size = constants_hyperparam["dev_size"]
elif config.data_mode == Config.TEST:
    dataset_size = constants_hyperparam["test_size"]
else:
    raise AssertionError("Unknown or unhandled data_mode. Found " + str(config.data_mode))

# If model file is None then model will use a randomly initialized model.
# Gpu memory fraction additionally take into account the % of GPU to occupy.
agent.init_session(model_file=None, gpu_memory_fraction=0.45)

# E.g., agent.init_session(model_file="./saved_mle/model_epoch_4.ckpt", gpu_memory_fraction=0.25)

# Train the agent
agent.train()
