# This class defines the data, model and learning constants that are used by the model.

# All consntants are stored in a dictionary called constants.
constants = dict()

constants["text_hidden_dim"] = 250
constants["image_hidden_dim"] = 200
constants["direction_dim"] = 24
constants["block_dim"] = 32
constants["rl_learning_rate"] = 0.0025
constants["mle_learning_rate"] = 0.001
constants["max_epoch"] = 30
constants["models_to_keep"] = 30
constants["max_patience"] = 30
constants["batch_size"] = 32

# *******Danger Zone*******:
# The constants below are fundamental and their change
# should be synchronized with the simulator or the data. Changing them
# may give undesired behaviour. Similarly if the dataset files are changed
# then you may want to change the constants defined below.
constants["train_size"] = 11871
constants["validation_size"] = 675
constants["dev_size"] = 1719
constants["test_size"] = 3177

constants["num_block"] = 20
constants["num_direction"] = 4
constants["use_stop"] = True
