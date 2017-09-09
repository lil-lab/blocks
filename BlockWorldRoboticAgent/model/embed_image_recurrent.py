import tensorflow as tf
import logger
import numpy as np


class EmbedImageRecurrent:
    """ Provides functionality for embedding images using a LSTM-RNN
        taking input from deep CNN
    """

    def get_images_data_and_mask(self):
        return self.images_data, self.mask

    def __init__(self, output_size, image_dim, image_embed, time_horizon, scope_name="embed_recurrent_image"):
        """ Embeds a batch of image using 2 layer convolutional neural network
         followed by a fully connected layer. """

        height = image_dim
        width = image_dim
        channels = 3

        self.variables = []

        self.images_data = tf.placeholder(dtype=tf.float32, shape=None, name=scope_name + "_input")
        batchsize = tf.shape(self.images_data)[0]
        float_images = tf.reshape(self.images_data, [batchsize * time_horizon, width, height, channels])

        # We instantiate all variables using tf.get_variable() instead of
        # tf.Variable() in order to share variables across multiple GPU training runs.
        # If we only ran this model on a single GPU, we could simplify this function
        # by replacing all instances of tf.get_variable() with tf.Variable().

        # conv + affine + relu
        with tf.variable_scope(scope_name + '_conv1') as scope:
            kernel = self._variable_with_weight_decay('weights', shape=[8, 8, channels, 32],
                                                      stddev=0.005, wd=0.0)
            conv = tf.nn.conv2d(float_images, kernel, [1, 4, 4, 1], padding='SAME')
            biases = self._create_variable_('biases', [32], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope.name)
            self.variables.extend([kernel, biases])

        # conv + affine + relu
        with tf.variable_scope(scope_name + '_conv2') as scope:
            kernel = self._variable_with_weight_decay('weights', shape=[8, 8, 32, 32],
                                                      stddev=0.005, wd=0.0)
            conv = tf.nn.conv2d(conv1, kernel, [1, 4, 4, 1], padding='SAME')
            biases = self._create_variable_('biases', [32], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope.name)
            self.conv2 = conv1
            self.variables.extend([kernel, biases])

        # conv + affine + relu
        with tf.variable_scope(scope_name + '_conv3') as scope:
            kernel = self._variable_with_weight_decay('weights', shape=[4, 4, 32, 32],
                                                      stddev=0.005, wd=0.0)
            conv = tf.nn.conv2d(conv2, kernel, [1, 2, 2, 1], padding='SAME')
            biases = self._create_variable_('biases', [32], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(bias, name=scope.name)
            self.variables.extend([kernel, biases])

        # affine
        with tf.variable_scope(scope_name + '_linear') as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            reshape = tf.reshape(conv3, [batchsize * time_horizon, -1])
            # Value before is hacked
            # Not sure how to fix it
            # It if based on image dimension
            dim = 512
            weights = self._variable_with_weight_decay('weights', [dim, image_embed],
                                                       stddev=0.004, wd=0.004)
            biases = self._create_variable_('biases', [image_embed],
                                            tf.constant_initializer(0.0))
            image_embedding = tf.add(tf.matmul(reshape, weights), biases, name=scope.name)
            self.variables.extend([weights, biases])

        reshaped_image_embeddings = tf.reshape(image_embedding, [batchsize, time_horizon, -1])

        # Create a Tracking RNN
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(output_size, forget_bias=1.0)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell])
        self._initial_state = cell.zero_state(batchsize, tf.float32)

        # Zero Masking for RNN
        self.mask = tf.placeholder(tf.float32, [None, time_horizon])

        # Pass the CNN embeddings through RNN
        outputs = []
        state = self._initial_state
        with tf.variable_scope(scope_name):
            for time_step in range(time_horizon):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(reshaped_image_embeddings[:, time_step, :], state)
                zero_mask = self.mask[:, time_step]
                zero_mask = tf.reshape(zero_mask, [batchsize, 1])
                masked_output = tf.mul(cell_output, zero_mask)
                outputs.append(masked_output)

        temporal_sum = tf.reduce_sum(outputs, 0)
        num_frames = tf.reduce_sum(self.mask, 1)
        num_frames = tf.reshape(num_frames, [batchsize, 1])
        self.output = tf.div(temporal_sum, num_frames)

        # Create mask.
        self.mask_ls = []
        for i in range(0, time_horizon + 1):
            maski = [[1.0] * i + [0.0] * (time_horizon - i)]
            self.mask_ls.append(maski)

        logger.Log.info("Created Embed Image Recurrent. Time horizon " + str(time_horizon)
                        + ", Image Dim " + str(image_dim) + ", Output size " + str(output_size))

    def _create_variable_(self, name, shape, initializer):
        var = tf.get_variable(name, shape, initializer=initializer)
        return var

    def _variable_with_weight_decay(self, name, shape, stddev, wd):
        var = self._create_variable_(name, shape, tf.truncated_normal_initializer(stddev=stddev))
        return var

    def __get_variables__(self):
        return self.variables

    def get_output(self):
        return self.output

    def get_mask_for_num_frames(self, num_frames):
        return self.mask_ls[num_frames]

    def copy_variables_to(self, other):
        """ We are given another object of this class and we want to copy
        variables from one to another. It returns a list of copy operations. """

        ops = []
        for v1, v2 in zip(self.variables, EmbedImageRecurrent.__get_variables__(other)):
            op = v2.assign(v1)
            ops.append(op)

        return ops
