import tensorflow as tf
import re
import numpy as np


class EmbedImage:
    """ Provides functionality for embedding images using a deep CNN.
    """

    def get_images_data(self):
        return self.images_data

    def __init__(self, output_size, image_dim, scope_name="embed_image"):
        """ Embeds a batch of image using 2 layer convolutional neural network
         followed by a fully connected layer. """

        self.output_size = output_size
        height = image_dim
        width = image_dim
        channels = 3 * 5

        self.variables = []

        self.images_data = tf.placeholder(dtype=tf.float32, shape=None, name=scope_name + "_input")
        batchsize = tf.shape(self.images_data)[1]
        float_images = tf.reshape(self.images_data, [batchsize, width, height, channels])

        # We instantiate all variables using tf.get_variable() instead of
        # tf.Variable() in order to share variables across multiple GPU training runs.
        # If we only ran this model on a single GPU, we could simplify this function
        # by replacing all instances of tf.get_variable() with tf.Variable().

        # conv + affine + relu
        with tf.variable_scope(scope_name + '_conv1') as scope:
            kernel = self._variable_with_weight_decay('weights', shape=[8, 8, channels, 32],
                                                      stddev=0.005, wd=0.0)
            conv = tf.nn.conv2d(float_images, kernel, [1, 4, 4, 1], padding='SAME')
            biases = self._variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope.name)
            self.variables.extend([kernel, biases])

        # conv + affine + relu
        with tf.variable_scope(scope_name + '_conv2') as scope:
            kernel = self._variable_with_weight_decay('weights', shape=[8, 8, 32, 32],
                                                      stddev=0.005, wd=0.0)
            conv = tf.nn.conv2d(conv1, kernel, [1, 4, 4, 1], padding='SAME')
            biases = self._variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope.name)
            self.conv2 = conv1
            self.variables.extend([kernel, biases])

        # conv + affine + relu
        with tf.variable_scope(scope_name + '_conv3') as scope:
            kernel = self._variable_with_weight_decay('weights', shape=[4, 4, 32, 32],
                                                      stddev=0.005, wd=0.0)
            conv = tf.nn.conv2d(conv2, kernel, [1, 2, 2, 1], padding='SAME')
            biases = self._variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(bias, name=scope.name)
            self.variables.extend([kernel, biases])

        # affine
        with tf.variable_scope(scope_name + '_linear') as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            reshape = tf.reshape(conv3, [batchsize, -1])
            # Value before is hacked
            # Not sure how to fix it
            # It if based on image dimension
            dim = 512
            weights = self._variable_with_weight_decay('weights', [dim, self.output_size],
                                                       stddev=0.004, wd=0.004)
            biases = self._variable_on_cpu('biases', [self.output_size],
                                           tf.constant_initializer(0.0))
            image_embedding = tf.add(tf.matmul(reshape, weights), biases, name=scope.name)
            self.variables.extend([weights, biases])

        self.debug1 = conv1
        self.debug2 = conv2
        self.output = image_embedding

        # Create 4 images for padding
        padding_image = np.zeros((image_dim, image_dim, 3), dtype=np.float32)
        self.padding_images = [padding_image, padding_image, padding_image, padding_image]
        print "Created Image Embedder"

    def get_dummy_images(self):
        return self.padding_images

    def get_debugs(self):
        return [self.debug1, self.debug2]

    def _variable_on_cpu(self, name, shape, initializer):
        """Helper to create a Variable stored on CPU memory.
        Args:
          name: name of the variable
          shape: list of ints
          initializer: initializer for Variable
        Returns:
          Variable Tensor
        """
        # with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
        return var

    def _variable_with_weight_decay(self, name, shape, stddev, wd):
        """Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.
        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.
        Returns:
          Variable Tensor
        """
        var = self._variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
        # if wd is not None:
        #     weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        #     tf.add_to_collection('losses', weight_decay)
        return var

    def __get_variables__(self):
        return self.variables

    def get_output(self):
        return self.output

    def get_debug_conv(self):
        return self.conv2

    def copy_variables_to(self, other):
        """ We are given another object of this class and we want to copy
        variables from one to another. It returns a list of ops to run. """

        ops = []
        for v1, v2 in zip(self.variables, EmbedImage.__get_variables__(other)):
            op = v2.assign(v1)
            ops.append(op)

        return ops
