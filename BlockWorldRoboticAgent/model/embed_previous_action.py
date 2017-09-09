import tensorflow as tf


class EmbedPreviousAction:
    """ Class that embeds the previous action. """

    def __init__(self, num_direction, dim_direction, num_blocks, dim_blocks, scope_name="previous_action"):
        """ A class that embeds previous action (given by direction) """

        # Last direction was taken
        self.last_direction_id = tf.placeholder(dtype=tf.int32, shape=None, name=scope_name + "_direction_input")

        # Last block chosen
        self.last_block_id = tf.placeholder(dtype=tf.int32, shape=None, name=scope_name + "_block_input")

        self.direction_embedding_matrix = tf.Variable(
            tf.random_normal(shape=(num_direction, dim_direction), mean=0.0, stddev=0.001, dtype=tf.float32),
            name=scope_name + "_direction_embedding_matrix")

        self.block_embedding_matrix = tf.Variable(
            tf.random_normal(shape=(num_blocks, dim_blocks), mean=0.0, stddev=0.001, dtype=tf.float32),
            name=scope_name + "_block_embedding_matrix")

        direction_output = tf.nn.embedding_lookup(self.direction_embedding_matrix, self.last_direction_id)
        block_output = tf.nn.embedding_lookup(self.block_embedding_matrix, self.last_block_id)
        self.output = tf.concat(1, [direction_output, block_output])

    def get_input(self):
        return self.last_direction_id, self.last_block_id

    def get_output(self):
        return self.output

    def copy_variables_to(self, other):

        op1 = other.block_embedding_matrix.assign(self.block_embedding_matrix)
        op2 = other.direction_embedding_matrix.assign(self.direction_embedding_matrix)

        return [op1, op2]
