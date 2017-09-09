import tensorflow as tf


class MixAndGenerateProbabilities:
    """ This class mixes embeddings of text and image
     and generates probabilities over action space.
     Specifically, given a,b as embedding of text and image
     respectively. It uses 2 layer MLP with relu-units to generate
     first an embedding c and then uses c to generate probabilties
     over block IDs and 4 actions.
    """

    def __init__(self, n_text, n_image, n_previous_action, text_embed, image_embed, previous_action_embed,
                 num_actions, use_softmax=True, scope_name="mix", create_copy=None):

        observed_state = tf.concat(1, [image_embed, text_embed, previous_action_embed])
        n_input = n_image + n_text + n_previous_action
        self.n_actions = num_actions
        dim = 120
        n_block = 20

        if create_copy is not None:
            self.weights = create_copy.weights
            self.biases = create_copy.biases
        else:
            with tf.name_scope(scope_name):
                # layers weight & bias
                self.weights = {
                    'w_1': tf.Variable(tf.random_normal([n_input, dim], stddev=0.01)),
                    'w_dir': tf.Variable(tf.random_normal([dim, self.n_actions], stddev=0.01)),
                    'w_block': tf.Variable(tf.random_normal([dim, n_block], stddev=0.01))
                }
                self.biases = {
                    'b_1': tf.Variable(tf.constant(0.0, dtype=None, shape=[dim])),
                    'b_dir': tf.Variable(tf.constant(0.0, dtype=None, shape=[self.n_actions])),
                    'b_block': tf.Variable(tf.constant(0.0, dtype=None, shape=[n_block]))
                }

        # Compute a common representation
        layer = tf.nn.relu(tf.add(tf.matmul(observed_state, self.weights["w_1"]), self.biases["b_1"]))

        # Direction logits 
        direction_logits = tf.add(tf.matmul(layer, self.weights["w_dir"]), self.biases["b_dir"])
        
        # Block logits
        block_logits = tf.add(tf.matmul(layer, self.weights["w_block"]), self.biases["b_block"])
        
        if use_softmax:
            self.direction_prob = tf.nn.softmax(direction_logits)
            self.block_prob = tf.nn.softmax(block_logits)

    def get_joined_probabilities(self):
        return self.block_prob, self.direction_prob

    def get_direction_weights(self):
        return self.weights["w_dir"], self.biases["b_dir"]
