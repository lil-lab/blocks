import tensorflow as tf


class MixAndGenerateQValues:
    """ This class mixes embeddings of text and image
     and generates probabilities over action space.
     Specifically, given a,b as embedding of text and image
     respectively. It uses 2 layer MLP with relu-units to generate
     first an embedding c and then uses c to generate probabilties
     over block IDs and 4 actions.
    """

    def __init__(self, n_text, n_image, n_previous_action, text_embed, image_embed, previous_action_embed,
                 num_actions, scope_name="mix_gen_qvalue", create_copy=None):

        observed_state = tf.concat(1, [image_embed, text_embed, previous_action_embed])
        n_input = n_image + n_text + n_previous_action
        self.n_actions = num_actions
        dim = 120

        if create_copy is not None:
            self.weights = create_copy.weights
            self.biases = create_copy.biases
        else:
            with tf.name_scope(scope_name):
                # layers weight & bias
                self.weights = {
                    'w_1': tf.Variable(tf.random_normal([n_input, dim], stddev=0.01)),
                    'w_q': tf.Variable(tf.random_normal([dim, self.n_actions], stddev=0.01))
                }
                self.biases = {
                    'b_1': tf.Variable(tf.constant(0.0, dtype=None, shape=[dim])),
                    'b_q': tf.Variable(tf.constant(0.0, dtype=None, shape=[self.n_actions]))
                }

        # Compute a common representation
        layer = tf.nn.relu(tf.add(tf.matmul(observed_state, self.weights["w_1"]), self.biases["b_1"]))

        # Direction logits 
        self.q_val = tf.add(tf.matmul(layer, self.weights["w_q"]), self.biases["b_q"])

    def get_q_val(self):
        return self.q_val

    def copy_variables_to(self, other):
        """ We are given another object of this class and we want to copy
        variables from one to another. It returns a list of ops to run. """

        op1 = other.weights['w_dir'].assign(self.weights['w_dir'])
        op2 = other.biases['b_dir'].assign(self.biases['b_dir'])

        return [op1, op2]
