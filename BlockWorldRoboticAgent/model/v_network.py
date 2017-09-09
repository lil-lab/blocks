import embed_image
import embed_previous_action as epa
import numpy as np
import tensorflow as tf
import replay_memory as rm
from model import embed_token_seq, image_preprocessing


class StateValueFunctionModel:
    """ Models the state value function V(s) """

    def __init__(self, n_text, image_dim, n_image,
                 n_direction_dim, n_block_dim, scope_name="state_value"):

        # Neural network for embedding text
        self.text_embedder = embed_token_seq.EmbedTokenSeq(n_text, scope_name=scope_name + "_RNN")
        text_embedding = self.text_embedder.get_output()

        # Image preprocessor
        self.image_preprocessor = image_preprocessing.ImagePreprocessing()

        # Neural network for embedding image
        self.image_embedder = embed_image.EmbedImage(n_image, image_dim, scope_name=scope_name + "_embed_image")
        image_embedding = self.image_embedder.get_output()

        # Network for embedding past action
        # 6 actions, one for no-action
        n_previous_action_embedding = n_direction_dim + n_block_dim
        self.previous_action_embedder = epa.EmbedPreviousAction(
            6, n_direction_dim, 21, n_block_dim, scope_name=scope_name + "_previous_action")
        previous_action_embedding = self.previous_action_embedder.get_output()

        # Concatenate them and pass them through a layer to generate V(s)
        observed_state = tf.concat(1, [image_embedding, text_embedding, previous_action_embedding])
        n_state_dim = n_text + n_image + n_previous_action_embedding
        dim = 120
        with tf.name_scope(scope_name):
            # layers weight & bias
            self.weights = {
                'w_hid': tf.Variable(tf.random_normal([n_state_dim, dim], stddev=0.01)),
                'w_out': tf.Variable(tf.random_normal([dim, 1], stddev=0.01))
            }
            self.biases = {
                'b_hid': tf.Variable(tf.constant(0.0, dtype=None, shape=[dim])),
                'b_out': tf.Variable(tf.constant(0.0, dtype=None, shape=[1]))
            }

        latent_vector = tf.nn.relu(tf.add(tf.matmul(observed_state, self.weights["w_hid"]), self.biases["b_hid"]))
        self.state_value = tf.add(tf.matmul(latent_vector, self.weights["w_out"]), self.biases["b_out"])

        # Placeholder for total reward
        self.total_exp_reward = tf.placeholder(dtype=tf.float32, shape=None, name=scope_name + "_total_exp_reward")
        self.loss = tf.reduce_mean(tf.square(tf.sub(self.state_value, self.total_exp_reward)))

        optimizer = tf.train.AdamOptimizer(0.001)

        using_grad_clip = True
        grad_clip_val = 5.0
        if not using_grad_clip:
            self.train_step = optimizer.minimize(self.loss)
        else:
            gvs = optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_norm(grad, grad_clip_val), var)
                          if grad is not None else (grad, var) for grad, var in gvs]
            self.train_step = optimizer.apply_gradients(capped_gvs)

    def evaluate_state_value_function(self, image_data, text_input_word_indices, text_mask, previous_action, sess):

        image_data = np.concatenate(list(image_data), 2)

        raw_image_input = self.image_preprocessor.get_raw_image_input()
        final_image_output = self.image_preprocessor.get_final_image()
        image_datas = [final_image_output.eval(session=sess, feed_dict={raw_image_input: image_data})]

        image_placeholder = self.image_embedder.get_images_data()
        text_input = self.text_embedder.get_input()
        mask = self.text_embedder.get_zero_mask()
        batch_size = self.text_embedder.get_batch_size()
        direction_input, block_input = self.previous_action_embedder.get_input()

        result = sess.run(self.state_value,
                          feed_dict={text_input: text_input_word_indices,
                                     mask: text_mask, batch_size: 1,
                                     image_placeholder: [image_datas],
                                     direction_input: [previous_action[0]],
                                     block_input: [previous_action[1]]})

        return result[0]

    def mc_train_iteration(self, sample, total_expected_rewards, sess):
        """ Performs Monte Carlo iteration with single sample to perform the following optimization:
                    min loss where loss = 1/Nsum_{i=1}^N(V(s_i) - sum_t r^(i)_t)^2
        """

        # Placeholders for the input
        text_input = self.text_embedder.get_input()
        mask = self.text_embedder.get_zero_mask()
        image_input = self.image_embedder.get_images_data()
        direction_input, block_input = self.previous_action_embedder.get_input()

        # Placeholders for image preprocessing
        raw_image_input = self.image_preprocessor.get_raw_image_input()
        final_image_output = self.image_preprocessor.get_final_image()

        image_datas = []
        text_input_word_indices = []
        input_mask = []
        block_indices_ = []
        direction_indices_ = []
        target_ = []
        directions_ = []
        block_inputs_ = []

        for replay_item in sample:
            instruction_word_indices = rm.ReplayMemory.get_instruction_word_indices(replay_item)
            instruction_mask = rm.ReplayMemory.get_instruction_mask(replay_item)
            action = rm.ReplayMemory.get_action(replay_item)
            target = rm.ReplayMemory.get_target(replay_item)
            state = rm.ReplayMemory.get_history_of_states(replay_item)
            previous_action = rm.ReplayMemory.get_previous_action_id(replay_item)

            text_input_word_indices.append(instruction_word_indices[0])
            input_mask.append(instruction_mask[0])
            block_indices_.append(action[0])
            direction_indices_.append(action[1])
            target_.append(target)
            directions_.append(previous_action[0])
            block_inputs_.append(previous_action[1])

            image_datas.append(final_image_output.eval(session=sess, feed_dict={raw_image_input: state}))

        result = sess.run([self.loss, self.train_step],
                          feed_dict={text_input: text_input_word_indices, mask: input_mask,
                                     self.text_embedder.get_batch_size(): len(sample),
                                     image_input: [image_datas], direction_input: directions_,
                                     block_input: block_inputs_, self.total_exp_reward: total_expected_rewards})
        # self.train_writer.add_summary(result[2], self.update_iter)
        # self.update_iter += 1

        return result[0]
