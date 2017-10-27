import embed_token_seq
import image_preprocessing
import embed_image
import mix_and_gen_prob
import embed_previous_action as epa
import tensorflow as tf
import numpy as np


class PolicyNetwork:
    """ Creates policy pi(s,a) that generates probability
    over actions for a given observed state s. """

    def __init__(self, image_dim, num_actions, constants):

        # Number of actions
        self.num_actions = num_actions

        # Neural network for embedding text
        self.n_text = constants["text_hidden_dim"]
        self.text_embedder = embed_token_seq.EmbedTokenSeq(self.n_text)
        text_embedding = self.text_embedder.get_output()

        # Create bucket network for RNN
        self.buckets = [15, 30, 45]
        self.embed_token_seq_buckets = []
        for bucket in self.buckets:
            embed_token_seq_bucket = \
                embed_token_seq.EmbedTokenSeq(self.n_text, num_steps=bucket, create_copy=self.text_embedder)
            self.embed_token_seq_buckets.append(embed_token_seq_bucket)

        # Image Preprocessing
        self.image_preprocessor = image_preprocessing.ImagePreprocessing()

        # Neural network for embedding image
        self.n_image = constants["image_hidden_dim"]
        self.image_embedder = embed_image.EmbedImage(self.n_image, image_dim)
        image_embedding = self.image_embedder.get_output()

        # Network for embedding past action
        # 6 actions, one for no-action
        num_blocks = constants["num_block"]
        self.num_directions = constants["num_direction"]
        self.n_direction_dim = constants["direction_dim"]
        self.n_blocks_dim = constants["block_dim"]
        self.n_previous_action_embedding = self.n_direction_dim + self.n_blocks_dim
        self.null_previous_action = (self.num_directions + 1, num_blocks)
        self.previous_action_embedder = epa.EmbedPreviousAction(
            self.num_directions + 2, self.n_direction_dim, num_blocks + 1, self.n_blocks_dim)
        previous_action_embedding = self.previous_action_embedder.get_output()

        # Neural network for mixing the embeddings of text
        # and image and generate probabilities over block-ids and direction
        use_softmax = True
        self.mix_and_gen_prob = mix_and_gen_prob.MixAndGenerateProbabilities(
            self.n_text, self.n_image, self.n_previous_action_embedding,
            text_embedding, image_embedding, previous_action_embedding, self.num_directions + 1, use_softmax)

        # Create buckets
        self.mix_and_gen_prob_buckets = []
        for i in range(0, len(self.buckets)):
            mix_and_gen_prob_bucket = mix_and_gen_prob.MixAndGenerateProbabilities(
                self.n_text, self.n_image, self.n_previous_action_embedding,
                self.embed_token_seq_buckets[i].get_output(), image_embedding,
                previous_action_embedding, self.num_directions + 1, use_softmax, create_copy=self.mix_and_gen_prob)
            self.mix_and_gen_prob_buckets.append(mix_and_gen_prob_bucket)

        # Define input and output
        self.target = tf.placeholder(dtype=tf.float32, shape=None)
        self.block_indices = tf.placeholder(dtype=tf.int32, shape=None)
        self.direction_indices = tf.placeholder(dtype=tf.int32, shape=None)
        block_prob, direction_prob = self.mix_and_gen_prob.get_joined_probabilities()
        self.model_output = block_prob, direction_prob
        self.model_output_indices = self.block_indices, self.direction_indices

        summary_qval_min = tf.scalar_summary("Direction Prob Min", tf.reduce_min(direction_prob))
        summary_qval_max = tf.scalar_summary("Direction Prob Max", tf.reduce_max(direction_prob))
        summary_qval_mean = tf.scalar_summary("Direction Prob Mean", tf.reduce_mean(direction_prob))

        self.feed_forward_summary = tf.merge_summary([summary_qval_min, summary_qval_max, summary_qval_mean])
        self.feed_iter = 0

    def get_bucket_network(self, num_tokens):
        """ """
        for i in range(0, len(self.buckets)):
            if num_tokens <= self.buckets[i]:
                return self.mix_and_gen_prob_buckets[i], self.embed_token_seq_buckets[i]
        return self.mix_and_gen_prob, self.text_embedder

    def evaluate_policy(self, image_data, text_input_word_indices, text_mask, previous_action, sess):
        """ Compute policy over actions for a given observed state. This code does not work with batch. """

        mix_and_gen_prob_bucket, text_embedder_bucket = self.get_bucket_network(sum(text_mask[0]))

        image_data = np.concatenate(list(image_data), 2)
        block_prob, direction_prob = mix_and_gen_prob_bucket.get_joined_probabilities()

        raw_image_input = self.image_preprocessor.get_raw_image_input()
        final_image_output = self.image_preprocessor.get_final_image()
        image_datas = [final_image_output.eval(session=sess, feed_dict={raw_image_input: image_data})]

        image_placeholder = self.image_embedder.get_images_data()
        text_input = text_embedder_bucket.get_input()
        mask = text_embedder_bucket.get_zero_mask()
        batch_size = text_embedder_bucket.get_batch_size()
        direction_input, block_input = self.previous_action_embedder.get_input()

        result = sess.run([block_prob, direction_prob],
                          feed_dict={text_input: text_input_word_indices,
                                     mask: text_mask, batch_size: 1,
                                     image_placeholder: [image_datas],
                                     direction_input: [previous_action[0]],
                                     block_input: [previous_action[1]]})
        # self.train_writer.add_summary(result[1], self.feed_iter)
        # self.feed_iter += 1

        return result[0][0], result[1][0]

    def get_action_values(self, image_data, text_input_word_indices, text_mask, previous_action, sess):
        """ Perform test time inference and return the model scoring values
         over the action. These scoring values are probabilities for policy network
         but will be Q-values for a q-network. """

        block_prob, direction_prob = self.evaluate_policy(
            image_data, text_input_word_indices, text_mask, previous_action, sess)

        actions_values = [0] * self.num_actions

        for action_id in range(0, self.num_actions - 1):
            block_id = int(action_id/float(self.num_directions))
            direction_id = action_id % self.num_directions
            actions_values[action_id] = block_prob[block_id] * direction_prob[direction_id]

        # Stop probability. Assumes last direction is always STOP
        actions_values[self.num_actions - 1] = direction_prob[self.num_directions]

        return actions_values
