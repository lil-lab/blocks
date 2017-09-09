import embed_token_seq
import image_preprocessing
import embed_image
import mix_and_gen_q_values
import embed_previous_action as epa
import tensorflow as tf
import numpy as np


class ActionValueFunctionNetwork:
    """ Creates policy Q(s,a) that approximates Q values
    over actions for a given observed state s. """

    # def __init__(self, image_dim, scope_name="Q_network"):
    def __init__(self, n_text, image_dim, n_image,
                 n_direction_dim, n_block_dim, scope_name="Q_network"):

        # Neural network for embedding text
        self.n_text = n_text
        self.text_embedder = embed_token_seq.EmbedTokenSeq(self.n_text, scope_name=scope_name)
        text_embedding = self.text_embedder.get_output()

        ####################
        # Create bucket network
        self.buckets = [15, 30, 45]
        self.embed_token_seq_buckets = []
        for bucket in self.buckets:
            embed_token_seq_bucket = \
                embed_token_seq.EmbedTokenSeq(self.n_text, num_steps=bucket, create_copy=self.text_embedder,
                                              scope_name=scope_name)
            self.embed_token_seq_buckets.append(embed_token_seq_bucket)
        ####################

        # Image Preprocessing
        self.image_preprocessor = image_preprocessing.ImagePreprocessing()

        # Neural network for embedding image
        self.n_image = n_image
        self.image_embedder = embed_image.EmbedImage(self.n_image, image_dim, scope_name=scope_name)
        image_embedding = self.image_embedder.get_output()

        # Network for embedding past action
        # 6 actions, one for no-action
        self.n_direction_dim = n_direction_dim
        self.n_blocks_dim = n_block_dim
        self.n_previous_action_embedding = self.n_direction_dim + self.n_blocks_dim
        self.null_previous_action = (5, 20)
        self.previous_action_embedder = epa.EmbedPreviousAction(
            6, self.n_direction_dim, 21, self.n_blocks_dim, scope_name=scope_name)
        previous_action_embedding = self.previous_action_embedder.get_output()

        # Neural network for mixing the embeddings of text, image and previous action and generate q values
        self.mix_and_gen_q_val = mix_and_gen_q_values.MixAndGenerateQValues(
            self.n_text, self.n_image, self.n_previous_action_embedding,
            text_embedding, image_embedding, previous_action_embedding, 81, scope_name=scope_name)

        ####################
        # TODO BUG
        self.mix_and_gen_q_val_buckets = []
        for i in range(0, len(self.buckets)):
            mix_and_gen_q_val_bucket = mix_and_gen_q_values.MixAndGenerateQValues(
                self.n_text, self.n_image, self.n_previous_action_embedding,
                self.embed_token_seq_buckets[i].get_output(), image_embedding,
                previous_action_embedding, 81, create_copy=self.mix_and_gen_q_val, scope_name=scope_name)
            self.mix_and_gen_q_val_buckets.append(mix_and_gen_q_val_bucket)
        ####################

        # Define input and output
        self.target = tf.placeholder(dtype=tf.float32, shape=None)
        self.model_output = self.mix_and_gen_q_val.get_q_val()
        self.model_output_indices = tf.placeholder(dtype=tf.int32, shape=None)

        summary_qval_min = tf.scalar_summary("Q Val Min", tf.reduce_min(self.model_output))
        summary_qval_max = tf.scalar_summary("Q Val Max", tf.reduce_max(self.model_output))
        summary_qval_mean = tf.scalar_summary("Q Val Mean", tf.reduce_mean(self.model_output))

        self.feed_forward_summary = tf.merge_summary([summary_qval_min, summary_qval_max, summary_qval_mean])
        self.feed_iter = 0

    def get_bucket_network(self, num_tokens):

        for i in range(0, len(self.buckets)):
            if num_tokens <= self.buckets[i]:
                return self.mix_and_gen_q_val_buckets[i], self.embed_token_seq_buckets[i]
        return self.mix_and_gen_q_val, self.text_embedder

    def evaluate_qfunction(self, image_data, text_input_word_indices, text_mask, previous_action, sess):

        mix_and_gen_q_val_bucket, text_embedder_bucket = self.get_bucket_network(sum(text_mask[0]))

        image_data = np.concatenate(list(image_data), 2)
        q_val = mix_and_gen_q_val_bucket.get_q_val()

        raw_image_input = self.image_preprocessor.get_raw_image_input()
        final_image_output = self.image_preprocessor.get_final_image()
        image_datas = [final_image_output.eval(session=sess, feed_dict={raw_image_input: image_data})]

        image_placeholder = self.image_embedder.get_images_data()
        text_input = text_embedder_bucket.get_input()
        mask = text_embedder_bucket.get_zero_mask()
        batch_size = text_embedder_bucket.get_batch_size()
        direction_input, block_input = self.previous_action_embedder.get_input()

        result = sess.run(q_val, feed_dict={text_input: text_input_word_indices,
                                            mask: text_mask, batch_size: 1,
                                            image_placeholder: [image_datas],
                                            direction_input: [previous_action[0]],
                                            block_input: [previous_action[1]]})
        # self.train_writer.add_summary(result[1], self.feed_iter)
        # self.feed_iter += 1

        return result[0]

    def get_action_values(self, image_data, text_input_word_indices, text_mask, previous_action, sess):
        """ Return Q value for the given agent state. """

        q_values = self.evaluate_qfunction(
            image_data, text_input_word_indices, text_mask, previous_action, sess)

        return q_values
