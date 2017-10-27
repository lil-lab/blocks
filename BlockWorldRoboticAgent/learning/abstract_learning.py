import replay_memory as rm
import tensorflow as tf
from experiments.constants import constants


class AbstractLearning:
    """ Abstract learning class that handles functions for a general learning algorithm """

    batch_size = constants["batch_size"]
    mle_learning_rate = constants["mle_learning_rate"]
    rl_learning_rate = constants["rl_learning_rate"]
    max_epochs = constants["max_epoch"]
    models_to_keep = constants["models_to_keep"]
    max_patience = constants["max_patience"]
    dataset_size = constants["train_size"]
    validation_datasize = constants["validation_size"]

    def __init__(self, model, loss, train_step, update_summaries):
        """ Creates constructor for an abstract learning setup """

        self.model = model
        self.loss = loss
        self.train_step = train_step
        self.update_summary = tf.merge_summary(update_summaries)
        self.update_iter = 0

    def min_loss(self, sample, sess, train_writer, factorized_actions=True):
        """ Performs one iteration of backpropagation and minimizes the loss.
        @:param: sample
        @:param sess
        @:param train_writer
        @:param factorized_actions: True if the model contains actions which are factorized i.e.
                separate block and direction and False otherwise.
        """

        # Placeholders for the input
        text_input = self.model.text_embedder.get_input()
        mask = self.model.text_embedder.get_zero_mask()
        image_input = self.model.image_embedder.get_images_data()
        direction_input, block_input = self.model.previous_action_embedder.get_input()

        # Placeholders for image preprocessing
        raw_image_input = self.model.image_preprocessor.get_raw_image_input()
        final_image_output = self.model.image_preprocessor.get_final_image()

        batch_image_datas = []
        batch_text_input_word_indices = []
        batch_input_mask = []
        batch_block_index = []
        batch_direction_index = []
        batch_action_index = []
        batch_target = []
        batch_previous_direction = []
        batch_previous_block = []

        for replay_item in sample:
            instruction_word_indices = rm.ReplayMemory.get_instruction_word_indices(replay_item)
            instruction_mask = rm.ReplayMemory.get_instruction_mask(replay_item)
            action = rm.ReplayMemory.get_action(replay_item)
            target = rm.ReplayMemory.get_target(replay_item)
            state = rm.ReplayMemory.get_history_of_states(replay_item)
            previous_action = rm.ReplayMemory.get_previous_action_id(replay_item)

            batch_text_input_word_indices.append(instruction_word_indices[0])
            batch_input_mask.append(instruction_mask[0])
            if factorized_actions:
                batch_block_index.append(action[0])
                batch_direction_index.append(action[1])
            else:
                batch_action_index.append(action)
            batch_target.append(target)
            batch_previous_direction.append(previous_action[0])
            batch_previous_block.append(previous_action[1])

            batch_image_datas.append(final_image_output.eval(session=sess, feed_dict={raw_image_input: state}))

        if factorized_actions:
            result = sess.run([self.loss, self.train_step, self.update_summary],
                              feed_dict={text_input: batch_text_input_word_indices, mask: batch_input_mask,
                                         self.model.text_embedder.get_batch_size(): len(sample),
                                         image_input: [batch_image_datas],
                                         self.model.block_indices: batch_block_index,
                                         self.model.direction_indices: batch_direction_index,
                                         self.model.target: batch_target,
                                         block_input: batch_previous_block, direction_input: batch_previous_direction})
        else:
            result = sess.run([self.loss, self.train_step, self.update_summary],
                              feed_dict={text_input: batch_text_input_word_indices, mask: batch_input_mask,
                                         self.model.text_embedder.get_batch_size(): len(sample),
                                         image_input: [batch_image_datas],
                                         self.model.model_output_indices: batch_action_index,
                                         self.model.target: batch_target,
                                         block_input: batch_previous_block, direction_input: batch_previous_direction})

        train_writer.add_summary(result[2], self.update_iter)
        self.update_iter += 1

        return result[0]
