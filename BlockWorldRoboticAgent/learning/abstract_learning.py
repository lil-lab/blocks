import replay_memory as rm
import tensorflow as tf


class AbstractLearning:
    """ Abstract learning class that handles functions for a general learning algorithm """

    batch_size = 32
    mle_learning_rate = 0.001
    rl_learning_rate = 0.00025
    max_epochs = 1000
    models_to_keep = 200
    max_patience = 1000
    dataset_size = 6003
    validation_datasize = 351

    def __init__(self, model, loss, train_step, update_summaries):
        """ Creates constructor for an abstract learning setup """

        self.model = model
        self.loss = loss
        self.train_step = train_step
        self.update_summary = tf.merge_summary(update_summaries)
        self.update_iter = 0

    def min_loss(self, sample, sess, train_writer, factorized_actions=True):
        """ Performs one iteration of backpropagation and minimize the loss.
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

        image_datas = []
        text_input_word_indices = []
        input_mask = []
        block_indices_ = []
        direction_indices_ = []
        action_indices_ = []
        target_ = []
        directions_ = []
        blocks_ = []

        for replay_item in sample:
            instruction_word_indices = rm.ReplayMemory.get_instruction_word_indices(replay_item)
            instruction_mask = rm.ReplayMemory.get_instruction_mask(replay_item)
            action = rm.ReplayMemory.get_action(replay_item)
            target = rm.ReplayMemory.get_target(replay_item)
            state = rm.ReplayMemory.get_history_of_states(replay_item)
            previous_action = rm.ReplayMemory.get_previous_action_id(replay_item)

            text_input_word_indices.append(instruction_word_indices[0])
            input_mask.append(instruction_mask[0])
            if factorized_actions:
                block_indices_.append(action[0])
                direction_indices_.append(action[1])
            else:
                action_indices_.append(action)
            target_.append(target)
            directions_.append(previous_action[0])
            blocks_.append(previous_action[1])

            image_datas.append(final_image_output.eval(session=sess, feed_dict={raw_image_input: state}))

        if factorized_actions:
            result = sess.run([self.loss, self.train_step, self.update_summary],
                              feed_dict={text_input: text_input_word_indices, mask: input_mask,
                                         self.model.text_embedder.get_batch_size(): len(sample),
                                         image_input: [image_datas],
                                         self.model.block_indices: block_indices_,
                                         self.model.direction_indices: direction_indices_, self.model.target: target_,
                                         block_input: blocks_, direction_input: directions_})
        else:
            result = sess.run([self.loss, self.train_step, self.update_summary],
                              feed_dict={text_input: text_input_word_indices, mask: input_mask,
                                         self.model.text_embedder.get_batch_size(): len(sample),
                                         image_input: [image_datas],
                                         self.model.model_output_indices: action_indices_, self.model.target: target_,
                                         block_input: blocks_, direction_input: directions_})

        train_writer.add_summary(result[2], self.update_iter)
        self.update_iter += 1

        return result[0]
