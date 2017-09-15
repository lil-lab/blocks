import tensorflow as tf


class ImagePreprocessing:
    """ Class that performs preprocessing steps on the raw image received from the simulator."""

    def __init__(self):
        """ Computation graph for image processing """
        self.raw_image = tf.placeholder(shape=(None, None, 15), dtype=tf.float32)
        whitened_image = tf.image.per_image_standardization(self.raw_image)
        self.final_image = whitened_image

    def get_final_image(self):
        return self.final_image

    def get_raw_image_input(self):
        return self.raw_image

    def get_standardized_image(self, image, sess):
        return self.final_image.eval(session=sess, feed_dict={self.raw_image: image})
