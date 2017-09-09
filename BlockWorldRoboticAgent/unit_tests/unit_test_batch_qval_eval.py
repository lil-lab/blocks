import time

import embed_image
import tensorflow as tf

from model import embed_token_seq, image_preprocessing, mix_and_gen_prob

### Create the computation graph
n_text_output = 20
text_embedder = embed_token_seq.EmbedTokenSeq(n_text_output)
text_embed_input = text_embedder.get_input()
text_embed_output = text_embedder.get_output()
max_steps = text_embedder.get_max_time_step()
mask = text_embedder.get_zero_mask()
batch_size = text_embedder.get_batch_size()

image_preprocessing = image_preprocessing.ImagePreprocessing()

n_image_output = 250
image_embedder = embed_image.EmbedImage(n_image_output)
image_embed_input = image_embedder.get_images_data()
image_embed_output = image_embedder.get_output()

n_actions = 81
mix_text_image = mix_and_gen_prob.MixAndGenerateProbabilities(n_text_output, n_image_output, text_embed_output,
                                                              image_embed_output, n_actions)
output = mix_text_image.get_joined_probabilities()

## Do feed forwarding over a batch
sess = tf.Session()
sess.run(tf.initialize_all_variables())

start = time.time()
for i in range(1, 20):
    my_batch_size = 32

    image_data = tf.gfile.FastGFile("../img/Screenshot.png", 'r').read()
    all_image_data = [image_data] * my_batch_size

    raw_image_input = image_preprocessing.get_raw_image_input()
    final_image_output = image_preprocessing.get_final_image()

    image_datas = []
    for img_data in all_image_data:
        image_datas.append(final_image_output.eval(session=sess, feed_dict={raw_image_input: img_data}))

    text_input_word_indices = text_embedder.convert_text_to_indices("Hey there, lets do cha-cha cha!")
    text_input_word_indices_padded = text_embedder.pad_indices(text_input_word_indices)
    input_mask = [1.0] * len(text_input_word_indices) + [0.0] * (max_steps - len(text_input_word_indices))

    result = output.eval(session=sess,
                         feed_dict={text_embed_input: [text_input_word_indices_padded] * my_batch_size,
                                    mask: [input_mask] * my_batch_size, batch_size: my_batch_size, image_embed_input: [image_datas]})
    print result
    print result.shape

end = time.time()
sess.close()
print "Time taken " + str(end - start) + " seconds "
