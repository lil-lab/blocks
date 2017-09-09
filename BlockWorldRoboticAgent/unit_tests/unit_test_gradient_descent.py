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

n_actions = 80
mix_text_image = mix_and_gen_prob.MixAndGenerateProbabilities(n_text_output, n_image_output, text_embed_output,
                                                              image_embed_output, n_actions)
output = mix_text_image.get_joined_probabilities()

### Compute the loss
target = tf.placeholder(dtype=tf.float32, shape=None)
indices = tf.placeholder(dtype=tf.int32, shape=None)

indices_flattened = tf.range(0, tf.shape(output)[0]) * tf.shape(output)[1] + indices

optimizer = tf.train.AdamOptimizer(0.0001)

q_val_action = tf.gather(tf.reshape(output, [-1]),
              indices_flattened)
loss = tf.reduce_mean(tf.square(tf.sub(q_val_action, target)))
train_step = optimizer.minimize(loss)

## Perform optimization
sess = tf.Session()
sess.run(tf.initialize_all_variables())

my_batch_size = 6
indices_ = [0, 0, 0, 1, 1, 1]
target_ = [0.5, 0.5, 0.5, -0.5, -0.5, -0.5]

for t in range(0, 500):

    image_data = tf.gfile.FastGFile("../img/Screenshot.png", 'r').read()
    file_names = [image_data] * my_batch_size

    raw_image_input = image_preprocessing.get_raw_image_input()
    final_image_output = image_preprocessing.get_final_image()

    image_datas = []
    for file_name in file_names:
        image_datas.append(final_image_output.eval(session=sess, feed_dict={raw_image_input: file_name}))

    text_input_word_indices = text_embedder.convert_text_to_indices("Hey there, lets do cha-cha cha!")
    text_input_word_indices_padded = text_embedder.pad_indices(text_input_word_indices)
    input_mask = [1.0] * len(text_input_word_indices) + [0.0] * (max_steps - len(text_input_word_indices))

    result = sess.run([loss, q_val_action, train_step],
                       feed_dict={text_embed_input: [text_input_word_indices_padded] * my_batch_size,
                                  mask: [input_mask], batch_size: my_batch_size, image_embed_input: [image_datas],
                                  indices: indices_, target: target_})
    print "Epoch: " + str(t) + " Loss = " + str(result[0])  + " val " + str(result[1])

print "Target was " + str(target_)
sess.close()
