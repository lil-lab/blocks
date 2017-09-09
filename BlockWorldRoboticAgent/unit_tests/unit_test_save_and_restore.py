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

saver = tf.train.Saver()

## Do feed forwarding over a batch
with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())

    my_batch_size = 2

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

    result1 = output.eval(session=sess,
                         feed_dict={text_embed_input: [text_input_word_indices_padded] * my_batch_size,
                                    mask: [input_mask] * my_batch_size, batch_size: my_batch_size, image_embed_input: [image_datas]})
    print result1

    ## Dump the variables to a file
    save_path = saver.save(sess, "./saved/model.ckpt")
    print("Model saved in file: %s" % save_path)

    sess.close()

## Read the model from file
with tf.Session() as sess1:

    saver = tf.train.Saver()

    saver.restore(sess1, "./saved/model.ckpt")
    print("Model restored.")

    # sess1.run(tf.initialize_all_variables())

    my_batch_size = 2

    image_data = tf.gfile.FastGFile("../img/Screenshot.png", 'r').read()
    all_image_data = [image_data] * my_batch_size

    raw_image_input = image_preprocessing.get_raw_image_input()
    final_image_output = image_preprocessing.get_final_image()

    image_datas = []
    for img_data in all_image_data:
        image_datas.append(final_image_output.eval(session=sess1, feed_dict={raw_image_input: img_data}))

    text_input_word_indices = text_embedder.convert_text_to_indices("Hey there, lets do cha-cha cha!")
    text_input_word_indices_padded = text_embedder.pad_indices(text_input_word_indices)
    input_mask = [1.0] * len(text_input_word_indices) + [0.0] * (max_steps - len(text_input_word_indices))

    result2 = output.eval(session=sess1,
                         feed_dict={text_embed_input: [text_input_word_indices_padded] * my_batch_size,
                                    mask: [input_mask] * my_batch_size, batch_size: my_batch_size, image_embed_input: [image_datas]})
    print result2

    sess1.close()

flag = True

if result1.shape == result2.shape:
    for i in range(0, result1.shape[0]):
        for j in range(0, result1.shape[1]):
            if not (result1[i][j] == result2[i][j]):
                flag = False
                break
else:
    raise Exception("Shapes dont match")

print "Match occured " + str(flag)