import tensorflow as tf

from model import embed_token_seq

output_size = 50
embedTokenSeq = embed_token_seq.EmbedTokenSeq(output_size)
output = embedTokenSeq.get_output()
input = embedTokenSeq.get_input()
max_steps = embedTokenSeq.get_max_time_step()
mask = embedTokenSeq.get_zero_mask()
batch_size = embedTokenSeq.get_batch_size()

text_input_word_indices = embedTokenSeq.convert_text_to_indices("Hey there, lets do cha-cha cha!")
text_input_word_indices_padded = embedTokenSeq.pad_indices(text_input_word_indices)
input_mask = [1.0] * len(text_input_word_indices) + [0.0] * (max_steps - len(text_input_word_indices))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

result = output.eval(session=sess, feed_dict={input: [text_input_word_indices_padded, text_input_word_indices_padded],
                                              mask: [input_mask, input_mask], batch_size: 2})
print result
print result.shape
print "Verify that all rows are same"

sess.close()