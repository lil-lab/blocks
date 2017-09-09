import embed_image
import tensorflow as tf

from model import image_preprocessing

image_preprocessing = image_preprocessing.ImagePreprocessing()
embedImage = embed_image.EmbedImage(20)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

image_data = tf.gfile.FastGFile("../img/Screenshot.png", 'r').read()
file_names = [image_data, image_data, image_data]

raw_image_input = image_preprocessing.get_raw_image_input()
final_image_output = image_preprocessing.get_final_image()

image_datas = []
for file_name in file_names:
    image_datas.append(final_image_output.eval(session=sess, feed_dict={raw_image_input: file_name}))

input = embedImage.get_images_data()
output = embedImage.get_output()

result = output.eval(session=sess, feed_dict= {input: [image_datas]})
print result
print "Verify that all rows are same"

sess.close()