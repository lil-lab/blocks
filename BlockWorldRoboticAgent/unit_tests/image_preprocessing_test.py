import tensorflow as tf
import matplotlib.pyplot as plt

def image_preprocess(file_name):
    height = 50 # 410 # 785 # IMAGE_SIZE
    width =  50 # 410 # 421 # IMAGE_SIZE

    image_data = tf.gfile.FastGFile(file_name, 'r').read()
    u = tf.image.decode_png(image_data, channels=3)
    v = tf.image.convert_image_dtype(u, tf.float32)

    resized_image = tf.image.per_image_whitening(resized_image)

    # resized_image = resize_image_patch.resize_image_with_crop_or_pad(v,
    #                                     width, height, dynamic_shape=True)

    sess = tf.Session()
    result = resized_image.eval(session=sess)
    sess.close()

    print result

    return result

result1 = image_preprocess("../img/Screenshot.png")
#result2 = image_preprocess("../img/Screenshot_1.png")
#print np.linalg.norm(result1 - result2)

plt.imshow(result1)
plt.show()
#
# plt.imshow(result2)
# plt.show()
