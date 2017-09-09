import tensorflow as tf
import numpy as np

image_dim = 120
height = image_dim
width = image_dim
channels = 3
image_embed = 200
time_horizon = 3
output_size = image_embed

scope_name = "CNN"

def _variable_on_cpu(name, shape, initializer):
    var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    return var

def _create_variable_(name, shape, initializer):
    var = tf.get_variable(name, shape, initializer=initializer)
    return var

images_data = tf.placeholder(dtype=tf.float32, shape=None, name=scope_name + "_input")
batchsize = tf.shape(images_data)[0]
float_images = tf.reshape(images_data, [batchsize * time_horizon, width, height, channels])

# conv + affine + relu
with tf.variable_scope(scope_name + '_conv1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[8, 8, channels, 32], stddev=0.005, wd=0.0)
    conv = tf.nn.conv2d(float_images, kernel, [1, 4, 4, 1], padding='SAME')
    biases = _create_variable_('biases', [32], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)

# conv + affine + relu
with tf.variable_scope(scope_name + '_conv2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[8, 8, 32, 32], stddev=0.005, wd=0.0)
    conv = tf.nn.conv2d(conv1, kernel, [1, 4, 4, 1], padding='SAME')
    biases = _create_variable_('biases', [32], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)

# conv + affine + relu
with tf.variable_scope(scope_name + '_conv3') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[4, 4, 32, 32], stddev=0.005, wd=0.0)
    conv = tf.nn.conv2d(conv2, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _create_variable_('biases', [32], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope.name)

# affine
with tf.variable_scope(scope_name + '_linear') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(conv3, [batchsize * time_horizon, -1])
    # Value before is hacked
    # Not sure how to fix it
    # It if based on image dimension
    dim = 512
    weights = _variable_with_weight_decay('weights', [dim, image_embed], stddev=0.004, wd=0.004)
    biases = _create_variable_('biases', [image_embed], tf.constant_initializer(0.0))
    image_embedding = tf.add(tf.matmul(reshape, weights), biases, name=scope.name)

reshaped_image_embeddings = tf.reshape(image_embedding, [batchsize, time_horizon, -1])

# Create a Tracking RNN
'''lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(output_size, forget_bias=1.0, state_is_tuple=True)
cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell], state_is_tuple=True)
_initial_state = cell.zero_state(batchsize, tf.float32)

# Zero Masking for RNN
mask = tf.placeholder(tf.float32, [None, time_horizon])

# Pass the CNN embeddings through RNN
outputs = []
state = _initial_state
with tf.variable_scope(scope_name):
    for time_step in range(time_horizon):
        if time_step > 0:
            tf.get_variable_scope().reuse_variables()
        print reshaped_image_embeddings[:, 0, :]
        (cell_output, state) = cell(reshaped_image_embeddings[:, time_step, :], state)
        exit(0)
        zero_mask = mask[:, time_step]
        zero_mask = tf.reshape(zero_mask, [batchsize, 1])
        masked_output = tf.mul(cell_output, zero_mask)
        outputs.append(masked_output)

temporal_sum = tf.reduce_sum(outputs, 0)
num_frames = tf.reduce_sum(mask, 1)
num_frames = tf.reshape(num_frames, [batchsize, 1])
output = tf.div(temporal_sum, num_frames)'''

# Create mask.
mask_ls = []
for i in range(0, time_horizon + 1):
    maski = [[1.0] * i + [0.0] * (time_horizon - i)]
    mask_ls.append(maski)


#########################
sess = tf.Session()
sess.run(tf.initialize_all_variables())

e = np.full((image_dim, image_dim, 3), 1)
f = np.full((image_dim, image_dim, 3), 2)
g = np.full((image_dim, image_dim, 3), 3)

a = np.array([e, f, g])
b = np.array([f, e, g])
input_vector = np.array([a, b, a])
res = reshaped_image_embeddings.eval(session=sess, feed_dict={images_data: input_vector})

p = res[0]

p1 = res[0][0]
p2 = res[0][1]
p3 = res[0][2]

q = res[1]

q1 = res[1][0]
q2 = res[1][1]
q3 = res[1][2]

r = res[2]

r1 = res[2][0]
r2 = res[2][1]
r3 = res[2][2]

print str(np.linalg.norm(p - q)) + " should not be ideally 0"
print str(np.linalg.norm(p - r)) + " should be 0"

print str(np.linalg.norm(p1 - q2)) + " should be 0"
print str(np.linalg.norm(p2 - q1)) + " should be 0"
print str(np.linalg.norm(p3 - q3)) + " should be 0"

print str(np.linalg.norm(p1 - q1)) + " should not be ideally 0"
print str(np.linalg.norm(p2 - q2)) + " should not be ideally 0"
#########################