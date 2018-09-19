import tensorflow as tf
from spatial_transformer import transformer
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot

import os

train_data_path = os.environ["DATA_DIR"]+"/data/mnist_sequence1_sample_5distortions5x5.npz"


mnist_cluttered = np.load(train_data_path)

X_train = mnist_cluttered['X_train']
y_train = mnist_cluttered['y_train']
X_valid = mnist_cluttered['X_valid']
y_valid = mnist_cluttered['y_valid']
X_test = mnist_cluttered['X_test']
y_test = mnist_cluttered['y_test']


Y_train = dense_to_one_hot(y_train, n_classes=10)
Y_valid = dense_to_one_hot(y_valid, n_classes=10)
Y_test = dense_to_one_hot(y_test, n_classes=10)

x = tf.placeholder(tf.float32, [None, 1600])
y = tf.placeholder(tf.float32, [None, 10])
x_tensor = tf.reshape(x, [-1, 40, 40, 1])

W_fc_loc1 = weight_variable([1600, 20])
b_fc_loc1 = bias_variable([20])

W_fc_loc2 = weight_variable([20, 6])

initial = np.array([[1., 0, 0], [0, 1., 0]])
initial = initial.astype('float32')
initial = initial.flatten()
b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')


h_fc_loc1 = tf.nn.tanh(tf.matmul(x, W_fc_loc1) + b_fc_loc1)

keep_prob = tf.placeholder(tf.float32)
h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)

h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)

out_size = (40, 40)
h_trans = transformer(x_tensor, h_fc_loc2, out_size)

filter_size = 3
n_filters_1 = 16
W_conv1 = weight_variable([filter_size, filter_size, 1, n_filters_1])

b_conv1 = bias_variable([n_filters_1])

h_conv1 = tf.nn.relu(tf.nn.conv2d(input=h_trans, filter=W_conv1,strides=[1, 2, 2, 1],padding='SAME') +b_conv1)


n_filters_2 = 16
W_conv2 = weight_variable([filter_size, filter_size, n_filters_1, n_filters_2])
b_conv2 = bias_variable([n_filters_2])
h_conv2 = tf.nn.relu(tf.nn.conv2d(input=h_conv1,filter=W_conv2,strides=[1, 2, 2, 1],padding='SAME') + b_conv2)
h_conv2_flat = tf.reshape(h_conv2, [-1, 10 * 10 * n_filters_2])


n_fc = 1024
W_fc1 = weight_variable([10 * 10 * n_filters_2, n_fc])
b_fc1 = bias_variable([n_fc])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


W_fc2 = weight_variable([n_fc, 10])
b_fc2 = bias_variable([10])
y_logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y))
opt = tf.train.AdamOptimizer()
optimizer = opt.minimize(cross_entropy)
grads = opt.compute_gradients(cross_entropy, [b_fc_loc2])


correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))



sess = tf.Session()
sess.run(tf.global_variables_initializer())



iter_per_epoch = 100
n_epochs = 500
train_size = 10000

indices = np.linspace(0, 10000 - 1, iter_per_epoch)
indices = indices.astype('int')

for epoch_i in range(n_epochs):
    for iter_i in range(iter_per_epoch - 1):
        batch_xs = X_train[indices[iter_i]:indices[iter_i+1]]
        batch_ys = Y_train[indices[iter_i]:indices[iter_i+1]]

        if iter_i % 10 == 0:
            loss = sess.run(cross_entropy,feed_dict={x: batch_xs,y: batch_ys,keep_prob: 1.0})
            print('Iteration: ' + str(iter_i) + ' Loss: ' + str(loss))

        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.8})

    print('Accuracy (%d): ' % epoch_i + str(sess.run(accuracy,feed_dict={x: X_valid,y: Y_valid,keep_prob: 1.0})))


model_path = os.environ["RESULT_DIR"]+"/model"

classification_inputs = tf.saved_model.utils.build_tensor_info(X_train)
classification_outputs_classes = tf.saved_model.utils.build_tensor_info(y_train)

classification_signature = (tf.saved_model.signature_def_utils.build_signature_def(inputs={tf.saved_model.signature_constants.CLASSIFY_INPUTS:classification_inputs},outputs={tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:classification_outputs_classes},method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))
builder = tf.saved_model.builder.SavedModelBuilder(model_path)
legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],signature_def_map={'predict_images': classification_signature,},legacy_init_op=legacy_init_op)

save_path = str(builder.save())
