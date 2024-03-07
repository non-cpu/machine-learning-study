import tensorflow as tf

tf.compat.v1.disable_v2_behavior()

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('./mnist/data/', one_hot=True)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

X = tf.compat.v1.compat.v1.placeholder(tf.compat.v1.float32, [None, 28, 28, 1])
Y = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, 10])
keep_prob = tf.compat.v1.placeholder(tf.compat.v1.float32)

W1 = tf.compat.v1.Variable(
    tf.compat.v1.random_normal([3, 3, 1, 32], stddev=0.01))
L1 = tf.compat.v1.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.compat.v1.nn.relu(L1)
L1 = tf.compat.v1.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[
    1, 2, 2, 1], padding='SAME')

W2 = tf.compat.v1.Variable(
    tf.compat.v1.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.compat.v1.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.compat.v1.nn.relu(L2)
L2 = tf.compat.v1.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[
    1, 2, 2, 1], padding='SAME')

W3 = tf.compat.v1.Variable(
    tf.compat.v1.random_normal([7 * 7 * 64, 256], stddev=0.01))
L3 = tf.compat.v1.reshape(L2, [-1, 7 * 7 * 64])
L3 = tf.compat.v1.matmul(L3, W3)
L3 = tf.compat.v1.nn.relu(L3)
L3 = tf.compat.v1.nn.dropout(L3, keep_prob)

W4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([256, 10], stddev=0.01))
model = tf.compat.v1.matmul(L3, W4)

cost = tf.compat.v1.reduce_mean(
    tf.compat.v1.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.compat.v1.train.AdamOptimizer(0.001).minimize(cost)

init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)

    batch_size = 100
    total_batch = int(len(x_train) / batch_size)
    for epoch in range(15):
        total_cost = 0

        for i in range(total_batch):
            batch_xs = x_train[i * batch_size: (i+1)*batch_size]
            batch_ys = y_train[i * batch_size: (i+1)*batch_size]

            batch_xs = batch_xs.reshape(-1, 28, 28, 1)
            batch_ys = tf.one_hot(batch_ys, 10).eval()

            _, cost_val = sess.run(
                [optimizer, cost],
                feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7},
            )
            total_cost += cost_val

        print(
            'Epoch:', '%04d' % (epoch + 1),
            'Avg. cost =', '{:.3f}'.format(total_cost / total_batch)
        )

    print('Optimization Complete!')

    is_correct = tf.compat.v1.equal(
        tf.compat.v1.argmax(model, 1), tf.compat.v1.argmax(Y, 1))
    accuracy = tf.compat.v1.reduce_mean(
        tf.compat.v1.cast(is_correct, tf.compat.v1.float32))

    print('accuracy:', sess.run(
        accuracy,
        feed_dict={
            X: x_test.reshape(-1, 28, 28, 1), Y: tf.one_hot(y_test, 10).eval(), keep_prob: 1.0}
    )
    )
