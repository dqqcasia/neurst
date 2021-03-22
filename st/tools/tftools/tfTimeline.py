import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
from tensorflow.python.client import timeline


def testSimple():
    """
    https://towardsdatascience.com/howto-profile-tensorflow-1a49fb18073d
    """
    a = tf.random_normal([2000, 5000])
    b = tf.random_normal([5000, 1000])
    res = tf.matmul(a, b)

    with tf.Session() as sess:
        # add additional options to trace the session execution
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        sess.run(res, options=options, run_metadata=run_metadata)

        # Create the Timeline object, and write it to a json file
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('timeline_01.json', 'w') as f:
            f.write(chrome_trace)


def testComplicated():
    """
    https://towardsdatascience.com/howto-profile-tensorflow-1a49fb18073d
    """
    import tempfile

    import tensorflow as tf
    from tensorflow.contrib.layers import fully_connected as fc
    from tensorflow.examples.tutorials.mnist import input_data
    from tensorflow.python.client import timeline

    batch_size = 100

    inputs = tf.placeholder(tf.float32, [batch_size, 784])
    targets = tf.placeholder(tf.float32, [batch_size, 10])

    with tf.variable_scope("layer_1"):
        fc_1_out = fc(inputs, num_outputs=500, activation_fn=tf.nn.sigmoid)
    with tf.variable_scope("layer_2"):
        fc_2_out = fc(fc_1_out, num_outputs=784, activation_fn=tf.nn.sigmoid)
    with tf.variable_scope("layer_3"):
        logits = fc(fc_2_out, num_outputs=10)

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets))
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    mnist_save_dir = os.path.join(tempfile.gettempdir(), 'MNIST_data')
    mnist = input_data.read_data_sets(mnist_save_dir, one_hot=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        for i in range(3):
            batch_input, batch_target = mnist.train.next_batch(batch_size)
            feed_dict = {inputs: batch_input,
                         targets: batch_target}

            sess.run(train_op,
                     feed_dict=feed_dict,
                     options=options,
                     run_metadata=run_metadata)

            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open('timeline_02_step_%d.json' % i, 'w') as f:
                f.write(chrome_trace)


if __name__ == '__main__':
    testComplicated()
