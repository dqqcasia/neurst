#!/usr/bin/env
# coding=utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf
import logging
import sys
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')

from configs.arguments import args


# ==============================
# the TF dataset examples
# ==============================
def simple_dataset_with_error():
    """
    https://github.com/adventuresinML/adventures-in-ml-code/blob/master/tf_dataset_tutorial.py
    """
    x = np.arange(0, 10)
    # create dataset object from the numpy array
    dx = tf.data.Dataset.from_tensor_slices(x)
    # create a one-shot iterator
    iterator = dx.make_one_shot_iterator()
    # extract an element
    next_element = iterator.get_next()
    with tf.Session() as sess:
        for i in range(11):
            try:
                val = sess.run(next_element)
                print(val)
            except tf.errors.OutOfRangeError:
                print('out of range!!')
                break


def simple_dataset_initializer():
    x = np.arange(0, 10)
    dx = tf.data.Dataset.from_tensor_slices(x)
    # create an initializable iterator
    iterator = dx.make_initializable_iterator()
    # extract an element
    next_element = iterator.get_next()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in range(15):
            val = sess.run(next_element)
            print(val)
            if i % 9 == 0 and i > 0:
                sess.run(iterator.initializer)


def simple_dataset_batch():
    x = np.arange(0, 10)
    dx = tf.data.Dataset.from_tensor_slices(x).batch(3)
    # create a one-shot iterator
    iterator = dx.make_initializable_iterator()
    # extract an element
    next_element = iterator.get_next()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in range(15):
            val = sess.run(next_element)
            print(val)
            if (i + 1) % (10 // 3) == 0 and i > 0:
                sess.run(iterator.initializer)


def simple_zip_example():
    x = np.arange(0, 10)
    y = np.arange(1, 11)
    # create dataset objects from the arrays
    dx = tf.data.Dataset.from_tensor_slices(x)
    dy = tf.data.Dataset.from_tensor_slices(y)
    # zip the two datasets together
    dcomb = tf.data.Dataset.zip((dx, dy)).batch(3)
    iterator = dcomb.make_initializable_iterator()
    # extract an element
    next_element = iterator.get_next()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in range(15):
            val = sess.run(next_element)
            print(val)
            if (i + 1) % (10 // 3) == 0 and i > 0:
                sess.run(iterator.initializer)


# ==============================
# the TF queue examples
# ==============================
def FIFO_queue_demo_no_coord():
    # first let's create a simple random normal Tensor to act as dummy input data
    # this operation should be run more than once, everytime the queue needs filling
    # back up.  However, it isn't in this case, because of our lack of a co-ordinator/
    # proper threading
    dummy_input = tf.random_normal([3], mean=0, stddev=1)
    # let's print so we can see when this operation is called
    dummy_input = tf.Print(dummy_input, data=[dummy_input],
                           message='New dummy inputs have been created: ', summarize=6)
    # create a FIFO queue object
    q = tf.FIFOQueue(capacity=3, dtypes=tf.float32)
    # load up the queue with our dummy input data
    enqueue_op = q.enqueue_many(dummy_input)
    # grab some data out of the queue
    data = q.dequeue()
    # now print how much is left in the queue
    data = tf.Print(data, data=[q.size()], message='This is how many items are left in q: ')
    # create a fake graph that we can call upon
    fg = data + 1
    # now run some operations
    with tf.Session() as sess:
        # first load up the queue
        sess.run(enqueue_op)
        # now dequeue a few times, and we should see the number of items
        # in the queue decrease
        sess.run(fg)
        sess.run(fg)
        sess.run(fg)
        # by this stage the queue will be emtpy, if we run the next time, the queue
        # will block waiting for new data
        sess.run(fg)
        # this will never print:
        print("We're here!")


def simple_shuffle_batch(source, capacity, batch_size=10, num_threads=1, allow_smaller_final_batch=True):
    # Create a random shuffle queue.
    queue = tf.RandomShuffleQueue(capacity=capacity,
                                  min_after_dequeue=int(0.9*capacity),
                                  shapes=source.shape, dtypes=source.dtype)
    # Create an op to enqueue one item.
    enqueue = queue.enqueue(source)

    # Create a queue runner that, when started, will launch 4 threads applying
    # that enqueue op.
    # the num_thread won't eff
    qr = tf.train.QueueRunner(queue, [enqueue],
                              queue_closed_exception_types=(tf.errors.CancelledError,
                                                            tf.errors.OutOfRangeError))

    # Register the queue runner so it can be found and started by
    # `tf.train.start_queue_runners` later (the threads are not launched yet).
    tf.train.add_queue_runner(qr)

    # Create an op to dequeue a batch
    if not allow_smaller_final_batch:
        return queue.dequeue_many(batch_size), queue.dequeue()
    else:
        return queue.dequeue_up_to(batch_size), queue.dequeue()


def simple_FIFO_batch(source, capacity, batch_size=10, num_threads=1, allow_smaller_final_batch=True):
    # Create a random shuffle queue.
    queue = tf.FIFOQueue(capacity=capacity, dtypes=source.dtype)
    # Create an op to enqueue one item.
    enqueue = queue.enqueue(source)
    qr = tf.train.QueueRunner(queue, [enqueue])
    tf.train.add_queue_runner(qr)

    # Create an op to dequeue a batch
    if not allow_smaller_final_batch:
        return queue.dequeue_many(batch_size), queue.dequeue()
    else:
        return queue.dequeue_up_to(batch_size), queue.dequeue()


def testFIFOQueue():
    """
    FIFOQueue's DequeueMany and DequeueUpTo require the components to have specified shapes.
    """
    dataset = tf.data.Dataset.from_tensor_slices(tf.constant(list(range(31))))
    iter_data = dataset.make_one_shot_iterator().get_next()
    get_batch, get_one = simple_FIFO_batch(iter_data, capacity=40)

    with tf.train.MonitoredSession() as sess:
        for i in range(100):
            batch = sess.run(get_batch)
            logging.info("batch shape: {}".format(batch.shape))


def testPaddingFIFOQueue():
    """
    shapes: A list of TensorShape objects, with the same length as dtypes.
    Any dimension in the TensorShape containing value None is dynamic
    and allows values to be enqueued with variable size in that dimension.
    """
    dataset = tf.data.Dataset.from_tensor_slices(tf.constant(list(range(31))))
    iter_data = dataset.make_one_shot_iterator().get_next()
    queue = tf.PaddingFIFOQueue(capacity=30,
                                dtypes=(tf.int32, tf.int32),
                                shapes=([], [None]))
    # Create an op to enqueue one item.
    enqueue = queue.enqueue([iter_data, tf.ones([iter_data], dtype=tf.int32)])
    qr = tf.train.QueueRunner(queue, [enqueue])
    tf.train.add_queue_runner(qr)

    tensor_batch = queue.dequeue_up_to(10)
    # queue.dequeue_up_to(batch_size), queue.dequeue()

    with tf.train.MonitoredTrainingSession() as sess:
        for i in range(100):
            batch = sess.run(tensor_batch)
            logging.info("batch shape: {}".format(batch))


def assert_enqueue():
    """
    https://www.tensorflow.org/versions/r1.4/api_guides/python/threading_and_queues#Queue_usage_overview
    """

    # create a dataset that counts from 0 to 99
    dataset = tf.data.Dataset.from_tensor_slices(tf.constant(list(range(31))))
    iter_data = dataset.make_one_shot_iterator().get_next()

    # Create a slightly shuffled batch from the sorted elements
    get_batch, get_one = simple_shuffle_batch(iter_data, capacity=40)

    top_queue = tf.FIFOQueue(20, dtypes=(tf.int32))
    num_samples_in_queue = top_queue.size()
    op_top_enqueue = top_queue.enqueue_many(get_batch)
    qr = tf.train.QueueRunner(top_queue, [op_top_enqueue])
    tf.train.add_queue_runner(qr)
    data_t = top_queue.dequeue()

    # `MonitoredSession` will start and manage the `QueueRunner` threads.
    with tf.train.MonitoredSession() as sess:
        # Since the `QueueRunners` have been started, data is available in the
        # queue, so the `sess.run(get_batch)` call will not hang.

        for i in range(100):
            try:
                print('num_samples_in_queue:', sess.run(num_samples_in_queue))
                print(sess.run(data_t))
            except tf.errors.OutOfRangeError:
                print('{}   out of range!!'.format(i))
                break


def test_B():
    """
    http://adventuresinmachinelearning.com/tensorflow-dataset-tutorial/
    """
    dataset = tf.data.Dataset.range(5)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.train.MonitoredTrainingSession() as sess:
        max_value = tf.placeholder(tf.int64, shape=[])
        dataset = tf.data.Dataset.range(max_value)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        # Initialize an iterator over a dataset with 10 elements.
        sess.run(iterator.initializer, feed_dict={max_value: 10})
        for i in range(10):
            value = sess.run(next_element)
            assert i == value

        # Initialize the same iterator over a dataset with 100 elements.
        sess.run(iterator.initializer, feed_dict={max_value: 100})
        for i in range(100):
            value = sess.run(next_element)
            assert i == value


def test_queue_and_bucket():
    """
    https://zhuanlan.zhihu.com/p/27238630
    """
    from dataProcessing.tfRecoderData import TFReader

    dataReader = TFReader(args.dir_dev_data, args=args)
    tfqueue_filename = tf.train.string_input_producer(dataReader.list_tfdata_filenames,
                                                      num_epochs=2,
                                                      shuffle=False)
    reader_tfRecord = tf.TFRecordReader()
    _, serialized_example = reader_tfRecord.read(tfqueue_filename)
    raw_example = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'feat': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
        })
    tensor_seq_features = tf.reshape(tf.decode_raw(raw_example['feat'], tf.float32),
                              [-1, args.dim_feature])
    tensor_seq_labels = tf.decode_raw(raw_example['label'], tf.int32)

    with tf.train.MonitoredTrainingSession() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for _ in range(10):
            try:
                seq_labels = sess.run(tensor_seq_labels)
            except tf.errors.OutOfRangeError:
                logging.info("{out of range}")

        coord.request_stop()
        coord.join(threads)


def test_queue():
    """
    https://www.jianshu.com/p/d063804fb272
    """
    data = 10 * np.ones((100, 4))
    target = 10 * np.ones(100)

    queue = tf.FIFOQueue(capacity=200, dtypes=[tf.float32, tf.int32], shapes=[[4], []])
    op_close = queue.close()

    enqueue_op = queue.enqueue_many([data, target])
    data_sample, label_sample = queue.dequeue_up_to(60)

    qr = tf.train.QueueRunner(queue, [enqueue_op] * 2)
    tf.train.add_queue_runner(qr)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        data_batch, label_batch = sess.run([data_sample, label_sample])
        print('hello')
        coord.request_stop()
        coord.join(threads)

def test_filename_queue_reader():
    """
    https://www.jianshu.com/p/d063804fb272
    """
    import tensorflow as tf

    filename_queue = tf.train.string_input_producer(["data1.csv", "data2.csv"])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for _ in range(100):
            pass
        coord.request_stop()
        coord.join(threads)


def testBucketBySequenceLength(allow_small_batch, bucket_capacities=None):
    from tfTools.tfQueueBucket import bucket_by_sequence_length
    import time
    # All inputs must be identical lengths across tuple index.
    # The input reader will get input_length from the first tuple
    # entry.

    # Make capacity very large so we can feed all the inputs in the
    # main thread without blocking

    dataset = tf.data.Dataset.from_tensor_slices(tf.constant(list(range(1, 13))))

    def map_func(x):
        return x, tf.random_uniform(x.shape)
    dataset = dataset.map(map_func, 2)
    iter_data = dataset.make_one_shot_iterator().get_next()

    input_queue = tf.FIFOQueue(50, dtypes=(tf.int32, tf.float32), name='input_queue')
    num_in_input_queue = input_queue.size()
    is_input_queue_closed = input_queue.is_closed()
    op_close_input_queue = input_queue.close()
    input_enqueue_op = input_queue.enqueue(iter_data)
    qr = tf.train.QueueRunner(input_queue, [input_enqueue_op]*3,
                              queue_closed_exception_types=(tf.errors.CancelledError,
                                                            tf.errors.OutOfRangeError))
    tf.train.add_queue_runner(qr)
    lengths_t, data_t = input_queue.dequeue()
    lengths_t.set_shape(())
    data_t.set_shape(lengths_t.shape)

    out_lengths_t, data_and_labels_t, top_queue, list_bucket_queues =\
                                    bucket_by_sequence_length(
                                            input_length=lengths_t,
                                            tensors=[data_t],
                                            batch_size=[6, 4, 3, 1],
                                            bucket_boundaries=[3, 5, 10],
                                            capacity=20,
                                            allow_smaller_final_batch=allow_small_batch,
                                            num_threads=10)
    num_in_top_queue = top_queue.size()
    nums_in_bucket_queues = [queue.size() for queue in list_bucket_queues]

    is_top_queue_closed = top_queue.is_closed()
    is_bucket_queues_closed = [queue.is_closed() for queue in list_bucket_queues]

    list_op_close_bucket_queues = [queue.close() for queue in list_bucket_queues]
    with tf.Session() as sess:
        time.sleep(1)
        print('\nnum_in_input_queue: ', sess.run(num_in_input_queue),
              '\nnum_in_bucket_queues: ', sess.run(nums_in_bucket_queues),
              '\nnum_in_top_queue: ', sess.run(num_in_top_queue),
              '\n\ninput_queue state: ', sess.run(is_input_queue_closed),
              '\nbuckets queue state: ', sess.run(is_bucket_queues_closed),
              '\ntop queue state: ', sess.run(is_top_queue_closed))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        time.sleep(1)
        # for _ in range(32):
        #     sess.run(input_enqueue_op)
        # Read off the top of the bucket and ensure correctness of output
        print('\nnum_in_input_queue: ', sess.run(num_in_input_queue),
              '\nnum_in_bucket_queues: ', sess.run(nums_in_bucket_queues),
              '\nnum_in_top_queue: ', sess.run(num_in_top_queue),
              '\n\ninput_queue state: ', sess.run(is_input_queue_closed),
              '\nbuckets queue state: ', sess.run(is_bucket_queues_closed),
              '\ntop queue state: ', sess.run(is_top_queue_closed))

        coord.request_stop()
        coord.join(threads)


def testSpeedOfBucket():
    import tfTools.tfRecoderData as tfReader
    from tqdm import tqdm
    import time

    def filename_queue2iter_tensor(reader_tfRecord, tfqueue_filename):
        _, serialized_example = reader_tfRecord.read(tfqueue_filename)
        raw_example = tf.parse_single_example(serialized_example,
                                              features={'feat': tf.FixedLenFeature([], tf.string),
                                                        'label': tf.FixedLenFeature([], tf.string)})
        tensor_seq_features = tf.reshape(tf.decode_raw(raw_example['feat'], tf.float32),
                                         [-1, args.dim_raw_feature])
        tensor_seq_labels = tf.decode_raw(raw_example['label'], tf.int32)
        tensor_seq_features = tfReader.process_raw_feature(tensor_seq_features, args)

        return tensor_seq_features, tensor_seq_labels
    # create dev data queue
    reader_tfRecord = tf.TFRecordReader()
    dataReader_dev = tfReader.TFReader(args.dir_dev_data, args)
    tfqueue_filename_dev = tf.train.string_input_producer(dataReader_dev.list_tfdata_filenames,
                                                          num_epochs=2,
                                                          shuffle=False)
    tensor_seq_features, tensor_seq_labels = filename_queue2iter_tensor(reader_tfRecord,
                                                                        tfqueue_filename_dev)
    op_reset_reader_dev = reader_tfRecord.reset()

    tensor_batch_dev = tensor_seq_features, tensor_seq_labels

    with tf.train.MonitoredTrainingSession() as sess:
        for _ in tqdm(range(5410)):
            sess.run(tensor_batch_dev)
        sess.run(op_reset_reader_dev)
        time.sleep(1)
        for _ in tqdm(range(5410)):
            sess.run(tensor_batch_dev)


if __name__ == '__main__':

    testPaddingFIFOQueue()
