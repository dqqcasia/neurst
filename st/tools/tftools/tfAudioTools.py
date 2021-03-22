#!/usr/bin/python
# coding=utf-8

import tensorflow as tf
import numpy as np
import sys
from st.layers.common_layers import shape_list

# tf fea opr
def tf_kaldi_fea_delt1(features):
    feats_padded = tf.pad(features, [[1, 1], [0, 0]], "SYMMETRIC")
    feats_padded = tf.pad(feats_padded, [[1, 1], [0, 0]], "SYMMETRIC")

    shape = tf.shape(features)
    l2 = tf.slice(feats_padded, [0, 0], shape)
    l1 = tf.slice(feats_padded, [1, 0], shape)
    r1 = tf.slice(feats_padded, [3, 0], shape)
    r2 = tf.slice(feats_padded, [4, 0], shape)

    delt1 = (r1 - l1) * 0.1 + (r2 - l2) * 0.2
    return delt1


def tf_kaldi_fea_delt2(features):
    feats_padded = tf.pad(features, [[1, 1], [0, 0]], "SYMMETRIC")
    feats_padded = tf.pad(feats_padded, [[1, 1], [0, 0]], "SYMMETRIC")
    feats_padded = tf.pad(feats_padded, [[1, 1], [0, 0]], "SYMMETRIC")
    feats_padded = tf.pad(feats_padded, [[1, 1], [0, 0]], "SYMMETRIC")

    shape = tf.shape(features)
    l4 = tf.slice(feats_padded, [0, 0], shape)
    l3 = tf.slice(feats_padded, [1, 0], shape)
    l2 = tf.slice(feats_padded, [2, 0], shape)
    l1 = tf.slice(feats_padded, [3, 0], shape)
    c = tf.slice(feats_padded, [4, 0], shape)
    r1 = tf.slice(feats_padded, [5, 0], shape)
    r2 = tf.slice(feats_padded, [6, 0], shape)
    r3 = tf.slice(feats_padded, [7, 0], shape)
    r4 = tf.slice(feats_padded, [8, 0], shape)

    delt2 = - 0.1 * c - 0.04 * (l1 + r1) + 0.01 * (l2 + r2) + 0.04 * (l3 + l4 + r4 + r3)
    return delt2


def add_delt(feature):
    fb = []
    fb.append(feature)
    delt1 = tf_kaldi_fea_delt1(feature)
    fb.append(delt1)
    delt2 = tf_kaldi_fea_delt2(feature)
    fb.append(delt2)
    return tf.concat(axis=1, values=fb)


def cmvn_global(feature, mean, var):
    fea = (feature - mean) / var
    return fea


def cmvn_utt(feature):
    fea_mean = tf.reduce_mean(feature, 0)
    fea_var = tf.reduce_mean(tf.square(feature), 0)
    fea_var = fea_var - fea_mean * fea_mean
    fea_ivar = tf.rsqrt(fea_var + 1E-12)
    fea = (feature - fea_mean) * fea_ivar
    return fea


def splice(features, left_num, right_num):
    """
    [[1,1,1], [2,2,2], [3,3,3], [4,4,4], [5,5,5], [6,6,6], [7,7,7]]
    left_num=0, right_num=2:
        [[1 1 1 2 2 2 3 3 3]
         [2 2 2 3 3 3 4 4 4]
         [3 3 3 4 4 4 5 5 5]
         [4 4 4 5 5 5 6 6 6]
         [5 5 5 6 6 6 7 7 7]
         [6 6 6 7 7 7 0 0 0]
         [7 7 7 0 0 0 0 0 0]]
    """
    shape = tf.shape(features)
    splices = []
    pp = tf.pad(features, [[left_num, right_num], [0, 0]])
    for i in range(left_num + right_num + 1):
        splices.append(tf.slice(pp, [i, 0], shape))
    splices = tf.concat(axis=1, values=splices)

    return splices


def down_sample(features, rate, axis=1):
    """
    features: batch x time x deep
    Notation: you need to set the shape of the output! tensor.set_shape(None, dim_input)
    """
    len_seq = tf.shape(features)[axis]

    return tf.gather(features, tf.range(len_seq, delta=rate), axis=axis)


def target_delay(features, num_target_delay):
    seq_len = tf.shape(features)[0]
    feats_part1 = tf.slice(features, [num_target_delay, 0], [seq_len-num_target_delay, -1])
    frame_last = tf.slice(features, [seq_len-1, 0], [1, -1])
    feats_part2 = tf.concat([frame_last for _ in range(num_target_delay)], axis=0)
    features = tf.concat([feats_part1, feats_part2], axis=0)

    return features


def batch_splice(features, left_num, right_num, jump=True):
    """
    [[[1,1,1], [2,2,2], [3,3,3], [4,4,4], [5,5,5], [6,6,6], [7,7,7]],
     [[1,1,1], [2,2,2], [3,3,3], [4,4,4], [5,5,5], [6,6,6], [7,7,7]]])
    left_num=1, right_num=1:
    array([[[0, 0, 0, 1, 1, 1, 2, 2, 2],
            [3, 3, 3, 4, 4, 4, 5, 5, 5],
            [6, 6, 6, 7, 7, 7, 0, 0, 0]],

           [[0, 0, 0, 1, 1, 1, 2, 2, 2],
            [3, 3, 3, 4, 4, 4, 5, 5, 5],
            [6, 6, 6, 7, 7, 7, 0, 0, 0]]])>)
    """
    shape = tf.shape(features)
    splices = []
    pp = tf.pad(features, [[0, 0], [left_num, right_num], [0, 0]])
    for i in range(left_num + right_num + 1):
        splices.append(tf.slice(pp, [0, i, 0], shape))
    splices = tf.concat(axis=-1, values=splices)

    return splices[:, ::(left_num + right_num + 1), :] if jump else splices


def apply_specaugmant(inputs, F, T, mF, mT, p):
    """ applying specaugment

    Args:
      inputs: feature batch with shape: [batch_size, input_length, fbank_dim, fbank_channel]
      F: max width masked on the frequency dim
      T: max width masked on the time dim
      mF: the time of masking
      mT: the time of masking
      p: max probability masked on the time dim

    Returns:
      return the inputs after applying specaugment
    """

    batch_size, input_length, v, channel_nums = shape_list(inputs)
    inputs_real_length = tf.reduce_sum(tf.to_int32(tf.not_equal(
        tf.reduce_sum(tf.abs(inputs[:, :, :, 0]), axis=-1), 0)), axis=-1)

    def specaugment_to_sentence(j, prev_masks):
        # step 1: Time warping
        # After time warping, the produced delta and delta-delta may be wrong,
        # thus we choose to bypass it at start.

        # step 2: Frequency masking
        f = tf.random_uniform([], minval=0, maxval=F, dtype=tf.int32)
        f0 = tf.random_uniform([], minval=0, maxval=(v - f), dtype=tf.int32)
        freq_mask = tf.concat([tf.ones([1, input_length, f0, channel_nums]),
                               tf.zeros([1, input_length, f, channel_nums]),
                               tf.ones([1, input_length, v - f - f0, channel_nums])], axis=2)

        if mF == 2:
            f = tf.random_uniform([], minval=0, maxval=F, dtype=tf.int32)
            f0 = tf.random_uniform([], minval=0, maxval=(v - f), dtype=tf.int32)
            freq_mask_2 = tf.concat([tf.ones([1, input_length, f0, channel_nums]),
                                     tf.zeros([1, input_length, f, channel_nums]),
                                     tf.ones([1, input_length, v - f - f0, channel_nums])], axis=2)
            freq_mask *= freq_mask_2

        # step 3: Time masking
        t = tf.random_uniform([], minval=0, maxval=T, dtype=tf.int32)
        tau = inputs_real_length[j]
        t = tf.minimum(t, tf.to_int32(p * tf.to_float(tau)))
        t0 = tf.random_uniform([], minval=0, maxval=tau - t, dtype=tf.int32)
        time_mask = tf.concat([tf.ones([1, t0, v, channel_nums]),
                               tf.zeros([1, t, v, channel_nums]),
                               tf.ones([1, input_length - t - t0, v, channel_nums])], axis=1)

        if mT == 2:
            prev_t = t
            t = tf.random_uniform([], minval=0, maxval=T, dtype=tf.int32)
            t = tf.maximum(tf.minimum(t + prev_t, tf.to_int32(p * tf.to_float(tau))) - prev_t, 0)
            t0 = tf.random_uniform([], minval=0, maxval=tau - t, dtype=tf.int32)
            time_mask_2 = tf.concat([tf.ones([1, t0, v, channel_nums]),
                                     tf.zeros([1, t, v, channel_nums]),
                                     tf.ones([1, input_length - t - t0, v, channel_nums])], axis=1)
            time_mask *= time_mask_2

        mask = freq_mask * time_mask
        prev_masks = tf.concat([prev_masks, mask], axis=0)

        return j + 1, prev_masks

    masks = tf.zeros([0, input_length, v, channel_nums], dtype=tf.float32)
    _, masks = tf.while_loop(
        lambda j, *_: tf.less(j, batch_size),
        specaugment_to_sentence,
        [tf.constant(0), masks],
        shape_invariants=[
            tf.TensorShape([]),
            tf.TensorShape([None, None, v, channel_nums])
        ]
    )
    handled_inputs = inputs * masks
    ishape_static = inputs.get_shape()
    inputs = handled_inputs
    inputs.set_shape([ishape_static[0], ishape_static[1], ishape_static[2], ishape_static[3]])

    return inputs


if __name__ == '__main__':
    x = tf.placeholder(tf.float32)  # 1-D tensor
    i = tf.placeholder(tf.float32)

    # y = splice_features(x,1,1)
    y = add_delt(x)
    # y = tf.slice(x, i, [1,1])
    # y = cmvn_features(x)

    # initialize
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # run
    result = sess.run(y, feed_dict={x: [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]})
    print(result)

    result = sess.run(y, feed_dict={x: [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]})
    print(result)
