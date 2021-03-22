#!/usr/bin/python3
# -*- coding:utf8 -*-
from __future__ import print_function
import six
import yaml
import math
import random
import tensorflow.contrib.crf as crf
import pickle
import nltk
from tensorflow.python.client import device_lib
import functools

import logging
import os
import time
from tempfile import mkstemp

import numpy as np
import tensorflow as tf
import tensorflow.contrib.framework as tff
from tensorflow.python.layers import base as base_layer

from st.layers import common_layers_v1, common_attention_v1
from st.layers import common_attention, common_attention_v1, expert_utils
from collections import defaultdict
import struct

common_layers_v1.allow_defun = False

FLAGS = tf.app.flags.FLAGS

PAD_INDEX = 0
UNK_INDEX = 1
BOS_INDEX = 2
EOS_INDEX = 3

PAD = u'<PAD>'
UNK = u'<UNK>'
BOS = u'<S>'
EOS = u'</S>'

INT_TYPE = np.int32
FLOAT_TYPE = np.float32


def load_vocab(path, vocab_size=None):
    with tf.io.gfile.GFile(path, 'r') as f:
        vocab = [line.strip().split()[0] for line in f]
    vocab = vocab[:vocab_size] if vocab_size else vocab
    id_unk = vocab.index('<unk>')
    token2idx = defaultdict(lambda: id_unk)
    idx2token = defaultdict(lambda: '<unk>')
    token2idx.update({token: idx for idx, token in enumerate(vocab)})
    idx2token.update({idx: token for idx, token in enumerate(vocab)})
    if '<space>' in vocab:
        idx2token[token2idx['<space>']] = ' '
    if '<blk>' in vocab:
        idx2token[token2idx['<blk>']] = ''
    if '<unk>' in vocab:
        idx2token[token2idx['<unk>']] = '<UNK>'

    assert len(token2idx) == len(idx2token)

    return token2idx, idx2token


def update_params(config):
    """
    Update params defined in config file
    """
    config.src_vocab = os.path.join(config.data_dir, config.src_vocab)
    config.dst_vocab = os.path.join(config.data_dir, config.dst_vocab)

    config.train.src_path = os.path.join(config.data_dir, config.train.src_path)
    config.train.dst_path = os.path.join(config.data_dir, config.train.dst_path)

    if config.train.eval_on_dev:
        config.dev.src_path = os.path.join(config.data_dir, config.dev.src_path)
        config.dev.dst_path = os.path.join(config.data_dir, config.dev.dst_path)

        config.test.src_path = os.path.join(
            config.data_dir, config.test.src_path)
        config.test.dst_path = os.path.join(
            config.data_dir, config.test.dst_path)

        config.dev.output_path = os.path.join(
            config.model_dir, config.dev.output_path)
        config.test.output_path = os.path.join(
            config.model_dir, config.test.output_path)

    if config.dev.ref_path:
        config.dev.ref_path = os.path.join(config.data_dir, config.dev.ref_path)
    if config.test.ref_path:
        config.test.ref_path = os.path.join(
            config.data_dir, config.test.ref_path)

    return config


class FlushFile():
    """
    A wrapper for File, allowing users see result immediately.
    """

    def __init__(self, f):
        self.f = f

    def write(self, x):
        self.f.write(x)
        self.f.flush()

    def flush(self):
        self.f.flush()


class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if item not in self:
            logging.warning(
                '{} is not in the dict. None is returned as default.'.format(item))
            return None
        if isinstance(self[item], dict):
            self[item] = AttrDict(self[item])
        return self[item]


class DataReader(object):
    """
    Read data and create batches for training and testing.
    """

    def __init__(self, config):
        self._config = config
        self._tmps = set()
        self.load_vocab()

    def __del__(self):
        for fname in self._tmps:
            if os.path.exists(fname):
                os.remove(fname)

    def load_vocab(self):
        """
        Load vocab from disk.
        The first four items in the vocab should be <PAD>, <UNK>, <S>, </S>
        """

        def load_vocab_(path, vocab_size):
            vocab = [line.split()[0]
                     for line in tf.io.gfile.GFile(path, 'r')]
            vocab = vocab[:vocab_size]
            assert len(vocab) == vocab_size
            word2idx = {word: idx for idx, word in enumerate(vocab)}
            idx2word = {idx: word for idx, word in enumerate(vocab)}
            return word2idx, idx2word

        logging.info('Load vocabularies %s and %s.' %
                     (self._config.dirs.src_vocab, self._config.dirs.dst_vocab))
        self.src2idx, self.idx2src = load_vocab_(
            self._config.dirs.src_vocab, self._config.data.src_vocab_size)
        self.dst2idx, self.idx2dst = load_vocab_(
            self._config.dirs.dst_vocab, self._config.data.dst_vocab_size)

    def get_training_batches(self, shuffle=False, epoches=None):
        """
        Generate batches according to bucket setting.
        """
        buckets = [(i, i) for i in range(10, 1000000, 3)]

        def select_bucket(sl, dl):
            for l1, l2 in buckets:
                if sl < l1 and dl < l2:
                    return l1, l2
            raise Exception("The sequence is too long: ({}, {})".format(sl, dl))

        # Shuffle the training files.
        src_path = self._config.dirs.train.src_path
        dst_path = self._config.dirs.train.dst_path
        max_length = self._config.dirs.train.max_length

        epoch = [0]

        def stop_condition():
            if epoches is None:
                return True
            else:
                epoch[0] += 1
                return epoch[0] < epoches + 1

        while stop_condition():
            if shuffle:
                logging.info('Shuffle files %s and %s.' % (src_path, dst_path))
                src_shuf_path, dst_shuf_path = self.shuffle(
                    [src_path, dst_path])
                self._tmps.add(src_shuf_path)
                self._tmps.add(dst_shuf_path)
            else:
                src_shuf_path = src_path
                dst_shuf_path = dst_path

            caches = {}
            for bucket in buckets:
                # src sentences, dst sentences, src tokens, dst tokens
                caches[bucket] = [[], [], 0, 0]

            for src_sent, dst_sent in zip(
                    tf.io.gfile.GFile(src_shuf_path, 'r'), tf.io.gfile.GFile(dst_shuf_path, 'r')):
                src_sent, dst_sent = src_sent, dst_sent

                src_sent = src_sent.split()
                dst_sent = dst_sent.split()


                if len(src_sent) > max_length or len(dst_sent) > max_length:
                    continue

                bucket = select_bucket(len(src_sent), len(dst_sent))
                if bucket is None:  # No bucket is selected when the sentence length exceed the max length.
                    continue

                caches[bucket][0].append(src_sent)
                caches[bucket][1].append(dst_sent)
                caches[bucket][2] += len(src_sent)
                caches[bucket][3] += len(dst_sent)

                if max(caches[bucket][2], caches[bucket][3]
                       ) >= self._config.num_batch_tokens:
                    batch = (
                        self.create_batch(
                            caches[bucket][0], o='src'), self.create_batch(
                            caches[bucket][1], o='dst'))
                    # logging.debug(
                    #     'Yield batch with source shape %s and target shape %s.' % (batch[0].shape, batch[1].shape))
                    yield batch
                    caches[bucket] = [[], [], 0, 0]

            # Clean remain sentences.
            for bucket in buckets:
                # Ensure each device at least get one sample.
                if len(caches[bucket][0]) >= max(
                        1, self._config.train.num_gpus):
                    batch = (
                        self.create_batch(
                            caches[bucket][0], o='src'), self.create_batch(
                            caches[bucket][1], o='dst'))
                    # logging.debug(
                    #     'Yield batch with source shape %s and target shape %s.' % (batch[0].shape, batch[1].shape))
                    yield batch

            # Remove shuffled files when epoch finished.
            if shuffle:
                os.remove(src_shuf_path)
                os.remove(dst_shuf_path)
                self._tmps.remove(src_shuf_path)
                self._tmps.remove(dst_shuf_path)

    @staticmethod
    def shuffle(list_of_files):

        if not os.path.exists('./tmp'):
            os.makedirs('./tmp')

        tf_os, tpath = mkstemp(dir=os.path.join('./', 'tmp/'))
        tempf = tf.io.gfile.GFile(tpath, 'w')

        fds = [tf.io.gfile.GFile(ff) for ff in list_of_files]

        for l in fds[0]:
            lines = [l.strip()] + [ff.readline().strip() for ff in fds[1:]]
            print("<CONCATE4SHUF>".join(lines), file=tempf)

        [ff.close() for ff in fds]
        tempf.close()

        os.system('shuf %s > %s' % (tpath, tpath + '.shuf'))

        fnames = ['./tmp/{}.{}.{}.shuf'.format(i,
                                               os.getpid(),
                                               time.time()) for i,
                                                                ff in enumerate(list_of_files)]
        fds = [tf.io.gfile.GFile(fn, 'w') for fn in fnames]

        for l in tf.io.gfile.GFile(tpath + '.shuf'):
            s = l.strip().split('<CONCATE4SHUF>')
            for i, fd in enumerate(fds):
                print(s[i], file=fd)

        [ff.close() for ff in fds]

        os.remove(tpath)
        os.remove(tpath + '.shuf')

        return fnames

    def get_training_batches_without_buckets(
            self, shuffle=True, epoches=None, batch_size=256):
        """
        Generate batches according without bucket setting.
        """

        # Shuffle the training files.
        src_path = self._config.dirs.train.src_path
        dst_path = self._config.dirs.train.dst_path
        max_length = self._config.dirs.train.max_length

        epoch = [0]

        def stop_condition():
            if epoches is None:
                return True
            else:
                epoch[0] += 1
                return epoch[0] < epoches + 1

        while stop_condition():
            if shuffle:
                logging.info('Shuffle files %s and %s.' % (src_path, dst_path))
                src_shuf_path, dst_shuf_path = self.shuffle(
                    [src_path, dst_path])
                self._tmps.add(src_shuf_path)
                self._tmps.add(dst_shuf_path)
            else:
                src_shuf_path = src_path
                dst_shuf_path = dst_path

            src_sents, dst_sents = [], []
            for src_sent, dst_sent in zip(tf.io.gfile.GFile(src_shuf_path, 'r'), tf.io.gfile.GFile(dst_shuf_path, 'r')):
                src_sent, dst_sent = src_sent, dst_sent

                src_sent = src_sent.split()
                dst_sent = dst_sent.split()


                if len(src_sent) > max_length or len(dst_sent) > max_length:
                    continue

                src_sents.append(src_sent)
                dst_sents.append(dst_sent)
                # Create a padded batch.
                if len(src_sents) >= batch_size:
                    batch = (
                        self.create_batch(
                            src_sents, o='src'), self.create_batch(
                            dst_sents, o='dst'))
                    # logging.debug(
                    #     'Yield batch with source shape %s and target shape %s.' % (batch[0].shape, batch[1].shape))
                    yield batch
                    src_sents = []
                    dst_sents = []
            # Clean remain sentences.
            if src_sents:
                # Ensure each device at least get one sample.
                if len(src_sents) >= max(1, self._config.train.num_gpus):
                    batch = (
                        self.create_batch(
                            src_sents, o='src'), self.create_batch(
                            dst_sents, o='dst'))
                    # logging.debug(
                    #     'Yield batch with source shape %s and target shape %s.' % (batch[0].shape, batch[1].shape))
                    yield batch

            # Remove shuffled files when epoch finished.
            if shuffle:
                os.remove(src_shuf_path)
                os.remove(dst_shuf_path)
                self._tmps.remove(src_shuf_path)
                self._tmps.remove(dst_shuf_path)

    def get_test_batches(self, src_path, batch_size):
        # Read batches for testing.
        src_sents = []
        for src_sent in tf.io.gfile.GFile(src_path, 'r'):
            src_sent = src_sent
            src_sent = src_sent.split()
            src_sents.append(src_sent)
            # Create a padded batch.
            if len(src_sents) >= batch_size:
                yield self.create_batch(src_sents, o='src')
                src_sents = []
        if src_sents:
            # We ensure batch size not small than gpu number by padding
            # redundant samples.
            if len(src_sents) < self._config.infer.num_gpus:
                src_sents.extend([src_sents[-1]] * self._config.infer.num_gpus)
            yield self.create_batch(src_sents, o='src')

    def get_test_batches_with_target(self, src_path, dst_path, batch_size):
        """
        Usually we don't need target sentences for test unless we want to compute PPl.
        Returns:
            Paired source and target batches.
        """

        src_sents, dst_sents = [], []
        for src_sent, dst_sent in zip(tf.io.gfile.GFile(src_path, 'r'), tf.io.gfile.GFile(dst_path, 'r')):
            src_sent, dst_sent = src_sent, dst_sent
            src_sent = src_sent.split()
            dst_sent = dst_sent.split()
            src_sents.append(src_sent)
            dst_sents.append(dst_sent)
            # Create a padded batch.
            if len(src_sents) >= batch_size:
                yield self.create_batch(src_sents, o='src'), self.create_batch(dst_sents, o='dst')
                src_sents, dst_sents = [], []
        if src_sents:
            yield self.create_batch(src_sents, o='src'), self.create_batch(dst_sents, o='dst')

    def get_test_batches_with_target_with_max_tokens_num(
            self, src_path, dst_path):
        """
        Generate batches according to tokens_per_batch setting.
        """
        src_sents, dst_sents, src_nums, dst_nums = [], [], 0, 0
        for src_sent, dst_sent in zip(tf.io.gfile.GFile(src_path, 'r'), tf.io.gfile.GFile(dst_path, 'r')):
            src_sent, dst_sent = src_sent, dst_sent
            src_sent = src_sent.split()
            dst_sent = dst_sent.split()
            src_sents.append(src_sent)
            dst_sents.append(dst_sent)
            src_nums += len(src_sent)
            dst_nums += len(dst_sent)

            # Create a padded batch.
            if max(src_nums, dst_nums) >= self._config.num_batch_tokens:
                yield self.create_batch(src_sents, o='src'), self.create_batch(dst_sents, o='dst')
                src_sents, dst_sents, src_nums, dst_nums = [], [], 0, 0
        if src_sents:
            yield self.create_batch(src_sents, o='src'), self.create_batch(dst_sents, o='dst')

    def create_batch(self, sents, o):
        # Convert words to indices.
        assert o in ('src', 'dst')
        word2idx = self.src2idx if o == 'src' else self.dst2idx
        EOS_TOKEN = u"</S>" if "</S>" in word2idx else u"<eos>"
        indices = []
        for sent in sents:
            x = [
                word2idx.get(
                    word,
                    1) for word in (
                        sent +
                        [EOS_TOKEN])]  # 1: OOV, </S>: End of Text
            indices.append(x)

        # Pad to the same length.
        maxlen = max([len(s) for s in indices])
        X = np.zeros([len(indices), maxlen], np.int32)
        for i, x in enumerate(indices):
            X[i, :len(x)] = x

        return X

    def indices_to_words(self, Y, o='dst'):
        assert o in ('src', 'dst')
        idx2word = self.idx2src if o == 'src' else self.idx2dst
        sents = []
        for y in Y:  # for each sentence
            sent = []
            for i in y:  # For each word
                if i == 3:  # </S>
                    break
                w = idx2word[i]
                sent.append(w)
            sents.append(' '.join(sent))
        return sents

    def words_to_indices(self, sents, o='src'):
        assert o in ('src', 'dst')
        word2idx = self.src2idx if o == 'src' else self.dst2idx
        indices = []
        for sent in sents:
            x = [
                word2idx.get(
                    word,
                    1) for word in (
                        sent +
                        [u"<eos>"])]  # 1: OOV, </S>: End of Tex
            indices.append(x)

        # Pad to the same length.
        maxlen = max([len(s) for s in indices])
        X = np.zeros([len(indices), maxlen], np.int32)
        for i, x in enumerate(indices):
            X[i, :len(x)] = x

        return X


def expand_feed_dict(feed_dict):
    """If the key is a tuple of placeholders,
    split the input data then feed them into these placeholders.
    """

    new_feed_dict = {}
    for k, v in feed_dict.items():
        if not isinstance(k, tuple):
            new_feed_dict[k] = v
        else:
            # Split v along the first dimension.
            n = len(k)
            batch_size = v.shape[0]
            assert batch_size > 0
            span = batch_size // n
            remainder = batch_size % n
            base = 0
            for i, p in enumerate(k):
                if i < remainder:
                    end = base + span + 1
                else:
                    end = base + span
                new_feed_dict[p] = v[base: end]
                base = end
    return new_feed_dict


def available_variables(checkpoint_dir):
    all_vars = tf.global_variables()
    all_available_vars = tff.list_variables(checkpoint_dir=checkpoint_dir)
    all_available_vars = dict(all_available_vars)
    available_vars = []
    for v in all_vars:
        vname = v.name.split(':')[0]
        if vname in all_available_vars and v.get_shape(
        ) == all_available_vars[vname]:
            available_vars.append(v)
    return available_vars


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        else:
            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    return average_grads


def residual(inputs, outputs, dropout_rate):
    """Residual connection.

    Args:
        inputs: A Tensor.
        outputs: A Tensor.
        dropout_rate: A float range from [0, 1).

    Returns:
        A Tensor.
    """
    outputs = inputs + tf.nn.dropout(outputs, 1 - dropout_rate)
    outputs = common_layers_v1.layer_norm(outputs)
    return outputs


def learning_rate_decay(config, global_step):
    """Inverse-decay learning rate until warmup_steps, then decay."""
    warmup_steps = tf.to_float(config.train.warmup_steps)
    global_step = tf.to_float(global_step)
    return config.model.hidden_units ** -0.5 * tf.minimum(
        (global_step + 1.0) * warmup_steps ** -1.5, (global_step + 1.0) ** -0.5)


def shift_right(input, pad=2):
    """Shift input tensor right to create decoder input. '2' denotes <S>"""
    return tf.concat((tf.ones_like(input[:, :1]) * pad, input[:, :-1]), 1)


def source_padding(input, pad=0):
    """Shift input tensor right to create decoder input. '2' denotes <S>"""
    return tf.concat((tf.concat((tf.ones_like(
        input[:, :1]) * pad, input[:, :-1]), 1), tf.ones_like(input[:, :1]) * pad), 1)


def embedding(x, vocab_size, dense_size, name=None,
              reuse=None, kernel=None, multiplier=1.0):
    """Embed x of type int64 into dense vectors."""
    with tf.variable_scope(
            name, default_name="embedding", values=[x], reuse=reuse):
        if kernel is not None:
            embedding_var = kernel
        else:
            embedding_var = tf.get_variable("kernel", [vocab_size, dense_size])
        output = tf.gather(embedding_var, x)
        if multiplier != 1.0:
            output *= multiplier
        return output


def dense(inputs,
          units,
          activation=tf.identity,
          use_bias=True,
          kernel=None,
          reuse=None,
          name=None):
    argcount = activation.__code__.co_argcount
    if activation.__defaults__:
        argcount -= len(activation.__defaults__)
    assert argcount in (0, 1, 2)
    with tf.variable_scope(name, "dense", reuse=reuse):
        if argcount <= 1:
            input_size = inputs.get_shape().as_list()[-1]
            inputs_shape = tf.unstack(tf.shape(inputs))
            inputs = tf.reshape(inputs, [-1, input_size])
            if kernel is not None:
                assert kernel.get_shape().as_list()[0] == units
                w = kernel
            else:
                with tf.variable_scope(tf.get_variable_scope()):
                    w = tf.get_variable("kernel", [units, input_size])
            outputs = tf.matmul(inputs, w, transpose_b=True)
            if use_bias:
                b = tf.get_variable("bias", [units], initializer=tf.zeros_initializer)
                outputs += b
            outputs = activation(outputs)
            return tf.reshape(outputs, inputs_shape[:-1] + [units])
        else:
            arg1 = dense(inputs, units, tf.identity, use_bias, name='arg1')
            arg2 = dense(inputs, units, tf.identity, use_bias, name='arg2')
            return activation(arg1, arg2)


def ff_hidden(inputs, hidden_size, output_size, activation,
              use_bias=True, reuse=None, name=None):
    with tf.variable_scope(name, "ff_hidden", reuse=reuse):
        hidden_outputs = dense(inputs, hidden_size, activation, use_bias)
        outputs = dense(hidden_outputs, output_size, tf.identity, use_bias)
        return outputs


def multihead_attention(query_antecedent,
                        memory_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        num_queries=None,
                        query_eq_key=False,
                        summaries=False,
                        image_shapes=None,
                        name=None):
    """Multihead scaled-dot-product attention with input/output transformations.

    Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels]
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    num_queries: a int or None
    query_eq_key: a boolean
    summaries: a boolean
    image_shapes: optional quadruple of integer scalars for image summary.
        If the query positions and memory positions represent the
        pixels of a flattened image, then pass in their dimensions:
          (query_rows, query_cols, memory_rows, memory_cols).
    name: an optional string

    Returns:
    A Tensor.
    """
    with tf.variable_scope(name, default_name="multihead_attention",
                           values=[query_antecedent, memory_antecedent]):

        if not query_eq_key:
            if memory_antecedent is None:
                # Q = K = V
                # self attention
                combined = dense(
                    query_antecedent,
                    total_key_depth * 2 + total_value_depth,
                    name="qkv_transform")
                q, k, v = tf.split(
                    combined, [
                        total_key_depth, total_key_depth, total_value_depth],
                    axis=2)
            else:
                # Q != K = V
                q = dense(query_antecedent, total_key_depth, name="q_transform")
                combined = dense(
                    memory_antecedent,
                    total_key_depth +
                    total_value_depth,
                    name="kv_transform")
                k, v = tf.split(
                    combined, [
                        total_key_depth, total_value_depth], axis=2)
        else:
            # In this setting, we use query_antecedent as the query and key,
            # and use memory_antecedent as the value.
            assert memory_antecedent is not None
            combined = dense(
                query_antecedent,
                total_key_depth * 2,
                name="qk_transform")
            q, k = tf.split(
                combined, [total_key_depth, total_key_depth],
                axis=2)
            v = dense(memory_antecedent, total_value_depth, name='v_transform')

        if num_queries:
            q = q[:, -num_queries:, :]

        q = common_attention_v1.split_heads(q, num_heads)
        k = common_attention_v1.split_heads(k, num_heads)
        v = common_attention_v1.split_heads(v, num_heads)
        key_depth_per_head = total_key_depth // num_heads
        q *= key_depth_per_head**-0.5
        x = common_attention_v1.dot_product_attention(
            q, k, v, bias, dropout_rate, summaries, image_shapes)
        x = common_attention_v1.combine_heads(x)
        x = dense(x, output_depth, name="output_transform")
        return x


class AttentionGRUCell(tf.nn.rnn_cell.GRUCell):
    def __init__(self,
                 num_units,
                 attention_memories,
                 attention_bias=None,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 name=None):

        super(AttentionGRUCell, self).__init__(
            num_units=num_units,
            activation=activation,
            reuse=reuse,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer)
        with tf.variable_scope(name, "AttentionGRUCell", reuse=reuse):
            self._attention_keys = dense(
                attention_memories, num_units, name='attention_key')
            self._attention_values = dense(
                attention_memories, num_units, name='attention_value')
        self._attention_bias = attention_bias

    def attention(self, inputs, state):
        attention_query = tf.matmul(
            tf.concat([inputs, state], 1), self._attention_query_kernel)
        attention_query = tf.nn.bias_add(
            attention_query, self._attention_query_bias)

        alpha = tf.tanh(attention_query[:, None, :] + self._attention_keys)
        alpha = dense(alpha, 1, kernel=self._alpha_kernel, name='attention')
        if self._attention_bias is not None:
            alpha += self._attention_bias

        alpha = tf.nn.softmax(alpha, axis=1)

        context = tf.multiply(self._attention_values, alpha)
        context = tf.reduce_sum(context, axis=1)

        return context

    def call(self, inputs, state):
        context = self.attention(inputs, state)
        inputs = tf.concat([inputs, context], axis=1)
        return super(AttentionGRUCell, self).call(inputs, state)

    def build(self, inputs_shape):
        if inputs_shape[-1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value
        self._gate_kernel = self.add_variable(
            "gates/weights",
            shape=[input_depth + 2 * self._num_units, 2 * self._num_units],
            initializer=self._kernel_initializer)
        self._gate_bias = self.add_variable(
            "gates/bias",
            shape=[2 * self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else tf.constant_initializer(1.0, dtype=self.dtype)))
        self._candidate_kernel = self.add_variable(
            "candidate/weights",
            shape=[input_depth + 2 * self._num_units, self._num_units],
            initializer=self._kernel_initializer)
        self._candidate_bias = self.add_variable(
            "candidate/bias",
            shape=[self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else tf.zeros_initializer(dtype=self.dtype)))

        self._attention_query_kernel = self.add_variable(
            "attention_query/weight",
            shape=[input_depth + self._num_units, self._num_units],
            initializer=self._kernel_initializer)
        self._attention_query_bias = self.add_variable(
            "attention_query/bias",
            shape=[self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else tf.constant_initializer(1.0, dtype=self.dtype)))
        self._alpha_kernel = self.add_variable(
            'alpha_kernel',
            shape=[1, self._num_units],
            initializer=self._kernel_initializer)
        self.built = True


class IndRNNCell(tf.nn.rnn_cell.RNNCell):
    """The independent RNN cell."""

    def __init__(self, num_units, recurrent_initializer=None,
                 reuse=None, name=None):
        super(IndRNNCell, self).__init__(_reuse=reuse, name=name)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._recurrent_initializer = recurrent_initializer
        self._kernel_initializer = None
        self._bias_initializer = tf.constant_initializer(0.0, dtype=tf.float32)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value
        self._recurrent_kernel = self.add_variable(
            "recurrent/weights",
            shape=[self._num_units],
            initializer=self._recurrent_initializer)
        epsilon = np.power(2.0, 1.0 / 50.0)
        self._recurrent_kernel = tf.clip_by_value(
            self._recurrent_kernel, -epsilon, epsilon)

        self._input_kernel = self.add_variable(
            "input/weights",
            shape=[input_depth, self._num_units],
            initializer=self._kernel_initializer)

        self._bias = self.add_variable(
            "bias",
            shape=[self._num_units],
            initializer=self._bias_initializer)

        self.built = True

    def call(self, inputs, state):
        inputs = tf.matmul(inputs, self._input_kernel)
        state = tf.multiply(state, self._recurrent_kernel)
        output = inputs + state
        output = tf.nn.bias_add(output, self._bias)
        output = tf.nn.relu(output)
        return output, output


class AttentionIndRNNCell(IndRNNCell):
    def __init__(self,
                 num_units,
                 attention_memories,
                 attention_bias=None,
                 recurrent_initializer=None,
                 reuse=None,
                 name=None):
        super(AttentionIndRNNCell, self).__init__(num_units,
                                                  recurrent_initializer=recurrent_initializer,
                                                  reuse=reuse, name=name)
        with tf.variable_scope(name, "AttentionIndRNNCell", reuse=reuse):
            self._attention_keys = dense(
                attention_memories, num_units, name='attention_key')
            self._attention_values = dense(
                attention_memories, num_units, name='attention_value')
        self._attention_bias = attention_bias

    def attention(self, inputs, state):
        attention_query = tf.matmul(
            tf.concat([inputs, state], 1), self._attention_query_kernel)
        attention_query = tf.nn.bias_add(
            attention_query, self._attention_query_bias)

        alpha = tf.tanh(attention_query[:, None, :] + self._attention_keys)
        alpha = dense(alpha, 1, kernel=self._alpha_kernel, name='attention')
        if self._attention_bias is not None:
            alpha += self._attention_bias
        alpha = tf.nn.softmax(alpha, axis=1)
        self._alpha = alpha

        context = tf.multiply(self._attention_values, alpha)
        context = tf.reduce_sum(context, axis=1)

        return context

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value
        self._recurrent_kernel = self.add_variable(
            "recurrent/weights",
            shape=[self._num_units],
            initializer=self._recurrent_initializer)
        epsilon = np.power(2.0, 1.0 / 50.0)
        self._recurrent_kernel = tf.clip_by_value(
            self._recurrent_kernel, -epsilon, epsilon)

        self._input_kernel = self.add_variable(
            "input/weights",
            shape=[input_depth + self._num_units, self._num_units],
            initializer=self._kernel_initializer)

        self._bias = self.add_variable(
            "bias",
            shape=[self._num_units],
            initializer=self._bias_initializer)

        self._attention_query_kernel = self.add_variable(
            "attention_query/weight",
            shape=[input_depth + self._num_units, self._num_units],
            initializer=self._kernel_initializer)
        self._attention_query_bias = self.add_variable(
            "attention_query/bias",
            shape=[self._num_units],
            initializer=self._bias_initializer)
        self._alpha_kernel = self.add_variable(
            'alpha_kernel',
            shape=[1, self._num_units],
            initializer=self._kernel_initializer)

        self.built = True

    def call(self, inputs, state):
        context = self.attention(inputs, state)
        inputs = tf.concat([inputs, context], axis=1)
        return super(AttentionIndRNNCell, self).call(inputs, state)

    def get_attention_weights(self):
        return self._alpha


# add for universal transformer

def gated_linear_units(inputs):
    input_shape = inputs.get_shape().as_list()
    assert len(input_shape) == 3
    input_pass = inputs[:, :, 0:int(input_shape[2] / 2)]
    input_gate = inputs[:, :, int(input_shape[2] / 2):]
    input_gate = tf.sigmoid(input_gate)
    return tf.multiply(input_pass, input_gate)


def linear_mapping_weightnorm(
        inputs, out_dim, in_dim=None, dropout=1.0, var_scope_name="linear_mapping"):
    with tf.variable_scope(var_scope_name):
        input_shape = inputs.get_shape().as_list()  # static shape. may has None
        input_shape_tensor = tf.shape(inputs)

        V = tf.get_variable('V',
                            shape=[int(input_shape[-1]),
                                   out_dim],
                            dtype=tf.float32,
                            trainable=True)


        V_norm = tf.norm(V, axis=0)  # V shape is M*N,  V_norm shape is N


        g = tf.get_variable(
            'g',
            dtype=tf.float32,
            shape=[out_dim],
            trainable=True)

        b = tf.get_variable('b', shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(),
                            trainable=True)

        assert len(input_shape) == 3
        inputs = tf.reshape(inputs, [-1, input_shape[-1]])
        inputs = tf.matmul(inputs, V)
        inputs = tf.reshape(inputs, [input_shape_tensor[0], -1, out_dim])


        scaler = tf.div(g, tf.norm(V, axis=0))

        inputs = tf.reshape(scaler, [1, out_dim]) * \
                 inputs + tf.reshape(b, [1, out_dim])

        return inputs


def create_position_embedding(pos_embed, lengths, maxlen):
    # Slice to size of current sequence
    pe_slice = pos_embed[2:maxlen + 2, :]
    # Replicate encodings for each element in the batch
    batch_size = tf.shape(lengths)[0]
    pe_batch = tf.tile([pe_slice], [batch_size, 1, 1])

    # Mask out positions that are padded
    positions_mask = tf.sequence_mask(
        lengths=lengths, maxlen=maxlen, dtype=tf.float32)
    positions_embed = pe_batch * tf.expand_dims(positions_mask, 2)



    positions_embed = tf.reverse_sequence(positions_embed, lengths, batch_dim=0,
                                          seq_dim=1)
    positions_embed = tf.reverse(positions_embed, [1])


    return positions_embed


def add_position_embedding(pos_embed, inputs):
    time = tf.cast(tf.shape(inputs), dtype=tf.int32)[1] - 1

    batch_size = tf.cast(tf.shape(inputs), dtype=tf.int32)[0]
    seq_pos_embed = pos_embed[2:time + 1 + 2, :]

    seq_pos_embed = tf.expand_dims(seq_pos_embed, axis=0)

    seq_pos_embed_batch = tf.tile(seq_pos_embed, [batch_size, 1, 1])

    return tf.add(inputs, tf.cast(seq_pos_embed_batch, dtype=tf.float32))


def conv1d_weightnorm(inputs, layer_idx, out_dim, kernel_size, padding="SAME", dropout=1.0,
                      var_scope_name="conv_layer"):
    # padding should take attention

    with tf.variable_scope("conv_layer_" + str(layer_idx)):
        in_dim = int(inputs.get_shape()[-1])

        V = tf.get_variable(
            'V',
            shape=[
                kernel_size,
                in_dim,
                out_dim],
            dtype=tf.float32,
            trainable=True)


        V_norm = tf.norm(V, axis=[0, 1])  # V shape is M*N*k,  V_norm shape is k

        g = tf.get_variable(
            'g',
            dtype=tf.float32,
            shape=[out_dim],
            trainable=True)
        b = tf.get_variable('b', shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(),
                            trainable=True)

        W = tf.reshape(g, [1, 1, out_dim]) * tf.nn.l2_normalize(V, [0, 1])
        inputs = tf.nn.bias_add(
            tf.nn.conv1d(
                value=inputs,
                filters=W,
                stride=1,
                padding=padding),
            b)
        return inputs


def make_attention(attention_values, target_embed,
                   encoder_output, decoder_hidden, layer_idx):
    with tf.variable_scope("attention_layer_" + str(layer_idx)):
        embed_size = target_embed.get_shape().as_list()[-1]
        dec_hidden_proj = linear_mapping_weightnorm(decoder_hidden, embed_size,
                                                    var_scope_name="linear_mapping_att_query")
        dec_rep = (dec_hidden_proj + target_embed) * \
                  tf.sqrt(0.5)


        encoder_output_a = encoder_output
        encoder_output_c = attention_values

        att_score = tf.matmul(dec_rep, encoder_output_a, transpose_b=True)
        att_score = tf.nn.softmax(att_score)

        length = tf.cast(tf.shape(encoder_output_c), tf.float32)

        att_out = tf.matmul(att_score, encoder_output_c) * length[1] * tf.sqrt(
            1.0 / length[1])

        att_out = linear_mapping_weightnorm(att_out, decoder_hidden.get_shape().as_list()[-1],
                                            var_scope_name="linear_mapping_att_out")
    return att_out


def conv_encoder_stack(inputs, nhids_list, kwidths_list, dropout_dict, mode):
    next_layer = inputs
    for layer_idx in range(len(nhids_list)):
        nin = nhids_list[layer_idx] if layer_idx == 0 else nhids_list[layer_idx - 1]
        nout = nhids_list[layer_idx]
        if nin != nout:
            # mapping for res add
            res_inputs = linear_mapping_weightnorm(next_layer, nout, dropout=dropout_dict['src'],
                                                   var_scope_name="linear_mapping_cnn_" + str(layer_idx))
        else:
            res_inputs = next_layer
        # dropout before input to conv
        next_layer = tf.contrib.layers.dropout(
            inputs=next_layer,
            keep_prob=dropout_dict['hid'],
            is_training=mode)

        next_layer = conv1d_weightnorm(inputs=next_layer, layer_idx=layer_idx, out_dim=nout * 2,
                                       kernel_size=kwidths_list[layer_idx], padding="SAME",
                                       dropout=dropout_dict['hid'], var_scope_name="conv_layer_" + str(layer_idx))

        next_layer = gated_linear_units(next_layer)
        next_layer = (next_layer + res_inputs) * tf.sqrt(0.5)

    return next_layer


def conv_decoder_stack(attention_values, target_embed, enc_output,
                       inputs, nhids_list, kwidths_list, dropout_dict, mode):
    next_layer = inputs
    for layer_idx in range(len(nhids_list)):
        nin = nhids_list[layer_idx] if layer_idx == 0 else nhids_list[layer_idx - 1]
        nout = nhids_list[layer_idx]
        if nin != nout:
            # mapping for res add
            res_inputs = linear_mapping_weightnorm(next_layer, nout, dropout=dropout_dict['src'],
                                                   var_scope_name="linear_mapping_cnn_" + str(layer_idx))
        else:
            res_inputs = next_layer
        # dropout before input to conv
        next_layer = tf.contrib.layers.dropout(
            inputs=next_layer,
            keep_prob=dropout_dict['hid'],
            is_training=mode)
        # special process here, first padd then conv, because tf does not suport
        # padding other than SAME and VALID
        next_layer = tf.pad(next_layer,
                            [[0, 0], [kwidths_list[layer_idx] - 1,
                                      kwidths_list[layer_idx] - 1], [0, 0]],
                            "CONSTANT")

        next_layer = conv1d_weightnorm(inputs=next_layer, layer_idx=layer_idx, out_dim=nout * 2,
                                       kernel_size=kwidths_list[layer_idx], padding="VALID",
                                       dropout=dropout_dict['hid'], var_scope_name="conv_layer_" + str(layer_idx))

        layer_shape = next_layer.get_shape().as_list()
        assert len(layer_shape) == 3
        # to avoid using future information
        next_layer = next_layer[:, 0:-kwidths_list[layer_idx] + 1, :]

        next_layer = gated_linear_units(next_layer)


        att_out = make_attention(
            attention_values,
            target_embed,
            enc_output,
            next_layer,
            layer_idx)
        next_layer = (next_layer + att_out) * tf.sqrt(0.5)


        next_layer += (next_layer + res_inputs) * tf.sqrt(0.5)
    return next_layer


def top(body_output, vocab_size, dense_size, shared_embedding=True, reuse=None):
    with tf.variable_scope('embedding', reuse=reuse):
        if shared_embedding:
            with tf.variable_scope("shared", reuse=True):
                shape = tf.shape(body_output)[:-1]
                body_output = tf.reshape(body_output, [-1, dense_size])
                # embedding_var = embedding(vocab_size, dense_size)
                embedding_var = tf.get_variable(
                    "kernel", [vocab_size, dense_size])
                logits = tf.matmul(body_output, embedding_var, transpose_b=True)
                logits = tf.reshape(logits, tf.concat([shape, [vocab_size]], 0))
        else:
            with tf.variable_scope("softmax", reuse=None):

                embedding_var = tf.get_variable(
                    "kernel", [vocab_size, dense_size])
                shape = tf.shape(body_output)[:-1]
                body_output = tf.reshape(body_output, [-1, dense_size])
                logits = tf.matmul(body_output, embedding_var, transpose_b=True)
                logits = tf.reshape(logits, tf.concat([shape, [vocab_size]], 0))
    return logits


def get_config():
    """Returns config for tf.session"""
    config = tf.ConfigProto(allow_soft_placement=True)

    config.gpu_options.allow_growth = True
    return config


def load_ckpt(saver, sess, ckpt_dir="train"):
    """Load checkpoint from the ckpt_dir (if unspecified, this is train dir) and restore it to saver and sess, waiting 10
    secs in the case of failure. Also returns checkpoint name."""
    while True:
        try:
            latest_filename = "checkpoint_best" if ckpt_dir == "eval" else None
            ckpt_dir = os.path.join(FLAGS.log_root, ckpt_dir)
            ckpt_state = tf.train.get_checkpoint_state(
                ckpt_dir, latest_filename=latest_filename)
            tf.logging.info(
                'Loading checkpoint %s',
                ckpt_state.model_checkpoint_path)
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            return ckpt_state.model_checkpoint_path
        except BaseException:
            tf.logging.info(
                "Failed to load checkpoint from %s. Sleeping for %i secs...",
                ckpt_dir,
                10)
            time.sleep(10)


def load_dqn_ckpt(saver, sess):
    """Load checkpoint from the ckpt_dir (if unspecified, this is train dir) and restore it to saver and sess, waiting 10
    secs in the case of failure. Also returns checkpoint name."""
    while True:
        try:
            ckpt_dir = os.path.join(FLAGS.log_root, "dqn", "train")
            ckpt_state = tf.train.get_checkpoint_state(ckpt_dir)
            tf.logging.info(
                'Loading checkpoint %s',
                ckpt_state.model_checkpoint_path)
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            return ckpt_state.model_checkpoint_path
        except BaseException:
            tf.logging.info(
                "Failed to load checkpoint from %s. Sleeping for %i secs...",
                ckpt_dir,
                10)
            time.sleep(10)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])


def get_batches(data, batch_size, batches=0, allow_smaller=True):
    """
    Segment `data` into a given number of fixed-size batches. The dataset is automatically shuffled.

    This function is for smaller datasets, when you need access to the entire dataset at once (e.g. dev set).
    For larger (training) datasets, where you may want to lazily iterate over batches
    and cycle several times through the entire dataset, prefer batch iterators
    (such as `cycling_batch_iterator`).

    :param data: the dataset to segment into batches (a list of data points)
    :param batch_size: the size of a batch
    :param batches: number of batches to return (0 for the largest possible number)
    :param allow_smaller: allow the last batch to be smaller
    :return: a list of batches (which are lists of `batch_size` data points)
    """
    if not allow_smaller:
        max_batches = len(data) // batch_size
    else:
        max_batches = int(math.ceil(len(data) / batch_size))

    if batches < 1 or batches > max_batches:
        batches = max_batches

    random.shuffle(data)
    batches = [data[i * batch_size:(i + 1) * batch_size]
               for i in range(batches)]
    return batches


def inference(scores, sequence_lengths, transitions):
    paths = np.zeros(scores.shape[:2], dtype=INT_TYPE)
    for i in range(len(scores)):
        tag_score, length = scores[i], sequence_lengths[i]
        if length == 0:
            continue
        path, _ = crf.viterbi_decode(tag_score[:length], transitions)
        paths[i, :length] = path
    return paths


def split_array(array, num):
    if num <= 1:
        return [array]
    lin_space = np.linspace(start=0, stop=len(array), num=num + 1, dtype=np.int)
    blocks = []
    for i in range(num):
        blocks.append(array[lin_space[i]: lin_space[i + 1]])
    return blocks


def create_dic(item_list, add_unk=False, add_pad=False,
               lower=False, threshold=1, dic_size=None):
    """
    Create a dictionary of items from a list of list of items.
    """

    def merge_(dic1, dic2):
        for k, v in dic2.items():
            if k in dic1:
                dic1[k] += v
            else:
                dic1[k] = v

    def create_dic_(items):
        dic = {}
        if type(items) in (list, tuple):
            for t in items:
                merge_(dic, create_dic_(t))
        else:
            if lower:
                items = items.lower()
            dic[items] = 1
        return dic

    dic = create_dic_(item_list)
    # Make sure that <PAD> have a id 0.
    if add_pad:
        dic['<PAD>'] = 1e20
    # If specified, add a special item <UNK>.
    if add_unk:
        dic['<UNK>'] = 1e10

    if dic_size:
        num = 0
        new_dic = {}
        sorted_dic = [(k, dic[k])
                      for k in sorted(dic, key=dic.get, reverse=True)]
        for key, value in sorted_dic:
            new_dic[key] = value
            num += 1
            if num == dic_size:
                return new_dic

    for k, v in dic.copy().items():
        if v < threshold:
            dic.pop(k)
    return dic


def create_mapping(items):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    if isinstance(items, dict):
        sorted_items = sorted(items.items(), key=lambda x: (-x[1], x[0]))
        id2item = {i: v[0] for i, v in enumerate(sorted_items)}
        item2id = {v: k for k, v in id2item.items()}
        return item2id, id2item
    elif isinstance(items, list):
        id2item = {i: v for i, v in enumerate(items)}
        item2id = {v: k for k, v in id2item.items()}
        return item2id, id2item


def create_input(batch):
    """
    Take each sentence data in batch and return an input for
    the training or the evaluation function.
    """

    def pad_one_sentence_(seq, max_word_len, max_char_len):
        assert len(seq) <= max_word_len
        if max_char_len == 0:
            return seq + [0] * (max_word_len - len(seq))
        else:
            ret = []
            for w in seq:
                ret.append(w[:max_char_len] + [0] *
                           max(0, max_char_len - len(w)))
            return ret + [[0] * max_char_len] * (max_word_len - len(seq))

    assert len(batch) > 0
    word_lengths = [len(seq) for seq in batch[0]]
    max_word_len = max(1, max(word_lengths))
    max_char_len = 30  # Accept no more than 30 characters in a word.
    ret = []
    for i, d in enumerate(batch):
        dd = []
        for seq_id, pos in zip(d, word_lengths):
            assert len(seq_id) == pos
            dd.append(pad_one_sentence_(seq_id, max_word_len=max_word_len,
                                        max_char_len=max_char_len if i == 1 else 0))
        ret.append(np.array(dd))
    ret.append(word_lengths)
    return ret


def full2half(ustring):
    rstring = []
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
            # rstring.append(unichr(inside_code))
            rstring.append(chr(inside_code))
        elif 65281 <= inside_code <= 65374:
            inside_code -= 65248
            # rstring.append(unichr(inside_code))
            rstring.append(chr(inside_code))
        else:
            rstring.append(uchar)
    return ''.join(rstring)


def half2full(ustring):
    rstring = []
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 32:
            inside_code = 12288
            # rstring.append(unichr(inside_code))
            rstring.append(chr(inside_code))
        elif 32 <= inside_code <= 126:
            inside_code += 65248
            # rstring.append(unichr(inside_code))
            rstring.append(chr(inside_code))
        else:
            rstring.append(uchar)
    return ''.join(rstring)


def convert_data(data, mappings):
    return [map_seq(seqs, mapping) for seqs, mapping in zip(data, mappings)]


def map_seq(seqs, mapping):
    """
    Map text data to ids.
    """
    if type(seqs) in (list, tuple):
        return [map_seq(seq, mapping) for seq in seqs]
    if seqs in mapping:
        return mapping[seqs]
    if seqs.lower() in mapping:
        return mapping[seqs.lower()]
    return mapping['<UNK>']


def data_iterator(inputs, batch_size, shuffle=True, max_length=200):
    """
    A simple iterator for generating dynamic mini batches.
    """

    assert len(inputs) > 0
    assert batch_size > 0
    assert all([len(item) == len(inputs[0]) for item in inputs])
    inputs = list(zip(*inputs))

    if shuffle:
        np.random.shuffle(inputs)

    batch = []
    bs = batch_size
    for d in inputs:
        if len(d[0]) > max_length:
            bs = max(1, min(batch_size * max_length / len(d[0]), bs))
        if len(batch) < bs:
            batch.append(d)
        else:
            yield list(zip(*batch))
            batch = [d]
            if len(d[0]) < max_length:
                bs = batch_size
            else:
                bs = max(1, batch_size * max_length / len(d[0]))
    if batch:
        yield list(zip(*batch))


def set_graph(func):
    def wrapper(self, *args, **kargs):
        with self.graph.as_default():
            return func(self, *args, **kargs)
    return wrapper


def layer_normalize(inputs, epsilon=1e-8, scope="ln", reuse=None):
    """Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    """
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
        outputs = gamma * normalized + beta

    return outputs


def load_config(config_paths, config=None):
    """Loads configuration files.
    Args:
      config_paths: A list of configuration files.
      config: A (possibly non empty) config dictionary to fill.
    Returns:
      The configuration dictionary.
    """
    if config is None:
        config = {}

    for config_path in config_paths:
        with tf.io.gfile.GFile(config_path, "r") as config_file:
            subconfig = yaml.load(config_file.read())
            # Add or update section in main configuration.
            merge_dict(config, subconfig)

    return config


def merge_dict(dict1, dict2):
    """Merges :obj:`dict2` into :obj:`dict1`.
    Args:
      dict1: The base dictionary.
      dict2: The dictionary to merge.
    Returns:
      The merged dictionary :obj:`dict1`.
    """
    for key, value in six.iteritems(dict2):
        if isinstance(value, dict):
            dict1[key] = merge_dict(dict1.get(key, {}), value)
        else:
            dict1[key] = value
    return dict1


def store_2d(array, fw):
    """
    fw = tf.io.gfile.GFile('distribution.bin', 'wb')
    array: np.array([])
    """
    fw.write(struct.pack('I', len(array)))
    for i, distrib in enumerate(array):
        for p in distrib:
            p = struct.pack('f', p)
            fw.write(p)