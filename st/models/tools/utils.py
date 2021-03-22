from __future__ import print_function

import codecs
import logging
import os
import time
from tempfile import mkstemp

import numpy as np
import tensorflow as tf
import tensorflow.contrib.framework as tff
from tensorflow.python.layers import base as base_layer

from st.layers import common_layers_v1, common_attention_v1
common_layers_v1.allow_defun = False


class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if item not in self:
            logging.warning('{} is not in the dict. None is returned as default.'.format(item))
            return None
        if type(self[item]) is dict:
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
            vocab = [line.split()[0] for line in codecs.open(path, 'r', 'utf-8')]
            vocab = vocab[:vocab_size]
            assert len(vocab) == vocab_size
            word2idx = {word: idx for idx, word in enumerate(vocab)}
            idx2word = {idx: word for idx, word in enumerate(vocab)}
            return word2idx, idx2word

        logging.debug('Load vocabularies %s and %s.' % (self._config.src_vocab, self._config.dst_vocab))
        self.src2idx, self.idx2src = load_vocab_(self._config.src_vocab, self._config.src_vocab_size)
        self.dst2idx, self.idx2dst = load_vocab_(self._config.dst_vocab, self._config.dst_vocab_size)

    def get_training_batches(self, shuffle=True, epoches=None):
        """
        Generate batches according to bucket setting.
        """
        buckets = [(i, i) for i in range(5, 1000000, 3)]

        def select_bucket(sl, dl):
            for l1, l2 in buckets:
                if sl < l1 and dl < l2:
                    return l1, l2
            raise Exception("The sequence is too long: ({}, {})".format(sl, dl))

        # Shuffle the training files.
        src_path = self._config.train.src_path
        dst_path = self._config.train.dst_path
        max_length = self._config.train.max_length

        epoch = [0]

        def stop_condition():
            if epoches is None:
                return True
            else:
                epoch[0] += 1
                return epoch[0] < epoches + 1

        while stop_condition():
            if shuffle:
                logging.debug('Shuffle files %s and %s.' % (src_path, dst_path))
                src_shuf_path, dst_shuf_path = self.shuffle([src_path, dst_path])
                self._tmps.add(src_shuf_path)
                self._tmps.add(dst_shuf_path)
            else:
                src_shuf_path = src_path
                dst_shuf_path = dst_path

            caches = {}
            for bucket in buckets:
                caches[bucket] = [[], [], 0, 0]  # src sentences, dst sentences, src tokens, dst tokens

            for src_sent, dst_sent in zip(open(src_shuf_path, 'r'), open(dst_shuf_path, 'r')):
                src_sent, dst_sent = src_sent.decode('utf8'), dst_sent.decode('utf8')

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

                if max(caches[bucket][2], caches[bucket][3]) >= self._config.train.tokens_per_batch:
                    batch = (self.create_batch(caches[bucket][0], o='src'), self.create_batch(caches[bucket][1], o='dst'))
                    logging.debug(
                        'Yield batch with source shape %s and target shape %s.' % (batch[0].shape, batch[1].shape))
                    yield batch
                    caches[bucket] = [[], [], 0, 0]

            # Clean remain sentences.
            for bucket in buckets:
                # Ensure each device at least get one sample.
                if len(caches[bucket][0]) >= max(1, self._config.train.num_gpus):
                    batch = (self.create_batch(caches[bucket][0], o='src'), self.create_batch(caches[bucket][1], o='dst'))
                    logging.debug(
                        'Yield batch with source shape %s and target shape %s.' % (batch[0].shape, batch[1].shape))
                    yield batch

            # Remove shuffled files when epoch finished.
            if shuffle:
                os.remove(src_shuf_path)
                os.remove(dst_shuf_path)
                self._tmps.remove(src_shuf_path)
                self._tmps.remove(dst_shuf_path)

    @staticmethod
    def shuffle(list_of_files):
        tf_os, tpath = mkstemp()
        tf = open(tpath, 'w')

        fds = [open(ff) for ff in list_of_files]

        for l in fds[0]:
            lines = [l.strip()] + [ff.readline().strip() for ff in fds[1:]]
            print("<CONCATE4SHUF>".join(lines), file=tf)

        [ff.close() for ff in fds]
        tf.close()

        os.system('shuf %s > %s' % (tpath, tpath + '.shuf'))

        fnames = ['/tmp/{}.{}.{}.shuf'.format(i, os.getpid(), time.time()) for i, ff in enumerate(list_of_files)]
        fds = [open(fn, 'w') for fn in fnames]

        for l in open(tpath + '.shuf'):
            s = l.strip().split('<CONCATE4SHUF>')
            for i, fd in enumerate(fds):
                print(s[i], file=fd)

        [ff.close() for ff in fds]

        os.remove(tpath)
        os.remove(tpath + '.shuf')

        return fnames

    def get_test_batches(self, src_path, batch_size):
        # Read batches for testing.
        src_sents = []
        for src_sent in open(src_path, 'r'):
            src_sent = src_sent.decode('utf8')
            src_sent = src_sent.split()
            src_sents.append(src_sent)
            # Create a padded batch.
            if len(src_sents) >= batch_size:
                yield self.create_batch(src_sents, o='src')
                src_sents = []
        if src_sents:
            # We ensure batch size not small than gpu number by padding redundant samples.
            if len(src_sents) < self._config.test.num_gpus:
                src_sents.extend([src_sents[-1]] * self._config.test.num_gpus)
            yield self.create_batch(src_sents, o='src')

    def get_test_batches_with_target(self, src_path, dst_path, batch_size):
        """
        Usually we don't need target sentences for test unless we want to compute PPl.
        Returns:
            Paired source and target batches.
        """

        src_sents, dst_sents = [], []
        for src_sent, dst_sent in zip(open(src_path, 'r'), open(dst_path, 'r')):
            src_sent, dst_sent = src_sent.decode('utf8'), dst_sent.decode('utf8')
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

    def create_batch(self, sents, o):
        # Convert words to indices.
        assert o in ('src', 'dst')
        word2idx = self.src2idx if o == 'src' else self.dst2idx
        indices = []
        for sent in sents:
            x = [word2idx.get(word, 1) for word in (sent + [u"<eos>"])]
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
        for y in Y: # for each sentence
            sent = []
            for i in y:  # For each word
                if i == 3:  # </S>
                    break
                w = idx2word[i]
                sent.append(w)
            sents.append(' '.join(sent))
        return sents


def expand_feed_dict(feed_dict):
    """If the key is a tuple of placeholders,
    split the input data then feed them into these placeholders.
    """
    new_feed_dict = {}
    for k, v in feed_dict.items():
        if type(k) is not tuple:
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
        if vname in all_available_vars and v.get_shape() == all_available_vars[vname]:
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
    return config.hidden_units ** -0.5 * tf.minimum(
        (global_step + 1.0) * warmup_steps ** -1.5, (global_step + 1.0) ** -0.5)


def shift_right(input, pad=2):
    """Shift input tensor right to create decoder input. '2' denotes <S>"""
    return tf.concat((tf.ones_like(input[:, :1]) * pad, input[:, :-1]), 1)


def embedding(x, vocab_size, dense_size, name=None, reuse=None, kernel=None, multiplier=1.0):
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


def ff_hidden(inputs, hidden_size, output_size, activation, use_bias=True, reuse=None, name=None):
    """
    eq 2:
        hidden_outputs = activation(x * W1 + b1)
        outputs = hidden_outputs * W2 + b2
    """
    with tf.variable_scope(name, "ff_hidden", reuse=reuse):
        hidden_outputs = dense(inputs, hidden_size, activation, use_bias)
        outputs = dense(hidden_outputs, output_size, tf.identity, use_bias)
        return outputs


def multihead_attention_lh(query_antecedent,
                            memory_antecedent,
                            bias,
                            total_key_depth,
                            total_value_depth,
                            output_depth,
                            num_heads,
                            dropout_rate,
                            num_queries=None,
                            summaries=False,
                            image_shapes=None,
                            name=None):
    with tf.variable_scope(
        name,
        default_name="multihead_attention",
        values=[query_antecedent, memory_antecedent]):

        if memory_antecedent is None:
            # Q = K = V
            # self attention
            q = dense(query_antecedent, total_key_depth, use_bias=False, name="q")
            k = dense(query_antecedent, total_key_depth, use_bias=False, name="k")
            v = dense(query_antecedent, total_value_depth, use_bias=False, name="v")

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
        x = dense(x, output_depth, use_bias=False, name="output_transform")
        return x

def ff_hidden_lh(inputs, hidden_size, output_size, activation, use_bias=True, reuse=None, name=None):
    """
    eq 2:
        hidden_outputs = activation(x * W1 + b1)
        outputs = hidden_outputs * W2 + b2
    """
    with tf.variable_scope(name, "ffn", reuse=reuse):
        hidden_outputs = dense(inputs, hidden_size, activation, use_bias, name='conv1')
        outputs = dense(hidden_outputs, output_size, tf.identity, use_bias, name='conv2')
        return outputs

def residual_lh(inputs, outputs, dropout_rate):
    """Residual connection.

    Args:
        inputs: A Tensor.
        outputs: A Tensor.
        dropout_rate: A float range from [0, 1).

    Returns:
        A Tensor.
    """
    outputs = inputs + tf.nn.dropout(outputs, 1 - dropout_rate)
    with tf.variable_scope("layer_prepostprocess"):
        outputs = common_layers_v1.layer_norm(outputs)
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
    with tf.variable_scope(
        name,
        default_name="multihead_attention",
        values=[query_antecedent, memory_antecedent]):

        if not query_eq_key:
            if memory_antecedent is None:
                # Q = K = V
                # self attention
                combined = dense(query_antecedent, total_key_depth * 2 + total_value_depth, name="qkv_transform")
                q, k, v = tf.split(
                  combined, [total_key_depth, total_key_depth, total_value_depth],
                  axis=2)
            else:
                # Q != K = V
                q = dense(query_antecedent, total_key_depth, name="q_transform")
                combined = dense(memory_antecedent, total_key_depth + total_value_depth, name="kv_transform")
                k, v = tf.split(combined, [total_key_depth, total_value_depth], axis=2)
        else:
            # In this setting, we use query_antecedent as the query and key,
            # and use memory_antecedent as the value.
            assert memory_antecedent is not None
            combined = dense(query_antecedent, total_key_depth * 2, name="qk_transform")
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
            bias_initializer=bias_initializer,
            name=name)
        with tf.variable_scope(name, "AttentionGRUCell", reuse=reuse):
            self._attention_keys = dense(attention_memories, num_units, name='attention_key')
            self._attention_values = dense(attention_memories, num_units, name='attention_value')
        self._attention_bias = attention_bias

    def attention(self, inputs, state):
        attention_query = tf.matmul(
            tf.concat([inputs, state], 1), self._attention_query_kernel)
        attention_query = tf.nn.bias_add(attention_query, self._attention_query_bias)

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

    def __init__(self, num_units, recurrent_initializer=None, reuse=None, name=None):
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
        epsilon = np.power(2.0, 1.0/50.0)
        self._recurrent_kernel = tf.clip_by_value(self._recurrent_kernel, -epsilon, epsilon)

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
            self._attention_keys = dense(attention_memories, num_units, name='attention_key')
            self._attention_values = dense(attention_memories, num_units, name='attention_value')
        self._attention_bias = attention_bias

    def attention(self, inputs, state):
        attention_query = tf.matmul(
            tf.concat([inputs, state], 1), self._attention_query_kernel)
        attention_query = tf.nn.bias_add(attention_query, self._attention_query_bias)

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
        epsilon = np.power(2.0, 1.0/50.0)
        self._recurrent_kernel = tf.clip_by_value(self._recurrent_kernel, -epsilon, epsilon)

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
