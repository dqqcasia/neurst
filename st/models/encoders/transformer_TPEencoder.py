'''@file listener.py
contains the listener code'''

import tensorflow as tf
import numpy as np
from .encoder import Encoder
from st.models.layers import residual, conv_lstm
from st.layers import common_attention_v1
from ..tools.utils import residual, multihead_attention, ff_hidden

from st.layers.common_layers_v1 import layer_norm


class Transformer_TPEncoder(Encoder):
    '''Test Model'''

    def __init__(self, args, is_train, embed_table=None, name=None):
        self.name = name
        super().__init__(args, is_train, embed_table=embed_table, name=self.name)

    @staticmethod
    def normal_conv(inputs, filter_num, kernel, stride, padding, use_relu, name,
                    w_initializer=None, norm_type="batch"):
        with tf.variable_scope(name):
            net = tf.layers.conv2d(inputs, filter_num, kernel, stride, padding,
                                   kernel_initializer=w_initializer, name="conv")
            if norm_type == "batch":
                net = tf.layers.batch_normalization(net, name="bn")
            elif norm_type == "layer":
                net = layer_norm(net)
            else:
                net = net
            output = tf.nn.relu(net) if use_relu else net

        return output

    def pooling(self, x, len_sequence, type, name, method=None):
        num_cell_units = self.args.model.encoder.num_cell_units

        x = tf.expand_dims(x, axis=2)
        x = self.normal_conv(
            x,
            num_cell_units,
            (1, 1),
            (1, 1),
            'SAME',
            'True',
            name="tdnn_"+str(name),
            norm_type='layer')
        if method == 'average':
            x = tf.layers.average_pooling2d(x, (2, 2), (3, 3), padding='same', data_format='channels_last', name=None)
        else:
            if type == 'SAME':
                x = tf.layers.max_pooling2d(x, (1, 1), (1, 1), 'SAME')
            elif type == 'HALF':
                x = tf.layers.max_pooling2d(x, (2, 1), (2, 1), 'SAME')
                len_sequence = tf.cast(tf.ceil(tf.cast(len_sequence, tf.float32)/2), tf.int32)

        x = tf.squeeze(x, axis=2)

        return x, len_sequence

    def encode(self, features, len_feas):

        attention_dropout_rate = self.args.model.encoder2.attention_dropout_rate if self.is_train else 0.0
        residual_dropout_rate = self.args.model.encoder2.residual_dropout_rate if self.is_train else 0.0
        hidden_units = self.args.model.encoder2.num_cell_units
        num_heads = self.args.model.encoder2.num_heads
        num_blocks = self.args.model.encoder2.num_blocks
        self._ff_activation = tf.nn.relu

        # Mask
        encoder_padding = tf.equal(tf.sequence_mask(len_feas, maxlen=tf.shape(features)[1]), False)  # bool tensor
        encoder_attention_bias = common_attention_v1.attention_bias_ignore_padding(encoder_padding)

        # Add positional signal
        encoder_output = common_attention_v1.add_timing_signal_1d(features)
        # Dropout
        encoder_output = tf.layers.dropout(encoder_output,
                                           rate=residual_dropout_rate,
                                           training=self.is_train)

        # Blocks
        for i in range(num_blocks):
            with tf.variable_scope("block_{}".format(i)):
                # Multihead Attention
                encoder_output = residual(encoder_output,
                                          multihead_attention(
                                              query_antecedent=encoder_output,
                                              memory_antecedent=None,
                                              bias=encoder_attention_bias,
                                              total_key_depth=hidden_units,
                                              total_value_depth=hidden_units,
                                              output_depth=hidden_units,
                                              num_heads=num_heads,
                                              dropout_rate=attention_dropout_rate,
                                              name='encoder_self_attention',
                                              summaries=True),
                                          dropout_rate=residual_dropout_rate)

                # Feed Forward
                encoder_output = residual(encoder_output,
                                          ff_hidden(
                                              inputs=encoder_output,
                                              hidden_size=4 * hidden_units,
                                              output_size=hidden_units,
                                              activation=self._ff_activation),
                                          dropout_rate=residual_dropout_rate)
        # Mask padding part to zeros.
        encoder_output *= tf.expand_dims(1.0 - tf.to_float(encoder_padding), axis=-1)

        return encoder_output, len_feas
