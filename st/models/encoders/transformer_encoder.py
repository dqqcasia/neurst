'''@file listener.py
contains the listener code'''

import tensorflow as tf
import numpy as np
from .encoder import Encoder
from st.models.layers import residual, conv_lstm
from st.layers import common_attention_v1
from ..tools.utils import residual, multihead_attention, ff_hidden

from st.layers.common_layers_v1 import layer_norm


class Transformer_Encoder(Encoder):
    '''VERY DEEP CONVOLUTIONAL NETWORKS FOR END-TO-END SPEECH RECOGNITION
    '''

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

        attention_dropout_rate = self.args.model.encoder.attention_dropout_rate if self.is_train else 0.0
        residual_dropout_rate = self.args.model.encoder.residual_dropout_rate if self.is_train else 0.0
        hidden_units = self.args.model.encoder.num_cell_units
        num_heads = self.args.model.encoder.num_heads
        num_blocks = self.args.model.encoder.num_blocks
        self._ff_activation = tf.nn.relu
        # CNN sampling
        if self.args.model.use_cnn_resample:

            batch_size, length, feat_size = tf.shape(features)[0], \
                                            tf.shape(features)[1], \
                                            self.args.data.dim_input
            feat_size = int(feat_size / self.args.data.feature_channel)
            encoder_input = tf.reshape(features,
                                       [batch_size, length, feat_size, self.args.data.feature_channel])

            kernel = tuple([int(i) for i in self.args.model.cnn.kernel.split(',')])
            stride = tuple([int(i) for i in self.args.model.cnn.stride.split(',')])
            for i in range(self.args.model.cnn.layers):
                encoder_input = self.normal_conv(
                    inputs=encoder_input,
                    filter_num=self.args.model.num_filters,
                    kernel=kernel,
                    stride=stride,
                    padding='SAME',
                    use_relu=True,
                    name="conv_"+str(i),
                    w_initializer=None,
                    norm_type='layer')
                length = tf.cast(tf.ceil(tf.cast(length, tf.float32) / stride[0]), tf.int32)
                len_feas = tf.cast(tf.ceil(tf.cast(len_feas, tf.float32) / stride[0]), tf.int32)
                feat_size = int(np.ceil(feat_size / stride[1]))
            feat_size = feat_size * self.args.model.num_filters

            encoder_input = tf.layers.max_pooling2d(
                inputs=encoder_input,
                pool_size=(3, 3),
                strides=(1, 1),
                padding='SAME')

            encoder_input = tf.reshape(encoder_input, [batch_size, length, feat_size])
            features = encoder_input

        # Mask
        encoder_padding = tf.equal(tf.sequence_mask(len_feas, maxlen=tf.shape(features)[1]), False)
        encoder_attention_bias = common_attention_v1.attention_bias_ignore_padding(encoder_padding)

        # Add positional signal
        encoder_output = common_attention_v1.add_timing_signal_1d(features)
        
        # Dropout
        encoder_output = tf.layers.dropout(encoder_output,
                                           rate=residual_dropout_rate,
                                           training=self.is_train)
        # Dense
        encoder_output = tf.layers.dense(
            inputs=encoder_output,
            units=hidden_units,
            activation=None,
            use_bias=False,
            name='encoder_fc')

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
