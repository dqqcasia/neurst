import logging
import tensorflow as tf
from st.layers.common_layers_v1 import conv_lstm, layer_norm
from tensorflow.contrib.rnn import BasicLSTMCell, GRUCell, LayerNormBasicLSTMCell, DropoutWrapper, ResidualWrapper, MultiRNNCell, OutputProjectionWrapper
from st.models.layers import pooling
from st.models.tools.utils import residual, multihead_attention, ff_hidden


def compute_kd_loss(bert_feat, encoded, loss_type):
    if loss_type == 'mean_squared_error':
        loss = tf.losses.mean_squared_error(bert_feat, encoded)
    elif loss_type == 'absolute_difference':
        loss = tf.losses.absolute_difference(bert_feat, encoded)
    elif loss_type == 'cosine_distance':
        loss = tf.losses.cosine_distance(bert_feat, encoded)
    elif loss_type == 'hinge_loss':
        loss = tf.losses.hinge_loss(bert_feat, encoded)
    elif loss_type == 'huber_loss':
        loss = tf.losses.huber_loss(bert_feat, encoded)
    elif loss_type == 'log_loss':
        loss = tf.losses.log_loss(bert_feat, encoded)
    elif loss_type == 'mean_pairwise_squared_error':
        loss = tf.losses.mean_pairwise_squared_error(bert_feat, encoded)
    elif loss_type == 'cross_entropy':
        tf.logging.error("Not implemented loss type!")
        loss = 0
    else:
        tf.logging.error("Not implemented loss type!")
        loss = 0
    return loss


def make_attention(Q, V, args=None):
    # Multihead Attention (vanilla attention)
    Q = multihead_attention(
          query_antecedent=Q,
          memory_antecedent=V,
          bias=None,
          # bias=None,
          total_key_depth=args.model.kd.num_cell_units,
          total_value_depth=args.model.kd.num_cell_units,
          output_depth=args.model.kd.num_cell_units,
          num_heads=args.model.kd.num_heads,
          dropout_rate=args.model.kd.attention_dropout_rate,
          name="decoder_vanilla_attention",
          summaries=True)
    return Q


def extract_feat(encoded, bert_feat, args):
    with tf.name_scope('encoded_feat_layer'):
        type = args.model.kd.feat_extract_type
        if type == 'reduce_mean':
            encoded_feat = tf.reduce_mean(encoded, axis=1, keep_dims=True)
        elif type == 'cnn_pooling':
            if args.model.kd.filter_num:
                num_cell_units = args.model.kd.filter_num
            else:
                raise NotImplementedError("Not assign filter_num")
            encoded_feat, _ = pooling(encoded, tf.shape(encoded)[1], "HALF", 1, num_cell_units, method='average')
            encoded_feat = tf.reduce_mean(encoded_feat, axis=1, keep_dims=True)

        elif type == 'self_attention':
            encoded_feat = make_attention(bert_feat, encoded, args)
        else:
            raise NotImplementedError('Not implemented extract_feat type!')

    return encoded_feat



