'''@file rnn_decoder.py
contains the general recurrent decoder class'''

import tensorflow as tf
import os
from .decoder import Decoder
# from ..tools import helpers


class FCDecoder(Decoder):
    '''a follly connected decoder for the CTC architecture'''

    def _decode(self, encoded, len_encoded):
        '''
        Create the variables and do the forward computation to decode an entire
        sequence

        Args:
            encoded: the encoded inputs, this is a list of
                [batch_size x ...] tensors
            encoded_seq_length: the sequence lengths of the encoded inputs
                as a list of [batch_size] vectors

        Returns:
            - the output logits of the decoder as a list of
                [batch_size x ...] tensors
            - the logit sequence_lengths as a list of [batch_size] vectors
            - the final state of the decoder as a possibly nested tupple
                of [batch_size x ... ] tensors
        '''
        # dim_output = self.args.dim_output
        dim_output = self.args.src_dim_output if self.args.src_dim_output else self.args.dim_output
        logits = tf.layers.dense(
            inputs=encoded,
            units=dim_output,
            activation=None,
            use_bias=False,
            name='fully_connected')

        preds = tf.argmax(logits, -1)

        return logits, preds, len_encoded

    def decode(self, encoded, len_encoded):
        '''
        Create the variables and do the forward computation to decode an entire
        sequence

        Args:
            encoded: the encoded inputs, this is a list of
                [batch_size x ...] tensors
            encoded_seq_length: the sequence lengths of the encoded inputs
                as a list of [batch_size] vectors

        Returns:
            - the output logits of the decoder as a list of
                [batch_size x ...] tensors
            - the logit sequence_lengths as a list of [batch_size] vectors
            - the final state of the decoder as a possibly nested tupple
                of [batch_size x ... ] tensors
        '''
        # dim_output = self.args.dim_output
        dim_output = self.args.dim_ctc_output if self.args.dim_ctc_output else self.args.dim_output
        logits = tf.layers.dense(
            inputs=encoded,
            units=dim_output,
            activation=None,
            use_bias=False,
            name='fully_connected')

        preds = tf.argmax(logits, -1)

        return logits, preds, len_encoded
