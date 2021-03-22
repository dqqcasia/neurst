'''@file ed_encoder.py
contains the EDEncoder class'''

from abc import ABCMeta, abstractmethod
import tensorflow as tf
from collections import namedtuple


class Encoder(object):
    '''a general encoder for an encoder decoder system

    transforms input features into a high level representation'''

    __metaclass__ = ABCMeta

    def __init__(self, args, is_train, embed_table=None, name=None):
        '''EDEncoder constructor

        Args:
            args: the encoder configuration
            name: the encoder name
            constraint: the constraint for the variables
        '''
        self.args = args
        self.name = name
        self.is_train = is_train
        self.embed_table = embed_table

    def __call__(self, features, len_feas):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the inputs to the neural network, this is a dictionary of
                [batch_size x time x ...] tensors
            input_seq_length: The sequence lengths of the input utterances, this
                is a dictionary of [batch_size] vectors
            is_train: whether or not the network is in training mode

        Returns:
            - the outputs of the encoder as a dictionary of
                [bath_size x time x ...] tensors
            - the sequence lengths of the outputs as a dictionary of
                [batch_size] tensors
        '''
        with tf.variable_scope(self.name or 'encoder'):
            outputs, output_seq_length = self.encode(features, len_feas)

        return outputs, output_seq_length

    @abstractmethod
    def encode(self, features, len_feas):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the inputs to the neural network, this is a dictionary of
                [batch_size x time x ...] tensors
            input_seq_length: The sequence lengths of the input utterances, this
                is a dictionary of [batch_size] vectors
            is_train: whether or not the network is in training mode

        Returns:
            - the outputs of the encoder as a dictionary of
                [bath_size x time x ...] tensors
            - the sequence lengths of the outputs as a dictionary of
                [batch_size] tensors
        '''

    def embedding(self, ids):
        if self.embed_table:
            embeded = tf.nn.embedding_lookup(self.embed_table, ids)
        else:
            embeded = tf.one_hot(ids, self.args.dim_output, dtype=tf.float32)

        return embeded

    @property
    def variables(self):
        '''get a list of the models's variables'''
        variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=self.name + '/')

        if hasattr(self, 'wrapped'):
            #pylint: disable=E1101
            variables += self.wrapped.variables

        return variables
