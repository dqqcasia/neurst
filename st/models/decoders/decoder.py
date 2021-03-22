'''@file asr_decoder.py
contains the EDDecoder class'''

from abc import ABCMeta, abstractmethod
import tensorflow as tf
from collections import namedtuple

from st.tools.tftools.tfTools import right_shift_rows


class Decoder(object):
    '''a general decoder for an encoder decoder system
    converts the high level features into output logits
    '''

    __metaclass__ = ABCMeta

    def __init__(self, args, is_train, global_step, embed_table=None, name=None):
        '''EDDecoder constructor
        Args:
            conf: the decoder configuration as a configparser
            outputs: the name of the outputs of the model
            constraint: the constraint for the variables

            self.start_token is used in the infer_graph, for auto feed the first
            <sos> tokens to the decoder, while in the train_graph, you need to
            pad the <sos> for the decoder input manually!
            Also, in the infer_graph, decoder should know when to stop, so the
            decoder need to specify the <eos> in the helper or BeamSearchDecoder.
        '''
        self.args = args
        self.name = name
        self.is_train = is_train
        self.start_token = args.token2idx['<sos>'] # tf.fill([self.batch_size], args.token2idx['<sos>'])
        self.end_token = args.token2idx['<eos>']
        self.embed_table = embed_table
        self.global_step = global_step
        self.start_warmup_steps = self.args.model.start_warmup_steps
        self.schedule = self.linear_increase(
            prob_start=self.args.model.prob_start,
            prob_end=self.args.model.prob_end,
            global_step=self.global_step,
            start_warmup_steps=self.args.model.start_warmup_steps,
            interim_steps=self.args.model.interim_steps)

    def __call__(self, encoded, len_encoded):
        '''
        Create the variables and do the forward computation to decode an entire
        sequence

        Returns:
            - the output logits of the decoder as a dictionary of
                [batch_size x time x ...] tensors
            - the logit sequence_lengths as a dictionary of [batch_size] vectors
            - the final state of the decoder as a possibly nested tupple
                of [batch_size x ... ] tensors
        '''
        with tf.variable_scope(self.name or 'decoder'):
            logits, sample_id, len_decode = self._decode(encoded, len_encoded)

        return logits, sample_id, len_decode

    def build_input(self, id_gpu, tensors_input):
        """
        the decoder label input is tensors_input.labels left concat <sos>,
        the lengths correspond add 1.
        Create a tgt_input prefixed with <sos> and
        PLEASE create a tgt_output suffixed with <eos> in the ce_loss.

        we need to pass the tensors_input in to judge whether there is
        tensors_input.label_splits
        """
        decoder_input = namedtuple('decoder_input',
            'input_labels, output_labels, len_labels')

        assert self.start_token, self.end_token

        if tensors_input.label_splits:
            # in the training mode, so that label is provided
            decoder_input.output_labels = tensors_input.label_splits[id_gpu]
            decoder_input.input_labels = right_shift_rows(
                p=tensors_input.label_splits[id_gpu],
                shift=1,
                pad=self.start_token)
            decoder_input.len_labels = tensors_input.len_label_splits[id_gpu]
        else:
            # in the infer mode, so no label is provided
            decoder_input.output_labels = None
            decoder_input.input_labels = None
            decoder_input.len_labels = None

        return decoder_input

    def build_mtl_input(self, id_gpu, tensors_input):
        """
        the decoder label input is tensors_input.labels left concat <sos>,
        the lengths correspond add 1.
        Create a tgt_input prefixed with <sos> and
        PLEASE create a tgt_output suffixed with <eos> in the ce_loss.

        we need to pass the tensors_input in to judge whether there is
        tensors_input.label_splits
        """
        decoder_input = namedtuple('decoder_input',
            'input_labels, output_labels, len_labels, input_labels_aux, output_labels_aux, len_labels_aux')

        assert self.start_token, self.end_token

        if tensors_input.label_splits:
            # in the training mode, so that label is provided
            decoder_input.output_labels = tensors_input.label_splits[id_gpu]
            decoder_input.input_labels = right_shift_rows(
                p=tensors_input.label_splits[id_gpu],
                shift=1,
                pad=self.start_token)
            decoder_input.len_labels = tensors_input.len_label_splits[id_gpu]
        else:
            # in the infer mode, so no label is provided
            decoder_input.output_labels = None
            decoder_input.input_labels = None
            decoder_input.len_labels = None

        if tensors_input.aux_label_splits:
            decoder_input.output_labels_aux = tensors_input.aux_label_splits[id_gpu]
            decoder_input.input_labels_aux = right_shift_rows(
                p=tensors_input.aux_label_splits[id_gpu],
                shift=1,
                pad=self.start_token)
            decoder_input.len_labels_aux = tensors_input.len_aux_label_splits[id_gpu]
        else:
            # in the infer mode, so no label is provided
            decoder_input.output_labels_aux = None
            decoder_input.input_labels_aux = None
            decoder_input.len_labels_aux = None

        return decoder_input

    def build_input_ptdec(self, id_gpu, tensors_input):
        """
        the decoder label input is tensors_input.labels left concat <sos>,
        the lengths correspond add 1.
        Create a tgt_input prefixed with <sos> and
        PLEASE create a tgt_output suffixed with <eos> in the ce_loss.

        we need to pass the tensors_input in to judge whether there is
        tensors_input.label_splits
        """
        decoder_input = namedtuple('decoder_input',
            'input_labels, output_labels, len_labels')

        assert self.start_token, self.end_token

        if self.args.model.shift_right:
            if tensors_input.label_splits:
                # in the training mode, so that label is provided
                decoder_input.output_labels = tensors_input.label_splits[id_gpu]
                decoder_input.input_labels = right_shift_rows(
                    p=tensors_input.label_splits[id_gpu],
                    shift=1,
                    pad=self.start_token)
                decoder_input.len_labels = tensors_input.len_label_splits[id_gpu]
            else:
                # in the infer mode, so no label is provided
                decoder_input.output_labels = None
                decoder_input.input_labels = None
                decoder_input.len_labels = None

        else:

            if tensors_input.label_splits:
                # in the training mode, so that label is provided
                decoder_input.output_labels = tensors_input.bert_feat_splits[id_gpu]
                decoder_input.input_labels = tensors_input.label_splits[id_gpu]
                decoder_input.len_labels = tensors_input.len_label_splits[id_gpu]
            else:
                # in the infer mode, so no label is provided
                decoder_input.output_labels = None
                decoder_input.input_labels = None
                decoder_input.len_labels = None
        return decoder_input

    def teacher_forcing(self, encoded, len_encoded, target_labels, max_len):
        with tf.variable_scope(self.name or 'decoder'):
            logits = self.teacherforcing_decode(encoded, len_encoded, target_labels, max_len)
        return logits

    def max_decoder_len(self, len_src=None):
        if self.args.model.decoder.len_max_decoder:
            len_max_decode = self.args.model.decoder.len_max_decoder
        else:
            assert len_src
            decoding_length_factor = 2.0
            len_max_decode = tf.to_int32(tf.round(
                tf.to_float(len_src) * decoding_length_factor))

        return len_max_decode

    def embedding(self, ids):
        if self.embed_table:
            embeded = tf.nn.embedding_lookup(self.embed_table, ids)
        else:
            embeded = tf.one_hot(ids, self.args.dim_output, dtype=tf.float32)

        return embeded

    @abstractmethod
    def _decode(self, encoded, len_encoded, labels, len_labels):
        '''
        Create the variables and do the forward computation to decode an entire
        sequence

        Args:
            encoded: the encoded inputs, this is a dictionary of
                [batch_size x time x ...] tensors
            encoded_seq_length: the sequence lengths of the encoded inputs
                as a dictionary of [batch_size] vectors
            targets: the targets used as decoder inputs as a dictionary of
                [batch_size x time x ...] tensors
            target_seq_length: the sequence lengths of the targets
                as a dictionary of [batch_size] vectors

        Returns:
            - the output logits of the decoder as a dictionary of
                [batch_size x time x ...] tensors
            - the logit sequence_lengths as a dictionary of [batch_size] vectors
            - the final state of the decoder as a possibly nested tupple
                of [batch_size x ... ] tensors
        '''

    @abstractmethod
    def zero_state(self, encoded_dim, batch_size):
        '''get the decoder zero state

        Args:
            encoded_dim: the dimension of the encoded sequence as a list of
                integers
            batch size: the batch size as a scalar Tensor

        Returns:
            the decoder zero state as a possibly nested tupple
                of [batch_size x ... ] tensors
        '''

    @property
    def variables(self):
        '''
        get a list of the models's variables
        '''
        variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=self.name + '/')

        if hasattr(self, 'wrapped'):
            #pylint: disable=E1101
            variables += self.wrapped.variables

        return variables

    @abstractmethod
    def get_output_dims(self):
        '''get the decoder output dimensions

        args:
            trainlabels: the number of extra labels the trainer needs

        Returns:
            a dictionary containing the output dimensions
        '''

    @staticmethod
    def linear_increase(global_step, start_warmup_steps, prob_start, prob_end, interim_steps):
        global_step = tf.to_float(global_step)
        step_increasement = tf.to_float((prob_end-prob_start)/interim_steps)
        schedule = step_increasement * (global_step-start_warmup_steps) + prob_start

        # if prob_start > prob_end:
        #     schedule = tf.minimum(tf.maximum(schedule, prob_end), prob_start)
        # else:
        #     schedule = tf.minimum(tf.maximum(schedule, prob_start), prob_end)
        schedule = tf.clip_by_value(schedule, min(prob_start, prob_end), max(prob_start, prob_end))

        return schedule
