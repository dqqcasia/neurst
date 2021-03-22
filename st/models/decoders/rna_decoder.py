'''@file rnn_decoder.py
the while_loop implementation'''

import tensorflow as tf
from .decoder import Decoder
from tensorflow.python.util import nest
from st.layers import common_layers
from st.models.coldFusion import cold_fusion
from st.models.tools.utils import dense
import logging

inf = 1e10


class RNADecoder(Decoder):
    """language model cold fusion
    """

    def __init__(self, args, is_train, global_step, embed_table=None, name=None):
        self.num_layers = args.model.decoder.num_layers
        self.num_cell_units_de = args.model.decoder.num_cell_units
        self.dropout = args.model.decoder.dropout
        self.num_cell_units_en = args.model.encoder.num_cell_units
        self.size_embedding = args.model.decoder.size_embedding
        self.dim_output = args.dim_output
        self.beam_size = args.beam_size
        self.softmax_temperature = args.model.decoder.softmax_temperature
        if args.model.shallow_fusion or args.model.rerank or args.model.cold_fusion:
            logging.info('load language model object: {}'.format(args.lm_obj))
            self.lm = args.lm_obj
        super().__init__(args, is_train, global_step, embed_table, name)


    def _decode(self, encoded, len_encoded):
        batch_size = tf.shape(len_encoded)[0]
        blank_id = self.dim_output-1
        token_init = tf.fill([batch_size, 1], blank_id)
        logits_init = tf.zeros([batch_size, 1, self.dim_output], dtype=tf.float32)

        self.cell = self.create_cell()
        # collect the initial states of lstms used in decoder.
        all_initial_states = {}
        all_initial_states["state_decoder"] = self.zero_state(batch_size, dtype=tf.float32)
        if self.args.model.decoder.cold_fusion:
            all_initial_states["state_lm"] = self.lm.zero_state(batch_size, dtype=tf.float32)
            all_initial_states["logit_lm"] = tf.zeros([batch_size, self.dim_output], dtype=tf.float32)

        def step(i, preds, all_states, logits):
            state_decoder = all_states["state_decoder"]
            prev_emb = self.embedding(preds[:, -1])
            decoder_input = tf.concat([encoded[:, i, :], prev_emb], axis=1)
            decoder_input.set_shape([None, self.size_embedding + self.num_cell_units_en])

            # Lstm part
            with tf.variable_scope("decoder_lstms"):
                output_decoder, state_decoder = tf.contrib.legacy_seq2seq.rnn_decoder(
                    decoder_inputs=[decoder_input],
                    initial_state=state_decoder,
                    cell=self.cell)
                all_states["state_decoder"] = state_decoder
                output_decoder = [tf.concat([output_decoder[0], encoded[:, i, :]], axis=1)]

            if self.args.model.decoder.cold_fusion:
                logit_lm, state_lm = self.lm.forward(
                    input=preds[:, -1],
                    state=all_states["state_lm"],
                    stop_gradient=True)

                logit_lm, state_lm = self.update_lm(
                    preds=preds[:, -1],
                    blank_id=blank_id,
                    logit_lm=logit_lm[0],
                    state_lm=state_lm,
                    logit_lm_pre=all_initial_states["logit_lm"],
                    state_lm_pre=all_initial_states["state_lm"],
                    num_cell_units_lm=self.args.model.lm.model.decoder.num_cell_units)

                all_initial_states["logit_lm"] = logit_lm
                all_states["state_lm"] = state_lm

                cur_logit = cold_fusion(
                    logit_lm=logit_lm,
                    state_decoder=state_decoder,
                    num_cell_units=self.num_cell_units_lm,
                    dim_output=self.dim_output)
            else:
                cur_logit = tf.layers.dense(
                    inputs=output_decoder[0],
                    units=self.dim_output,
                    activation=None,
                    use_bias=False,
                    name='fully_connected'
                    )

            if self.is_train and self.args.model.decoder.sample_decoder:
                cur_ids = tf.distributions.Categorical(logits=cur_logit/self.softmax_temperature).sample()
            else:
                cur_ids = tf.to_int32(tf.argmax(cur_logit, -1))
            preds = tf.concat([preds, tf.expand_dims(cur_ids, 1)], axis=1)
            logits = tf.concat([logits, tf.expand_dims(cur_logit, 1)], 1)

            return i+1, preds, all_states, logits

        _, preds, _, logits = tf.while_loop(
            cond=lambda i, *_: tf.less(i, tf.shape(encoded)[1]),
            body=step,
            loop_vars=[0, token_init, all_initial_states, logits_init],
            shape_invariants=[tf.TensorShape([]),
                              tf.TensorShape([None, None]),
                              nest.map_structure(lambda t: tf.TensorShape(t.shape), all_initial_states),
                              tf.TensorShape([None, None, self.dim_output])]
            )

        logits = logits[:, 1:, :]
        preds = preds[:, 1:]
        not_padding = tf.to_int32(tf.sequence_mask(len_encoded, maxlen=tf.shape(encoded)[1]))
        preds = tf.multiply(tf.to_int32(preds), not_padding)

        return logits, preds, len_encoded

    def beam_decode_rerank(self, encoded, len_encoded):
        """
        beam search rerank at end with language model integration (self-attention model)
        the input to te score is <sos> + tokens !!!
        """
        from tfTools.tfTools import alignment_shrink
        # from tfTools.tfMath import sum_log

        lambda_lm = self.args.lambda_lm
        beam_size = self.beam_size
        batch_size = tf.shape(len_encoded)[0]
        blank_id = self.dim_output-1

        # beam search Initialize
        # repeat each sample in batch along the batch axis [1,2,3,4] -> [1,1,2,2,3,3,4,4]
        encoded = tf.tile(encoded[:, None, :, :],
                          multiples=[1, beam_size, 1, 1]) # [batch_size, beam_size, *, hidden_units]
        encoded = tf.reshape(encoded,
                             [batch_size * beam_size, -1, encoded.get_shape()[-1].value])
        # [[<blk>, <blk>, ..., <blk>,]], shape: [batch_size * beam_size, 1]
        token_init = tf.fill([batch_size * beam_size, 1], self.args.sos_idx)
        logits_init = tf.zeros([batch_size * beam_size, 1, self.dim_output], dtype=tf.float32)
        # the score must be [0, -inf, -inf, ...] at init, for the preds in beam is same in init!!!
        scores = tf.constant([0.0] + [-inf] * (beam_size - 1), dtype=tf.float32)  # [beam_size]
        scores = tf.tile(scores, multiples=[batch_size])  # [batch_size * beam_size]
        cache_init = tf.zeros([batch_size*beam_size,
                               0,
                               self.lm.args.model.decoder.num_blocks,
                               self.lm.args.model.decoder.num_cell_units])

        # create decoder cell
        self.cell = self.create_cell()

        # collect the initial states of lstms used in decoder.
        all_initial_states = {}
        all_initial_states["state_decoder"] = self.zero_state(batch_size * beam_size, dtype=tf.float32)
        base_indices = tf.reshape(tf.tile(tf.range(batch_size)[:, None], multiples=[1, beam_size]), shape=[-1])

        def step(i, preds, scores, all_states, logits, cache):
            """
            the cache has no specific shape, so no can be put in the all_states
            """
            state_decoder = all_states["state_decoder"]
            prev_emb = self.embedding(preds[:, -1])
            decoder_input = tf.concat([encoded[:, i, :], prev_emb], axis=1)
            decoder_input.set_shape([None, self.size_embedding + self.num_cell_units_en])

            # Lstm part
            with tf.variable_scope("decoder_lstms"):
                output_decoder, state_decoder = tf.contrib.legacy_seq2seq.rnn_decoder(
                    decoder_inputs=[decoder_input],
                    initial_state=state_decoder,
                    cell=self.cell)
                all_states["state_decoder"] = state_decoder
                output_decoder = [tf.concat([output_decoder[0], encoded[:, i, :]], axis=1)]

            cur_logit = tf.layers.dense(
                inputs=output_decoder[0],
                units=self.dim_output,
                activation=None,
                use_bias=False,
                name='fully_connected')

            logits = tf.concat([logits, cur_logit[:, None]], 1)
            z = tf.nn.log_softmax(cur_logit) # [batch*beam, size_output]

            # the langueage model infer
            if self.args.model.shallow_fusion:
                # must use the `blank_id` to pad the pred otherwise the pad whould be preserved
                preds, len_noblank = alignment_shrink(preds, blank_id, blank_id)
                preds = tf.concat([token_init, preds], -1)
                preds_emb = self.lm.decoder.embedding(preds)

                with tf.variable_scope(self.args.top_scope, reuse=True):
                    with tf.variable_scope(self.args.lm_scope):
                        lm_output, cache = self.lm.decoder.decoder_with_caching_impl(preds_emb, cache)
                        logit_lm = dense(
                            inputs=lm_output[:, -1, :],
                            units=self.dim_output,
                            kernel=tf.transpose(self.lm.decoder.fully_connected),
                            use_bias=False)
                z_lm = lambda_lm * tf.nn.log_softmax(logit_lm) # [batch*beam, size_output]
            else:
                z_lm = tf.zeros_like(z)

            z_lm = tf.where(
                tf.equal(tf.argmax(z, -1), blank_id),
                x=tf.zeros_like(z_lm),
                y=z_lm)

            # rank the combined scores
            next_scores, next_preds = tf.nn.top_k(z+z_lm, k=beam_size, sorted=True)
            next_preds = tf.to_int32(next_preds)

            # beamed scores & Pruning
            scores = scores[:, None] + next_scores  # [batch_size * beam_size, beam_size]
            scores = tf.reshape(scores, shape=[batch_size, beam_size * beam_size])

            _, k_indices = tf.nn.top_k(scores, k=beam_size)
            k_indices = base_indices * beam_size * beam_size + tf.reshape(k_indices, shape=[-1])  # [batch_size * beam_size]
            # Update scores.
            scores = tf.reshape(scores, [-1])
            scores = tf.gather(scores, k_indices)
            # Update predictions.
            next_preds = tf.gather(tf.reshape(next_preds, shape=[-1]), indices=k_indices)
            # k_indices: [0~batch*beam*beam], preds: [0~batch*beam]
            preds = tf.gather(preds, indices=k_indices // beam_size)
            preds = tf.concat((preds, next_preds[:, None]), axis=1)  # [batch_size * beam_size, i]

            return i+1, preds, scores, all_states, logits, cache

        _, preds, score_org, _, logits, cache = tf.while_loop(
            cond=lambda i, *_: tf.less(i, tf.shape(encoded)[1]),
            body=step,
            loop_vars=[0, token_init, scores, all_initial_states, logits_init, cache_init],
            shape_invariants=[tf.TensorShape([]),
                              tf.TensorShape([None, None]),
                              tf.TensorShape([None]),
                              nest.map_structure(lambda t: tf.TensorShape(t.shape), all_initial_states),
                              tf.TensorShape([None, None, self.dim_output]),
                              tf.TensorShape([None, None, None, None])]
            )

        preds, len_encoded = alignment_shrink(preds, blank_id)
        preds = tf.concat([token_init, preds], -1)

        with tf.variable_scope(self.args.top_scope, reuse=True):
            with tf.variable_scope(self.args.lm_scope):
                # the len_encoded not counts the token_init
                score_rerank, distribution = self.lm.decoder.score(preds, len_encoded)

        # [batch_size * beam_size, ...]
        preds = preds[:, 1:]
        logits = logits[:, 1:, :]
        not_padding = tf.to_int32(tf.sequence_mask(len_encoded, maxlen=tf.shape(preds)[1]))
        preds *= not_padding

        # [batch_size , beam_size, ...]
        score_rerank = self.args.lambda_rerank * score_rerank
        # scores += score_rerank
        scores = score_org + score_rerank

        # [batch_size, beam_size, ...]
        scores_sorted, sorted = tf.nn.top_k(tf.reshape(scores, [batch_size, beam_size]),
                                            k=beam_size,
                                            sorted=True)
        # [batch_size * beam_size, ...]
        base_indices = tf.reshape(tf.tile(tf.range(batch_size)[:, None],
                                          multiples=[1, beam_size]), shape=[-1])
        sorted = base_indices * beam_size + tf.reshape(sorted, shape=[-1])  # [batch_size * beam_size]

        preds_sorted = tf.gather(preds, sorted)
        score_rerank_sorted = tf.gather(score_rerank, sorted)
        score_org_sorted = tf.gather(score_org, sorted)

        # [batch_size, beam_size, ...]
        preds_sorted = tf.reshape(preds_sorted, shape=[batch_size, beam_size, -1])
        scores_sorted = tf.reshape(scores_sorted, shape=[batch_size, beam_size])
        score_rerank_sorted = tf.reshape(score_rerank_sorted, shape=[batch_size, beam_size])
        score_org_sorted = tf.reshape(score_org_sorted, shape=[batch_size, beam_size])

        return preds_sorted[:, 0, :], [preds_sorted, score_org_sorted, score_rerank_sorted]

    def create_cell(self):
        cell = common_layers.lstm_cells(
            self.num_layers,
            self.num_cell_units_de,
            initializer=None,
            dropout=self.dropout)

        return cell

    def zero_state(self, batch_size, dtype):
        return self.cell.zero_state(batch_size, dtype)

    @staticmethod
    def update_lm(preds, blank_id, logit_lm, state_lm, logit_lm_pre, state_lm_pre, num_cell_units_lm):
        is_blank = tf.equal(preds, tf.fill(tf.shape(preds), blank_id))

        # logit update
        updated_logit_lm = tf.where(
            condition=is_blank,
            x=logit_lm_pre,
            y=logit_lm)

        # state update
        is_blank = tf.stack([tf.to_float(is_blank)]*num_cell_units_lm, 1)
        updated_state_lm = []
        for cell_pre, cell in zip(state_lm_pre, state_lm):
            h_states = is_blank * cell_pre.h + (1-is_blank) * cell.h
            c_states = is_blank * cell_pre.c + (1-is_blank) * cell.c
            updated_state_lm.append(tf.contrib.rnn.LSTMStateTuple(c_states, h_states))
        updated_state_lm = tuple(updated_state_lm)

        return updated_logit_lm, updated_state_lm
