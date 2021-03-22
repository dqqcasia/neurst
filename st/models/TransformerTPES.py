'''@file model.py
contains de Model class
During the training , using greadysearch decoder, there is loss.
During the dev/infer, using beamsearch decoder, there is no logits, therefor loss, only preds self.
because we cannot access label during the dev set and need to depend on last time decision.

so, there is only infer and training
'''

"""
Trained with both ASR and AST dataset. (finished)
"""

import tensorflow as tf
import logging
from collections import namedtuple

from st.models.TransformerTPE import TransformerTPE
from st.models.model_tools import choose_device, smoothing_cross_entropy
from st.models.kd_layer import compute_kd_loss, extract_feat
from st.tools.tftools.tfTools import dense_sequence_to_sparse
from st.tools.tftools.gradientTools import average_gradients, handle_gradients
import sys
from st.tools.tftools.tfAudioTools import apply_specaugmant


class TransformerTPES(TransformerTPE):
    '''a general class for an encoder decoder system
    '''

    def __init__(self, tensor_global_step, encoder, decoder, is_train, args, summary=None,
                 batch=None, batch_asr=None, embed_table_encoder=None, embed_table_decoder=None,
                 name='transformer'):
        '''Model constructor

        Args:
        '''

        self.args = args
        self.name = name
        self.ctc_decoder = self.args.model.decoder2.type
        self.mt_encoder = self.args.model.encoder2.type
        self.global_step = tensor_global_step
        self.summary = summary
        self.size_embedding = args.model.decoder.size_embedding
        self.embed_table_decoder = self.get_embedding(
            embed_table=None,
            size_input=args.dim_output,
            size_embedding=self.size_embedding)
        self.ctc_merge_repeated = args.model.decoder2.ctc_merge_repeated

        self.batch_asr = batch_asr
        self.batch = batch

        super().__init__(tensor_global_step, encoder, decoder, is_train, args,
                         batch=batch,
                         embed_table_encoder=None,
                         embed_table_decoder=None,
                         name=name)

    def build_tf_input_ctcbert_asr(self, batch):
        """
                stand training input
                """
        tensors_input_asr = namedtuple('tensors_input_asr', 'feature_splits, label_splits, aux_label_splits, bert_feat_splits, len_fea_splits, '
                                                            'len_label_splits, len_aux_label_splits, len_bert_feat_splits, shape_batch')

        with tf.device(self.center_device):
            with tf.name_scope("inputs"):
                # split input data alone batch axis to gpus
                tensors_input_asr.feature_splits = tf.split(batch[1], self.num_gpus, name="feature_splits")
                tensors_input_asr.label_splits = tf.split(batch[2], self.num_gpus, name="label_splits")
                tensors_input_asr.aux_label_splits = tf.split(batch[3], self.num_gpus, name="aux_label_splits")
                tensors_input_asr.bert_feat_splits = tf.split(batch[4], self.num_gpus, name="bert_feat_splits")
                tensors_input_asr.len_fea_splits = tf.split(batch[5], self.num_gpus, name="len_fea_splits")
                tensors_input_asr.len_label_splits = tf.split(batch[6], self.num_gpus, name="len_label_splits")
                tensors_input_asr.len_aux_label_splits = tf.split(batch[7], self.num_gpus, name='len_aux_label_splits')
                tensors_input_asr.len_bert_feat_splits = tf.split(batch[8], self.num_gpus, name='len_bert_feat_splits')
        tensors_input_asr.shape_batch = tf.shape(self.batch_asr[0])

        return tensors_input_asr

    def ctc_loss(self, logits, len_logits, labels, len_labels):
        """
        No valid path found: It is possible that no valid path is found if the
        activations for the targets are zero.
        """
        with tf.name_scope("ctc_loss"):
            if self.args.model.use_wrapctc:
                import warpctc_tensorflow
                from st.tools.tftools.tfTools import get_indices

                indices = get_indices(len_labels)
                flat_labels = tf.gather_nd(labels, indices)
                ctc_loss = warpctc_tensorflow.ctc(
                    activations=tf.transpose(logits, [1, 0, 2]),
                    flat_labels=flat_labels,
                    label_lengths=len_labels,
                    input_lengths=len_logits,
                    blank_label=self.args.dim_output)
            else:
                labels_sparse = dense_sequence_to_sparse(
                    labels,
                    len_labels)
                ctc_loss = tf.nn.ctc_loss(
                    labels_sparse,
                    logits,
                    sequence_length=len_logits,
                    ctc_merge_repeated=self.ctc_merge_repeated,
                    ignore_longer_outputs_than_inputs=True,
                    time_major=False)

        return ctc_loss

    def ctc_decode(self, logits, len_logits):
        if self.args.ctc_beam_size:
            beam_size = self.args.ctc_beam_size
        else:
            beam_size = 1
        logits_timeMajor = tf.transpose(logits, [1, 0, 2])

        if beam_size == 1:
            decoded_sparse = tf.to_int32(tf.nn.ctc_greedy_decoder(
                logits_timeMajor,
                len_logits,
                merge_repeated=True)[0][0])
        else:
            decoded_sparse = tf.to_int32(tf.nn.ctc_beam_search_decoder(
                logits_timeMajor,
                len_logits,
                beam_width=beam_size,
                merge_repeated=True)[0][0])

        return decoded_sparse

    def build_graph(self):
        # cerate input tensors in the cpu
        tensors_input = self.build_input()
        tensors_input_asr = None
        # create optimizer
        self.optimizer = self.build_optimizer()
        if 'horovod' in sys.modules:
            import horovod.tensorflow as hvd
            logging.info('wrap the optimizer with horovod!')
            self.optimizer = hvd.DistributedOptimizer(self.optimizer)

        loss_step = []
        loss_ast_step = []
        loss_ctc_step = []
        loss_bert_step = []
        tower_grads = []
        list_debug = []
        # the outer scope is necessary for the where the reuse scope need to be limited whthin
        # or reuse=tf.get_variable_scope().reuse
        for id_gpu, name_gpu in enumerate(self.list_gpu_devices):
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                loss, loss_ast, loss_ctc, loss_bert, gradients, debug = self.build_single_graph_schedule(
                    id_gpu, name_gpu, tensors_input, tensors_input_asr)

                loss_step.append(loss)
                loss_ast_step.append(loss_ast)
                loss_ctc_step.append(loss_ctc)
                loss_bert_step.append(loss_bert)
                tower_grads.append(gradients)
                list_debug.append(debug)

        # mean the loss
        loss = tf.reduce_mean(loss_step)
        loss_ast = tf.reduce_mean(loss_ast_step)
        loss_ctc = tf.reduce_mean(loss_ctc_step)
        loss_bert = tf.reduce_mean(loss_bert_step)
        # merge gradients, update current model
        with tf.device(self.center_device):
            # computation relevant to gradient
            averaged_grads = average_gradients(tower_grads)
            handled_grads = handle_gradients(averaged_grads, self.args)

            # filter frozen variables:
            if self.args.frozen_scope:
                filtered_grads = []
                frozen_layers = self.args.frozen_scope.split(',')
                for grad, var in handled_grads:
                    flag = False
                    for layer in frozen_layers:
                        if layer in var.name:
                            flag = True
                    if not flag:
                        filtered_grads.append([grad, var])
                        print([grad, var])
            else:
                filtered_grads = handled_grads

            op_optimize = self.optimizer.apply_gradients(filtered_grads, self.global_step)

        self.__class__.num_Instances += 1
        logging.info("built {} {} instance(s).".format(self.__class__.num_Instances, self.__class__.__name__))
        self.summary_op = tf.summary.merge_all()
        self.loss_ast = loss_ast
        self.loss_ctc = loss_ctc
        self.loss_bert = loss_bert
        return loss, tensors_input.shape_batch, op_optimize, [x for x in zip(*list_debug)]

    def build_single_graph_schedule(self, id_gpu, name_gpu, tensors_input_ast, tensors_input_asr):
        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            encoder = self.gen_encoder(
                is_train=self.is_train,
                embed_table=None,
                args=self.args)
            encoder_mt = self.mt_encoder(
                is_train=self.is_train,
                embed_table=None,
                args=self.args,
                name='TPEncoder')
            decoder = self.gen_decoder(
                is_train=self.is_train,
                embed_table=self.embed_table_decoder,
                global_step=self.global_step,
                args=self.args)
            decoder_ctc = self.ctc_decoder(
                is_train=self.is_train,
                embed_table=self.src_embed_table,
                global_step=self.global_step,
                args=self.args)

            def func_ast():
                tensors_input = tensors_input_ast

            def func_asr():
                ensors_input = tensors_input_asr

            batch = []

            iter_flag = tf.equal(
                tf.mod(self.global_step,
                       tf.constant(2, dtype=tf.int64)),
                tf.constant(0, dtype=tf.int64))

            for i in range(len(self.batch)):
                batch_item = tf.cond(
                    iter_flag,
                    lambda: self.batch[i],
                    lambda: self.batch_asr[i]
                )
                batch.append(batch_item)

            tensors_input = self.build_tf_input_ctcbert_asr(batch)

            # Step0: front-end
            feature = tensors_input.feature_splits[id_gpu]
            if self.is_train and self.args.model.use_specAug:
                batch_size, length, feat_size = tf.shape(feature)[0], \
                                                tf.shape(feature)[1], \
                                                self.args.data.dim_input
                feat_size = int(feat_size / self.args.data.feature_channel)
                feature = tf.reshape(feature,
                                     [batch_size, length, feat_size, self.args.data.feature_channel])
                feature = apply_specaugmant(feature, 30, 40, 2, 2, 0.2)
                feature = tf.reshape(feature,
                                     [batch_size, length, feat_size * self.args.data.feature_channel])

            encoded_asr, len_encoded_asr = encoder(
                    features=feature,
                    len_feas=tensors_input.len_fea_splits[id_gpu])

            encoded, len_encoded = encoder_mt(
                features=encoded_asr,
                len_feas=len_encoded_asr
            )

            with tf.variable_scope(decoder.name or 'decoder'):
                decoder_input = decoder.build_input(
                    id_gpu=id_gpu,
                    tensors_input=tensors_input)
                
                if (not self.is_train) or (self.args.model.loss_type == 'OCD'):
                    # infer phrases
                    if self.args.dirs.lm_checkpoint and self.args.beam_size > 1:
                        logging.info('beam search with language model ...')
                        logits, preds, len_decoded = decoder.beam_decode_rerank(
                            encoded,
                            len_encoded)
                    else:
                        logging.info('gready search ...')
                        logits, preds, len_decoded = decoder.decoder_with_caching(
                            encoded,
                            len_encoded)
                    # asr predict
                    logits_asr, preds_asr, len_logits_asr = decoder_ctc(
                    encoded=encoded_asr,
                    len_encoded=len_encoded_asr)
                else:
                    # train phrase
                    logging.info('teacher-forcing training ...')

                    logits_asr, preds_asr, len_logits_asr = decoder_ctc(
                        encoded=encoded_asr,
                        len_encoded=len_encoded_asr)

                    # ast
                    decoder_input_labels = decoder_input.input_labels * tf.sequence_mask(
                        decoder_input.len_labels,
                        maxlen=tf.shape(decoder_input.input_labels)[1],
                        dtype=tf.int32)
                    logits, preds, _ = tf.cond(
                        iter_flag,
                        lambda: decoder.decode(
                            encoded=encoded,
                            len_encoded=len_encoded,
                            decoder_input=decoder_input_labels),
                        lambda: (tf.zeros_like(logits_asr, dtype=tf.float32), tf.zeros_like(preds_asr, dtype=tf.int32), len_logits_asr))

                    decoded_sparse = self.ctc_decode(logits_asr, len_logits_asr)
                    decoded = tf.sparse_to_dense(
                        sparse_indices=decoded_sparse.indices,
                        output_shape=decoded_sparse.dense_shape,
                        sparse_values=decoded_sparse.values,
                        default_value=0,
                        validate_indices=True)
                    distribution = tf.nn.softmax(logits)

            if self.is_train:
                if self.args.model.loss_type == 'OCD':
                    """
                    constrain the max decode length for ocd training since model
                    will decode to that long at beginning. Recommend 30.
                    """
                    loss, _ = self.ocd_loss(
                        logits=logits,
                        len_logits=len_decoded,
                        labels=tensors_input.label_splits[id_gpu],
                        preds=preds)

                elif self.args.model.loss_type == 'CE':
                    loss_ast = self.ce_loss(
                        logits=logits,
                        labels=decoder_input.output_labels,
                        len_labels=decoder_input.len_labels)
                    loss = loss_ast
                elif self.args.model.loss_type == 'Premium_CE':
                    table_targets_distributions = tf.nn.softmax(tf.constant(self.args.table_targets))
                    loss = self.premium_ce_loss(
                        logits=logits,
                        labels=tensors_input.label_splits[id_gpu],
                        table_targets_distributions=table_targets_distributions,
                        len_labels=tensors_input.len_label_splits[id_gpu])

                elif self.args.model.loss_type == 'kd_ce':
                    bert_feat = tensors_input.bert_feat_splits[id_gpu]
                    encoded_feat = extract_feat(encoded, bert_feat, self.args)

                    loss_bert = compute_kd_loss(bert_feat, encoded_feat, self.args.model.kd.loss_type)
                    loss_name = "loss_bert" + str(id_gpu)
                    self.summary_op = tf.summary.merge([tf.summary.scalar(loss_name, tf.reduce_mean(loss_bert))])
                    self.summary_loss_bert = tf.summary.scalar(loss_name, tf.reduce_mean(loss_bert))
                    loss_ast = self.ce_loss(
                        logits=logits,
                        labels=decoder_input.output_labels,
                        len_labels=decoder_input.len_labels)
                    loss = self.args.model.kd.kd_factor * loss_bert + (1 - self.args.model.kd.kd_factor) * loss_ast
                    loss_ctc = 0
                elif self.args.model.loss_type == 'ce_ctc':
                    loss_ctc = self.ctc_loss(
                        logits=logits_asr,
                        len_logits=len_logits_asr,
                        labels=tensors_input.aux_label_splits[id_gpu],
                        len_labels=tensors_input.len_aux_label_splits[id_gpu])

                    loss_name = "ctc_loss" + str(id_gpu)

                    loss_ast = tf.cond(
                        iter_flag,
                        lambda: self.ce_loss(
                            logits=logits,
                            labels=decoder_input.output_labels,
                            len_labels=decoder_input.len_labels),
                        lambda: tf.zeros_like(loss_ctc, dtype=tf.float32))

                    self.summary_op = tf.summary.merge([tf.summary.scalar(loss_name, tf.reduce_mean(loss_ctc))])
                    self.summary_loss_ctc = tf.summary.scalar(loss_name, tf.reduce_mean(loss_ctc))
                    loss_bert = 0
                    loss = self.args.model.decoder2.kd_factor * loss_ctc + \
                           (1 - self.args.model.decoder2.kd_factor) * loss_ast

                elif self.args.model.loss_type == 'ctc_kd_ce':

                    loss_ctc = self.ctc_loss(
                        logits=logits_asr,
                        len_logits=len_logits_asr,
                        labels=tensors_input.aux_label_splits[id_gpu],
                        len_labels=tensors_input.len_aux_label_splits[id_gpu])

                    loss_name_ctc = "ctc_loss" + str(id_gpu)

                    bert_feat = tensors_input.bert_feat_splits[id_gpu]

                    encoded_feat = extract_feat(encoded, bert_feat, self.args)
                    loss_bert = compute_kd_loss(bert_feat, encoded_feat, self.args.model.kd.loss_type)
                    loss_name_bert = "bert_loss" + str(id_gpu)

                    self.summary_op = tf.summary.merge([
                        tf.summary.scalar(loss_name_ctc, tf.reduce_mean(loss_ctc)),
                        tf.summary.scalar(loss_name_bert, tf.reduce_mean(loss_bert))])
                    self.summary_loss_ctc = tf.summary.scalar(loss_name_ctc, tf.reduce_mean(loss_ctc))
                    self.summary_loss_bert = tf.summary.scalar(loss_name_bert, tf.reduce_mean(loss_bert))

                    loss_ast = tf.cond(
                        iter_flag,
                        lambda: self.ce_loss(
                                        logits=logits,
                                        labels=decoder_input.output_labels,
                                        len_labels=decoder_input.len_labels),
                        lambda: tf.zeros_like(loss_ctc, dtype=tf.float32))

                    loss = self.args.model.decoder2.kd_factor * loss_ctc + \
                           self.args.model.kd.kd_factor * loss_bert + \
                           (1 - self.args.model.decoder2.kd_factor - self.args.model.kd.kd_factor) * loss_ast

                elif self.args.model.loss_type == 'ctc_kd-tok_ce':
                    loss_ast = self.ce_loss(
                        logits=logits,
                        labels=decoder_input.output_labels,
                        len_labels=decoder_input.len_labels)
                    loss_ctc = self.ctc_loss(
                        logits=logits_asr,
                        len_logits=len_logits_asr,
                        labels=tensors_input.aux_label_splits[id_gpu],
                        len_labels=tensors_input.len_aux_label_splits[id_gpu])

                    loss_name_ctc = "ctc_loss" + str(id_gpu)

                    bert_feat = tensors_input.bertfull_feat_splits[id_gpu]

                    encoded_feat = extract_feat(encoded, bert_feat, self.args)
                    loss_bert = compute_kd_loss(bert_feat, encoded_feat, self.args.model.kd.loss_type)
                    loss_name_bert = "bert_loss" + str(id_gpu)

                    self.summary_op = tf.summary.merge([
                        tf.summary.scalar(loss_name_ctc, tf.reduce_mean(loss_ctc)),
                        tf.summary.scalar(loss_name_bert, tf.reduce_mean(loss_bert))])
                    self.summary_loss_ctc = tf.summary.scalar(loss_name_ctc, tf.reduce_mean(loss_ctc))
                    self.summary_loss_bert = tf.summary.scalar(loss_name_bert, tf.reduce_mean(loss_bert))

                    loss = self.args.model.decoder2.kd_factor * loss_ctc + \
                           self.args.model.kd.kd_factor * loss_bert + \
                           (1 - self.args.model.decoder2.kd_factor - self.args.model.kd.kd_factor) * loss_ast

                elif self.args.model.loss_type == 'CTC_BERT':

                    loss_ctc = self.ctc_loss(
                        logits=logits_asr,
                        len_logits=len_logits_asr,
                        labels=tensors_input.aux_label_splits[id_gpu],
                        len_labels=tensors_input.len_aux_label_splits[id_gpu])

                    bert_feat = tensors_input.bert_feat_splits[id_gpu]

                    encoded_feat = extract_feat(encoded, bert_feat, self.args)
                    loss_bert = compute_kd_loss(bert_feat, encoded_feat, self.args.model.kd.loss_type)

                    loss_name_ctc = "ctc_loss" + str(id_gpu)
                    loss_name_bert = "bert_loss" + str(id_gpu)
                    self.summary_op = tf.summary.merge([
                        tf.summary.scalar(loss_name_ctc, tf.reduce_mean(loss_ctc)),
                        tf.summary.scalar(loss_name_bert, tf.reduce_mean(loss_bert))])
                    self.summary_loss_ctc = tf.summary.scalar(loss_name_ctc, tf.reduce_mean(loss_ctc))
                    self.summary_loss_bert = tf.summary.scalar(loss_name_bert, tf.reduce_mean(loss_bert))

                    loss = self.args.model.decoder2.kd_factor * loss_ctc + \
                           self.args.model.kd.kd_factor * loss_bert
                    loss_ast = 0

                else:
                    raise NotImplemented('NOT found loss type!')

                with tf.name_scope("gradients"):
                    assert loss.get_shape().ndims == 1
                    loss = tf.reduce_mean(loss)
                    gradients = self.optimizer.compute_gradients(loss)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))
        if self.is_train:
            # no_op is preserved for debug info to pass
            # return loss, gradients, tf.no_op()
            return loss, loss_ast, loss_ctc, loss_bert, gradients, [tf.no_op(), preds, tensors_input.label_splits[id_gpu]]
        elif self.args.model.loss_type == "ctc_kd_ce" or self.args.model.loss_type == "CTC_BERT":
            return logits, len_decoded, preds, logits_asr, preds_asr
        else:
            return logits, len_decoded, preds
