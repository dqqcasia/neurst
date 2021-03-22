'''@file model.py
contains de Model class
During the training , using greadysearch decoder, there is loss.
During the dev/infer, using beamsearch decoder, there is no logits, therefor loss, only preds self.
because we cannot access label during the dev set and need to depend on last time decision.

so, there is only infer and training
'''

"""Doublely supervised encoder for ST. (finished)"""


import tensorflow as tf
import logging
from collections import namedtuple

from st.models.seq2seqModel import Seq2SeqModel
from st.models.model_tools import choose_device, smoothing_cross_entropy
from st.models.kd_layer import compute_kd_loss, extract_feat
from st.tools.tftools.tfTools import dense_sequence_to_sparse
from st.tools.tftools.tfAudioTools import apply_specaugmant
from st.tools.tftools.gradientTools import average_gradients, handle_gradients
import sys
from st.models.tools.blocks import shrink_layer


class TransformerTPE(Seq2SeqModel):
    '''a general class for an encoder-decoder system with a two-pass encoder.
    '''

    def __init__(self, tensor_global_step, encoder, decoder, is_train, args, summary=None,
                 batch=None, embed_table_encoder=None, embed_table_decoder=None,
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
        if self.args.dirs.src_vocab:
            self.src_embed_table = self.get_embedding(
                embed_table=None,
                size_input=args.src_dim_output,
                size_embedding=self.size_embedding,
                name='embedding_src')
        else:
            self.src_embed_table = self.embed_table_decoder

        self.ctc_merge_repeated = args.model.decoder2.ctc_merge_repeated

        if args.model.is_multilingual or args.model.is_cotrain:
            self.lang_embedding = self.get_embedding(
                embed_table=None,
                size_input=args.data.lang_num,
                size_embedding=args.data.dim_raw_input * 6,
                name="lang_embedding"
            )

        super().__init__(tensor_global_step, encoder, decoder, is_train, args,
                         batch,
                         embed_table_encoder=None,
                         embed_table_decoder=None,
                         name=name)

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
        # create input tensors in the cpu
        tensors_input = self.build_input()
        # create optimizer
        self.optimizer = self.build_optimizer()
        # if 'horovod' in sys.modules:
        if self.args.use_horovod:
            import horovod.tensorflow as hvd
            # hvd.init()
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
                loss, loss_ast, loss_ctc, loss_bert, gradients, debug = self.build_single_graph(
                    id_gpu, name_gpu, tensors_input)

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
        self.loss_am = loss_ctc
        self.loss_bert = loss_bert
        return loss, tensors_input.shape_batch, op_optimize, [x for x in zip(*list_debug)]

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):
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
            # Step1: acoustic encoder
            if self.args.model.encoder_concat_tag:

                # concat language embedding before encoder
                lang_id = tensors_input.lang_label_splits[id_gpu]

                lang_emd = tf.nn.embedding_lookup(self.lang_embedding, lang_id)
                feature = tf.concat([lang_emd, feature], 1)
                encoded_am, len_encoded_am = encoder(
                    features=feature,
                    len_feas=tensors_input.len_fea_splits[id_gpu])
            else:
                encoded_am, len_encoded_am = encoder(
                    features=feature,
                    len_feas=tensors_input.len_fea_splits[id_gpu])

            # Step2: acoustic decoder
            with tf.variable_scope(decoder.name or 'decoder'):
                logits_am, preds_ad, len_logits_am = decoder_ctc(
                    encoded=encoded_am,
                    len_encoded=len_encoded_am)

                decoded_sparse = self.ctc_decode(logits_am, len_logits_am)
                decoded = tf.sparse_to_dense(
                    sparse_indices=decoded_sparse.indices,
                    output_shape=decoded_sparse.dense_shape,
                    sparse_values=decoded_sparse.values,
                    default_value=0,
                    validate_indices=True)
                distribution = tf.nn.softmax(logits_am)
                preds_am = decoded


            if self.args.model.shrink_layer:
                # shrink layer
                encoded_shrunk, len_encoded_shrunk = shrink_layer(
                    encoded_am, len_encoded_am, logits_am,
                    encoded_am.get_shape()[-1])

                encoded_am, len_encoded_am = encoded_shrunk, len_encoded_shrunk

            # Step3: Semantic Encoder
            if self.args.model.use_semantic_encoder:
                if self.args.model.semantic_encoder_pretrain:

                    source_input = tf.nn.embedding_lookup(self.src_embed_table, tensors_input.aux_label_splits[id_gpu])
                    encoded, len_encoded = encoder_mt(
                        features=source_input,
                        len_feas=tensors_input.len_aux_label_splits[id_gpu])
                else:
                    encoded, len_encoded = encoder_mt(
                        features=encoded_am,
                        len_feas=len_encoded_am)

            else:
                encoded, len_encoded = encoded_am, len_encoded_am

            if self.args.model.variational_inference:
                # variational inference
                # reparameter
                z_mu = tf.layers.dense(encoded, self.args.model.latent_size)
                z_logvar = tf.layers.dense(encoded, self.args.model.latent_size)
                # sample
                eps = tf.random_normal(shape=tf.shape(z_mu))
                z_sample = z_mu + tf.exp(z_logvar / 2) * eps

                # sample encoder
                encoded_z = tf.layers.dense(z_sample, self.args.model.latent_size)
                encoded = tf.layers.dense(tf.concat([encoded, encoded_z], -1), self.args.model.latent_size)

            # Step4: Semantic Decoder
            with tf.variable_scope(decoder.name or 'decoder'):
                if self.args.model.decoder_concat_tag:
                    decoder_input = decoder.build_input_withtag(
                        id_gpu=id_gpu,
                        tensors_input=tensors_input)
                elif self.args.model.pt_decoder:
                    decoder_input = decoder.build_input_ptdec(
                        id_gpu=id_gpu,
                        tensors_input=tensors_input)
                else:
                    decoder_input = decoder.build_input(
                        id_gpu=id_gpu,
                        tensors_input=tensors_input)

                if (not self.is_train) or (self.args.model.loss_type == 'OCD'):
                    # infer phrases
                    if self.args.dirs.lm_checkpoint or self.args.beam_size > 1:
                        logging.info('beam search with language model ...')
                        logits, preds, len_decoded = decoder.beam_decode_rerank(
                            encoded,
                            len_encoded)
                    else:
                        logging.info('gready search ...')
                        logits, preds, len_decoded = decoder.decoder_with_caching(
                            encoded,
                            len_encoded)
                else:
                    # train phrases
                    logging.info('teacher-forcing training ...')
                    if self.args.model.pt_decoder:
                        # ignore encoded during pre-training decoder
                        encoded = tf.cast(tf.tile(tf.expand_dims(tf.ones_like(tensors_input.feature_splits[id_gpu][:, :, 1]), axis=-1),
                                                  [1, 1, self.args.model.encoder2.num_cell_units]), dtype=tf.float32)

                    # ast
                    decoder_input_labels = decoder_input.input_labels * tf.sequence_mask(
                        decoder_input.len_labels,
                        maxlen=tf.shape(decoder_input.input_labels)[1],
                        dtype=tf.int32)

                    logits, preds, _ = decoder.decode(
                        encoded=encoded,
                        len_encoded=len_encoded,
                        decoder_input=decoder_input_labels)

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
                    loss_ctc, loss_bert = 0, 0
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
                    loss_ast = self.ce_loss(
                        logits=logits,
                        labels=decoder_input.output_labels,
                        len_labels=decoder_input.len_labels)
                    loss_ctc = self.ctc_loss(
                        logits=logits_am,
                        len_logits=len_logits_am,
                        labels=tensors_input.aux_label_splits[id_gpu],
                        len_labels=tensors_input.len_aux_label_splits[id_gpu])

                    loss = self.args.model.decoder2.kd_factor * loss_ctc + \
                           (1 - self.args.model.decoder2.kd_factor) * loss_ast
                    loss_bert = 0

                elif self.args.model.loss_type == 'ctc_kd_ce':
                    loss_ast = self.ce_loss(
                        logits=logits,
                        labels=decoder_input.output_labels,
                        len_labels=decoder_input.len_labels)
                    loss_ctc = self.ctc_loss(
                        logits=logits_am,
                        len_logits=len_logits_am,
                        labels=tensors_input.aux_label_splits[id_gpu],
                        len_labels=tensors_input.len_aux_label_splits[id_gpu])

                    bert_feat = tensors_input.bert_feat_splits[id_gpu]

                    encoded_feat = extract_feat(encoded, bert_feat, self.args)
                    loss_bert = compute_kd_loss(bert_feat, encoded_feat, self.args.model.kd.loss_type)

                    loss = self.args.model.decoder2.kd_factor * loss_ctc + \
                           self.args.model.kd.kd_factor * loss_bert + \
                           (1 - self.args.model.decoder2.kd_factor - self.args.model.kd.kd_factor) * loss_ast

                elif self.args.model.loss_type == 'ctc_kd-tok_ce':
                    loss_ast = self.ce_loss(
                        logits=logits,
                        labels=decoder_input.output_labels,
                        len_labels=decoder_input.len_labels)
                    loss_ctc = self.ctc_loss(
                        logits=logits_am,
                        len_logits=len_logits_am,
                        labels=tensors_input.aux_label_splits[id_gpu],
                        len_labels=tensors_input.len_aux_label_splits[id_gpu])

                    bert_feat = tensors_input.bertfull_feat_splits[id_gpu]

                    encoded_feat = extract_feat(encoded, bert_feat, self.args)
                    loss_bert = compute_kd_loss(bert_feat, encoded_feat, self.args.model.kd.loss_type)

                    loss = self.args.model.decoder2.kd_factor * loss_ctc + \
                           self.args.model.kd.kd_factor * loss_bert + \
                           (1 - self.args.model.decoder2.kd_factor - self.args.model.kd.kd_factor) * loss_ast

                elif self.args.model.loss_type == 'CTC_BERT':

                    loss_ctc = self.ctc_loss(
                        logits=logits_am,
                        len_logits=len_logits_am,
                        labels=tensors_input.aux_label_splits[id_gpu],
                        len_labels=tensors_input.len_aux_label_splits[id_gpu])

                    bert_feat = tensors_input.bert_feat_splits[id_gpu]

                    encoded_feat = extract_feat(encoded, bert_feat, self.args)
                    loss_bert = compute_kd_loss(bert_feat, encoded_feat, self.args.model.kd.loss_type)

                    loss = self.args.model.decoder2.kd_factor * loss_ctc + \
                           self.args.model.kd.kd_factor * loss_bert
                    loss_ast = 0

                elif self.args.model.loss_type == 'CTC':
                    loss_ctc = self.ctc_loss(
                        logits=logits_am,
                        len_logits=len_logits_am,
                        labels=tensors_input.aux_label_splits[id_gpu],
                        len_labels=tensors_input.len_aux_label_splits[id_gpu])
                    loss_bert, loss_ast = 0, 0
                    loss = loss_ctc

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
        elif self.args.model.loss_type == "ctc_kd_ce" or self.args.model.loss_type == "CTC_BERT" or self.args.model.loss_type == "ce_ctc":
            return logits, len_decoded, preds, logits_am, preds_am
        else:
            return logits, len_decoded, preds

    def build_infer_graph(self):
        if self.args.mode == "infer":
            tensors_input = self.build_input()
        else:
            if self.args.model.is_multilingual or self.args.model.is_cotrain:
                tensors_input = self.build_infer_input_multilingual()
            elif self.args.model.use_multilabel:
                tensors_input = self.build_infer_input_multilabel()
            else:
                tensors_input = self.build_infer_input()
        with tf.variable_scope(self.name, reuse=bool(self.__class__.num_Model)):
            if self.args.model.loss_type == "ctc_kd_ce" or self.args.model.loss_type == "CTC_BERT" or self.args.model.loss_type == "ce_ctc":
                logits, len_logits, preds, logits_am, preds_am = self.build_single_graph(
                    id_gpu=0,
                    name_gpu=self.list_gpu_devices[0],
                    tensors_input=tensors_input)
            else:
                logits, len_logits, preds = self.build_single_graph(
                    id_gpu=0,
                    name_gpu=self.list_gpu_devices[0],
                    tensors_input=tensors_input)
        if preds.get_shape().ndims == 3:
            preds = preds[:, :, 0]
        self.summary_op = tf.summary.merge_all()
        if self.args.model.loss_type == "ctc_kd_ce" or self.args.model.loss_type == "CTC_BERT" or self.args.model.loss_type == "ce_ctc":
            return preds, preds_am, tensors_input.shape_batch, tf.no_op()
        else:
            return preds, tensors_input.shape_batch, tf.no_op()

