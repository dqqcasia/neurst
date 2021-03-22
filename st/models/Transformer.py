'''@file model.py
contains de Model class
During the training , using greadysearch decoder, there is loss.
During the dev/infer, using beamsearch decoder, there is no logits, therefor loss, only predsself.
because we cannot access label during the dev set and need to depend on last time decision.

so, there is only infer and training
'''

import tensorflow as tf
import logging
from collections import namedtuple

from st.models.seq2seqModel import Seq2SeqModel
from st.models.model_tools import choose_device, smoothing_cross_entropy
from st.models.kd_layer import compute_kd_loss, extract_feat
from st.tools.tftools.tfAudioTools import apply_specaugmant

class Transformer(Seq2SeqModel):
    '''a general class for an encoder decoder system
    '''

    def __init__(self, tensor_global_step, encoder, decoder, is_train, args, summary=None,
                 batch=None, embed_table_encoder=None, embed_table_decoder=None,
                 name='transformer'):
        '''Model constructor

        Args:
        '''
        self.args = args
        self.name = name
        self.global_step = tensor_global_step
        self.summary = summary
        self.size_embedding = args.model.decoder.size_embedding
        self.embed_table_decoder = self.get_embedding(
            embed_table=None,
            size_input=args.dim_output,
            size_embedding=self.size_embedding)

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

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):
        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            encoder = self.gen_encoder(
                is_train=self.is_train,
                embed_table=None,
                args=self.args)
            decoder = self.gen_decoder(
                is_train=self.is_train,
                embed_table=self.embed_table_decoder,
                global_step=self.global_step,
                args=self.args)

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

            if self.args.model.is_multilingual:

                # concat language embedding before encoder
                lang_id = tensors_input.lang_label_splits[id_gpu]

                lang_emd = tf.nn.embedding_lookup(self.lang_embedding, lang_id)

                feature = tf.concat([lang_emd, feature], 1)
                encoded, len_encoded = encoder(
                    features=feature,
                    len_feas=tensors_input.len_fea_splits[id_gpu])
            else:
                feature = tensors_input.feature_splits[id_gpu]
                encoded, len_encoded = encoder(
                    features=feature,
                    len_feas=tensors_input.len_fea_splits[id_gpu])

            if self.args.kd_preasr:
                with tf.variable_scope('asr_encoder'):
                    encoder_asr = self.gen_encoder(
                    is_train=self.is_train,
                    embed_table=None,
                    args=self.args)
                    encoded_asr, len_encoded_asr = encoder_asr(
                        features=tensors_input.feature_splits[id_gpu],
                        len_feas=tensors_input.len_fea_splits[id_gpu])

            with tf.variable_scope(decoder.name or 'decoder'):
                decoder_input = decoder.build_input(
                    id_gpu=id_gpu,
                    tensors_input=tensors_input)
                
                if (not self.is_train) or (self.args.model.loss_type == 'OCD'):
                    # infer phrases
                    if self.args.beam_size > 1:
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
                    logging.info('teacher-forcing training ...')
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
                    loss = self.ce_loss(
                        logits=logits,
                        labels=decoder_input.output_labels,
                        len_labels=decoder_input.len_labels)

                elif self.args.model.loss_type == 'Premium_CE':
                    table_targets_distributions = tf.nn.softmax(tf.constant(self.args.table_targets))
                    loss = self.premium_ce_loss(
                        logits=logits,
                        labels=tensors_input.label_splits[id_gpu],
                        table_targets_distributions=table_targets_distributions,
                        len_labels=tensors_input.len_label_splits[id_gpu])

                elif self.args.model.loss_type == 'CE_KD':
                    bert_feat = tensors_input.bert_feat_splits[id_gpu]
                    bert_feat = tf.expand_dims(bert_feat, axis=1)
                    encoded_feat = extract_feat(encoded, bert_feat, self.args)

                    kd_loss = compute_kd_loss(bert_feat, encoded_feat, self.args.model.kd.loss_type)
                    loss_name = "kd_loss" + str(id_gpu)
                    tf.summary.scalar(loss_name, kd_loss)
                    loss_ce = self.ce_loss(
                        logits=logits,
                        labels=decoder_input.output_labels,
                        len_labels=decoder_input.len_labels)
                    loss = self.args.model.kd.kd_factor * kd_loss + (1 - self.args.model.kd.kd_factor) * loss_ce

                elif self.args.model.loss_type == 'CE_KD_ASR':
                    loss_asr = self.ce_loss(
                        logits=logits,
                        labels=decoder_input.output_labels,
                        len_labels=decoder_input.len_labels)
                    loss_kd = 0

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
            return loss, gradients, [tf.no_op(), preds, tensors_input.label_splits[id_gpu]]
        else:
            return logits, len_decoded, preds


