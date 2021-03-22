#!/usr/bin/env python
from datetime import datetime
from time import time
import os
import logging
import tensorflow as tf
from st.layers.arguments import args
from tqdm import tqdm
import numpy as np
import editdistance as ed

from st.tools.utils import get_session
from st.tools.utils import create_embedding, size_variables

from st.tools.tftools.tfRecord import TFReader, readTFRecord, readTFRecord_multilabel, readTFRecord_kdbert, readTFRecord_ctcbert, \
    readTFRecord_ctcbertfull, readTFRecord_multilingual, readTFRecord_multilingual_multilabel
from st.layers.data_helper import ASRDataLoader, ASRMultiLabelDataLoader, ASRKdBertDataLoader, ASRCtcBertDataLoader, \
    ASRCtcBertFullDataLoader, ASRMultiLingualDataLoader, ASRMultiLingualMultilabelDataLoader
from st.tools.tftools.checkpointTools import list_variables

from st.tools.summaryTools import Summary
from st.tools.performanceTools import decode_test
from st.tools.textTools import array_idx2char, array2text
from st.tools.tools import check_to_stop
import json
from tensorflow.contrib import slim
import subprocess
from st.tools.wer import calc_wer, calc_per


def train():

    logging.info('reading data from %s'.format(args.dirs.train.tfdata))

    def prepare_train(args):
        if args.dirs.train.tfdata_list:
            dataReader_train = TFReader(args.dirs.train.tfdata_list, args=args)
        else:
            dataReader_train = TFReader(args.dirs.train.tfdata, args=args)
        if args.model.use_multilabel:
            batch_train = dataReader_train.fentch_multi_label_batch_bucket()

            id_dev, feat_dev, label_dev, auxlabel_dev = readTFRecord_multilabel(args.dirs.dev.tfdata, args, _shuffle=False, transform=True)
            dataloader_dev = ASRMultiLabelDataLoader(args.dataset_dev, args, feat_dev, label_dev, auxlabel_dev, batch_size=args.batch_size,
                                                     num_loops=1)

            id_test, feat_test, label_test, auxlabel_test = readTFRecord_multilabel(args.dirs.test.tfdata, args, _shuffle=False, transform=True)
            dataloader_test = ASRMultiLabelDataLoader(args.dataset_test, args, feat_test, label_test, auxlabel_test, batch_size=args.batch_size,
                                                      num_loops=1)

        elif args.model.use_bert:
            batch_train = dataReader_train.fentch_kdbert_batch_bucket()

            id_dev, feat_dev, label_dev, bertfeat_dev = readTFRecord_kdbert(args.dirs.dev.tfdata, args, _shuffle=False, transform=True)
            dataloader_dev = ASRKdBertDataLoader(args.dataset_dev, args, feat_dev, label_dev, bertfeat_dev, batch_size=args.batch_size, num_loops=1)

            id_test, feat_test, label_test, bertfeat_test = readTFRecord_kdbert(args.dirs.test.tfdata, args, _shuffle=False, transform=True)
            dataloader_test = ASRKdBertDataLoader(args.dataset_test, args, feat_test, label_test, bertfeat_test, batch_size=args.batch_size,
                                                  num_loops=1)

        elif args.model.use_ctc_bert:
            batch_train = dataReader_train.fentch_ctcbert_batch_bucket()

            id_dev, feat_dev, label_dev, auxlabel_dev, bertfeat_dev = readTFRecord_ctcbert(args.dirs.dev.tfdata, args, _shuffle=False, transform=True)
            dataloader_dev = ASRCtcBertDataLoader(args.dataset_dev, args, feat_dev, label_dev, bertfeat_dev, batch_size=args.batch_size, num_loops=1)

            id_test, feat_test, label_test, auxlabel_test, bertfeat_test = readTFRecord_ctcbert(args.dirs.test.tfdata, args, _shuffle=False, transform=True)
            dataloader_test = ASRCtcBertDataLoader(args.dataset_test, args, feat_test, label_test, bertfeat_test, batch_size=args.batch_size,
                                                   num_loops=1)

        elif args.model.use_ctc_bertfull:
            batch_train = dataReader_train.fentch_ctcbertfull_batch_bucket()

            id_dev, feat_dev, label_dev, auxlabel_dev, bertfullfeat_dev = readTFRecord_ctcbertfull(args.dirs.dev.tfdata, args, _shuffle=False, transform=True)
            dataloader_dev = ASRCtcBertFullDataLoader(args.dataset_dev, args, feat_dev, label_dev, bertfullfeat_dev, batch_size=args.batch_size,
                                                      num_loops=1)

            id_test, feat_test, label_test, auxlabel_test, bertfullfeat_test = readTFRecord_ctcbertfull(args.dirs.test.tfdata, args, _shuffle=False,
                                                                                               transform=True)
            dataloader_test = ASRCtcBertFullDataLoader(args.dataset_test, args, feat_test, label_test, bertfullfeat_test, batch_size=args.batch_size,
                                                       num_loops=1)

        elif args.model.is_multilingual:
            batch_train = dataReader_train.fentch_multilingual_batch_bucket()

            id_dev, feat_dev, label_dev, auxlabel_dev = readTFRecord_multilingual(args.dirs.dev.tfdata, args, _shuffle=False, transform=True)
            dataloader_dev = ASRMultiLingualDataLoader(args.dataset_dev, args, feat_dev, label_dev, auxlabel_dev, batch_size=args.batch_size,
                                                       num_loops=1)

            id_test, feat_test, label_test, auxlabel_test = readTFRecord_multilingual(args.dirs.test.tfdata, args, _shuffle=False, transform=True)
            dataloader_test = ASRMultiLingualDataLoader(args.dataset_test, args, feat_test, label_test, auxlabel_test, batch_size=args.batch_size,
                                                        num_loops=1)
        elif args.model.is_cotrain:
            batch_train = dataReader_train.fentch_multilingual_multilabel_batch_bucket()

            id_dev, feat_dev, label_dev, label_dev_2, auxlabel_dev = readTFRecord_multilingual_multilabel(args.dirs.dev.tfdata, args, _shuffle=False, transform=True)
            dataloader_dev = ASRMultiLingualMultilabelDataLoader(args.dataset_dev, args, feat_dev, label_dev, label_dev_2, auxlabel_dev, batch_size=args.batch_size, num_loops=1)

            id_test, feat_test, label_test, label_test_2, auxlabel_test = readTFRecord_multilingual_multilabel(args.dirs.test.tfdata, args, _shuffle=False, transform=True)
            dataloader_test = ASRMultiLingualMultilabelDataLoader(args.dataset_test, args, feat_test, label_test, label_test_2, auxlabel_test, batch_size=args.batch_size, num_loops=1)

        else:
            batch_train = dataReader_train.fentch_batch_bucket()

            id_dev, feat_dev, label_dev = readTFRecord(args.dirs.dev.tfdata, args, _shuffle=False, transform=True)
            dataloader_dev = ASRDataLoader(args.dataset_dev, args, feat_dev, label_dev, batch_size=args.batch_size, num_loops=1)

            id_test, feat_test, label_test = readTFRecord(args.dirs.test.tfdata, args, _shuffle=False, transform=True)
            dataloader_test = ASRDataLoader(args.dataset_test, args, feat_test, label_test, batch_size=args.batch_size, num_loops=1)

        return batch_train, dataReader_train, dataloader_dev, dataloader_test
    batch_train, dataReader_train, dataloader_dev, dataloader_test = prepare_train(args)

    tensor_global_step = tf.train.get_or_create_global_step()

    summary = Summary(str(args.dir_log))

    model = args.Model(
        tensor_global_step,
        encoder=args.model.encoder.type,
        decoder=args.model.decoder.type,
        batch=batch_train,
        is_train=True,
        args=args)

    model_infer = args.Model(
        tensor_global_step,
        encoder=args.model.encoder.type,
        decoder=args.model.decoder.type,
        is_train=False,
        args=args)

    size_variables()
    start_time = datetime.now()

    checker = check_to_stop()

    saver = tf.train.Saver(max_to_keep=15, save_relative_paths=True)
    saver_init = tf.train.Saver(slim.get_variables_to_restore(exclude=['global_step']))

    if args.dirs.lm_checkpoint:
        list_lm_vars_pretrained = list_variables(args.dirs.lm_checkpoint)
        list_lm_vars = model.decoder.lm.variables

        dict_lm_vars = {}
        for var in list_lm_vars:
            if 'embedding' in var.name:
                for var_pre in list_lm_vars_pretrained:
                    if 'embedding' in var_pre[0]:
                        break
            else:
                name = var.name.split(model.decoder.lm.name)[1].split(':')[0]
                for var_pre in list_lm_vars_pretrained:
                    if name in var_pre[0]:
                        break
            dict_lm_vars[var_pre[0]] = var

        saver_lm = tf.train.Saver(dict_lm_vars)

    if args.dirs.checkpoint_pretrain:
        pretrain_vars = list_variables(args.dirs.checkpoint_pretrain)

        if args.pretrain_scope:
            pretrain_layers = args.pretrain_scope.split(',')
            pretrain_vars_list = []
            for var_pre in pretrain_vars:
                for layer in pretrain_layers:
                    if var_pre[0].startswith(layer):
                        pretrain_vars_list.append(var_pre[0])
        else:
            pretrain_vars_list = [var_pre[0] for var_pre in pretrain_vars]

        vars_list = tf.all_variables()
        restored_list = []
        for var in vars_list:
            if var.name.split(':')[0] in pretrain_vars_list and var.name.split(':')[0] != 'global_step':
                restored_list.append(var)

        saver_pretrain = tf.train.Saver(restored_list)

    if args.dirs.pretrain_enc and args.dirs.pretrain_enc.checkpoint:
        pretrain_vars = list_variables(args.dirs.pretrain_enc.checkpoint)
        if args.dirs.pretrain_enc.scope:
            pretrain_layers = args.dirs.pretrain_enc.scope.split(',')
            pretrain_vars_enc_list = []
            for var_pre in pretrain_vars:
                for layer in pretrain_layers:
                    if var_pre[0].startswith(layer):
                        pretrain_vars_enc_list.append(var_pre[0])
        else:
            pretrain_vars_enc_list = [var_pre[0] for var_pre in pretrain_vars]

        vars_list = tf.all_variables()
        restored_list = []
        for var in vars_list:
            if var.name.split(':')[0] in pretrain_vars_enc_list and var.name.split(':')[0] != 'global_step':
                restored_list.append(var)

        saver_pretrain_enc = tf.train.Saver(restored_list)

    if args.dirs.pretrain_dec and args.dirs.pretrain_dec.checkpoint:
        pretrain_vars = list_variables(args.dirs.pretrain_dec.checkpoint)
        if args.dirs.pretrain_dec.scope:
            pretrain_layers = args.dirs.pretrain_dec.scope.split(',')
            pretrain_vars_dec_list = []
            for var_pre in pretrain_vars:
                for layer in pretrain_layers:
                    if var_pre[0].startswith(layer):
                        pretrain_vars_dec_list.append(var_pre[0])
        else:
            pretrain_vars_dec_list = [var_pre[0] for var_pre in pretrain_vars]

        vars_list = tf.all_variables()
        restored_list = []
        for var in vars_list:
            if var.name.split(':')[0] in pretrain_vars_dec_list and var.name.split(':')[0] != 'global_step':
                restored_list.append(var)

        saver_pretrain_dec = tf.train.Saver(restored_list)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.log_device_placement = False

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    with tf.train.MonitoredTrainingSession(config=config) as sess:
        summary.writer.add_graph(sess.graph)
        if args.dirs.checkpoint_init:
            saver_init.restore(sess, args.dirs.checkpoint_init)

        elif args.dirs.lm_checkpoint:
            lm_checkpoint = tf.train.latest_checkpoint(args.dirs.lm_checkpoint)
            saver_lm.restore(sess, lm_checkpoint)

        elif args.dirs.checkpoint_pretrain:
            saver_pretrain.restore(sess, args.dirs.checkpoint_pretrain)

        elif args.dirs.checkpoint_latest:
            saver.restore(sess, args.dirs.checkpoint_latest)

        elif args.dirs.pretrain_enc or args.dirs.pretrain_dec:
            if args.dirs.pretrain_enc.checkpoint:
                saver_pretrain_enc.restore(sess, args.dirs.pretrain_enc.checkpoint)
            if args.dirs.pretrain_dec.checkpoint:
                saver_pretrain_dec.restore(sess, args.dirs.pretrain_dec.checkpoint)

        elif tf.train.latest_checkpoint(args.dir_checkpoint):
            saver.restore(sess, tf.train.latest_checkpoint(args.dir_checkpoint))

        else:
            logger.info('Nothing to be reload from disk.')

        dataloader_dev.sess = sess
        dataloader_test.sess = sess

        num_processed = 0
        progress = 0
        global_wer = float('inf')
        global_bleu = float('-inf')

        bad_count = 0

        def evaluate_decode(args, model_infer, sess, global_step, dataloader_dev):

            def dev_test(args, dataloader, model, sess):
                start_time, batch_time = time(), time()
                processed = 0

                with tf.io.gfile.GFile('dev_wav.txt', 'w') as fw, \
                        tf.io.gfile.GFile('dev_ast_res.txt', 'w') as fastres, \
                        tf.io.gfile.GFile('dev_ast_ref.txt', 'w') as fastref, \
                        tf.io.gfile.GFile('dev_asr_res.txt', 'w') as fasrres, \
                        tf.io.gfile.GFile('dev_asr_ref.txt', 'w') as fasrref:
                    for batch in dataloader:
                        if not batch: continue
                        if args.model.use_multilabel:
                            dict_feed = {model.list_pl[0]: batch[0], model.list_pl[1]: batch[3]}
                        elif args.model.use_bert:
                            dict_feed = {model.list_pl[0]: batch[0], model.list_pl[1]: batch[3]}
                        elif args.model.use_ctc_bert:
                            dict_feed = {model.list_pl[0]: batch[0], model.list_pl[1]: batch[4]}
                        elif args.model.is_multilingual:
                            dict_feed = {model.list_pl[0]: batch[0], model.list_pl[1]: batch[3], model.list_pl[2]: batch[2]}
                        elif args.model.is_cotrain:
                            dict_feed = {model.list_pl[0]: batch[0], model.list_pl[1]: batch[4], model.list_pl[2]: batch[3]}
                        else:
                            dict_feed = {model.list_pl[0]: batch[0], model.list_pl[1]: batch[2]}

                        if args.model.loss_type == "ctc_kd_ce" or args.model.loss_type == "CTC_BERT" or args.model.loss_type == "ce_ctc":
                            decoded, decoded_asr, shape_batch, _ = sess.run(model.list_run, feed_dict=dict_feed)
                        else:
                            decoded, shape_batch, _ = sess.run(model.list_run, feed_dict=dict_feed)

                        used_time = time() - batch_time
                        batch_time = time()
                        processed += shape_batch[0]
                        progress = processed / len(dataloader)
                        logging.info('batch: {}\t time:{:.2f}s {:.3f}%'.format(shape_batch, used_time, progress * 100.0))

                        def split_res(res_txt, ref_txt):
                            try:
                                ast_res, asr_res = res_txt.split('<AST>')[-1], res_txt.split('<AST>')[-2].split('<ASR>')[-1]
                                ast_ref, asr_ref = ref_txt.split('<AST>')[-1], ref_txt.split('<AST>')[-2].split('<ASR>')[-1]
                            except:
                                res_tokens = res_txt.split()
                                ast_res, asr_res = ' '.join(res_tokens[int(0.5 * len(res_tokens)):]), ' '.join(
                                    res_tokens[0: int(0.5 * len(res_tokens))])
                                ast_ref, asr_ref = ref_txt.split('<AST>')[-1], ref_txt.split('<AST>')[-2].split('<ASR>')[-1]
                            return ast_res, asr_res, ast_ref, asr_ref

                        for lres, lref in zip(decoded, batch[1]):
                            tokens_res = array2text(lres, args.data.unit, args.idx2token, min_idx=0, max_idx=args.dim_output - 1, args=args)
                            tokens_fef = array2text(lref, args.data.unit, args.idx2token, min_idx=0, max_idx=args.dim_output - 1, args=args)
                            ast_res, asr_res, ast_ref, asr_ref = split_res(tokens_res, tokens_fef)
                            fastres.write(''.join(ast_res).replace("<eos>", '') + '\n')
                            fastref.write(''.join(ast_ref).replace("<eos>", '') + '\n')
                            fasrres.write(''.join(asr_res).replace("<eos>", '') + '\n')
                            fasrref.write(''.join(asr_ref).replace("<eos>", '') + '\n')

                used_time = time() - start_time

                cer = 0

                # calculate wer
                wer = calc_wer('dev_asr_ref.txt', 'dev_asr_res.txt')

                if args.bleu_cmd:
                    cmd = args.bleu_cmd
                else:
                    cmd = "perl st/tools/multi-bleu.perl {ref} " \
                          "< {output} 2>/dev/null | awk '{{print($3)}}' | awk -F, '{{print $1}}'"
                cmd = cmd.strip()
                bleu = subprocess.getoutput(cmd.format(**{'ref': 'dev_ast_ref.txt',
                                                          'output': 'dev_ast_res.txt'}))

                try:
                    bleu = float(bleu)
                except:
                    bleu = 0

                logging.info(
                    '=====dev info, total used time {:.2f}h==== \n'
                    'CER: {:.4f}\n'
                    'WER: {:.4f}\n'
                    'BLEU: {:.4f}'.format(used_time / 3600, cer, wer, bleu))
                return cer, wer, bleu

            dev_cer, dev_wer, dev_bleu = dev_test(
                args=args,
                dataloader=dataloader_dev,
                model=model_infer,
                sess=sess)

            summary.summary_scalar('dev_cer', dev_cer, global_step)
            summary.summary_scalar('dev_wer', dev_wer, global_step)
            summary.summary_scalar('dev_bleu', dev_bleu, global_step)
            return dev_cer, dev_wer, dev_bleu

        def check_save(dev_bleu, dev_wer, global_bleu, global_wer, bad_count):
            if args.metrics and args.metrics == "BLEU":
                if dev_bleu >= global_bleu:
                    global_bleu = dev_bleu
                    model_path = args.dir_checkpoint + '/model'
                    saver.save(
                        get_session(sess),
                        model_path,
                        global_step=global_step,
                        write_meta_graph=True)
                    logging.info('Save model in %s_%s.' % (model_path, str(global_step)))
                else:
                    logging.info('Model not saved, best bleu is %s.' % str(global_bleu))
                    bad_count += 1

            else:
                if dev_wer <= global_wer:
                    global_wer = dev_wer
                    model_path = args.dir_checkpoint + '/model'
                    saver.save(
                        get_session(sess),
                        model_path,
                        global_step=global_step,
                        write_meta_graph=True)
                    logging.info('Save model in %s_%s.' % (model_path, str(global_step)))
                else:
                    logging.info('Model not saved, best wer is %s.' % str(global_wer))
                    bad_count += 1
            return global_bleu, global_wer, bad_count

        while progress < args.num_epochs:
            batch_time = time()
            global_step, lr = sess.run([tensor_global_step, model.learning_rate])
            if args.model.use_multilabel:
                [loss, shape_batch, _, _], loss_ast, loss_asr = sess.run([model.list_run,
                                                                     model.loss_ast,
                                                                     model.loss_am])
                loss_bert = 0
                loss_ctc = loss_asr
                summary.summary_scalar('loss_asr', loss_asr, global_step)
                summary.summary_scalar('loss_ast', loss_ast, global_step)
            elif args.model.use_bert:
                [loss, shape_batch, _, _] = sess.run(model.list_run)
                loss_ast, loss_ctc, loss_bert = loss, 0, 0
                summary.summary_scalar('loss_bert', loss_bert, global_step)
            elif args.model.use_ctc_bert:
                [loss, shape_batch, _, _], loss_ast, loss_ctc, loss_bert = sess.run([model.list_run,
                                                                     model.loss_ast,
                                                                     model.loss_am,
                                                                     model.loss_bert])
                summary.summary_scalar('loss_ctc', loss_ctc, global_step)
                summary.summary_scalar('loss_bert', loss_bert, global_step)
            elif args.model.use_ctc_bertfull:
                [loss, shape_batch, _, _], loss_ast, loss_ctc, loss_bert = sess.run([model.list_run,
                                                                                     model.loss_ast,
                                                                                     model.loss_am,
                                                                                     model.loss_bert])
                summary.summary_scalar('loss_ctc', loss_ctc, global_step)
                summary.summary_scalar('loss_bert', loss_bert, global_step)
            elif args.model.is_cotrain:
                [loss, shape_batch, _, _], loss_ast, loss_ctc = sess.run([model.list_run,
                                                                     model.loss_ast,
                                                                     model.loss_am])
                summary.summary_scalar('loss_ctc', loss_ctc, global_step)
                loss_bert = 0
            else:
                loss, shape_batch, _, _ = sess.run(model.list_run)
                loss_ast, loss_ctc, loss_bert = loss, 0, 0
            used_time = time() - batch_time

            num_processed += shape_batch[0]
            progress = num_processed/args.data.train.size_dataset

            if global_step % 1 == 0:
                logging.info(
                    'loss:{:.3f}\tloss_ast:{:.3f}\tloss_ctc:{:.3f}\tloss_bert:{:.3f}\tbatch:{}\tlr:{:.6f}\ttime:{:.2f}s\tprocess:{:.3f}%\tstep:{}'.format(
                        loss, loss_ast, loss_ctc, loss_bert, shape_batch, lr, used_time, progress*100.0, global_step))

                summary.summary_scalar('loss', loss, global_step)

                summary.summary_scalar('learning_rate', lr, global_step)

            if args.begin_dev_step and global_step < args.begin_dev_step:
                if global_step % 10000 == 0 and global_step != 0:
                    dev_cer, dev_wer, dev_bleu = evaluate_decode(args, model_infer, sess, global_step, dataloader_dev)
                    global_bleu, global_wer, bad_count = check_save(dev_bleu, dev_wer, global_bleu, global_wer, bad_count)
                    if args.early_stop and bad_count > args.early_stop:
                        logging.info('Model training is early stopped!')
                        break

                elif global_step % 5000 == 0 and global_step != 0:
                    model_path = args.dir_checkpoint + '/model'
                    saver.save(
                        get_session(sess),
                        model_path,
                        global_step=global_step,
                        write_meta_graph=True)
                    logging.info('Save model in %s_%s.' % (model_path, str(global_step)))

            elif args.dev_step and args.dev_step > 0 and global_step % args.dev_step == 0 and global_step != 0:
                dev_cer, dev_wer, dev_bleu = evaluate_decode(args, model_infer, sess, global_step, dataloader_dev)
                global_bleu, global_wer, bad_count = check_save(dev_bleu, dev_wer, global_bleu, global_wer, bad_count)
                if args.early_stop and bad_count > args.early_stop:
                    logging.info('Model training is early stopped!')
                    break
            elif not args.dev_step and global_step % 1000 ==0 and global_step != 0:
                model_path = args.dir_checkpoint + '/model'
                saver.save(
                    get_session(sess),
                    model_path,
                    global_step=global_step,
                    write_meta_graph=True)
                logging.info('Save model in %s_%s.' % (model_path, str(global_step)))

            if args.decode_step and args.decode_step > 0 and global_step > 0 and global_step % args.decode_step == 0:

                decode_test(
                    step=global_step,
                    sample=args.dataset_test[10],
                    model=model_infer,
                    sess=sess,
                    unit=args.data.unit,
                    idx2token=args.idx2token,
                    eos_idx=None,
                    min_idx=0,
                    max_idx=None,
                    args=args)
                logging.info("decode_test for sample: {}".format(args.dataset_test[10]['id']))

            if args.num_steps and global_step > args.num_steps:
                sys.exit()
    logging.info('training duration: {:.2f}h'.format(
        (datetime.now()-start_time).total_seconds()/3600))


def infer():
    def prepare_infer(args):
        if args.dirs.test.tfdata_list:
            dataReader_test = TFReader(args.dirs.test.tfdata_list, args=args, is_train=False)
        else:
            dataReader_test = TFReader(args.dirs.test.tfdata, args=args, is_train=False)
        if args.model.use_multilabel:
            batch_test = dataReader_test.fentch_multi_label_batch_bucket()
        elif args.model.use_bert:
            batch_test = dataReader_test.fentch_kdbert_batch_bucket()
        elif args.model.use_ctc_bert:
            batch_test = dataReader_test.fentch_ctcbert_batch_bucket()
        elif args.model.use_ctc_bertfull:
            batch_test = dataReader_test.fentch_ctcbertfull_batch_bucket()
        elif args.model.is_multilingual:
            batch_test = dataReader_test.fentch_multilingual_batch_bucket()
        elif args.model.is_cotrain:
            batch_test = dataReader_test.fentch_multilingual_multilabel_batch_bucket()
        else:
            batch_test = dataReader_test.fentch_batch_bucket()
        return batch_test, dataReader_test
    batch_test, dataReader_test = prepare_infer(args)

    tensor_global_step = tf.train.get_or_create_global_step()

    model_infer = args.Model(
        tensor_global_step,
        encoder=args.model.encoder.type,
        decoder=args.model.decoder.type,
        batch=batch_test,
        is_train=False,
        args=args)

    saver = tf.train.Saver(max_to_keep=40)
    size_variables()

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    with tf.train.MonitoredTrainingSession(config=config) as sess:
        if args.dirs.checkpoint_infer:
            saver.restore(sess, args.dirs.checkpoint_infer)
            checkpoint_name = args.dirs.checkpoint_infer
        else:
            checkpoint_name = tf.train.latest_checkpoint(args.dir_checkpoint)
            saver.restore(sess, checkpoint_name)

        def run_decode(args, model, sess):
            batch_time = time()
            start_time = time()

            with tf.io.gfile.GFile(args.dir_model + '/test_wav.txt', 'w') as fwav, \
                    tf.io.gfile.GFile(args.dir_model + '/test_res.txt', 'w') as fres, \
                    tf.io.gfile.GFile(args.dir_model + '/test_ref.txt', 'w') as fref, \
                    tf.io.gfile.GFile(args.dir_model + '/test_ast_res.txt', 'w') as fastres, \
                    tf.io.gfile.GFile(args.dir_model + '/test_ast_ref.txt', 'w') as fastref, \
                    tf.io.gfile.GFile(args.dir_model + '/test_asr_res.txt', 'w') as fasrres, \
                    tf.io.gfile.GFile(args.dir_model + '/test_asr_ref.txt', 'w') as fasrref, \
                    tf.io.gfile.GFile(args.dir_model + '/test_ac_res.txt', 'w') as facres, \
                    tf.io.gfile.GFile(args.dir_model + '/test_ac_ref.txt', 'w') as facref:

                num_processed = 0
                progress = 0

                while progress < 1:

                    if args.model.loss_type == 'ctc_kd_ce' or args.model.loss_type == "CTC_BERT" or args.model.loss_type == "ce_ctc" or args.model.loss_type == "ce_ce_ctc":
                        [decoded, decoded_am, shape_batch, _], batch = sess.run([model.list_run, model.batch])
                    else:
                        [decoded, shape_batch, _], batch = sess.run([model.list_run, model.batch])

                    current_time = time() - batch_time
                    total_time = time() - start_time
                    batch_time = time()

                    num_processed += shape_batch[0]
                    progress = num_processed / args.data.test.size_dataset

                    logging.info('batch: {}\t batch_time:{:.2f}s \t total_time:{:.2f}s \t {:.3f}%'.format(
                        shape_batch, current_time, total_time, progress * 100.0))

                    for id_line in decoded:
                        token_line = array2text(id_line, args.data.unit, args.idx2token, min_idx=0, max_idx=args.dim_output - 1, args=args).replace("<eos>", '')

                        try:
                            ast_res, asr_res = token_line.split('<AST>')[-1], token_line.split('<AST>')[-2].split('<ASR>')[-1]
                        except:
                            res_tokens = token_line.split()
                            ast_res, asr_res = ' '.join(res_tokens[int(0.5 * len(res_tokens)):]), ' '.join(res_tokens[0: int(0.5 * len(res_tokens))])
                        fres.write(token_line + '\n')
                        fastres.write(''.join(ast_res) + '\n')
                        fasrres.write(''.join(asr_res) + '\n')

                    for id_line in batch[2]:
                        token_line = array2text(id_line, args.data.unit, args.idx2token, min_idx=0, max_idx=args.dim_output - 1, args=args).replace("<eos>", '')

                        ast_ref, asr_ref = token_line.split('<AST>')[-1], token_line.split('<AST>')[-2].split('<ASR>')[-1]

                        fref.write(token_line + '\n')
                        fastref.write(''.join(ast_ref) + '\n')
                        fasrref.write(''.join(asr_ref) + '\n')

                    for id_line in batch[0]:
                        if args.data.withoutid:
                            fwav.write('\n')
                        else:
                            wav_id = array_idx2char(id_line, args.idx2wav)
                            fwav.write(wav_id + '\n')

                    if args.model.loss_type == 'ctc_kd_ce' or args.model.loss_type == "CTC_BERT" or args.model.loss_type == "ce_ctc":
                        for id_line in decoded_am:
                            if args.dirs.src_vocab:
                                token_line = array2text(id_line, args.data.unit, args.src_idx2token, min_idx=0, max_idx=args.dim_output - 1, args=args)
                            else:
                                token_line = array2text(id_line, args.data.unit, args.idx2token, min_idx=0, max_idx=args.dim_output - 1, args=args)
                            facres.write(''.join(token_line).replace("<eos>", '') + '\n')
                        for id_line in batch[3]:
                            if args.dirs.src_vocab:
                                token_line = array2text(id_line, args.data.unit, args.src_idx2token, min_idx=0, max_idx=args.dim_output - 1, args=args)
                            else:
                                token_line = array2text(id_line, args.data.unit, args.idx2token, min_idx=0, max_idx=args.dim_output - 1, args=args)
                            facref.write(''.join(token_line).replace("<eos>", '') + '\n')

            cer = 0

            # copy to local
            tf.io.gfile.copy(args.dir_model+'/test_wav.txt', 'test_wav.txt', overwrite=True)
            tf.io.gfile.copy(args.dir_model+'/test_ast_res.txt', 'test_ast_res.txt', overwrite=True)
            tf.io.gfile.copy(args.dir_model+'/test_ast_ref.txt', 'test_ast_ref.txt', overwrite=True)
            tf.io.gfile.copy(args.dir_model+'/test_asr_res.txt', 'test_asr_res.txt', overwrite=True)
            tf.io.gfile.copy(args.dir_model+'/test_asr_ref.txt', 'test_asr_ref.txt', overwrite=True)
            tf.io.gfile.copy(args.dir_model+'/test_res.txt', 'test_res.txt', overwrite=True)
            tf.io.gfile.copy(args.dir_model+'/test_ref.txt', 'test_ref.txt', overwrite=True)

            # calculate bleu
            if args.bleu_cmd:
                cmd = args.bleu_cmd
            else:
                cmd = "perl st/tools/multi-bleu.perl {ref} " \
                      "< {output} 2>/dev/null | awk '{{print($3)}}' | awk -F, '{{print $1}}'"
            cmd = cmd.strip()
            bleu = subprocess.getoutput(cmd.format(**{'ref': 'test_ast_ref.txt',
                                                      'output': 'test_ast_res.txt'}))

            bleu = float(bleu)
            logging.info('BLEU:{:.3f}'.format(bleu))

            # calculate wer
            wer = calc_wer('test_asr_ref.txt', 'test_asr_res.txt')

            if args.model.loss_type == 'ctc_kd_ce' or args.model.loss_type == "CTC_BERT" or args.model.loss_type == "ce_ctc":
                tf.io.gfile.copy(args.dir_model+'/test_ac_ref.txt', 'test_ac_ref.txt', overwrite=True)
                tf.io.gfile.copy(args.dir_model+'/test_ac_res.txt', 'test_ac_res.txt', overwrite=True)

                per = calc_per('test_ac_ref.txt', 'test_ac_res.txt')
                logging.info('infer checkpoint name: {}'.format(checkpoint_name))
                logging.info(
                    'beam_size: {} \t length_penalty_weight: {} \t max_len: {}'.format(args.beam_size, args.length_penalty_weight, args.max_len))
                logging.info('infer CER: {:.3f} \t WER: {:.3f} \t PER:{:.3f}\tBLEU: {:.3f}'.format(
                    cer,
                    wer,
                    per,
                    bleu))
                return cer, wer, per, bleu
            else:
                logging.info('infer checkpoint name: {}'.format(checkpoint_name))
                logging.info('beam_size: {} \t length_penalty_weight: {} \t max_len: {}'.format(args.beam_size, args.length_penalty_weight, args.max_len))
                logging.info('infer CER: {:.3f} \t WER: {:.3f} \t BLEU: {:.3f}'.format(
                    cer,
                    wer,
                    bleu))
                return cer, wer, bleu

        if args.model.loss_type == 'ctc_kd_ce' or args.model.loss_type == "CTC_BERT" or args.model.loss_type == "ce_ctc" or args.model.loss_type == "ctc":
            test_cer, test_wer, test_per, test_bleu = run_decode(args, model_infer, sess)

            logging.info('infer checkpoint name: {}'.format(checkpoint_name))
            logging.info('beam_size: {} \t length_penalty_weight: {} \t max_len: {}'.format(args.beam_size, args.length_penalty_weight, args.max_len))
            logging.info('Test Infer CER: {:.3f} \t WER: {:.3f} \t PER: {:.3f} \t BLEU: {:.3f}'.format(test_cer, test_wer, test_per, test_bleu))

        else:
            test_cer, test_wer, test_bleu = run_decode(args, model_infer, sess)

            logging.info('infer checkpoint name: {}'.format(checkpoint_name))
            logging.info('beam_size: {} \t length_penalty_weight: {} \t max_len: {}'.format(args.beam_size, args.length_penalty_weight, args.max_len))
            logging.info('Test Infer CER: {:.3f} \t WER: {:.3f} \t BLEU: {:.3f}'.format(test_cer, test_wer, test_bleu))


def infer_sample():
    tensor_global_step = tf.train.get_or_create_global_step()

    model_infer = args.Model(
        tensor_global_step,
        encoder=args.model.encoder.type,
        decoder=args.model.decoder.type,
        is_train=False,
        args=args)

    saver = tf.train.Saver(max_to_keep=40)
    size_variables()

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    with tf.train.MonitoredTrainingSession(config=config) as sess:
        if args.dirs.checkpoint_infer:

            saver.restore(sess, args.dirs.checkpoint_infer)
            checkpoint_name = args.dirs.checkpoint_infer
        else:
            saver.restore(sess, tf.train.latest_checkpoint(args.dir_checkpoint))

            checkpoint_name = tf.train.latest_checkpoint(args.dir_checkpoint)
            saver.restore(sess, checkpoint_name)
        total_cer_dist = 0
        total_cer_len = 0
        total_wer_dist = 0
        total_wer_len = 0
        # with open(args.dir_model.name+'_decode.txt', 'w') as fw:
        with tf.io.gfile.GFile(args.dir_model + '/test_wav.txt', 'w') as fwav, \
                tf.io.gfile.GFile(args.dir_model + '/test_res.txt', 'w') as fres, \
                tf.io.gfile.GFile(args.dir_model + '/test_ref.txt', 'w') as fref, \
                tf.io.gfile.GFile(args.dir_model + '/test_ast_res.txt', 'w') as fastres, \
                tf.io.gfile.GFile(args.dir_model + '/test_ast_ref.txt', 'w') as fastref, \
                tf.io.gfile.GFile(args.dir_model + '/test_asr_res.txt', 'w') as fasrres, \
                tf.io.gfile.GFile(args.dir_model + '/test_asr_ref.txt', 'w') as fasrref, \
                tf.io.gfile.GFile(args.dir_model + '/test_ac_res.txt', 'w') as facres, \
                tf.io.gfile.GFile(args.dir_model + '/test_ac_ref.txt', 'w') as facref:

            for sample in tqdm(args.dataset_test):
                if not sample:
                    logging.error("not sample error when infer!")
                    # continue
                dict_feed = {model_infer.list_pl[0]: np.expand_dims(sample['feature'], axis=0),
                             model_infer.list_pl[1]: np.array([len(sample['feature'])])}

                if args.model.loss_type == 'ctc_kd_ce' or args.model.loss_type == "CTC_BERT" or args.model.loss_type == "ce_ctc" or args.model.loss_type == "ce_ce_ctc":
                    decoded, decoded_am, shape_batch, _ = sess.run(model_infer.list_run, feed_dict=dict_feed)
                else:
                    decoded, shape_batch, _ = sess.run(model_infer.list_run, feed_dict=dict_feed)


                res_txt = array2text(
                    decoded[0],
                    args.data.unit,
                    args.idx2token,
                    eos_idx=args.eos_idx,
                    min_idx=0,
                    max_idx=args.dim_output-1)

                ref_txt = array2text(
                    sample['label'],
                    args.data.unit,
                    args.idx2token,
                    eos_idx=args.eos_idx,
                    min_idx=0,
                    max_idx=args.dim_output-1)

                try:
                    ast_res, asr_res = res_txt.split('<AST>')[-1], res_txt.split('<AST>')[-2].split('<ASR>')[-1]
                except:
                    res_tokens = res_txt.split()
                    ast_res, asr_res = ' '.join(res_tokens[int(0.5 * len(res_tokens)):]), ' '.join(res_tokens[0: int(0.5 * len(res_tokens))])

                ast_ref, asr_ref = ref_txt.split('<AST>')[-1], ref_txt.split('<AST>')[-2].split('<ASR>')[-1]

                fwav.write(str(sample['id']) + '\n')
                fres.write(res_txt + '\n')
                fref.write(ref_txt + '\n')
                fastres.write(''.join(ast_res) + '\n')
                fasrres.write(''.join(asr_res) + '\n')
                fastref.write(''.join(ast_ref) + '\n')
                fasrref.write(''.join(asr_ref) + '\n')

                if args.model.loss_type == 'ctc_kd_ce' or args.model.loss_type == "CTC_BERT" or args.model.loss_type == "ce_ctc":
                    token_line = array2text(decoded_am[0], args.data.unit, args.idx2token, min_idx=0, max_idx=args.dim_output - 1, args=args)
                    facres.write(''.join(token_line).replace("<eos>", '') + '\n')
                    token_line = array2text(sample['aux_label'], args.data.unit, args.idx2token, min_idx=0, max_idx=args.dim_output - 1, args=args)
                    facref.write(''.join(token_line).replace("<eos>", '') + '\n')

        # copy to local
        tf.io.gfile.copy(args.dir_model+'/test_wav.txt', 'test_wav.txt', overwrite=True)
        tf.io.gfile.copy(args.dir_model+'/test_ast_res.txt', 'test_ast_res.txt', overwrite=True)
        tf.io.gfile.copy(args.dir_model+'/test_ast_ref.txt', 'test_ast_ref.txt', overwrite=True)
        tf.io.gfile.copy(args.dir_model+'/test_asr_res.txt', 'test_asr_res.txt', overwrite=True)
        tf.io.gfile.copy(args.dir_model+'/test_asr_ref.txt', 'test_asr_ref.txt', overwrite=True)
        tf.io.gfile.copy(args.dir_model+'/test_res.txt', 'test_res.txt', overwrite=True)
        tf.io.gfile.copy(args.dir_model+'/test_ref.txt', 'test_ref.txt', overwrite=True)


        # calculate bleu
        if args.bleu_cmd:
            cmd = args.bleu_cmd
        else:
            cmd = "perl st/tools/multi-bleu.perl {ref} " \
                  "< {output} 2>/dev/null | awk '{{print($3)}}' | awk -F, '{{print $1}}'"
        cmd = cmd.strip()

        bleu = subprocess.getoutput(cmd.format(**{'ref': 'test_ast_ref.txt',
                                                  'output': 'test_ast_res.txt'}))

        bleu = float(bleu)
        logging.info('BLEU:{:.3f}'.format(bleu))

        # calculate wer
        wer = calc_wer('test_asr_ref.txt', 'test_asr_res.txt')

        if args.model.loss_type == 'ctc_kd_ce' or args.model.loss_type == "CTC_BERT" or args.model.loss_type == "ce_ctc":
            tf.io.gfile.copy(args.dir_model+'/test_ac_ref.txt', 'test_ac_ref.txt', overwrite=True)
            tf.io.gfile.copy(args.dir_model+'/test_ac_res.txt', 'test_ac_res.txt', overwrite=True)

            per = calc_per('test_ac_ref.txt', 'test_ac_res.txt')
            logging.info('infer checkpoint name: {}'.format(checkpoint_name))
            logging.info(
                'beam_size: {} \t length_penalty_weight: {} \t max_len: {}'.format(args.beam_size, args.length_penalty_weight, args.max_len))
            logging.info('infer WER: {:.3f} \t PER:{:.3f}\tBLEU: {:.3f}'.format(
                wer,
                per,
                bleu))
        else:
            logging.info('infer checkpoint name: {}'.format(checkpoint_name))
            logging.info('beam_size: {} \t length_penalty_weight: {} \t max_len: {}'.format(args.beam_size, args.length_penalty_weight, args.max_len))
            logging.info('infer WER: {:.3f} \t BLEU: {:.3f}'.format(
                wer,
                bleu))


def infer_lm():
    tensor_global_step = tf.train.get_or_create_global_step()
    dataset_dev = args.dataset_test if args.dataset_test else args.dataset_dev

    model_lm = args.Model_LM(
        tensor_global_step,
        is_train=False,
        args=args.args_lm)

    args.lm_obj = model_lm
    saver_lm = tf.train.Saver(model_lm.variables())

    args.top_scope = tf.get_variable_scope()
    args.lm_scope = model_lm.decoder.scope

    model_infer = args.Model(
        tensor_global_step,
        encoder=args.model.encoder.type,
        decoder=args.model.decoder.type,
        is_train=False,
        args=args)

    saver = tf.train.Saver(model_infer.variables())

    size_variables()

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    with tf.train.MonitoredTrainingSession(config=config) as sess:
        checkpoint = tf.train.latest_checkpoint(args.dirs.checkpoint_init)
        checkpoint_lm = tf.train.latest_checkpoint(args.dirs.lm_checkpoint)
        saver.restore(sess, checkpoint)
        saver_lm.restore(sess, checkpoint_lm)

        total_cer_dist = 0
        total_cer_len = 0
        total_wer_dist = 0
        total_wer_len = 0
        with open(args.dir_model.name+'_decode.txt', 'w') as fw:
            for sample in tqdm(dataset_dev):
                if not sample:
                    continue
                dict_feed = {model_infer.list_pl[0]: np.expand_dims(sample['feature'], axis=0),
                             model_infer.list_pl[1]: np.array([len(sample['feature'])])}
                sample_id, shape_batch, beam_decoded = sess.run(model_infer.list_run, feed_dict=dict_feed)

                res_txt = array2text(sample_id[0], args.data.unit, args.idx2token, min_idx=0, max_idx=args.dim_output-1, args=args)
                ref_txt = array2text(sample['label'], args.data.unit, args.idx2token, min_idx=0, max_idx=args.dim_output-1, args=args)

                list_res_char = list(res_txt)
                list_ref_char = list(ref_txt)
                list_res_word = res_txt.split()
                list_ref_word = ref_txt.split()
                cer_dist = ed.eval(list_res_char, list_ref_char)
                cer_len = len(list_ref_char)
                wer_dist = ed.eval(list_res_word, list_ref_word)
                wer_len = len(list_ref_word)
                total_cer_dist += cer_dist
                total_cer_len += cer_len
                total_wer_dist += wer_dist
                total_wer_len += wer_len
                if cer_len == 0:
                    cer_len = 1000
                    wer_len = 1000
                if wer_dist/wer_len > 0:
                    print('ref  ' , ref_txt)
                    for i, decoded, score, rerank_score in zip(range(10), beam_decoded[0][0], beam_decoded[1][0], beam_decoded[2][0]):
                        candidate = array2text(decoded, args.data.unit, args.idx2token, min_idx=0, max_idx=args.dim_output-1, args=args)
                        print('res' ,i , candidate, score, rerank_score)
                        fw.write('res: {}; ref: {}\n'.format(candidate, ref_txt))
                    fw.write('id:\t{} \nres:\t{}\nref:\t{}\n\n'.format(sample['id'], res_txt, ref_txt))
                logging.info('current cer: {:.3f}, wer: {:.3f};\tall cer {:.3f}, wer: {:.3f}'.format(
                    cer_dist/cer_len, wer_dist/wer_len, total_cer_dist/total_cer_len, total_wer_dist/total_wer_len))
        logging.info('dev CER {:.3f}:  WER: {:.3f}'.format(total_cer_dist/total_cer_len, total_wer_dist/total_wer_len))


def save(gpu, name=0):
    from st.layers.utils import store_2d

    tensor_global_step = tf.train.get_or_create_global_step()

    embed = create_embedding(
        name='embedding_table',
        size_vocab=args.dim_output,
        size_embedding=args.model.decoder.size_embedding)

    model_infer = args.Model(
        tensor_global_step,
        encoder=args.model.encoder.type,
        decoder=args.model.decoder.type,
        embed_table_decoder=embed,
        is_train=False,
        args=args)

    dataset_dev = args.dataset_test if args.dataset_test else args.dataset_dev

    saver = tf.train.Saver(max_to_keep=15)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    with tf.train.MonitoredTrainingSession(config=config) as sess:
        checkpoint = tf.train.latest_checkpoint(args.dirs.checkpoint_init)
        saver.restore(sess, checkpoint)
        if not name:
            name = args.dir_model.name
        with open('outputs/distribution_'+name+'.bin', 'wb') as fw, \
            open('outputs/res_ref_'+name+'.txt', 'w') as fw2:

            for i, sample in enumerate(tqdm(dataset_dev)):

                if not sample: continue
                dict_feed = {model_infer.list_pl[0]: np.expand_dims(sample['feature'], axis=0),
                             model_infer.list_pl[1]: np.array([len(sample['feature'])])}
                decoded, _, distribution = sess.run(model_infer.list_run, feed_dict=dict_feed)
                store_2d(distribution[0], fw)

                result_txt = array_idx2char(decoded, args.idx2token, seperator=' ')
                ref_txt = array_idx2char(sample['label'], args.idx2token, seperator=' ')
                fw2.write('{}_res: {}\n{}_ref: {}\n'.format(i, result_txt[0], i, ref_txt))


def draw_attention():
    tensor_global_step = tf.train.get_or_create_global_step()

    model = args.Model(
        tensor_global_step,
        encoder=args.model.encoder.type,
        decoder=args.model.decoder.type,
        is_train=False,
        args=args)

    saver = tf.train.Saver(max_to_keep=40)
    size_variables()

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.log_device_placement = False

    summary = Summary(str(args.dir_log)+'/infer_log')

    with tf.train.MonitoredTrainingSession(config=config) as sess:
        if args.dirs.checkpoint_infer:

            saver.restore(sess, args.dirs.checkpoint_infer)
            checkpoint_name = args.dirs.checkpoint_infer
        else:
            checkpoint_name = tf.train.latest_checkpoint(args.dir_checkpoint)
            saver.restore(sess, checkpoint_name)

        def run_decode(dataset):

            idx = 0
            for sample in tqdm(dataset):
                print(sample['id'])

                dict_feed = {model.list_pl[0]: np.expand_dims(sample['feature'], axis=0),
                             model.list_pl[1]: np.array([len(sample['feature'])])}
                if args.model.loss_type == 'ctc_kd_ce' or args.model.loss_type == "CTC_BERT" or args.model.loss_type == "ce_ctc" or args.model.loss_type == "ce_ce_ctc":
                    decoded, decoded_am, shape_batch, _ = sess.run(model.list_run, feed_dict=dict_feed)
                else:
                    decoded, shape_batch, _ = sess.run(model.list_run, feed_dict=dict_feed)

                res_txt = array2text(
                    decoded[0],
                    "word",
                    args.idx2token,
                    eos_idx=args.eos_idx,
                    min_idx=0,
                    max_idx=args.dim_output-1)

                ref_txt = array2text(
                    sample['label'],
                    "word",
                    args.idx2token,
                    eos_idx=args.eos_idx,
                    min_idx=0,
                    max_idx=args.dim_output-1)

                print("res_txt", res_txt)
                print("ref_txt", ref_txt)

                idx += 1
                break

            logging.info('infer checkpoint name: {}'.format(checkpoint_name))
            logging.info('log dir: {}'.format(args.dir_log))
            logging.info('beam_size: {} \t length_penalty_weight: {} \t max_len: {}'.format(args.beam_size, args.length_penalty_weight, args.max_len))

        run_decode(args.dataset_test)


if __name__ == '__main__':
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', type=str, dest='mode', default='train')
    parser.add_argument('--name', type=str, dest='name', default=None)
    parser.add_argument('--gpu', type=str, dest='gpu', default="0")
    parser.add_argument('-task', dest='task', default='asr')
    parser.add_argument('-c', type=str, dest='config', nargs="+", help="Configuration files.")

    param = parser.parse_args()

    logging.basicConfig(format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
                        datefmt='%Y-%m-%d %A %H:%M:%S',
                        filename=args.dir_model + '/train.log',
                        level=logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger = logging.getLogger('')

    logging.getLogger().setLevel(logging.INFO)

    print('CUDA_VISIBLE_DEVICES: ', args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    args.mode = param.mode

    try:

        if param.mode == 'infer':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
            logging.info('enter the INFERING phrase')
            infer()
        elif param.mode == 'infer_sample':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
            logging.info('enter the INFERING phrase')
            infer_sample()

        elif param.mode == 'infer_lm':
            logging.info('enter the INFERING phrase')
            infer_lm()

        elif param.mode == 'save':
            logging.info('enter the SAVING phrase')
            save(gpu=param.gpu, name=param.name)

        elif param.mode == 'attend':
            logging.info('enter the ATTEND phrase')
            draw_attention()

        elif param.mode == 'train':
            # save config setting
            json.dump(args, tf.io.gfile.GFile(args.dirs.models + '/config.json', 'w'), indent=2)

            logging.info('enter the TRAINING phrase')
            train()
    except:
        logger.exception("Exception Logged")

