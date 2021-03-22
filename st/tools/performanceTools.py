#!/usr/bin/env python
import numpy as np
from time import time
import logging
import subprocess

from st.tools.textTools import batch_wer, batch_cer, array2text, array_idx2char
import tensorflow as tf

def dev_test(args, step, dataloader, model, sess, unit, idx2token, eos_idx=None, min_idx=0, max_idx=None):
    start_time, batch_time = time(), time()
    processed = 0

    total_cer_dist = 0
    total_cer_len = 0

    total_wer_dist = 0
    total_wer_len = 0

    with tf.io.gfile.GFile('dev_wav.txt', 'w') as fw, \
            tf.io.gfile.GFile('dev_ast_res.txt', 'w') as fastres, \
            tf.io.gfile.GFile('dev_ast_ref.txt', 'w') as fastref, \
            tf.io.gfile.GFile('dev_asr_res.txt', 'w') as fasrres, \
            tf.io.gfile.GFile('dev_asr_ref.txt', 'w') as fasrref:
        for batch in dataloader:
            if not batch: continue
            if args.model.use_multilabel:
                dict_feed = {model.list_pl[0]: batch[0], model.list_pl[1]: batch[3], model.list_pl[2]: batch[2], model.list_pl[3]: batch[5]}
            elif args.model.use_bert:
                dict_feed = {model.list_pl[0]: batch[0], model.list_pl[1]: batch[3]}
            elif args.model.use_ctc_bert or args.model.use_ctc_bertfull:
                dict_feed = {model.list_pl[0]: batch[0], model.list_pl[1]: batch[4]}
            elif args.model.is_multilingual:
                dict_feed = {model.list_pl[0]: batch[0], model.list_pl[1]: batch[3], model.list_pl[2]: batch[2]}
            elif args.model.is_cotrain:
                dict_feed = {model.list_pl[0]: batch[0], model.list_pl[1]: batch[4], model.list_pl[2]: batch[3]}
            else:
                dict_feed = {model.list_pl[0]: batch[0], model.list_pl[1]: batch[2]}

            if args.model.loss_type == "ctc_kd_ce" or args.model.loss_type == "CTC_BERT" or args.model.loss_type == "ce_ctc" or args.model.loss_type == "ce_ce_ctc":
                decoded, decoded_asr, shape_batch, _ = sess.run(model.list_run, feed_dict=dict_feed)
            else:
                decoded, shape_batch, _ = sess.run(model.list_run, feed_dict=dict_feed)

            batch_cer_dist, batch_cer_len = batch_cer(result=decoded, reference=batch[1], eos_idx=eos_idx, min_idx=min_idx, max_idx=max_idx)
            _cer = batch_cer_dist/batch_cer_len
            total_cer_dist += batch_cer_dist
            total_cer_len += batch_cer_len

            batch_wer_dist, batch_wer_len = batch_wer(result=decoded, reference=batch[1], idx2token=idx2token, unit=unit, eos_idx=eos_idx,
                                                      min_idx=min_idx, max_idx=max_idx, args=args)
            _wer = batch_wer_dist/batch_wer_len
            total_wer_dist += batch_wer_dist
            total_wer_len += batch_wer_len

            used_time = time()-batch_time
            batch_time = time()
            processed += shape_batch[0]
            progress = processed/len(dataloader)
            logging.info('batch cer: {:.3f}\twer: {:.3f} batch: {}\t time:{:.2f}s {:.3f}%'.format(
                         _cer, _wer, shape_batch, used_time, progress*100.0))
            for id_line in decoded:
                token_line = array2text(id_line, args.data.unit, args.idx2token, min_idx=0, max_idx=args.dim_output - 1, args=args)
                fastres.write(''.join(token_line).replace("<eos>", '') + '\n')

            for id_line in batch[1]:
                token_line = array2text(id_line, args.data.unit, args.idx2token, min_idx=0, max_idx=args.dim_output - 1, args=args)
                fastref.write(''.join(token_line).replace("<eos>", '') + '\n')
            if args.model.loss_type == "ctc_kd_ce" or args.model.loss_type == "CTC_BERT" or args.model.loss_type == "ce_ctc":
                for id_line in decoded_asr:
                    if args.dirs.src_vocab:
                        token_line = array2text(id_line, args.data.unit, args.src_idx2token, min_idx=0, max_idx=args.src_dim_output - 1, args=args, delimiter=args.data.src_delimiter)
                    else:
                        token_line = array2text(id_line, args.data.unit, args.idx2token, min_idx=0, max_idx=args.dim_output - 1, args=args)
                    fasrres.write(''.join(token_line).replace("<eos>", '') + '\n')
                for id_line in batch[2]:
                    if args.dirs.src_vocab:
                        token_line = array2text(id_line, args.data.unit, args.src_idx2token, min_idx=0, max_idx=args.src_dim_output - 1, args=args, delimiter=args.data.src_delimiter)
                    else:
                        token_line = array2text(id_line, args.data.unit, args.idx2token, min_idx=0, max_idx=args.dim_output - 1, args=args)
                    fasrref.write(''.join(token_line).replace("<eos>", '') + '\n')

    used_time = time() - start_time
    cer = total_cer_dist/total_cer_len
    wer = total_wer_dist/total_wer_len

    # calculate bleu
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
        'WER: {:.4f}\n'
        'total_wer_len: {}\n'
        'BLEU: {:.4f}'.format(used_time/3600, wer, total_wer_len, bleu))
    return cer, wer, bleu


def decode_test(step, sample, model, sess, unit, idx2token, eos_idx=None, min_idx=0, max_idx=None, args=None):
    if args.model.is_multilingual or args.model.is_cotrain:
        dict_feed = {model.list_pl[0]: np.expand_dims(sample['feature'], axis=0),
                     model.list_pl[1]: np.array([len(sample['feature'])]),
                     model.list_pl[2]: np.expand_dims(sample['lang_label'], axis=0)}
    elif args.model.use_multilabel:
        dict_feed = {model.list_pl[0]: np.expand_dims(sample['feature'], axis=0),
                     model.list_pl[1]: np.array([len(sample['feature'])]),
                     model.list_pl[2]: np.expand_dims(sample['aux_label'], axis=0),
                     model.list_pl[3]: np.array([len(sample['aux_label'])])}
    else:
        dict_feed = {model.list_pl[0]: np.expand_dims(sample['feature'], axis=0),
                     model.list_pl[1]: np.array([len(sample['feature'])])}
    if args.model.loss_type == "ctc_kd_ce" or args.model.loss_type == "CTC_BERT" or args.model.loss_type == "ce_ctc":
        sampled_id, sample_id_am, shape_sample, _ = sess.run(model.list_run, feed_dict=dict_feed)
    else:
        sampled_id, shape_sample, _ = sess.run(model.list_run, feed_dict=dict_feed)

    res_txt = array2text(sampled_id[0], unit, idx2token, eos_idx, min_idx, max_idx, args=args)
    ref_txt = array2text(sample['label'], unit, idx2token, eos_idx, min_idx, max_idx, args=args)

    logging.info('length: {}, res: \n{}\nref: \n{}'.format(
                 shape_sample[1], res_txt, ref_txt))


def cls_dev_test(args, step, dataloader, model, sess, unit, idx2token, eos_idx=None, min_idx=0, max_idx=None):
    start_time = time()
    with open(args.dir_model + '/dev_decode_res.txt', 'w') as fres, open(args.dir_model + '/dev_decode_ref.txt', 'w') as fref:
        for batch in dataloader:
            if not batch: continue

            dict_feed = {model.list_pl[0]: batch[0], model.list_pl[1]: batch[2]}

            decoded, shape_batch, _ = sess.run(model.list_run, feed_dict=dict_feed)

            for id_line in decoded:
                token_line = array2text(id_line, args.data.unit, args.idx2token, min_idx=0, max_idx=args.dim_output - 1, args=args)
                fres.write(''.join(token_line).replace("<eos>", '') + '\n')
            for id_line in batch[1]:
                token_line = array2text(id_line, args.data.unit, args.idx2token, min_idx=0, max_idx=args.dim_output - 1, args=args)
                fref.write(''.join(token_line).replace("<eos>", '') + '\n')

    used_time = time() - start_time
    # calculate acc
    with open(args.dir_model + '/dev_decode_res.txt', 'r') as fres, open(args.dir_model + '/dev_decode_ref.txt', 'r') as fref:
        acc_num = 0
        total_num = 0
        for l1, l2 in zip(fres, fref):
            if l1.strip() == l2.strip():
                acc_num += 1
            total_num += 1
        acc = acc_num / total_num

    logging.info(
        '=====dev info, total used time {:.2f}h==== \n'
        'acc_num: {}\n'
        'total_num: {}\n'
        'acc: {:.4f}'.format(used_time/3600, acc_num, total_num, acc))
    return acc
