#!/usr/bin/env
# coding=utf-8

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    import os

import tensorflow as tf
import logging
from pathlib import Path
from random import shuffle

from . import tfAudioTools as tfAudio
from multiprocessing import Process, Queue
from tqdm import tqdm


def _bytes_feature(value):
    """Returns a bytes_list from a list of string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class TFDataSaver:
    """
    test on TF2.0-alpha
    """
    def __init__(self, dataset, dir_save, args, size_file=5000000, max_feat_len=3000):
        self.dataset = dataset
        self.max_feat_len = max_feat_len
        self.dir_save = dir_save
        self.args = args
        self.add_eos = args.data.add_eos
        self.size_file = size_file
        self.dim_feature = dataset[0]['feature'].shape[-1]

    def get_example(self, sample):
        if self.args.model.use_multilabel:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={'id': _bytes_feature(sample['id'].tostring()),
                             'feature': _bytes_feature(sample['feature'].tostring()),
                             'label': _bytes_feature(sample['label'].tostring()),
                             'aux_label': _bytes_feature(sample['aux_label'].tostring())}
                )
            )
        elif self.args.model.use_bert:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={'id': _bytes_feature(sample['id'].tostring()),
                             'feature': _bytes_feature(sample['feature'].tostring()),
                             'label': _bytes_feature(sample['label'].tostring()),
                             'bert_feature': _bytes_feature(sample['bert_feature'].tostring())}
                )
            )
        elif self.args.model.use_ctc_bert:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={'id': _bytes_feature(sample['id'].tostring()),
                             'feature': _bytes_feature(sample['feature'].tostring()),
                             'label': _bytes_feature(sample['label'].tostring()),
                             'aux_label': _bytes_feature(sample['aux_label'].tostring()),
                             'bert_feature': _bytes_feature(sample['bert_feature'].tostring())}
                )
            )
        elif self.args.model.use_ctc_bertfull:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={'id': _bytes_feature(sample['id'].tostring()),
                             'feature': _bytes_feature(sample['feature'].tostring()),
                             'label': _bytes_feature(sample['label'].tostring()),
                             'aux_label': _bytes_feature(sample['aux_label'].tostring()),
                             'bert_feature': _bytes_feature(sample['bert_feature'].tostring()),
                             'bertfull_feature': _bytes_feature(sample['bertfull_feature'].tostring())}
                )
            )
        elif self.args.model.is_multilingual:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={'id': _bytes_feature(sample['id'].tostring()),
                             'feature': _bytes_feature(sample['feature'].tostring()),
                             'label': _bytes_feature(sample['label'].tostring()),
                             'lang_label': _bytes_feature(sample['lang_label'].tostring())}
                )
            )
        elif self.args.model.is_cotrain:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={'id': _bytes_feature(sample['id'].tostring()),
                             'feature': _bytes_feature(sample['feature'].tostring()),
                             'label': _bytes_feature(sample['label'].tostring()),
                             'label_2': _bytes_feature(sample['label_2'].tostring()),
                             'lang_label': _bytes_feature(sample['lang_label'].tostring())}
                )
            )
        else:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={'id': _bytes_feature(sample['id'].tostring()),
                             'feature': _bytes_feature(sample['feature'].tostring()),
                             'label': _bytes_feature(sample['label'].tostring())}
                )
            )
        return example

    def save(self):
        num_token = 0
        idx_file = -1
        num_damaged_sample = 0

        assert self.dataset.transform == False
        with tf.io.gfile.GFile(self.dir_save + '/feature_length.txt', 'w') as fw:
            for i, sample in enumerate(tqdm(self.dataset)):
                if not sample:
                    num_damaged_sample += 1
                    continue
                dim_feature = sample['feature'].shape[-1]
                if (num_token // self.size_file) > idx_file:
                    idx_file = num_token // self.size_file
                    print('saving to file {}/{}.recode'.format(self.dir_save, idx_file))
                    writer = tf.python_io.TFRecordWriter(str(self.dir_save+'/{}.recode'.format(idx_file)))

                example = self.get_example(sample)

                writer.write(example.SerializeToString())
                num_token += len(sample['feature'])
                line = str(sample['id']) + ' ' + str(len(sample['feature']))
                fw.write(line + '\n')

        with tf.io.gfile.GFile(self.dir_save+'/tfdata.info', 'w') as fw:
            # print('data_file {}'.format(dataset.list_files), file=fw)
            print('dim_feature {}'.format(dim_feature), file=fw)
            print('num_tokens {}'.format(num_token), file=fw)
            print('size_dataset {}'.format(i-num_damaged_sample), file=fw)
            print('damaged samples: {}'.format(num_damaged_sample), file=fw)

    def split_save(self):
        output = Queue()
        coord = tf.train.Coordinator()
        assert self.dataset.transform == False

        def gen_recoder(i):
            num_saved = 0
            num_damaged_sample = 0
            idx_start = i * self.size_file
            idx_end = min((i + 1) * self.size_file, len(self.dataset))
            if not tf.io.gfile.isdir(self.dir_save):
                tf.io.gfile.mkdir(self.dir_save)
            writer = tf.io.TFRecordWriter(str(self.dir_save + '/{}.recode'.format(i)))
            print('saving dataset[{}: {}] to file {}/{}.recode'.format(idx_start, idx_end, self.dir_save, i))
            with tf.io.gfile.GFile(self.dir_save + '/feature_length.{}.txt'.format(i), 'w') as fw:
                if i == 0:
                    m = tqdm(range(idx_start, idx_end))
                else:
                    m = range(idx_start, idx_end)
                for j in m:
                    sample = self.dataset[j]
                    if not sample:
                        num_damaged_sample += 1
                        continue

                    example = self.get_example(sample)
                    writer.write(example.SerializeToString())
                    line = str(sample['id']) + ' ' + str(len(sample['feature']))
                    fw.write(line + '\n')
                    num_saved += 1
            print('{}.recoder finished, {} saved, {} damaged. '.format(i, num_saved, num_damaged_sample))
            output.put((i, num_damaged_sample, num_saved))

        processes = []
        workers = len(self.dataset) // self.size_file + 1
        print('save {} samples to {} record files'.format(len(self.dataset), workers))
        for i in range(workers):
            p = Process(target=gen_recoder, args=(i,))
            p.start()
            processes.append(p)
        print('generating ...')
        coord.join(processes)
        print('save recode files finished.')

        res = [output.get() for _ in processes]
        num_saved = sum([x[2] for x in res])
        num_damaged = sum([x[1] for x in res])
        # TODO: concat feature length file
        with tf.io.gfile.GFile(str(self.dir_save + '/tfdata.info'), 'w') as fw:
            fw.write('data_file {}\n'.format(self.dataset.list_files))
            fw.write('dim_feature {}\n'.format(self.dataset[0]['feature'].shape[-1]))
            fw.write('size_dataset {}\n'.format(num_saved))
            fw.write('damaged samples: {}\n'.format(num_damaged))

        os.system('cat {}/feature_length.*.txt > {}/feature_length.txt'.format(self.dir_save, self.dir_save))

        print('ALL FINISHED.')


def save2tfrecord(dataset, dir_save, size_file=5000000):
    """
    Args:
        dataset = ASRdataSet(data_file, args)
        dir_save: the dir to save the tfdata files
    Return:
        Nothing but a folder consist of `tfdata.info`, `*.recode`

    Notice: the feats.scp file is better to cluster by ark file and sort by the index in the ark files
    For example, '...split16/1/final_feats.ark:143468' the paths share the same arkfile '1/final_feats.ark' need to close with each other,
    Meanwhile, these files need to be sorted by the index ':143468'
    ther sorted scp file will be 10x faster than the unsorted one.
    """

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    num_token = 0
    idx_file = -1
    num_damaged_sample = 0

    assert dataset.transform == False

    for i, sample in enumerate(tqdm(dataset)):
        if not sample:
            num_damaged_sample += 1
            continue
        dim_feature = sample['feature'].shape[-1]
        if (num_token // size_file) > idx_file:
            idx_file = num_token // size_file
            print('saving to file {}/{}.recode'.format(dir_save, idx_file))
            writer = tf.python_io.TFRecordWriter(str(dir_save / '{}.recode'.format(idx_file)))

        example = tf.train.Example(
            features=tf.train.Features(
                feature={'id': _bytes_feature(sample['id'].tostring()),
                         'feature': _bytes_feature(sample['feature'].tostring()),
                         'label': _bytes_feature(sample['label'].tostring())}
            )
        )
        writer.write(example.SerializeToString())
        # num_token += len(sample['feature'])

    with tf.io.gfile.GFile(dir_save / 'tfdata.info', 'w') as fw:
        print('data_file {}'.format(dataset.list_files), file=fw)
        print('dim_feature {}'.format(dim_feature), file=fw)
        print('num_tokens {}'.format(num_token), file=fw)
        print('size_dataset {}'.format(len(dataset)-num_damaged_sample), file=fw)
        print('damaged samples: {}'.format(num_damaged_sample), file=fw)

    return


def save2tfrecord_multilabel(dataset, dir_save, size_file=5000000):
    """
    use for multi-label
    """

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    num_token = 0
    idx_file = -1
    num_damaged_sample = 0

    assert dataset.transform == False
    for i, sample in enumerate(tqdm(dataset)):
        if not sample:
            num_damaged_sample += 1
            continue
        dim_feature = sample['feature'].shape[-1]
        if (num_token // size_file) > idx_file:
            idx_file = num_token // size_file
            print('saving to file {}/{}.recode'.format(dir_save, idx_file))
            writer = tf.python_io.TFRecordWriter(str(dir_save/'{}.recode'.format(idx_file)))

        example = tf.train.Example(
            features=tf.train.Features(
                feature={'id': _bytes_feature(sample['id'].tostring()),
                         'feature': _bytes_feature(sample['feature'].tostring()),
                         'label': _bytes_feature(sample['label'].tostring()),
                         'aux_label': _bytes_feature(sample['aux_label'].tostring())}
            )
        )
        writer.write(example.SerializeToString())
        num_token += len(sample['feature'])

    with tf.io.gfile.GFile(dir_save/'tfdata.info', 'w') as fw:
        print('data_file {}'.format(dataset.list_files), file=fw)
        print('dim_feature {}'.format(dim_feature), file=fw)
        print('num_tokens {}'.format(num_token), file=fw)
        print('size_dataset {}'.format(i-num_damaged_sample+1), file=fw)
        print('damaged samples: {}'.format(num_damaged_sample), file=fw)

    return


def save2tfrecord_kdbert(dataset, dir_save, size_file=5000000):
    """
    use for multi-label
    """

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    num_token = 0
    idx_file = -1
    num_damaged_sample = 0

    assert dataset.transform == False

    for i, sample in enumerate(tqdm(dataset)):
        if not sample:
            num_damaged_sample += 1
            continue
        dim_feature = sample['feature'].shape[-1]
        if (num_token // size_file) > idx_file:
            idx_file = num_token // size_file
            print('saving to file {}/{}.recode'.format(dir_save, idx_file))
            writer = tf.python_io.TFRecordWriter(str(dir_save/'{}.recode'.format(idx_file)))

        example = tf.train.Example(
            features=tf.train.Features(
                feature={'id': _bytes_feature(sample['id'].tostring()),
                         'feature': _bytes_feature(sample['feature'].tostring()),
                         'label': _bytes_feature(sample['label'].tostring()),
                         'bert_feature': _bytes_feature(sample['bert_feature'].tostring())}
            )
        )
        writer.write(example.SerializeToString())
        num_token += len(sample['feature'])

    with tf.io.gfile.GFile(dir_save/'tfdata.info', 'w') as fw:
        print('data_file {}'.format(dataset.list_files), file=fw)
        print('dim_feature {}'.format(dim_feature), file=fw)
        print('num_tokens {}'.format(num_token), file=fw)
        print('size_dataset {}'.format(i-num_damaged_sample+1), file=fw)
        print('damaged samples: {}'.format(num_damaged_sample), file=fw)

    return


def save2tfrecord_ctcbert(dataset, dir_save, size_file=5000000):
    """
        use for multi-label
        """

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    num_token = 0
    idx_file = -1
    num_damaged_sample = 0

    assert dataset.transform == False
    for i, sample in enumerate(tqdm(dataset)):
        if not sample:
            num_damaged_sample += 1
            continue
        dim_feature = sample['feature'].shape[-1]
        if (num_token // size_file) > idx_file:
            idx_file = num_token // size_file
            print('saving to file {}/{}.recode'.format(dir_save, idx_file))
            writer = tf.python_io.TFRecordWriter(str(dir_save / '{}.recode'.format(idx_file)))

        example = tf.train.Example(
            features=tf.train.Features(
                feature={'id': _bytes_feature(sample['id'].tostring()),
                         'feature': _bytes_feature(sample['feature'].tostring()),
                         'label': _bytes_feature(sample['label'].tostring()),
                         'aux_label': _bytes_feature(sample['aux_label'].tostring()),
                         'bert_feature': _bytes_feature(sample['bert_feature'].tostring())}
            )
        )
        writer.write(example.SerializeToString())
        num_token += len(sample['feature'])

    with tf.io.gfile.GFile(dir_save / 'tfdata.info', 'w') as fw:
        print('data_file {}'.format(dataset.list_files), file=fw)
        print('dim_feature {}'.format(dim_feature), file=fw)
        print('num_tokens {}'.format(num_token), file=fw)
        print('size_dataset {}'.format(i - num_damaged_sample + 1), file=fw)
        print('damaged samples: {}'.format(num_damaged_sample), file=fw)

    return


def save2tfrecord_ctcbertfull(dataset, dir_save, size_file=5000000):
    """
        use for multi-label
        """

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    num_token = 0
    idx_file = -1
    num_damaged_sample = 0

    assert dataset.transform == False
    for i, sample in enumerate(tqdm(dataset)):
        if not sample:
            num_damaged_sample += 1
            continue
        dim_feature = sample['feature'].shape[-1]
        if (num_token // size_file) > idx_file:
            idx_file = num_token // size_file
            print('saving to file {}/{}.recode'.format(dir_save, idx_file))
            writer = tf.python_io.TFRecordWriter(str(dir_save / '{}.recode'.format(idx_file)))

        example = tf.train.Example(
            features=tf.train.Features(
                feature={'id': _bytes_feature(sample['id'].tostring()),
                         'feature': _bytes_feature(sample['feature'].tostring()),
                         'label': _bytes_feature(sample['label'].tostring()),
                         'aux_label': _bytes_feature(sample['aux_label'].tostring()),
                         'bert_feature': _bytes_feature(sample['bert_feature'].tostring()),
                         'bertfull_feature': _bytes_feature(sample['bertfull_feature'].tostring())}
            )
        )
        writer.write(example.SerializeToString())
        num_token += len(sample['feature'])

    with tf.io.gfile.GFile(dir_save / 'tfdata.info', 'w') as fw:
        print('data_file {}'.format(dataset.list_files), file=fw)
        print('dim_feature {}'.format(dim_feature), file=fw)
        print('num_tokens {}'.format(num_token), file=fw)
        print('size_dataset {}'.format(i - num_damaged_sample + 1), file=fw)
        print('damaged samples: {}'.format(num_damaged_sample), file=fw)

    return


def save2tfrecord_multilingual(dataset, dir_save, size_file=5000000):
    """
    use for multi-label
    """

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    num_token = 0
    idx_file = -1
    num_damaged_sample = 0

    assert dataset.transform == False
    for i, sample in enumerate(tqdm(dataset)):
        if not sample:
            num_damaged_sample += 1
            continue
        dim_feature = sample['feature'].shape[-1]
        if (num_token // size_file) > idx_file:
            idx_file = num_token // size_file
            print('saving to file {}/{}.recode'.format(dir_save, idx_file))
            writer = tf.python_io.TFRecordWriter(str(dir_save/'{}.recode'.format(idx_file)))

        example = tf.train.Example(
            features=tf.train.Features(
                feature={'id': _bytes_feature(sample['id'].tostring()),
                         'feature': _bytes_feature(sample['feature'].tostring()),
                         'label': _bytes_feature(sample['label'].tostring()),
                         'lang_label': _bytes_feature(sample['lang_label'].tostring())}
            )
        )
        writer.write(example.SerializeToString())
        num_token += len(sample['feature'])

    with tf.io.gfile.GFile(dir_save/'tfdata.info', 'w') as fw:
        print('data_file {}'.format(dataset.list_files), file=fw)
        print('dim_feature {}'.format(dim_feature), file=fw)
        print('num_tokens {}'.format(num_token), file=fw)
        print('size_dataset {}'.format(i-num_damaged_sample+1), file=fw)
        print('damaged samples: {}'.format(num_damaged_sample), file=fw)

    return


def readTFRecord(dir_data, args, _shuffle=False, transform=False):
    """
    the tensor could run unlimitatly
    """

    list_filenames = fentch_filelist(dir_data)
    if _shuffle:
        shuffle(list_filenames)
    else:
        list_filenames.sort()

    num_epochs = 1 if args.mode == 'infer' else None
    filename_queue = tf.train.string_input_producer(
        list_filenames, num_epochs=num_epochs, shuffle=shuffle)

    reader_tfRecord = tf.TFRecordReader()
    _, serialized_example = reader_tfRecord.read(filename_queue)
    if args.data.withoutid:
        features = tf.parse_single_example(
            serialized_example,
            features={'feature': tf.FixedLenFeature([], tf.string),
                      'label': tf.FixedLenFeature([], tf.string)}
        )
        id = tf.convert_to_tensor([1], dtype=tf.int32)
    else:
        features = tf.parse_single_example(
            serialized_example,
            features={'feature': tf.FixedLenFeature([], tf.string),
                      'id': tf.FixedLenFeature([], tf.string),
                      'label': tf.FixedLenFeature([], tf.string)}
        )
        id = tf.decode_raw(features['id'], tf.int32)

    if args.mode == 'train':
        feature = tf.reshape(tf.decode_raw(features['feature'], tf.float32),
                             [-1, args.data.dim_feature])[:args.data.max_train_len, :]
    else:
        feature = tf.reshape(tf.decode_raw(features['feature'], tf.float32),
                             [-1, args.data.dim_feature])

    label = tf.decode_raw(features['label'], tf.int32)

    if transform:
        feature = process_raw_feature(feature, args)

    return id, feature, label


def readTFRecord_multilabel(dir_data, args, _shuffle=False, transform=False):
    """
    use for multi-label
    """
    list_filenames = fentch_filelist(dir_data)
    if _shuffle:
        shuffle(list_filenames)
    else:
        list_filenames.sort()
    num_epochs = 1 if args.mode == 'infer' else None
    filename_queue = tf.train.string_input_producer(
        list_filenames, num_epochs=num_epochs, shuffle=shuffle)

    reader_tfRecord = tf.TFRecordReader()
    _, serialized_example = reader_tfRecord.read(filename_queue)

    if args.data.withoutid:
        features = tf.parse_single_example(
            serialized_example,
            features={'feature': tf.FixedLenFeature([], tf.string),
                      'label': tf.FixedLenFeature([], tf.string),
                      'aux_label': tf.FixedLenFeature([], tf.string)}
        )
        id = tf.convert_to_tensor([1], dtype=tf.int32)
    else:
        features = tf.parse_single_example(
            serialized_example,
            features={'feature': tf.FixedLenFeature([], tf.string),
                      'id': tf.FixedLenFeature([], tf.string),
                      'label': tf.FixedLenFeature([], tf.string),
                      'aux_label': tf.FixedLenFeature([], tf.string)}
        )
        id = tf.decode_raw(features['id'], tf.int32)
    if args.mode == 'train':
        feature = tf.reshape(tf.decode_raw(features['feature'], tf.float32),
            [-1, args.data.dim_feature])[:args.data.max_train_len, :]
    else:
        feature = tf.reshape(tf.decode_raw(features['feature'], tf.float32),
                             [-1, args.data.dim_feature])
    label = tf.decode_raw(features['label'], tf.int32)
    aux_label = tf.decode_raw(features['aux_label'], tf.int32)

    if transform:
        feature = process_raw_feature(feature, args)

    return id, feature, label, aux_label


def readTFRecord_kdbert(dir_data, args, _shuffle=False, transform=False):
    """
    use for kd bert
    """
    list_filenames = fentch_filelist(dir_data)
    if _shuffle:
        shuffle(list_filenames)
    else:
        list_filenames.sort()

    num_epochs = 1 if args.mode == 'infer' else None
    filename_queue = tf.train.string_input_producer(
        list_filenames, num_epochs=num_epochs, shuffle=shuffle)

    reader_tfRecord = tf.TFRecordReader()
    _, serialized_example = reader_tfRecord.read(filename_queue)

    if args.data.withoutid:
        features = tf.parse_single_example(
            serialized_example,
            features={'feature': tf.FixedLenFeature([], tf.string),
                      'label': tf.FixedLenFeature([], tf.string),
                      'bert_feature': tf.FixedLenFeature([], tf.string)}
        )
        id = tf.convert_to_tensor([1], dtype=tf.int32)
    else:
        features = tf.parse_single_example(
            serialized_example,
            features={'feature': tf.FixedLenFeature([], tf.string),
                      'id': tf.FixedLenFeature([], tf.string),
                      'label': tf.FixedLenFeature([], tf.string),
                      'bert_feature': tf.FixedLenFeature([], tf.string)}
        )
        id = tf.decode_raw(features['id'], tf.string)
    if args.mode == 'train':
        feature = tf.reshape(tf.decode_raw(features['feature'], tf.float32),
            [-1, args.data.dim_feature])[:args.data.max_train_len, :]
    else:
        feature = tf.reshape(tf.decode_raw(features['feature'], tf.float32),
                             [-1, args.data.dim_feature])
    label = tf.decode_raw(features['label'], tf.int32)
    bert_feature = tf.decode_raw(features['bert_feature'], tf.float32)

    if transform:
        feature = process_raw_feature(feature, args)

    return id, feature, label, bert_feature


def readTFRecord_ctcbert(dir_data, args, _shuffle=False, transform=False):
    """
        use for ctc and kd bert
        """
    list_filenames = fentch_filelist(dir_data)
    if _shuffle:
        shuffle(list_filenames)
    else:
        list_filenames.sort()

    num_epochs = 1 if args.mode == 'infer' else None
    filename_queue = tf.train.string_input_producer(
        list_filenames, num_epochs=num_epochs, shuffle=shuffle)

    reader_tfRecord = tf.TFRecordReader()
    _, serialized_example = reader_tfRecord.read(filename_queue)

    if args.data.withoutid:
        features = tf.parse_single_example(
            serialized_example,
            features={'feature': tf.FixedLenFeature([], tf.string),
                      'label': tf.FixedLenFeature([], tf.string),
                      'aux_label': tf.FixedLenFeature([], tf.string),
                      'bert_feature': tf.FixedLenFeature([], tf.string)}
        )
        id = tf.convert_to_tensor([1], dtype=tf.int32)
    else:
        features = tf.parse_single_example(
            serialized_example,
            features={'feature': tf.FixedLenFeature([], tf.string),
                      'id': tf.FixedLenFeature([], tf.string),
                      'label': tf.FixedLenFeature([], tf.string),
                      'aux_label': tf.FixedLenFeature([], tf.string),
                      'bert_feature': tf.FixedLenFeature([], tf.string)}
        )
        id = tf.decode_raw(features['id'], tf.int32)
    if args.mode == 'train':
        feature = tf.reshape(tf.decode_raw(features['feature'], tf.float32),
                             [-1, args.data.dim_feature])[:args.data.max_train_len, :]
    else:
        feature = tf.reshape(tf.decode_raw(features['feature'], tf.float32),
                             [-1, args.data.dim_feature])
    aux_label = tf.decode_raw(features['aux_label'], tf.int32)
    label = tf.decode_raw(features['label'], tf.int32)
    if args.model.pt_decoder:
        bert_feature = tf.decode_raw(features['bert_feature'], tf.int32)
    else:
        bert_feature = tf.reshape(tf.decode_raw(features['bert_feature'], tf.float32),
                                  [-1, args.data.bert_feat_size])

    if transform:
        feature = process_raw_feature(feature, args)

    return id, feature, label, aux_label, bert_feature


def readTFRecord_ctcbertfull(dir_data, args, _shuffle=False, transform=False):
    """
            use for ctc and kd full
            """
    list_filenames = fentch_filelist(dir_data)
    if _shuffle:
        shuffle(list_filenames)
    else:
        list_filenames.sort()

    num_epochs = 1 if args.mode == 'infer' else None
    filename_queue = tf.train.string_input_producer(
        list_filenames, num_epochs=num_epochs, shuffle=shuffle)

    reader_tfRecord = tf.TFRecordReader()
    _, serialized_example = reader_tfRecord.read(filename_queue)
    if args.data.withoutid:
        features = tf.parse_single_example(
            serialized_example,
            features={'feature': tf.FixedLenFeature([], tf.string),
                      'label': tf.FixedLenFeature([], tf.string),
                      'aux_label': tf.FixedLenFeature([], tf.string),
                      'bertfull_feature': tf.FixedLenFeature([], tf.string)}
        )
        id = tf.convert_to_tensor([1], dtype=tf.int32)
    else:
        features = tf.parse_single_example(
            serialized_example,
            features={'feature': tf.FixedLenFeature([], tf.string),
                      'id': tf.FixedLenFeature([], tf.string),
                      'label': tf.FixedLenFeature([], tf.string),
                      'aux_label': tf.FixedLenFeature([], tf.string),
                      'bertfull_feature': tf.FixedLenFeature([], tf.string)}
        )
        id = tf.decode_raw(features['id'], tf.string)
    if args.mode == 'train':
        feature = tf.reshape(tf.decode_raw(features['feature'], tf.float32),
                             [-1, args.data.dim_feature])[:args.data.max_train_len, :]
    else:
        feature = tf.reshape(tf.decode_raw(features['feature'], tf.float32),
                             [-1, args.data.dim_feature])
    aux_label = tf.decode_raw(features['aux_label'], tf.int32)
    label = tf.decode_raw(features['label'], tf.int32)
    bertfull_feature = tf.reshape(tf.decode_raw(features['bertfull_feature'], tf.float32),
                              [-1, args.data.bert_feat_size])

    if transform:
        feature = process_raw_feature(feature, args)

    return id, feature, label, aux_label, bertfull_feature


def readTFRecord_multilingual(dir_data, args, _shuffle=False, transform=False):
    """
    use for multi-langual
    """
    list_filenames = fentch_filelist(dir_data)
    if _shuffle:
        shuffle(list_filenames)
    else:
        list_filenames.sort()

    num_epochs = 1 if args.mode == 'infer' else None
    filename_queue = tf.train.string_input_producer(
        list_filenames, num_epochs=num_epochs, shuffle=shuffle)

    reader_tfRecord = tf.TFRecordReader()
    _, serialized_example = reader_tfRecord.read(filename_queue)
    if args.data.withoutid:
        features = tf.parse_single_example(
            serialized_example,
            features={'feature': tf.FixedLenFeature([], tf.string),
                      'id': tf.FixedLenFeature([], tf.string),
                      'label': tf.FixedLenFeature([], tf.string),
                      'lang_label': tf.FixedLenFeature([], tf.string)}
        )
        id = tf.convert_to_tensor([1], dtype=tf.int32)
    else:
        features = tf.parse_single_example(
            serialized_example,
            features={'feature': tf.FixedLenFeature([], tf.string),
                      'id': tf.FixedLenFeature([], tf.string),
                      'label': tf.FixedLenFeature([], tf.string),
                      'lang_label': tf.FixedLenFeature([], tf.string)}
        )
        id = tf.decode_raw(features['id'], tf.string)
    if args.mode == 'train':
        feature = tf.reshape(tf.decode_raw(features['feature'], tf.float32),
            [-1, args.data.dim_feature])[:args.data.max_train_len, :]
    else:
        feature = tf.reshape(tf.decode_raw(features['feature'], tf.float32),
                             [-1, args.data.dim_feature])
    label = tf.decode_raw(features['label'], tf.int32)
    aux_label = tf.decode_raw(features['lang_label'], tf.int32)

    if transform:
        feature = process_raw_feature(feature, args)

    return id, feature, label, aux_label


def readTFRecord_multilingual_multilabel(dir_data, args, _shuffle=False, transform=False):
    """
    use for multi-langual with multi-label
    """
    list_filenames = fentch_filelist(dir_data)
    if _shuffle:
        shuffle(list_filenames)
    else:
        list_filenames.sort()

    num_epochs = 1 if args.mode == 'infer' else None
    filename_queue = tf.train.string_input_producer(
        list_filenames, num_epochs=num_epochs, shuffle=shuffle)

    reader_tfRecord = tf.TFRecordReader()
    _, serialized_example = reader_tfRecord.read(filename_queue)
    if args.data.withoutid:
        features = tf.parse_single_example(
                serialized_example,
                features={'feature': tf.FixedLenFeature([], tf.string),
                          'label': tf.FixedLenFeature([], tf.string),
                          'label_2': tf.FixedLenFeature([], tf.string),
                          'lang_label': tf.FixedLenFeature([], tf.string)}
            )
        id = tf.convert_to_tensor([1], dtype=tf.int32)
    else:
        features = tf.parse_single_example(
            serialized_example,
            features={'feature': tf.FixedLenFeature([], tf.string),
                      'id': tf.FixedLenFeature([], tf.string),
                      'label': tf.FixedLenFeature([], tf.string),
                      'label_2': tf.FixedLenFeature([], tf.string),
                      'lang_label': tf.FixedLenFeature([], tf.string)}
        )
    if args.mode == 'train':
        feature = tf.reshape(tf.decode_raw(features['feature'], tf.float32),
            [-1, args.data.dim_feature])[:args.data.max_train_len, :]
    else:
        feature = tf.reshape(tf.decode_raw(features['feature'], tf.float32),
                             [-1, args.data.dim_feature])
    id = tf.decode_raw(features['id'], tf.string)
    label = tf.decode_raw(features['label'], tf.int32)
    label_2 = tf.decode_raw(features['label_2'], tf.int32)
    aux_label = tf.decode_raw(features['lang_label'], tf.int32)

    if transform:
        feature = process_raw_feature(feature, args)

    return id, feature, label, label_2, aux_label


def process_raw_feature(seq_raw_features, args):
    # 1-D, 2-D

    if args.data.add_delta:
        seq_raw_features = tfAudio.add_delt(seq_raw_features)

    # Splice
    fea = tfAudio.splice(
        seq_raw_features,
        left_num=0,
        right_num=args.data.num_context)
    # downsample
    fea = tfAudio.down_sample(
        fea,
        rate=args.data.downsample,
        axis=0)
    fea.set_shape([None, args.data.dim_input])

    return fea


def fentch_filelist(dir_data):
    file_list = []
    for dir_split in str(dir_data).split(','):

        assert tf.io.gfile.isdir(dir_split)
        file_list = file_list + tf.io.gfile.glob(os.path.join(dir_split, '*.recode'))

    return file_list


class TFReader:
    def __init__(self, dir_tfdata, args, is_train=True):
        self.is_train = is_train
        self.args = args
        self.sess = None
        self.list_batch_size = self.args.list_batch_size
        self.list_bucket_boundaries = self.args.list_bucket_boundaries
        if args.model.use_multilabel:
            self.id, self.feat, self.label, self.aux_label = readTFRecord_multilabel(
                dir_tfdata,
                args,
                _shuffle=is_train,
                transform=True)
        elif args.model.use_bert:
            self.id, self.feat, self.label, self.bert_feature = readTFRecord_kdbert(
                dir_tfdata,
                args,
                _shuffle=is_train,
                transform=True)
        elif args.model.use_ctc_bert:
            self.id, self.feat, self.label, self.aux_label, self.bert_feature = readTFRecord_ctcbert(
                dir_tfdata,
                args,
                _shuffle=is_train,
                transform=True
            )

        elif args.model.use_ctc_bertfull:
           self.id, self.feat, self.label, self.aux_label, self.bertfull_feature = readTFRecord_ctcbertfull(
                dir_tfdata,
                args,
                _shuffle=is_train,
                transform=True
            )
        elif args.model.is_multilingual:
            self.id, self.feat, self.label, self.lang_label = readTFRecord_multilingual(
                dir_tfdata,
                args,
                _shuffle=is_train,
                transform=True)

        elif args.model.is_cotrain:
            self.id, self.feat, self.label, self.label_2, self.lang_label = readTFRecord_multilingual_multilabel(
                dir_tfdata,
                args,
                _shuffle=is_train,
                transform=True)

        else:
           self.id, self.feat, self.label = readTFRecord(
                dir_tfdata,
                args,
                _shuffle=is_train,
                transform=True)

    def __iter__(self):
        """It is only a demo! Using `fentch_batch_with_TFbuckets` in practice."""
        if not self.sess:
            raise NotImplementedError('please assign sess to the TFReader! ')

        for i in range(len(self.args.data.size_dev)):
            yield self.sess.run([self.feat, self.label])

    def fentch_batch(self):
        list_inputs = [self.feat, self.label, tf.shape(self.feat)[0], tf.shape(self.label)[0]]
        list_outputs = tf.train.batch(
            tensors=list_inputs,
            batch_size=16,
            num_threads=8,
            capacity=2000,
            dynamic_pad=True,
            allow_smaller_final_batch=True
        )
        seq_len_feats = tf.reshape(list_outputs[2], [-1])
        seq_len_label = tf.reshape(list_outputs[3], [-1])

        return list_outputs[0], list_outputs[1], seq_len_feats, seq_len_label

    def fentch_batch_bucket(self):
        """
        the input tensor length is not equal,
        so will add the len as a input tensor
        list_inputs: [tensor1, tensor2]
        added_list_inputs: [tensor1, tensor2, len_tensor1, len_tensor2]
        """
        list_inputs = [self.id, self.feat, self.label, tf.shape(self.feat)[0], tf.shape(self.label)[0]]
        _, list_outputs = tf.contrib.training.bucket_by_sequence_length(
            input_length=list_inputs[3],
            tensors=list_inputs,
            batch_size=self.list_batch_size,
            bucket_boundaries=self.list_bucket_boundaries,
            num_threads=64,
            bucket_capacities=[i*2 for i in self.list_batch_size],
            capacity=50,
            dynamic_pad=True,
            allow_smaller_final_batch=True)
        seq_len_feats = tf.reshape(list_outputs[3], [-1])
        seq_len_label = tf.reshape(list_outputs[4], [-1])
        return list_outputs[0], list_outputs[1], list_outputs[2], seq_len_feats, seq_len_label

    def fentch_multi_label_batch_bucket(self):
        """
        the input tensor length is not equal,
        so will add the len as a input tensor
        list_inputs: [tensor1, tensor2]
        added_list_inputs: [tensor1, tensor2, len_tensor1, len_tensor2]
        """
        list_inputs = [self.id, self.feat, self.label, self.aux_label,
                       tf.shape(self.feat)[0], tf.shape(self.label)[0], tf.shape(self.aux_label)[0]]
        _, list_outputs = tf.contrib.training.bucket_by_sequence_length(
            input_length=list_inputs[4],
            tensors=list_inputs,
            batch_size=self.args.list_batch_size,
            bucket_boundaries=self.args.list_bucket_boundaries,
            num_threads=64,
            bucket_capacities=[i*2 for i in self.args.list_batch_size],
            capacity=50,
            dynamic_pad=True,
            allow_smaller_final_batch=True)
        seq_len_feats = tf.reshape(list_outputs[4], [-1])
        seq_len_label = tf.reshape(list_outputs[5], [-1])
        seq_len_aux_label = tf.reshape(list_outputs[6], [-1])

        return list_outputs[0], list_outputs[1], list_outputs[2], list_outputs[3],\
seq_len_feats, seq_len_label, seq_len_aux_label

    def fentch_multi_label_batch_bucket_nmt(self):
        """
        the input tensor length is not equal,
        so will add the len as a input tensor
        list_inputs: [tensor1, tensor2]
        added_list_inputs: [tensor1, tensor2, len_tensor1, len_tensor2]
        """
        list_inputs = [self.id, self.feat, self.label, self.aux_label,
                       tf.shape(self.feat)[0], tf.shape(self.label)[0], tf.shape(self.aux_label)[0]]
        _, list_outputs = tf.contrib.training.bucket_by_sequence_length(
            input_length=list_inputs[5],
            tensors=list_inputs,
            batch_size=self.args.list_batch_size,
            bucket_boundaries=self.args.list_bucket_boundaries,
            num_threads=64,
            bucket_capacities=[i * 2 for i in self.args.list_batch_size],
            capacity=50,
            dynamic_pad=True,
            allow_smaller_final_batch=True)
        seq_len_feats = tf.reshape(list_outputs[4], [-1])
        seq_len_label = tf.reshape(list_outputs[5], [-1])
        seq_len_aux_label = tf.reshape(list_outputs[6], [-1])

        return list_outputs[0], list_outputs[1], list_outputs[2], list_outputs[3], \
               seq_len_feats, seq_len_label, seq_len_aux_label


    def fentch_kdbert_batch_bucket(self):
        """
        the input tensor length is not equal,
        so will add the len as a input tensor
        list_inputs: [tensor1, tensor2]
        added_list_inputs: [tensor1, tensor2, len_tensor1, len_tensor2]
        """
        list_inputs = [self.id, self.feat, self.label, self.bert_feature,
                       tf.shape(self.feat)[0], tf.shape(self.label)[0], tf.shape(self.bert_feature)[0]]
        _, list_outputs = tf.contrib.training.bucket_by_sequence_length(
            input_length=list_inputs[4],
            tensors=list_inputs,
            batch_size=self.args.list_batch_size,
            bucket_boundaries=self.args.list_bucket_boundaries,
            num_threads=64,
            bucket_capacities=[i*2 for i in self.args.list_batch_size],
            capacity=50,
            dynamic_pad=True,
            allow_smaller_final_batch=True)
        seq_len_feats = tf.reshape(list_outputs[4], [-1])
        seq_len_label = tf.reshape(list_outputs[5], [-1])
        seq_len_bert_feat = tf.reshape(list_outputs[6], [-1])

        return list_outputs[0], list_outputs[1], list_outputs[2], list_outputs[3],\
                seq_len_feats, seq_len_label, seq_len_bert_feat

    def fentch_ctcbert_batch_bucket(self):
        """
                the input tensor length is not equal,
                so will add the len as a input tensor
                list_inputs: [tensor1, tensor2]
                added_list_inputs: [tensor1, tensor2, len_tensor1, len_tensor2]
                """
        list_inputs = [self.id, self.feat, self.label, self.aux_label, self.bert_feature,
                       tf.shape(self.feat)[0], tf.shape(self.label)[0], tf.shape(self.aux_label)[0], tf.shape(self.bert_feature)[0]]
        _, list_outputs = tf.contrib.training.bucket_by_sequence_length(
            input_length=list_inputs[5],
            tensors=list_inputs,
            batch_size=self.args.list_batch_size,
            bucket_boundaries=self.args.list_bucket_boundaries,
            num_threads=64,
            bucket_capacities=[i * 2 for i in self.args.list_batch_size],
            capacity=50,
            dynamic_pad=True,
            allow_smaller_final_batch=True)
        seq_len_feats = tf.reshape(list_outputs[5], [-1])
        seq_len_label = tf.reshape(list_outputs[6], [-1])
        seq_len_aux_label = tf.reshape(list_outputs[7], [-1])
        seq_len_bert_feat = tf.reshape(list_outputs[8], [-1])

        return list_outputs[0], list_outputs[1], list_outputs[2], list_outputs[3], list_outputs[4],\
               seq_len_feats, seq_len_label, seq_len_aux_label, seq_len_bert_feat

    def fentch_ctcbertfull_batch_bucket(self):
        """
                        the input tensor length is not equal,
                        so will add the len as a input tensor
                        list_inputs: [tensor1, tensor2]
                        added_list_inputs: [tensor1, tensor2, len_tensor1, len_tensor2]
                        """
        list_inputs = [self.id, self.feat, self.label, self.aux_label, self.bertfull_feature,
                       tf.shape(self.feat)[0], tf.shape(self.label)[0], tf.shape(self.aux_label)[0], tf.shape(self.bertfull_feature)[0]]
        _, list_outputs = tf.contrib.training.bucket_by_sequence_length(
            input_length=list_inputs[5],
            tensors=list_inputs,
            batch_size=self.args.list_batch_size,
            bucket_boundaries=self.args.list_bucket_boundaries,
            num_threads=64,
            bucket_capacities=[i * 2 for i in self.args.list_batch_size],
            capacity=50,
            dynamic_pad=True,
            allow_smaller_final_batch=True)
        seq_len_feats = tf.reshape(list_outputs[5], [-1])
        seq_len_label = tf.reshape(list_outputs[6], [-1])
        seq_len_aux_label = tf.reshape(list_outputs[7], [-1])
        seq_len_bertfull_feat = tf.reshape(list_outputs[8], [-1])

        return list_outputs[0], list_outputs[1], list_outputs[2], list_outputs[3], list_outputs[4],\
               seq_len_feats, seq_len_label, seq_len_aux_label, seq_len_bertfull_feat

    def fentch_multilingual_batch_bucket(self):
        """
        the input tensor length is not equal,
        so will add the len as a input tensor
        list_inputs: [tensor1, tensor2]
        added_list_inputs: [tensor1, tensor2, len_tensor1, len_tensor2]
        """
        list_inputs = [self.id, self.feat, self.label, self.lang_label,
                       tf.shape(self.feat)[0], tf.shape(self.label)[0], tf.shape(self.lang_label)[0]]
        _, list_outputs = tf.contrib.training.bucket_by_sequence_length(
            input_length=list_inputs[4],
            tensors=list_inputs,
            batch_size=self.args.list_batch_size,
            bucket_boundaries=self.args.list_bucket_boundaries,
            num_threads=64,
            bucket_capacities=[i*2 for i in self.args.list_batch_size],
            capacity=50,
            dynamic_pad=True,
            allow_smaller_final_batch=True)
        seq_len_feats = tf.reshape(list_outputs[4], [-1])
        seq_len_label = tf.reshape(list_outputs[5], [-1])
        seq_len_lang_label = tf.reshape(list_outputs[6], [-1])

        return list_outputs[0], list_outputs[1], list_outputs[2], list_outputs[3],\
                seq_len_feats, seq_len_label, seq_len_lang_label

    def fentch_multilingual_multilabel_batch_bucket(self):
        """
        the input tensor length is not equal,
        so will add the len as a input tensor
        list_inputs: [tensor1, tensor2]
        added_list_inputs: [tensor1, tensor2, len_tensor1, len_tensor2]
        """
        list_inputs = [self.id, self.feat, self.label, self.label_2, self.lang_label,
                       tf.shape(self.feat)[0], tf.shape(self.label)[0], tf.shape(self.label_2)[0], tf.shape(self.lang_label)[0]]
        _, list_outputs = tf.contrib.training.bucket_by_sequence_length(
            input_length=list_inputs[5],
            tensors=list_inputs,
            batch_size=self.args.list_batch_size,
            bucket_boundaries=self.args.list_bucket_boundaries,
            num_threads=64,
            bucket_capacities=[i*2 for i in self.args.list_batch_size],
            capacity=50,
            dynamic_pad=True,
            allow_smaller_final_batch=True)
        seq_len_feats = tf.reshape(list_outputs[5], [-1])
        seq_len_label = tf.reshape(list_outputs[6], [-1])
        seq_len_label_2 = tf.reshape(list_outputs[7], [-1])
        seq_len_lang_label = tf.reshape(list_outputs[8], [-1])

        return list_outputs[0], list_outputs[1], list_outputs[2], list_outputs[3], list_outputs[4],\
                seq_len_feats, seq_len_label, seq_len_label_2, seq_len_lang_label

if __name__ == '__main__':
    # from configs.arguments import args
    from tqdm import tqdm
    import sys

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')
    # test_bucket_boundaries(args=args)
    # test_tfdata_bucket(args=args, num_threads=args.num_parallel)
    # test_queue()
