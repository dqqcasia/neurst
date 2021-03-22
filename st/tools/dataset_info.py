#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
import sys


def get_tfdata_info(dir_tfdata, len_dataset, args, idx_init=300, dir_save_info='seq2seq/data'):
    """
    enlarge idx_init can shrink the num of buckets
    default is 150
    """
    print('get the dataset info')
    import tensorflow as tf
    from .tftools.tfRecord import readTFRecord, readTFRecord_ctcbert

    if args.model.use_ctc_bert:
        _, feat, label, _, _ = readTFRecord_ctcbert(dir_tfdata, args, transform=True)
    else:
        _, feat, label = readTFRecord(dir_tfdata, args, transform=True)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    # len_num_list = [0] * 20
    with tf.train.MonitoredTrainingSession(config=config) as sess:
        list_len = []
        for _ in tqdm(range(len_dataset)):
            feature = sess.run(feat)
            # if len(feature) > 2000:
            list_len.append(len(feature))

    histogram(list_len, dir_save_info)

    list_num = []
    list_length = []
    f_len_hist = dir_save_info + 'dataset_len_hist.txt'
    with open(f_len_hist) as f:
        for line in f:
            num, length = line.strip().split(':')
            list_num.append(int(num))
            list_length.append(int(length))

    def next_idx(idx, energy):
        for i in range(idx, len(list_num), 1):
            if list_length[i]*sum(list_num[idx+1:i+1]) > energy:
                return i-1
        return
    M = args.num_batch_tokens
    b0 = int(M / list_length[idx_init])
    k = b0/sum(list_num[:idx_init+1])
    energy = M/k

    list_batchsize = [b0]
    list_boundary = [list_length[idx_init]]

    idx = idx_init
    while idx < len(list_num):
        idx = next_idx(idx, energy)
        if not idx:
            break
        if idx == idx_init:
            print('enlarge the idx_init!')
            sys.exit()
        list_boundary.append(list_length[idx])
        list_batchsize.append(int(M / list_length[idx]))

    list_boundary.append(list_length[-1])
    list_batchsize.append(int(M/list_length[-1]))

    print('suggest boundaries: \n{}'.format(','.join(map(str, list_boundary))))
    print('corresponding batch size: \n{}'.format(','.join(map(str, list_batchsize))))


def histogram(list_len, dir_save=''):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel('Smarts')
    ax.set_ylabel('Probability')
    ax.set_title('Histogram dataset lenght')
    n, bins, patches = ax.hist(list_len, facecolor='g', alpha=0.75)
    fig.savefig(dir_save + 'hist_dataset_len')

    hist, edges = np.histogram(list_len, bins=(max(list_len)-min(list_len)+1))

    # save hist
    info_file = dir_save + 'dataset_len_hist.txt'
    with open(info_file, 'w') as fw:
        for num, edge in zip(hist, edges):
            fw.write('{}: {}\n'.format(str(num), str(int(np.ceil(edge)))))


if __name__ == '__main__':
    dataset = [2000,2001,2002,2003,2004]
    histogram(dataset)
