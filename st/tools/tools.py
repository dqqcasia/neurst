import numpy as np
import os
from random import random


class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    demo:
    a ={'length': 10, 'shape': (2,3)}
    config = AttrDict(a)
    config.length #10

    for i in read_tfdata_info(args.dirs.train.tfdata).items():
        args.data.train.i[0] = i[1]
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if type(self[item]) is dict:
            self[item] = AttrDict(self[item])
        return self[item]


class check_to_stop(object):
    def __init__(self):
        self.value_1 = 999
        self.value_2 = 999
        self.value_3 = 999

    def __call__(self, new_value):
        import sys

        self.value_1 = self.value_2
        self.value_2 = self.value_3
        self.value_3 = new_value

        if self.value_1 < self.value_2 < self.value_3 and new_value > self.value_2:
            print('force exit!')
            sys.exit()


def padding_list_seqs(list_seqs, dtype=np.float32, pad=0.):
    len_x = [len(s) for s in list_seqs]

    size_batch = len(list_seqs)
    maxlen = max(len_x)

    shape_feature = tuple()
    for s in list_seqs:
        if len(s) > 0:
            shape_feature = np.asarray(s).shape[1:]
            break

    x = (np.ones((size_batch, maxlen) + shape_feature) * pad).astype(dtype)
    for idx, s in enumerate(list_seqs):
        x[idx, :len(s)] = s

    return x, len_x


def pad_to_split(batch, num_split):
    num_pad = num_split - len(batch) % num_split
    if num_pad != 0:
        if batch.ndim > 1:
            pad = np.tile(np.expand_dims(batch[0,:], 0), [num_pad]+[1]*(batch.ndim-1))
        elif batch.ndim == 1:
            pad = np.asarray([batch[0]] * num_pad, dtype=batch[0].dtype)
        batch = np.concatenate([batch, pad], 0)

    return batch


def size_bucket_to_put(l, buckets):
    for i, l1 in enumerate(buckets):
        if l < l1: return i, l1
    return -1, 9999


def iter_filename(dataset_dir, suffix='*', sort=None):
    if not os.path.exists(dataset_dir):
        raise IOError("'%s' does not exist" % dataset_dir)
        exit()

    import glob
    iter_filename = glob.iglob(os.path.join(dataset_dir, suffix))

    if sort:
        SORTS = ['filesize_low_high', 'filesize_high_low', 'alpha', 'random']
        if sort not in SORTS:
            raise ValueError('sort must be one of [%s]', SORTS)
        reverse = False
        key = None
        if 'filesize' in sort:
            key = os.path.getsize
        if sort == 'filesize_high_low':
            reverse = True
        elif sort == 'random':
            key = lambda *args: random()

        iter_filename = iter(sorted(list(iter_filename), key=key, reverse=reverse))

    return iter_filename


class Sentence_iter(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.strip().split()


def sparse_tuple_from(sequences):
    """
    Create a sparse representention of ``sequences``.

    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.int32)
    shape = np.asarray([len(sequences), indices.max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


if __name__ == '__main__':
    checker = check_to_stop()
    for i in [5,4,3,3,2,1,1,2,2]:
        checker(i)
