import numpy as np
import logging
from queue import Queue
import threading
import time
import collections
from random import shuffle, randint
from concurrent.futures import ThreadPoolExecutor
from st.tools.audio import audio2vector, process_raw_feature
from abc import ABCMeta, abstractmethod

from st.tools.tools import size_bucket_to_put
import codecs
import json

import tensorflow as tf

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class DataSet:
    __metaclass__ = ABCMeta

    def __iter__(self):
        """
        utility the __getitem__ to impliment the __iter__
        """
        for idx in range(len(self)):
            yield self[idx]

    @abstractmethod
    def __getitem__(self, idx):
        """
        sample = dataset_obj[idx]
        sample = dataset_obj(idx)
        """

    @abstractmethod
    def __len__(self):
        """
        the length of the dataset
        """

    def __call__(self, idx):

        return self.__getitem__(idx)


class ASRDataSet(DataSet):
    def __init__(self, list_files, args, _shuffle, transform):
        self.list_files = list_files
        self.transform = transform
        self.args = args
        self._shuffle = _shuffle
        self.token2idx, self.idx2token = args.token2idx, args.idx2token
        self.src_token2idx, self.src_idx2token = args.src_token2idx, args.src_idx2token
        self.wav2idx, self.idx2wav = args.wav2idx, args.idx2wav
        self.end_id = self.gen_end_id(self.token2idx)
        if _shuffle:
            self.shuffle_list_files()

    def gen_end_id(self, token2idx):
        if '<eos>' in token2idx.keys():
            eos_id = [token2idx['<eos>']]
        else:
            eos_id = []

        return eos_id

    def shuffle_list_files(self):
        shuffle(self.list_files)


class ASR_csv_DataSet(ASRDataSet):
    def __init__(self, list_files, args, _shuffle, transform):
        super().__init__(list_files, args, _shuffle, transform)
        self.list_utterances = self.gen_utter_list(list_files)
        if _shuffle:
            self.shuffle_utts()

    def __getitem__(self, idx):
        utterance = self.list_utterances[idx]

        wav, seq_label = utterance.strip().split('\t')
        try:
            fea = audio2vector(wav, self.args.data.dim_raw_input)
        except:
            return False

        if self.transform:
            fea = process_raw_feature(fea, self.args)
        if self.args.model.structure == "ctcModel":
            seq_label = np.array(
                [self.token2idx.get(word, self.token2idx['<unk>']) for word in seq_label.split(' ')],
                dtype=np.int32)
        else:
            seq_label = np.array(
                [self.token2idx.get(word, self.token2idx['<unk>'])
                for word in seq_label.split(' ')] + self.end_id,
                dtype=np.int32)
        if self.args.data.withoutid:
            wav = np.array(0, dtype=np.int32)
        else:
            wav = np.array(self.wav2idx.get(wav), dtype=np.int32)
        sample = {'id': wav, 'feature': fea, 'label': seq_label}

        return sample

    @staticmethod
    def gen_utter_list(list_files):
        list_utter = []
        for file in list_files:
            with tf.io.gfile.GFile(file, 'r') as f:
                list_utter.extend(f.readlines())
        return list_utter

    def shuffle_utts(self):
        shuffle(self.list_utterances)

    def __len__(self):
        return len(self.list_utterances)


class ASR_multilabel_csv_DataSet(ASRDataSet):

    def __init__(self, list_files, args, _shuffle, transform):
        super().__init__(list_files, args, _shuffle, transform)
        self.list_utterances = self.gen_utter_list(list_files)
        if _shuffle:
            self.shuffle_utts()

    def __getitem__(self, idx):
        utterance = self.list_utterances[idx]

        wav, seq_label, aux_label = utterance.strip().split('\t')
        try:
            fea = audio2vector(wav, self.args.data.dim_raw_input)
        except:
            return False
        if self.transform:
            fea = process_raw_feature(fea, self.args)

        seq_label = np.array(
            [self.token2idx.get(word, self.token2idx['<unk>'])
            for word in seq_label.split(' ')] + self.end_id,
            dtype=np.int32)
        if self.args.data.withoutid:
            wav = np.array(0, dtype=np.int32)
        else:
            wav = np.array(self.wav2idx.get(wav), dtype=np.int32)

        aux_label = np.array(
            [self.src_token2idx.get(word, self.src_token2idx['<unk>']) for word in aux_label.split(' ')],
            dtype=np.int32)

        sample = {'id': wav, 'feature': fea, 'label': seq_label, 'aux_label': aux_label}

        return sample

    @staticmethod
    def gen_utter_list(list_files):
        list_utter = []
        for file in list_files:
            with tf.io.gfile.GFile(file, 'r') as f:
                list_utter.extend(f.readlines())
        return list_utter

    def shuffle_utts(self):
        shuffle(self.list_utterances)

    def __len__(self):
        return len(self.list_utterances)


class ASR_kdbert_csv_DataSet(ASRDataSet):

    def __init__(self, list_files, args, _shuffle, transform):
        super().__init__(list_files, args, _shuffle, transform)
        self.list_utterances = self.gen_utter_list(list_files)
        if _shuffle:
            self.shuffle_utts()

    def __getitem__(self, idx):
        utterance = self.list_utterances[idx]
        wav, seq_label, bert_feature_file = utterance.strip().split('\t')
        bert_feature = self.get_bert_feature(bert_feature_file)
        try:
            fea = audio2vector(wav, self.args.data.dim_raw_input)
        except:
            return False
        if self.transform:
            fea = process_raw_feature(fea, self.args)

        seq_label = np.array(
            [self.token2idx.get(word, self.token2idx['<unk>'])
            for word in seq_label.split(' ')] + self.end_id,
            dtype=np.int32)

        if self.args.data.withoutid:
            wav = np.array(0, dtype=np.int32)
        else:
            wav = np.array(self.wav2idx.get(wav), dtype=np.int32)

        bert_feature = np.array(bert_feature, dtype=np.float32)

        sample = {'id': wav, 'feature': fea, 'label': seq_label, 'bert_feature':bert_feature}

        return sample

    @staticmethod
    def gen_utter_list(list_files):
        list_utter = []
        for file in list_files:
            with tf.io.gfile.GFile(file, 'r') as f:
                list_utter.extend(f.readlines())
        return list_utter

    def shuffle_utts(self):
        shuffle(self.list_utterances)

    def get_bert_feature(self, path):
        file = codecs.open(path, 'r', 'utf8')
        line = file.readline()
        feat_dict = json.loads(line)
        cls_feature = feat_dict["features"][0]["layers"][0]["values"]

        return cls_feature

    def __len__(self):
        return len(self.list_utterances)


class ASR_ctcbert_csv_dataset(ASRDataSet):

    def __init__(self, list_files, args, _shuffle, transform):
        super().__init__(list_files, args, _shuffle, transform)
        self.list_utterances = self.gen_utter_list(list_files)
        self.args = args

        if _shuffle:
            self.shuffle_utts()

    def __getitem__(self, idx):
        utterance = self.list_utterances[idx]
        wav, seq_label, aux_label, bert_feature_file = utterance.strip().split('\t')
        if self.args.model.kd.use_bert_type == 'Full':
            bert_feature = self.get_bert_feature_full(bert_feature_file)
        elif self.args.model.kd.use_bert_type == 'Cls':
            bert_feature = self.get_bert_feature(bert_feature_file)


        try:
            fea = audio2vector(wav, self.args.data.dim_raw_input)
        except:
            return False

        if self.transform:
            fea = process_raw_feature(fea, self.args)

        seq_label = np.array(
            [self.token2idx.get(word, self.token2idx['<unk>'])
            for word in seq_label.split(' ')] + self.end_id,
            dtype=np.int32)

        if self.args.data.withoutid:
            wav = np.array(0, dtype=np.int32)
        else:
            wav = np.array(self.wav2idx.get(wav), dtype=np.int32)

        aux_label = np.array(
            [self.src_token2idx.get(word, self.src_token2idx['<unk>'])
             for word in aux_label.split(' ')] + self.end_id,
            dtype=np.int32)

        bert_feature = np.array(bert_feature, dtype=np.float32)

        sample = {'id': wav, 'feature': fea, 'label': seq_label, 'aux_label': aux_label, 'bert_feature':bert_feature}

        return sample

    @staticmethod
    def gen_utter_list(list_files):

        list_utter = []
        for file in list_files:
            with tf.io.gfile.GFile(file, 'r') as f:
                list_utter.extend(f.readlines())
        return list_utter

    def shuffle_utts(self):
        shuffle(self.list_utterances)

    def get_bert_feature(self, path):
        file = codecs.open(path, 'r', 'utf8')
        line = file.readline()
        feat_dict = json.loads(line)
        cls_feature = feat_dict["features"][0]["layers"][0]["values"]
        return cls_feature

    def get_bert_feature_full(self, path):
        file = codecs.open(path, 'r', 'utf8')
        line = file.readline()
        feat_dict = json.loads(line)
        bert_feat = [feat_dict["features"][i]["layers"][0]["values"] for i in range(len(feat_dict["features"]))]
        return bert_feat

    def __len__(self):
        return len(self.list_utterances)


class ASR_ctcbertfull_csv_dataset(ASRDataSet):

    def __init__(self, list_files, args, _shuffle, transform):
        super().__init__(list_files, args, _shuffle, transform)
        self.list_utterances = self.gen_utter_list(list_files)
        if _shuffle:
            self.shuffle_utts()

    def __getitem__(self, idx):
        utterance = self.list_utterances[idx]
        wav, seq_label, aux_label, bert_feature_file, bert_fullfeature_file = utterance.strip().split('\t')
        bert_feature = self.get_bert_feature(bert_feature_file)
        bertfull_feature =  self.get_bertfull_feature(bert_fullfeature_file)
        try:
            fea = audio2vector(wav, self.args.data.dim_raw_input)
        except:
            return False
        if self.transform:
            fea = process_raw_feature(fea, self.args)

        seq_label = np.array(
            [self.token2idx.get(word, self.token2idx['<unk>'])
            for word in seq_label.split(' ')] + self.end_id,
            dtype=np.int32)
        if self.args.data.withoutid:
            wav = np.array(0, dtype=np.int32)
        else:
            wav = np.array(self.wav2idx.get(wav), dtype=np.int32)
        aux_label = np.array(
            [self.src_token2idx.get(word, self.src_token2idx['<unk>'])
             for word in aux_label.split(' ')] + self.end_id,
            dtype=np.int32)

        bert_feature = np.array(bert_feature, dtype=np.float32)

        bertfull_feature = np.array(bertfull_feature, dtype=np.float32)

        sample = {'id': wav, 'feature': fea, 'label': seq_label, 'aux_label': aux_label, 'bert_feature': bert_feature, 'bertfull_feature': bertfull_feature}

        return sample

    @staticmethod
    def gen_utter_list(list_files):

        list_utter = []
        for file in list_files:
            with tf.io.gfile.GFile(file, 'r') as f:
                list_utter.extend(f.readlines())
        return list_utter

    def shuffle_utts(self):
        shuffle(self.list_utterances)

    def get_bert_feature(self, path):
        file = codecs.open(path, 'r', 'utf8')
        line = file.readline()
        feat_dict = json.loads(line)
        cls_feature = feat_dict["features"][0]["layers"][0]["values"]
        return cls_feature

    def get_bertfull_feature(self, path):
        file = codecs.open(path, 'r', 'utf8')
        line = file.readline()
        feat_dict = json.loads(line)
        bert_feat = [feat_dict["features"][i]["layers"][0]["values"] for i in range(len(feat_dict["features"]))]
        return bert_feat

    def __len__(self):
        return len(self.list_utterances)


class ASR_multilingual_csv_DataSet(ASRDataSet):

    def __init__(self, list_files, args, _shuffle, transform):
        super().__init__(list_files, args, _shuffle, transform)
        self.list_utterances = self.gen_utter_list(list_files)

        if _shuffle:
            self.shuffle_utts()

    def __getitem__(self, idx):
        utterance = self.list_utterances[idx]

        wav, seq_label, lang_label = utterance.strip().split('\t')
        try:
            fea = audio2vector(wav, self.args.data.dim_raw_input)
        except:
            return False

        if self.transform:
            fea = process_raw_feature(fea, self.args)

        seq_label = np.array(
            [self.token2idx.get(word, self.token2idx['<unk>'])
            for word in seq_label.split(' ')] + self.end_id,
            dtype=np.int32)
        if self.args.data.withoutid:
            wav = np.array(0, dtype=np.int32)
        else:
            wav = np.array(self.wav2idx.get(wav), dtype=np.int32)

        lang_label = np.array([self.token2idx.get(lang_label.strip())], dtype=np.int32)

        sample = {'id': wav, 'feature': fea, 'label': seq_label, 'lang_label': lang_label}

        return sample

    @staticmethod
    def gen_utter_list(list_files):
        list_utter = []
        for file in list_files:
            with tf.io.gfile.GFile(file, 'r') as f:
                list_utter.extend(f.readlines())
        return list_utter

    def shuffle_utts(self):
        shuffle(self.list_utterances)

    def __len__(self):
        return len(self.list_utterances)


class ASR_multilingual_multilabel_csv_DataSet(ASRDataSet):

    def __init__(self, list_files, args, _shuffle, transform):
        super().__init__(list_files, args, _shuffle, transform)
        self.list_utterances = self.gen_utter_list(list_files)

        if _shuffle:
            self.shuffle_utts()

    def __getitem__(self, idx):
        utterance = self.list_utterances[idx]

        wav, seq_label, seq_label_2, lang_label = utterance.strip().split('\t')
        try:
            fea = audio2vector(wav, self.args.data.dim_raw_input)
        except:
            return False

        if self.transform:
            fea = process_raw_feature(fea, self.args)

        seq_label = np.array(
            [self.token2idx.get(word, self.token2idx['<unk>'])
            for word in seq_label.split(' ')] + self.end_id,
            dtype=np.int32)
        if self.args.data.withoutid:
            wav = np.array(0, dtype=np.int32)
        else:
            wav = np.array(self.wav2idx.get(wav), dtype=np.int32)
        seq_label_2 = np.array(
            [self.src_token2idx.get(word, self.src_token2idx['<unk>'])
             for word in seq_label_2.split(' ')] + self.end_id,
            dtype=np.int32)


        lang_label = np.array([self.token2idx.get(lang_label.strip())], dtype=np.int32)

        sample = {'id': wav, 'feature': fea, 'label': seq_label, 'label_2': seq_label_2, 'lang_label': lang_label}

        return sample

    @staticmethod
    def gen_utter_list(list_files):
        list_utter = []
        for file in list_files:
            with tf.io.gfile.GFile(file, 'r') as f:
                list_utter.extend(f.readlines())
        return list_utter

    def shuffle_utts(self):
        shuffle(self.list_utterances)

    def __len__(self):
        return len(self.list_utterances)


class PTDEC_csv_dataset(ASRDataSet):

    def __init__(self, list_files, args, _shuffle, transform):
        super().__init__(list_files, args, _shuffle, transform)
        self.list_utterances = self.gen_utter_list(list_files)
        self.args = args

        if _shuffle:
            self.shuffle_utts()

    def __getitem__(self, idx):
        utterance = self.list_utterances[idx]
        seq_label, seq_label_2 = utterance.strip().split('\t')

        seq_label = np.array(
            [self.token2idx.get(word, self.token2idx['<unk>'])
            for word in seq_label.split(' ')] + self.end_id,
            dtype=np.int32)

        seq_label_2 = np.array(
            [self.src_token2idx.get(word, self.src_token2idx['<unk>'])
             for word in seq_label_2.split(' ')] + self.end_id,
            dtype=np.int32)

        aux_label = np.array([0], dtype=np.int32)

        fea = np.zeros((np.shape(seq_label)[0], self.args.data.dim_raw_input), dtype=np.float32)

        wav = np.array([0], dtype=np.int32)

        sample = {'id': wav, 'feature': fea, 'label': seq_label, 'aux_label': aux_label, 'bert_feature': seq_label_2}

        return sample

    @staticmethod
    def gen_utter_list(list_files):

        list_utter = []
        for file in list_files:
            with tf.io.gfile.GFile(file, 'r') as f:
                list_utter.extend(f.readlines())
        return list_utter

    def shuffle_utts(self):
        shuffle(self.list_utterances)

    def get_bert_feature(self, path):
        file = codecs.open(path, 'r', 'utf8')
        line = file.readline()
        feat_dict = json.loads(line)
        cls_feature = feat_dict["features"][0]["layers"][0]["values"]
        return cls_feature

    def get_bert_feature_full(self, path):
        file = codecs.open(path, 'r', 'utf8')
        line = file.readline()
        feat_dict = json.loads(line)
        bert_feat = [feat_dict["features"][i]["layers"][0]["values"] for i in range(len(feat_dict["features"]))]
        return bert_feat

    def __len__(self):
        return len(self.list_utterances)


class LMDataSet(DataSet):
    """
    dataset for language model. Refer to the PTB dataset
    """
    def __init__(self, list_files, args, _shuffle):
        self.list_files = list_files
        self.args = args
        self._shuffle = _shuffle
        self.token2idx, self.idx2token = args.token2idx, args.idx2token
        self.end_id = self.token2idx['<eos>'] if '<eos>' in self.token2idx else None
        self.start_id = self.token2idx['<sos>'] if '<sos>' in self.token2idx else self.token2idx['<blk>']
        if _shuffle:
            shuffle(self.list_files)
        self.size_dataset = self.get_size()

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return self.size_dataset

    def get_size(self):
        num_lines = 0
        for filename in self.list_files:
            num_lines += sum(1 for line in open(filename))

        return num_lines

    def __iter__(self):
        # while True:
        for filename in self.list_files:
            with open(filename) as f:
                for line in f:
                    line = line.strip().split()
                    if len(line) > self.args.list_bucket_boundaries[-1]:
                        continue
                    text_ids = [self.token2idx[word] for word in line]
                    src_ids = [self.start_id] + (text_ids if self.end_id else text_ids[:-1])
                    tar_ids = (text_ids + [self.end_id]) if self.end_id else text_ids
                    sample = {'feature': src_ids, 'label': tar_ids}
                    yield sample


class FakeDataSet(DataSet):
    def __init__(self):
        self.dim_feature = 3

    def __getitem__(self, idx):
        sample = {}
        sample['label'] = np.random.randint(self.dim_feature, size=randint(5, 10), dtype=np.int32)
        sample['feature'] = self.embedding(sample['label'])

        return sample

    def __len__(self):

        return 100

    def embedding(self, list_idx):
        list_embeded = []
        for idx in list_idx:
            embeded = np.zeros([self.dim_feature], dtype=np.float32)
            embeded[idx] = 1
            list_embeded.append(embeded)

        return list_embeded


class SimpleDataLoader:
    def __init__(self, dataset, num_loops=1, batch_size=10):
        self.dataset = dataset
        self.num_loops = num_loops
        self.batch_size = batch_size
        self.list_seq_features = []
        self.list_seq_labels = []

    def __iter__(self):
        return self.next_batch(self.batch_size)

    def next_batch(self, size_batch):
        for _ in range(self.num_loops):
            for sample in self.dataset:
                seq_features, seq_labels = sample['feature'], sample['label']

                self.list_seq_features.append(seq_features)
                self.list_seq_labels.append(seq_labels)

                if len(self.list_seq_features) >= size_batch:
                    yield self.padding_list_seq_with_labels(self.list_seq_features, self.list_seq_labels)
                    self.list_seq_features = []
                    self.list_seq_labels = []

    @staticmethod
    def padding_list_seqs(list_seqs, dtype=np.float32, pad=0.):
        '''
        Pads each sequence to the same length of the longest sequence.
        If maxlen is provided, any sequence longer than maxlen is truncated to
        maxlen. Truncation happens.

        Args:
            list_seqs: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            pad: float, value to pad the list_seqs to the desired value.

        Returns:
            numpy.ndarray: Padded list_seqs shape = (number_of_list_seqs, maxlen)
            list: original sequence lengths
        DEmo:
            >> padding_list_seqs([[21, 11, 3], [31,1]])
            >> (array([[ 21.,  11.,   3.],
                [ 31.,   1.,   0.]], dtype=float32), [3, 2])
        '''
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

    @staticmethod
    def padding_list_seq_with_labels(list_seqs_features,
                                     list_seqs_labels,
                                     dtype=np.float32,
                                     value1=0.,
                                     value2=0):
        x, len_x = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_features,
            dtype=dtype,
            pad=value1)
        y, len_y = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_labels,
            dtype=np.int32,
            pad=value2)

        return [x, y, len_x, len_y]


class DataLoader(SimpleDataLoader):
    __metaclass__ = ABCMeta

    '''
    Train/test/dev dataset API for loading via threads and delivering batches.
    '''
    def __init__(self, dataset, args, num_loops=1, num_thread=4, size_queue=2000):
        super().__init__(dataset, num_loops)
        self.args = args
        self.num_thread = num_thread
        self.queue_sample = Queue(maxsize=size_queue)

        self.thread_queue_put = threading.Thread(target=self.feed_queue)
        self.thread_queue_put.daemon = True

        self.num_batch_tokens = args.num_batch_tokens
        self.bucket_boundaries = args.bucket_boundaries

    @abstractmethod
    def __iter__(self):
        '''
        return a iterator of seq, which is used to fentch a batch(with or without bucket)
        yield (seq_features, seq_labels)
        '''

    def batch_with_buckets(self):
        '''
        use the iter_seq:
        ```python
        args.size_bucket_start =
        args.size_bucket_end =
        args.size_bucket_gap =
        args.self.num_batch_token =
        dataReader = DataLoader(...)
        for batch in DataLoader.fentch_batch_with_buckets(dataReader)
            ...
        ```
        caches:
            {5: [[], [], 0],
             8: [[], [], 0],
            11: [[], [], 0],
            14: [[], [], 0],
            17: [[], [], 0]}
            id: [list_feats, list_labels, num_frame]
        len(caches[bucket][0]) is the batch length, i.e. the num of sents in a batch,
        while caches[bucket][1] is the num of tokens in a batch
        '''
        buckets = self.args.list_bucket_boundaries
        # max_length = buckets[-1]
        caches = collections.defaultdict(lambda: [[], [], 0])

        for _ in range(self.num_loops):
            for sample in self.dataset:
                if not sample: continue
                seq_features, seq_labels = sample['feature'], sample['label']
                # assert len(seq_features) == len(seq_labels)
                id_bucket, bucket = size_bucket_to_put(len(seq_features), buckets)
                if bucket is None:
                    continue
                caches[bucket][0].append(seq_features)
                caches[bucket][1].append(seq_labels)

                caches[bucket][2] += 1
                if caches[bucket][2] >= self.args.list_batch_size[id_bucket]:
                    batch = (caches[bucket][0], caches[bucket][1])
                    yield self.padding_list_seq_with_labels(*batch)
                    caches[bucket] = [[], [], 0]

        # Clean remain samples.
        for bucket in buckets:
            if caches[bucket][2] > 0:
                batch = (caches[bucket][0], caches[bucket][1])
                yield self.padding_list_seq_with_labels(*batch)
                caches[bucket] = [[], [], 0]
                # logging.info('empty the bucket {}'.format(bucket))

    def feed_queue(self):
        logging.info('enter the feed queue thread!')
        with ThreadPoolExecutor(self.num_thread) as ex:
            for idx in range(0, len(self.dataset)-self.num_thread, self.num_thread):
                batch_samples = ex.map(self.dataset, range(idx, idx+self.num_thread))
                # logging.info('add success!')
                [self.queue_sample.put(sample) for sample in batch_samples]

        self.dataset.shuffle_list_files()

    def bucket_with_queue(self):
        '''
        caches: {bucket_size: [list_feats, list_labels, num_frame]}
        '''
        self.thread_queue_put.start()
        logging.info('the activate num threads to prepare data is: {}'.format(threading.active_count()-2))
        index_loop = 0
        # feed_queue()
        buckets = self.args.list_bucket_boundaries
        batch_size = self.args.list_batch_size

        # max_length = buckets[-1]
        caches = collections.defaultdict(lambda: [[], [], 0])

        logging.info("size of the dataset: {}".format(len(self.dataset)))

        while True:
            sample = self.queue_sample.get()
            seq_features, seq_labels = sample['feature'], sample['label']

            # assert len(seq_features) == len(seq_labels)
            id_bucket, bucket = size_bucket_to_put(len(seq_features), buckets)
            if bucket is None:
                continue
            caches[bucket][0].append(seq_features)
            caches[bucket][1].append(seq_labels)
            caches[bucket][2] += 1

            if caches[bucket][2] >= batch_size[id_bucket]:
                batch = (caches[bucket][0], caches[bucket][1])
                yield self.padding_list_seq_with_labels(*batch)
                caches[bucket] = [[], [], 0]

            if self.queue_sample.empty():
                if threading.active_count() > 2:
                    logging.info('waitting for sample into the queue...')

                    time.sleep(3)
                elif index_loop < self.num_loops-1:
                    index_loop +=1

                    self.thread_queue_put.join()

                    self.thread_queue_put = threading.Thread(target=self.feed_queue)
                    self.thread_queue_put.start()
                    logging.info('***=======  loop {}/{} for the dataset  =======***'.format(index_loop+1, self.num_loops))
                else:
                    logging.info('finish iter dataset {} times'.format(self.num_loops))
                    break

        # Clean remain samples.
        for bucket in buckets:
            if caches[bucket][2] > 0:
                batch = (caches[bucket][0], caches[bucket][1])
                yield self.padding_list_seq_with_labels(*batch)
                caches[bucket] = [[], [], 0]


        self.thread_queue_put.join()

    def batch_with_map(self, size_batch):
        with ThreadPoolExecutor(self.num_thread) as ex:
            for idx in range(0, len(self.dataset)-size_batch, size_batch):
                batch_samples = ex.map(self.dataset, range(idx, idx+size_batch))

                self.list_seq_features = []
                self.list_seq_labels = []

                for sample in batch_samples:
                    if not sample: continue
                    seq_features, seq_labels = sample['feature'], sample['label']

                    self.list_seq_features.append(seq_features)
                    self.list_seq_labels.append(seq_labels)

                yield self.padding_list_seq_with_labels(
                    self.list_seq_features,
                    self.list_seq_labels)

    def batch_with_tfReader(self, batch_size):
        for _ in range(len(self)*self.num_loops):
            seq_features, seq_labels = self.sess.run([self.feat, self.label])

            self.list_seq_features.append(seq_features)
            self.list_seq_labels.append(seq_labels)

            if len(self.list_seq_labels) >= batch_size:
                yield self.padding_list_seq_with_labels(
                    self.list_seq_features,
                    self.list_seq_labels)
                self.list_seq_features = []
                self.list_seq_labels = []

        logging.info("clean the rest of data")
        if len(self.list_seq_features) > 0:
            yield self.padding_list_seq_with_labels(
                self.list_seq_features,
                self.list_seq_labels)
            self.list_seq_features = []
            self.list_seq_labels = []

    def batch_with_tfReader_buckets(self):
        buckets = self.args.list_bucket_boundaries

        caches = collections.defaultdict(lambda: [[], [], 0])
        for _ in range(len(self)*self.num_loops):
            seq_features, seq_labels = self.sess.run([self.feat, self.label])

            id_bucket, bucket = size_bucket_to_put(len(seq_features), buckets)
            if bucket is None:
                continue
            caches[bucket][0].append(seq_features)
            caches[bucket][1].append(seq_labels)

            caches[bucket][2] += 1
            if caches[bucket][2] >= self.args.list_batch_size[id_bucket]:
                batch = (caches[bucket][0], caches[bucket][1])
                yield self.padding_list_seq_with_labels(*batch)
                caches[bucket] = [[], [], 0]

        # Clean remain samples.
        for bucket in buckets:
            if caches[bucket][2] > 0:
                batch = (caches[bucket][0], caches[bucket][1])
                yield self.padding_list_seq_with_labels(*batch)
                caches[bucket] = [[], [], 0]



class ASRDataLoader(DataLoader):
    def __init__(self, dataset, args, feat, label, batch_size, num_loops, num_thread=4, size_queue=2000):
        super().__init__(dataset, args, num_loops=num_loops, num_thread=num_thread, size_queue=size_queue)
        self.sess = None
        self.feat = feat
        self.label = label
        self.size_dataset = len(dataset)
        self.batch_size = batch_size

    def __iter__(self):

        return self.batch_with_tfReader_buckets()

    def __len__(self):
        return self.size_dataset


class ASRMultiLabelDataLoader(DataLoader):
    def __init__(self, dataset, args, feat, label, auxlabel, batch_size, num_loops, num_thread=4, size_queue=2000):
        super().__init__(dataset, args, num_loops=num_loops, num_thread=num_thread, size_queue=size_queue)
        self.sess = None
        self.feat = feat
        self.label = label
        self.auxlabel = auxlabel
        self.list_seq_auxlabels = []
        self.size_dataset = len(dataset)
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(len(self)*self.num_loops):
            seq_features, seq_labels, seq_auxlabels = self.sess.run([self.feat, self.label, self.auxlabel])

            self.list_seq_features.append(seq_features)
            self.list_seq_labels.append(seq_labels)
            self.list_seq_auxlabels.append(seq_auxlabels)

            if len(self.list_seq_labels) >= self.batch_size:
                yield self.padding_list_seq_with_multilabels(
                    self.list_seq_features,
                    self.list_seq_labels,
                    self.list_seq_auxlabels)
                self.list_seq_features = []
                self.list_seq_labels = []
                self.list_seq_auxlabels = []

        logging.info("clean the rest of data")
        if len(self.list_seq_features) > 0:
            yield self.padding_list_seq_with_multilabels(
                self.list_seq_features,
                self.list_seq_labels,
                self.list_seq_auxlabels)
            self.list_seq_features = []
            self.list_seq_labels = []
            self.list_seq_auxlabels = []

    def __len__(self):
        return self.size_dataset

    @staticmethod
    def padding_list_seq_with_multilabels(list_seqs_features,
                                          list_seqs_labels,
                                          list_seqs_auxlabels,
                                          dtype=np.float32,
                                          value1=0.,
                                          value2=0):
        x, len_x = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_features,
            dtype=dtype,
            pad=value1)
        y, len_y = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_labels,
            dtype=np.int32,
            pad=value2)
        y2, len_y2 = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_auxlabels,
            dtype=np.int32,
            pad=value2)

        return [x, y, y2, len_x, len_y, len_y2]


class ASRKdBertDataLoader(DataLoader):
    def __init__(self, dataset, args, feat, label, bertfeat, batch_size, num_loops, num_thread=4, size_queue=2000):
        super().__init__(dataset, args, num_loops=num_loops, num_thread=num_thread, size_queue=size_queue)
        self.sess = None
        self.feat = feat
        self.label = label
        self.bertfeat = bertfeat
        self.list_seq_bertfeats = []
        self.size_dataset = len(dataset)
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(len(self)*self.num_loops):
            seq_features, seq_labels, seq_bertfeats = self.sess.run([self.feat, self.label, self.bertfeat])

            self.list_seq_features.append(seq_features)
            self.list_seq_labels.append(seq_labels)
            self.list_seq_bertfeats.append(seq_bertfeats)

            if len(self.list_seq_labels) >= self.batch_size:
                yield self.padding_list_seq_with_multilabels(
                    self.list_seq_features,
                    self.list_seq_labels,
                    self.list_seq_bertfeats)
                self.list_seq_features = []
                self.list_seq_labels = []
                self.list_seq_bertfeats = []

        logging.info("clean the rest of data")
        if len(self.list_seq_features) > 0:
            yield self.padding_list_seq_with_multilabels(
                self.list_seq_features,
                self.list_seq_labels,
                self.list_seq_bertfeats)
            self.list_seq_features = []
            self.list_seq_labels = []
            self.list_seq_bertfeats = []

    def __len__(self):
        return self.size_dataset

    @staticmethod
    def padding_list_seq_with_multilabels(list_seqs_features,
                                          list_seqs_labels,
                                          list_seqs_auxlabels,
                                          dtype=np.float32,
                                          value1=0.,
                                          value2=0):
        x, len_x = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_features,
            dtype=dtype,
            pad=value1)
        y, len_y = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_labels,
            dtype=np.int32,
            pad=value2)
        y2, len_y2 = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_auxlabels,
            dtype=np.int32,
            pad=value2)

        return [x, y, y2, len_x, len_y, len_y2]


class ASRCtcBertDataLoader(DataLoader):
    def __init__(self, dataset, args, feat, label, bertfeat, batch_size, num_loops, num_thread=4, size_queue=2000):
        super().__init__(dataset, args, num_loops=num_loops, num_thread=num_thread, size_queue=size_queue)
        self.sess = None
        self.feat = feat
        self.label = label
        self.auxlabel = label
        self.bertfeat = bertfeat
        self.list_seq_bertfeats = []
        self.list_seq_auxlabels = []
        self.size_dataset = len(dataset)
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(len(self)*self.num_loops):
            seq_features, seq_labels, seq_auxlabels, seq_bertfeats = self.sess.run([self.feat, self.label, self.auxlabel, self.bertfeat])

            self.list_seq_features.append(seq_features)
            self.list_seq_labels.append(seq_labels)
            self.list_seq_auxlabels.append(seq_auxlabels)
            self.list_seq_bertfeats.append(seq_bertfeats)

            if len(self.list_seq_labels) >= self.batch_size:
                yield self.padding_list_seq_with_multilabels(
                    self.list_seq_features,
                    self.list_seq_labels,
                    self.list_seq_auxlabels,
                    self.list_seq_bertfeats)
                self.list_seq_features = []
                self.list_seq_labels = []
                self.list_seq_auxlabels = []
                self.list_seq_bertfeats = []

        logging.info("clean the rest of data")
        if len(self.list_seq_features) > 0:
            yield self.padding_list_seq_with_multilabels(
                self.list_seq_features,
                self.list_seq_labels,
                self.list_seq_auxlabels,
                self.list_seq_bertfeats)
            self.list_seq_features = []
            self.list_seq_labels = []
            self.list_seq_auxlabels = []
            self.list_seq_bertfeats = []

    def __len__(self):
        return self.size_dataset

    @staticmethod
    def padding_list_seq_with_multilabels(list_seqs_features,
                                          list_seqs_labels,
                                          list_seqs_auxlabels,
                                          list_seqs_bertfeats,
                                          dtype=np.float32,
                                          value1=0.,
                                          value2=0):
        x, len_x = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_features,
            dtype=dtype,
            pad=value1)
        y, len_y = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_labels,
            dtype=np.int32,
            pad=value2)
        y2, len_y2 = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_auxlabels,
            dtype=np.int32,
            pad=value2)

        y3, len_y3 = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_bertfeats,
            dtype=np.int32,
            pad=value2
            )

        return [x, y, y2, y3, len_x, len_y, len_y2, len_y3]


class ASRCtcBertFullDataLoader(DataLoader):
    def __init__(self, dataset, args, feat, label, bertfullfeat, batch_size, num_loops, num_thread=4, size_queue=2000):
        super().__init__(dataset, args, num_loops=num_loops, num_thread=num_thread, size_queue=size_queue)
        self.sess = None
        self.feat = feat
        self.label = label
        self.auxlabel = label
        self.bertfullfeat = bertfullfeat
        self.list_seq_bertfeats = []
        self.list_seq_auxlabels = []
        self.size_dataset = len(dataset)
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(len(self)*self.num_loops):
            seq_features, seq_labels, seq_auxlabels, seq_bertfullfeats = self.sess.run([self.feat, self.label, self.auxlabel, self.bertfullfeat])

            self.list_seq_features.append(seq_features)
            self.list_seq_labels.append(seq_labels)
            self.list_seq_auxlabels.append(seq_auxlabels)
            self.list_seq_bertfullfeats.append(seq_bertfullfeats)

            if len(self.list_seq_labels) >= self.batch_size:
                yield self.padding_list_seq_with_multilabels(
                    self.list_seq_features,
                    self.list_seq_labels,
                    self.list_seq_auxlabels,
                    self.list_seq_bertfullfeats)
                self.list_seq_features = []
                self.list_seq_labels = []
                self.list_seq_auxlabels = []
                self.list_seq_bertfullfeats = []

        logging.info("clean the rest of data")
        if len(self.list_seq_features) > 0:
            yield self.padding_list_seq_with_multilabels(
                self.list_seq_features,
                self.list_seq_labels,
                self.list_seq_auxlabels,
                self.list_seq_bertfullfeats)
            self.list_seq_features = []
            self.list_seq_labels = []
            self.list_seq_auxlabels = []
            self.list_seq_bertfullfeats = []

    def __len__(self):
        return self.size_dataset

    @staticmethod
    def padding_list_seq_with_multilabels(list_seqs_features,
                                          list_seqs_labels,
                                          list_seqs_auxlabels,
                                          list_seq_bertfullfeats,
                                          dtype=np.float32,
                                          value1=0.,
                                          value2=0):
        x, len_x = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_features,
            dtype=dtype,
            pad=value1)
        y, len_y = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_labels,
            dtype=np.int32,
            pad=value2)
        y2, len_y2 = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_auxlabels,
            dtype=np.int32,
            pad=value2)

        y3, len_y3 = DataLoader.padding_list_seqs(
            list_seqs=list_seq_bertfullfeats,
            dtype=np.int32,
            pad=value2
            )

        return [x, y, y2, y3, len_x, len_y, len_y2, len_y3]


class LMDataLoader(DataLoader):
    def __init__(self, dataset, num_loops, args):
        self.dataset = dataset
        self.num_batch_tokens = args.data.num_batch_tokens
        self.num_steps = args.data.num_steps
        self.batch_size = args.batch_size
        self.num_loops = num_loops
        self.size_dataset = len(dataset)
        self.args = args

    def __iter__(self):
        return self.batch_with_buckets()


class PTBDataLoader(DataLoader):
    def __init__(self, dataset, num_loops, args):
        self.dataset = dataset
        self.num_batch_tokens = args.num_batch_tokens
        self.num_steps = args.num_steps
        self.batch_size = args.num_batch_tokens // args.num_steps
        self.num_loops = num_loops
        self.size_dataset = len(dataset)

    def __iter__(self):
        for _ in range(self.num_loops):
            for filename in self.dataset.list_files:
                with open(filename, "r") as f:
                    raw_data = f.read().replace("\n", "<eos>").split()
                    raw_data = [self.dataset.token2idx[word] for word in raw_data if word in self.dataset.token2idx.keys()]
                    data_len = len(raw_data)
                    batch_len = data_len // self.batch_size
                    data = np.array(raw_data[:self.batch_size * batch_len]).reshape([self.batch_size, batch_len])
                    epoch_size = (batch_len - 1) // self.num_steps
                    for i in range(epoch_size):
                        x = data[:, i*self.num_steps: (i+1)*self.num_steps]
                        y = data[:, i*self.num_steps+1: (i+1)*self.num_steps+1]
                        yield x, y, [self.num_steps]*self.batch_size, [self.num_steps]*self.batch_size


class ASRMultiLingualDataLoader(DataLoader):
    def __init__(self, dataset, args, feat, label, langlabel, batch_size, num_loops, num_thread=4, size_queue=2000):
        super().__init__(dataset, args, num_loops=num_loops, num_thread=num_thread, size_queue=size_queue)
        self.sess = None
        self.feat = feat
        self.label = label
        self.langlabel = langlabel
        self.list_seq_langlabels = []
        self.size_dataset = len(dataset)
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(len(self)*self.num_loops):
            seq_features, seq_labels, seq_langlabels = self.sess.run([self.feat, self.label, self.langlabel])

            self.list_seq_features.append(seq_features)
            self.list_seq_labels.append(seq_labels)
            self.list_seq_langlabels.append(seq_langlabels)

            if len(self.list_seq_labels) >= self.batch_size:
                yield self.padding_list_seq_with_multilingual(
                    self.list_seq_features,
                    self.list_seq_labels,
                    self.list_seq_langlabels)
                self.list_seq_features = []
                self.list_seq_labels = []
                self.list_seq_langlabels = []

        logging.info("clean the rest of data")
        if len(self.list_seq_features) > 0:
            yield self.padding_list_seq_with_multilingual(
                self.list_seq_features,
                self.list_seq_labels,
                self.list_seq_langlabels)
            self.list_seq_features = []
            self.list_seq_labels = []
            self.list_seq_langlabels = []

    def __len__(self):
        return self.size_dataset

    @staticmethod
    def padding_list_seq_with_multilingual(list_seqs_features,
                                          list_seqs_labels,
                                          list_seqs_langlabels,
                                          dtype=np.float32,
                                          value1=0.,
                                          value2=0):
        x, len_x = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_features,
            dtype=dtype,
            pad=value1)
        y, len_y = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_labels,
            dtype=np.int32,
            pad=value2)
        y2, len_y2 = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_langlabels,
            dtype=np.int32,
            pad=value2)

        return [x, y, y2, len_x, len_y, len_y2]


class ASRMultiLingualMultilabelDataLoader(DataLoader):
    def __init__(self, dataset, args, feat, label, label_2, langlabel, batch_size, num_loops, num_thread=4, size_queue=2000):
        super().__init__(dataset, args, num_loops=num_loops, num_thread=num_thread, size_queue=size_queue)
        self.sess = None
        self.feat = feat
        self.label = label
        self.label_2 = label_2
        self.langlabel = langlabel
        self.list_seq_labels = []
        self.list_seq_labels_2 = []
        self.list_seq_langlabels = []
        self.size_dataset = len(dataset)
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(len(self)*self.num_loops):
            seq_features, seq_labels, seq_labels_2, seq_langlabels = self.sess.run([self.feat, self.label, self.label_2, self.langlabel])

            self.list_seq_features.append(seq_features)
            self.list_seq_labels.append(seq_labels)
            self.list_seq_labels_2.append(seq_labels_2)
            self.list_seq_langlabels.append(seq_langlabels)

            if len(self.list_seq_labels) >= self.batch_size:
                yield self.padding_list_seq_with_multilingual_multilabel(
                    self.list_seq_features,
                    self.list_seq_labels,
                    self.list_seq_labels_2,
                    self.list_seq_langlabels)
                self.list_seq_features = []
                self.list_seq_labels = []
                self.list_seq_labels_2 = []
                self.list_seq_langlabels = []

        logging.info("clean the rest of data")
        if len(self.list_seq_features) > 0:
            yield self.padding_list_seq_with_multilingual_multilabel(
                self.list_seq_features,
                self.list_seq_labels,
                self.list_seq_labels_2,
                self.list_seq_langlabels)
            self.list_seq_features = []
            self.list_seq_labels = []
            self.list_seq_labels_2 = []
            self.list_seq_langlabels = []

    def __len__(self):
        return self.size_dataset

    @staticmethod
    def padding_list_seq_with_multilingual_multilabel(list_seqs_features,
                                          list_seqs_labels,
                                          list_seq_labels_2,
                                          list_seqs_langlabels,
                                          dtype=np.float32,
                                          value1=0.,
                                          value2=0):
        x, len_x = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_features,
            dtype=dtype,
            pad=value1)
        y, len_y = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_labels,
            dtype=np.int32,
            pad=value2)
        y2, len_y2 = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_labels,
            dtype=np.int32,
            pad=value2)
        y3, len_y3 = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_langlabels,
            dtype=np.int32,
            pad=value2)

        return [x, y, y2, y3, len_x, len_y, len_y2, len_y3]


if __name__ == '__main__':
    dataset = FakeDataSet()
    dataloader = SimpleDataLoader(dataset, 1)

    for i in dataloader:
        print(i)
