# pylint: disable=W0401
import tensorflow as tf
import logging
import os
import sys

if 'horovod' in sys.modules:
    import horovod.tensorflow as hvd

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')

from st.tools.vocab import load_vocab
from st.layers.utils import load_config

from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())


class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    demo:
    a ={'length': 10, 'shape': (2,3)}
    config = AttrDict(a)
    config.length #10

    here we can recurrently use attribute to access confis
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, item):
        try:
            if type(self[item]) is dict:
                self[item] = AttrDict(self[item])
            res = self[item]
        except:
            # print('not found {}'.format(item))
            res = None
        return res


# CONFIG_FILE = sys.argv[-3:]
CONFIG_FILE = sys.argv[-1:]
args = AttrDict(load_config(CONFIG_FILE))

args.num_gpus = len(args.gpus.split(','))
args.list_gpus = ['/gpu:{}'.format(i) for i in range(args.num_gpus)]

# dirs
args.dir_model = args.dirs.models

# args.dir_log = args.dir_model + "/" + args.dirs.log + "/" + TIMESTAMP
args.dir_log = args.dir_model + "/" + args.dirs.log
args.dir_checkpoint = args.dir_model + "/" + args.dirs.checkpoint
args.path_vocab = args.dirs.vocab

try:
    if hvd is None or hvd.rank() == 0:
        if not tf.io.gfile.exists(args.dir_model):
            tf.io.gfile.makedirs(args.dir_model)
        if not tf.io.gfile.exists(args.dir_log):
            tf.io.gfile.makedirs(args.dir_log)
        if not tf.io.gfile.exists(args.dir_checkpoint):
            tf.io.gfile.makedirs(args.dir_checkpoint)
except:
    if not os.path.exists(args.dir_model):
        tf.io.gfile.makedirs(args.dir_model)
    if not os.path.exists(args.dir_log):
        tf.io.gfile.makedirs(args.dir_log)
    if not os.path.exists(args.dir_checkpoint):
        tf.io.gfile.makedirs(args.dir_checkpoint)


# bucket
if args.bucket_boundaries:
    args.list_bucket_boundaries = [int(int(i))
                                   for i in args.bucket_boundaries.split(',')]
else:
    args.list_bucket_boundaries = [i
                                   for i in range(args.size_bucket_start,
                                                  args.size_bucket_end,
                                                  args.size_bucket_gap)]

args.list_batch_size = ([int(args.num_batch_tokens / boundary) * args.num_gpus
                         for boundary in (args.list_bucket_boundaries)] + [args.num_gpus])
logging.info('\nbucket_boundaries: {} \nbatch_size: {}'.format(
    args.list_bucket_boundaries, args.list_batch_size))

# vocab
args.token2idx, args.idx2token = load_vocab(args.dirs.vocab)
args.dim_output = len(args.token2idx)

if args.dirs.src_vocab:
    args.src_token2idx, args.src_idx2token = load_vocab(args.dirs.src_vocab)
    args.src_dim_output = len(args.src_token2idx)
else:
    args.src_token2idx, args.src_idx2token = args.token2idx, args.idx2token
    args.src_dim_output = args.dim_output

if not args.data.src_delimiter:
    args.data.src_delimiter = args.data.delimiter

# wav_id_mapping
if args.data.withoutid:
    pass
else:
    args.wav2idx, args.idx2wav = load_vocab(args.dirs.wav_ids, have_unk=False)

if '<eos>' in args.token2idx.keys():
    args.eos_idx = args.token2idx['<eos>']
else:
    args.eos_idx = None

if '<sos>' in args.token2idx.keys():
    args.sos_idx = args.token2idx['<sos>']
elif '<blk>' in args.token2idx.keys():
    args.sos_idx = args.token2idx['<blk>']
else:
    args.sos_idx = None

args.dirs.train.list_files = args.dirs.train.data.split(',')
args.dirs.dev.list_files = args.dirs.dev.data.split(',')

if not tf.io.gfile.isdir(args.dirs.train.tfdata):
    tf.io.gfile.mkdir(args.dirs.train.tfdata)
if not tf.io.gfile.isdir(args.dirs.dev.tfdata):
    tf.io.gfile.mkdir(args.dirs.dev.tfdata)
if not tf.io.gfile.isdir(args.dirs.test.tfdata):
    tf.io.gfile.mkdir(args.dirs.test.tfdata)

if args.dirs.test:
    args.dirs.test.list_files = args.dirs.test.data.split(',')

# dataset
if args.dirs.type == 'scp':
    from st.layers.data_helper import ASR_scp_DataSet
    dataset_train = ASR_scp_DataSet(
        f_scp=args.dirs.train.data,
        f_trans=args.dirs.train.label,
        args=args,
        _shuffle=True,
        transform=False)
    dataset_dev = ASR_scp_DataSet(
        f_scp=args.dirs.dev.data,
        f_trans=args.dirs.dev.label,
        args=args,
        _shuffle=False,
        transform=False)
    dataset_test = ASR_scp_DataSet(
        f_scp=args.dirs.test.data,
        f_trans=args.dirs.test.label,
        args=args,
        _shuffle=False,
        transform=True)
elif args.dirs.type == 'csv':
    from st.layers.data_helper import ASR_csv_DataSet, ASR_multilabel_csv_DataSet, ASR_kdbert_csv_DataSet, ASR_ctcbert_csv_dataset, \
        ASR_ctcbertfull_csv_dataset, ASR_multilingual_csv_DataSet, ASR_multilingual_multilabel_csv_DataSet, PTDEC_csv_dataset
    if args.model.use_multilabel:
        csv_reader = ASR_multilabel_csv_DataSet
    elif args.model.use_bert:
        csv_reader = ASR_kdbert_csv_DataSet
    elif args.model.use_ctc_bert:
        csv_reader = ASR_ctcbert_csv_dataset
    elif args.model.use_ctc_bertfull:
        csv_reader = ASR_ctcbertfull_csv_dataset
    elif args.model.is_multilingual:
        csv_reader = ASR_multilingual_csv_DataSet
    elif args.model.is_cotrain:
        csv_reader = ASR_multilingual_multilabel_csv_DataSet
    else:
        csv_reader = ASR_csv_DataSet

    if args.model.pt_decoder:
        csv_reader = PTDEC_csv_dataset

    dataset_train = csv_reader(
        list_files=args.dirs.train.list_files,
        args=args,
        _shuffle=True,
        transform=False)
    dataset_dev = csv_reader(
        list_files=args.dirs.dev.list_files,
        args=args,
        _shuffle=False,
        transform=False)
    dataset_test_tfrecord = csv_reader(
        list_files=args.dirs.test.list_files,
        args=args,
        _shuffle=False,
        transform=False)

    dataset_test = csv_reader(
        list_files=args.dirs.test.list_files,
        args=args,
        _shuffle=False,
        transform=True)
    dataset_dev_infer = csv_reader(
        list_files=args.dirs.dev.list_files,
        args=args,
        _shuffle=False,
        transform=True)

else:
    raise NotImplementedError('not dataset type!')

args.dataset_train = dataset_train
args.dataset_dev = dataset_dev
args.dataset_test = dataset_test
args.dataset_dev_infer = dataset_dev_infer
args.dataset_test_tfrecord = dataset_test_tfrecord

# model
# encoder
if args.model.encoder.type == 'transformer_encoder':
    from st.models.encoders.transformer_encoder import Transformer_Encoder as encoder
elif args.model.encoder.type == 'BLSTM':
    from st.models.encoders.blstm import BLSTM as encoder
else:
    raise NotImplementedError('not found encoder type: {}'.format(args.model.encoder.type))
args.model.encoder.type = encoder

# encoder2
try:
    if args.model.encoder2.type == 'transformer_tpeencoder':
        from st.models.encoders.transformer_TPEencoder import Transformer_TPEncoder as encoder
    args.model.encoder2.type = encoder
except BaseException:
    print("not using encoder2!")
    pass

# decoder
try:
    if args.model.decoder.type == 'rna_decoder3':
        from st.models.decoders.rna_decoder import RNADecoder as decoder
    elif args.model.decoder.type == 'fc_decoder':
        from st.models.decoders.fc_decoder import FCDecoder as decoder
    elif args.model.decoder.type == 'transformer_decoder':
        from st.models.decoders.transformer_decoder import Transformer_Decoder as decoder
    args.model.decoder.type = decoder
except BaseException:
    print("not using decoder!")
    args.model.decoder = AttrDict()
    args.model.decoder.size_embedding = None
    args.model.decoder.type = None

# decoder2
try:
    if args.model.decoder2.type == 'rna_decoder3':
        from st.models.decoders.rna_decoder import RNADecoder as decoder
    elif args.model.decoder2.type == 'fc_decoder':
        from st.models.decoders.fc_decoder import FCDecoder as decoder
    args.model.decoder2.type = decoder
except BaseException:
    print("not using decoder2!")
    pass


# model
if args.model.structure == 'Seq2SeqModel':
    from st.models.seq2seqModel import Seq2SeqModel as Model
elif args.model.structure == 'transformer':
    from st.models.Transformer import Transformer as Model
elif args.model.structure == 'transformerTpe':
    from st.models.TransformerTPE import TransformerTPE as Model
elif args.model.structure == 'transformerTpes':
    from st.models.TransformerTPES import TransformerTPES as Model
else:
    raise NotImplementedError('not found Model type!')

args.Model = Model

# vocab
logging.info('using vocab: {}'.format(args.dirs.vocab))

if args.dirs.vocab_pinyin:
    from st.layers.utils import load_vocab
    logging.info('using pinyin vocab: {}'.format(args.dirs.vocab_pinyin))
    args.phone.token2idx, args.phone.idx2token = load_vocab(
        args.dirs.vocab_pinyin)
    args.phone.dim_output = len(args.phone.token2idx)
    args.phone.eos_idx = None
    args.phone.sos_idx = args.phone.token2idx['<blk>']


def read_tfdata_info(dir_tfdata):
    data_info = {}
    with tf.io.gfile.GFile(os.path.join(dir_tfdata, 'tfdata.info'), 'r') as f:
        for line in f:
            if 'dim_feature' in line or \
                'num_tokens' in line or \
                    'size_dataset' in line:
                line = line.strip().split(' ')
                data_info[line[0]] = int(line[1])

    return data_info

try:
    info_dev = read_tfdata_info(args.dirs.dev.tfdata)
    args.data.dev.dim_feature = info_dev['dim_feature']
    args.data.dev.size_dataset = info_dev['size_dataset']

    args.data.dim_feature = args.data.dev.dim_feature
    args.data.dim_input = args.data.dim_feature * \
        (args.data.num_context + 1) *\
        (3 if args.data.add_delta else 1)
except BaseException:
    print("Unexpected error: ", sys.exc_info())

try:
    """
    exists tfdata and will build model and training
    """
    # info_train
    info_train = read_tfdata_info(args.dirs.train.tfdata)
    args.data.train.dim_feature = info_train['dim_feature']
    args.data.train.size_dataset = info_train['size_dataset']
    # info_dev
    info_dev = read_tfdata_info(args.dirs.dev.tfdata)
    args.data.dev.size_dataset = info_dev['size_dataset']
    # info test
    info_test = read_tfdata_info(args.dirs.test.tfdata)
    args.data.test.size_dataset = info_test['size_dataset']

    args.data.dim_feature = args.data.train.dim_feature
    args.data.dim_input = args.data.dim_feature * \
        (args.data.num_context + 1) *\
        (3 if args.data.add_delta else 1)

    logging.info('feature dim: {}; input dim: {}; output dim: {}'.format(
        args.data.dim_feature, args.data.dim_input, args.dim_output))
except BaseException:
    """
    no found tfdata and will read data and save it into tfdata
    won't build model
    """
    print("Unexpected error:", sys.exc_info())
    print('no finding tfdata ...')

