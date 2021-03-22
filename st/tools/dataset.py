import logging
import sys
import tensorflow as tf

from st.tools.tftools.tfRecord import save2tfrecord, save2tfrecord_multilabel, save2tfrecord_kdbert, save2tfrecord_ctcbert, \
    save2tfrecord_ctcbertfull, save2tfrecord_multilingual
from st.tools.tftools.tfRecord import TFDataSaver
from st.tools.textTools import array2text
from st.layers.arguments import args

def showing_scp_data():
    from st.layers.data_helper import ASR_scp_DataSet
    dataset_train = ASR_scp_DataSet(
        f_scp=args.dirs.train.data,
        f_trans=args.dirs.train.label,
        args=args,
        _shuffle=False,
        transform=False)
    ref_txt = array2text(dataset_train[0]['label'], args.data.unit, args.idx2token)
    print(ref_txt)


def showing_csv_data():
    from st.layers.data_helper import ASR_csv_DataSet
    dataset_train = ASR_csv_DataSet(
        list_files=[args.dirs.train.data],
        args=args,
        _shuffle=False,
        transform=False)
    ref_txt = array2text(dataset_train[0]['label'], args.data.unit, args.idx2token)
    print(ref_txt)


if __name__ == '__main__':
    import sys

    """
    noitation: the sample fentch from the dataset can be none!
    the bucket size is related to the **transformed** feature length
    each time you change dataset or frame skipping strategy, you need to rerun this script
    """

    # confirm = input("You are going to generate new tfdata, may covering the existing one.\n press ENTER to continue. ")
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    if args.model.use_multilabel:
        tfrecord_saver = save2tfrecord_multilabel
    elif args.model.use_bert:
        tfrecord_saver = save2tfrecord_kdbert
    elif args.model.use_ctc_bert:
        tfrecord_saver = save2tfrecord_ctcbert
    elif args.model.use_ctc_bertfull:
        tfrecord_saver = save2tfrecord_ctcbertfull
    elif args.model.is_multilingual:
        tfrecord_saver = save2tfrecord_multilingual
    else:
        tfrecord_saver = save2tfrecord

    args.sess = tf.Session()
    TFDataSaver(args.dataset_dev, args.dirs.dev.tfdata, args, size_file=10000, max_feat_len=args.data.max_train_len).split_save()
    TFDataSaver(args.dataset_test_tfrecord, args.dirs.test.tfdata, args, size_file=10000, max_feat_len=args.data.max_train_len).split_save()
    TFDataSaver(args.dataset_train, args.dirs.train.tfdata, args, size_file=10000, max_feat_len=args.data.max_train_len).split_save()

    args.sess.close()
