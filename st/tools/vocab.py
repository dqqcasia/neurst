#!/usr/bin/env python
"""
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
Modified by Easton.
"""
import logging
from argparse import ArgumentParser
from collections import Counter, defaultdict
from tqdm import tqdm
import tensorflow as tf


def load_vocab(path, vocab_size=None, have_unk=True):
    with tf.io.gfile.GFile(path, 'r') as f:
        vocab = [line.strip().split()[0] for line in f]
    vocab = vocab[:vocab_size] if vocab_size else vocab
    if have_unk:
        id_unk = vocab.index('<unk>')
        token2idx = defaultdict(lambda: id_unk)
        idx2token = defaultdict(lambda: '<unk>')
    else:
        token2idx = {}
        idx2token = {}
    token2idx.update({token: idx for idx, token in enumerate(vocab)})
    idx2token.update({idx: token for idx, token in enumerate(vocab)})

    if '<space>' in vocab:
        idx2token[token2idx['<space>']] = ' '
    if '<blk>' in vocab:
        idx2token[token2idx['<blk>']] = ''
    # if '<pad>' in vocab:
    #     idx2token[token2idx['<pad>']] = ''
    if '<unk>' in vocab:
        idx2token[token2idx['<unk>']] = '<UNK>'
    assert len(token2idx) == len(idx2token)

    return token2idx, idx2token


def make_vocab(fpath, fname):
    """Constructs vocabulary.
    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.

    Writes vocabulary line by line to `fname`.
    """
    word2cnt = Counter()
    with open(fpath, encoding='utf-8') as f:
        for l in f:
            words = l.strip().split()[1:]
            # words = l.strip().split(',')[1].split()
            word2cnt.update(Counter(words))
    word2cnt.update({"<pad>": 1000000000,
                     "<unk>": 1000000000,
                     "<sos>": 1000000000,
                     "<eos>": 1000000000})
    with open(fname, 'w', encoding='utf-8') as fout:
        for word, cnt in word2cnt.most_common():
            fout.write(u"{}\t{}\n".format(word, cnt))
        fout.write(u"{}\t{}\n".format("<blk>", 0))
    logging.info('Vocab path: {}\t size: {}'.format(fname, len(word2cnt)+1))


def pre_processing(fpath, fname):
    import re
    with open(fpath, errors='ignore') as f, open(fname, 'w') as fw:
        for line in tqdm(f):
            line = line.strip().split(maxsplit=1)
            idx = line[0]
            list_tokens = re.findall('\[[^\[\]]+\]|[^x00-xff]|[A-Za-z]', line[1])
            list_tokens = [token.upper() for token in list_tokens]

            fw.write(idx+' '+' '.join(list_tokens)+'\n')


def scp_process(inpath, outpath):
    with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
        for line in fin:
            idx, utt = line.strip().split(maxsplit=1)
            new_utt = ''.join(utt.strip().split())
            print(new_utt, file=fout)


def split_word(inpath, outpath):
    with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
        for line in fin:
            chars = list(''.join(line.strip().split()))
            print(' '.join(chars), file=fout)


def word2char(inpath, outpath):
    with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
        for line in fin:
            idx, utt = line.strip().split(maxsplit=1)
            chars = ''.join(utt.split())
            new_utt = ' '.join(list(chars))
            new_line = ' '.join([idx, new_utt])
            print(new_line, file=fout)


def get_wav(inpath, outpath):
    with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
        for line in fin:
            wav, utt = line.strip().split(maxsplit=1)
            print(wav, file=fout)


if __name__ == '__main__':
    import sys
    # Read config
    logging.basicConfig(level=logging.INFO)
    make_vocab(sys.argv[1], sys.argv[2])
    logging.info("Done")





