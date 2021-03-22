#!/usr/bin/env python
'''
@file wer.py
compute the word error rate with insterions and deletions

usage python wer.py reference decoded
'''

from __future__ import division
import sys
import os
import numpy as np
import re
import threading
from queue import Queue
import tensorflow as tf

def calc_cer(reference, decoded):
    '''main function

        args:
            reference: the file containing the reference utterances
            decoded: the file containing  the decoded utterances
        '''

    substitutions = 0
    insertions = 0
    deletions = 0
    numwords = 0

    with tf.io.gfile.GFile(reference, 'r') as fid, tf.io.gfile.GFile(decoded, 'r') as did:
        for lref, lout in zip(fid, did):
            # read the reference and  output
            reftext, output = list(lref.strip().replace(' ', '')), list(lout.strip().replace(' ', ''))

            # compare output to reference
            s, i, d = wer(reftext, output)
            substitutions += s
            insertions += i
            deletions += d
            numwords += len(reftext)

    substitutions /= numwords
    deletions /= numwords
    insertions /= numwords
    error = substitutions + deletions + insertions

    print(
        'character error rate: %s\n\tsubstitutions: %s\n\tinsertions: %s\n\t'
        'deletions: %s' % (error, substitutions, insertions, deletions))
    return error


def calc_wer(reference, decoded):
    '''main function

    args:
        reference: the file containing the reference utterances
        decoded: the file containing  the decoded utterances
    '''

    substitutions = 0
    insertions = 0
    deletions = 0
    numwords = 0

    with tf.io.gfile.GFile(reference, 'r') as fid, tf.io.gfile.GFile(decoded, 'r') as did:
        for lref, lout in zip(fid, did):
            # read the reference and  output
            reftext, output = lref.strip().split(), lout.strip().split()

            # compare output to reference
            s, i, d = wer(reftext, output)
            substitutions += s
            insertions += i
            deletions += d
            numwords += len(reftext)

    substitutions /= numwords
    deletions /= numwords
    insertions /= numwords
    error = substitutions + deletions + insertions

    print('word error rate: %s\n\tsubstitutions: %s\n\tinsertions: %s\n\tdeletions: %s' % (error, substitutions, insertions, deletions))
    return error


def calc_per(reference, decoded):
    '''main function

    args:
        reference: the file containing the reference utterances
        decoded: the file containing  the decoded utterances
    '''

    substitutions = 0
    insertions = 0
    deletions = 0
    numwords = 0

    with tf.io.gfile.GFile(reference, 'r') as fid, tf.io.gfile.GFile(decoded, 'r') as did:
        for lref, lout in zip(fid, did):
            # read the reference and  output
            reftext, output = lref.strip().split(), lout.strip().split()

            # compare output to reference
            s, i, d = wer(reftext, output)
            substitutions += s
            insertions += i
            deletions += d
            numwords += len(reftext)

    substitutions /= numwords
    deletions /= numwords
    insertions /= numwords
    error = substitutions + deletions + insertions

    print(
        'phoneme error rate: %s\n\tsubstitutions: %s\n\tinsertions: %s\n\t'
        'deletions: %s' % (error, substitutions, insertions, deletions))
    return error


def wer(reference, decoded):
    '''
    compute the word error rate

    args:
        reference: a list of the reference words
        decoded: a list of the decoded words

    returns
        - number of substitutions
        - number of insertions
        - number of deletions
    '''

    errors = np.zeros([len(reference) + 1, len(decoded) + 1, 3])
    errors[0, :, 1] = np.arange(len(decoded) + 1)
    errors[:, 0, 2] = np.arange(len(reference) + 1)
    substitution = np.array([1, 0, 0])
    insertion = np.array([0, 1, 0])
    deletion = np.array([0, 0, 1])
    for r, ref in enumerate(reference):
        for d, dec in enumerate(decoded):
            errors[r + 1, d + 1] = min((
                errors[r, d] + (ref != dec) * substitution,
                errors[r + 1, d] + insertion,
                errors[r, d + 1] + deletion), key=np.sum)

    return tuple(errors[-1, -1])


class Consumer(threading.Thread):

    def run(self):
        global queue
        global substitutions, insertions, deletions, numwords
        while queue.qsize() > 0:
            # msg = self.name + '消费了 ' + str(queue.get())
            # print(msg)
            res, ref = queue.get()
            s, i, d = wer(ref, res)
            substitutions += s
            insertions += i
            deletions += d
            numwords += len(ref)


def calc_wer_multiprocessing(reference, decoded):
    global queue
    queue = Queue()
    with open(reference) as file_res, open(decoded) as file_ref:
        for l1, l2 in zip(file_res, file_ref):
            queue.put([l1.strip().split(), l2.strip().split()])

    global substitutions, insertions, deletions, numwords

    substitutions = 0
    insertions = 0
    deletions = 0
    numwords = 0
    threads = []
    for i in range(5):
        c = Consumer()
        threads.append(c)
        c.start()

    for t in threads:
        t.join()

    substitutions /= numwords
    deletions /= numwords
    insertions /= numwords
    error = substitutions + deletions + insertions
    print(
        'word error rate: %s\n\tsubstitutions: %s\n\tinsertions: %s\n\t'
        'deletions: %s' % (error, substitutions, insertions, deletions))
    return error


if __name__ == '__main__':
    calc_wer(sys.argv[1], sys.argv[2])