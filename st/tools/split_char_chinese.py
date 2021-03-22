#!/usr/bin/python3
import codecs
import sys

if __name__ == '__main__':
    with codecs.open(sys.argv[1], 'r', encoding='utf8') as f:
        for line in f:
             line = ''.join(line.strip().split())
             tokens = list(line)
             new_line = ' '.join(tokens)
             print(new_line)
