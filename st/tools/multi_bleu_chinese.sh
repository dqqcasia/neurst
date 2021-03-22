#!/usr/bin/env bash

ref=$1
hyp=$2

sd=`dirname $0`

python3 ${sd}/split_char_chinese.py $ref > /tmp/$$.ref
python3 ${sd}/split_char_chinese.py $hyp > /tmp/$$.hyp

perl ${sd}/multi-bleu.perl /tmp/$$.ref < /tmp/$$.hyp

#rm /tmp/$$*

