#!/bin/bash

if [ $# -lt 10 ]
then
        echo "Usage: $0 <document file (each doc in a line)> <sample size (abs number of docs to use for co-occurrence estimation)> <output file> <frequency cutoff min %ge (0-100)> (typical value: 5) <frequency cutoff max %ge (0-100) (typical value: 90)> <alpha>"
        exit
fi
#cat $1 | shuf | head -n100 > $1.sample
#cat ~/Node2VecData/a/wikitext.txt | shuf | head -n200000 >  ~/Node2VecData/wikitextSample.txt
nohup java Cooccur $1.sample $2 $3 $4 $5 $6 $7 $8 $9 ${10} 
#sort -nr -k3 $9 > $9.s
#java GraphProcess  output_$I".txt" output_modified_$I".txt" wordIds/vocab_$I".txt"
#head -n50 $3.s
