#!/bin/bash

if [ $# -lt 6 ]
then
        echo "Usage: $0 <document file (each doc in a line)><output file path> <head %-le> <tail %-le> <alpha> <usePosition><UseContext><Context FilePath> <Final OutputFile Path><alpha>"
        exit
fi
#cat $1 | shuf | head -n100 > $1.sample
#cat ~/Node2VecData/a/wikitext.txt | shuf | head -n200000 >  ~/Node2VecData/wikitextSample.txt
if [ $# -gt 6 ]
then
	nohup java Cooccur $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} 
  exit
fi
if [ $# -lt 7 ]
then
 nohup java Cooccur $1 $2 $3 $4 $5 $6 
fi

#sort -nr -k3 $9 > $9.s
#java GraphProcess  output_$I".txt" output_modified_$I".txt" wordIds/vocab_$I".txt"
#head -n50 $3.s
