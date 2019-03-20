This repository provides a reference implementation of the method described in NAACL 2019 paper "Word-Node2Vec: Improving Word Embedding with Document-Level Non-Local Word Co-occurrences"

Basic Usage:

run_cooccur.sh will give the co-occurance sstatistics for a sample corpus.

Example

./run_cooccur.sh wikitextSample.txt cooccur.txt 5 90 0.5  false


run_node2vec.sh will give the word emedding based on the co-occurance statistics.
