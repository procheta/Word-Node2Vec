# Word Embedding with Node Representation Learning

This repository provides a reference implementation of the word embedding method described in our NAACL 2019 paper *Word-Node2Vec: Improving Word Embedding with Document-Level Non-Local Word Co-occurrences*.

The main idea of the paper is to augment the *context* of a word in skip-gram with non-local word co-occurrences computed as *document-level word co-occurrences*.
A document could be *defined* to represent smaller units of text as well, e.g. paragraphs or sentences. 


The proposed approach involves the following steps.
 1. Given a collection of documents, compute a graph G, each node representing a word from the vocabulary. 
 1. Embed each **word-node** of G.
 1. Save the output vectors for each word-node as
    a. **.vec**: Each line comprised of `<word> <space> [<num>]+`
    b. **.bin**: Each line comprised of `<word> <space>` `chunks of <4 bytes>`, i.e. the [word2vec](https://github.com/tmikolov/word2vec) bin file format.
 
The implemenation of embedding the nodes is largely adapted from Mikolov's C code for [word2vec](https://github.com/tmikolov/word2vec).

### Constructing a Graph from a given Collection

The input to this step is a collection of documents as a single text file, each line representing a document. For convenience, we provide a script which can be invoked as
```
run_cooccur.sh <list of arguments>
```
which takes a list of arguments and writes out an edge list file.

1. A text file (each doc in a line)
2. The path of the output file to store the edge-list, each line comprised of `<word-1> <word-2> <edge-wt>`
3. Head and tail percentiles to prune off two frequent (less informative) or too infrequent words (likely noise).
4. *alpha*, the relative weight of **tf** (*alpha*) and **idf** (*1-alpha*).
<usePosition><UseContext> <Context FilePath> <Final OutputFile Path><alpha>
5. `<usePosition>`: whether to decay the contributions of co-occurrences between words pairs that are far apart.

A sample invocation of `run_cooccur.sh` is shown below.

```
./run_cooccur.sh data_node2vec/dbpedia.subset.txt.100000 cooccur.txt 5 90 0.5 false
```


run_node2vec.sh will give the word emedding based on the co-occurance statistics.

**Example**

./run_node2vec.sh 200 5 


**Evaluation Script:**

We used the following evaluation package for embedding 

git clone https://github.com/kudkudak/word-embeddings-benchmarks.git

