#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datasource.input_corpus import InputCorpus
from feature.unigram_tfidf import UnigramTfidf


def gen_unigram_tfidf():
    input_corpus = InputCorpus('./corpus_test', encoding='gb18030')
    unigram_tfidf = UnigramTfidf(input_corpus)
    unigram_tfidf.set_stopwords('./data/stop_words_zh.utf8.txt')
    unigram_tfidf.get_tfidf_mat()
    # unigram_tfidf.save_tfidf('./output/output.test.txt', 10000)


def gen_unigram_tf():
    input_corpus = InputCorpus('./corpus', encoding='gb18030')
    unigram_tfidf = UnigramTfidf(input_corpus)
    unigram_tfidf.set_stopwords('./data/stop_words_zh.utf8.txt')
    unigram_tfidf.save_tf('./output/output.tf.txt', 10000)


if __name__ == '__main__':
    gen_unigram_tf()
