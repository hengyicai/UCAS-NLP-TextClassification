#!/usr/bin/env python
# -*- coding: utf-8 -*-


def feature_test():
    from datasource.input_corpus import InputCorpus
    from feature.unigram_tfidf import UnigramTfidf

    input_corpus = InputCorpus('./corpus', encoding='gb18030')
    unigram_tfidf = UnigramTfidf(input_corpus)
    unigram_tfidf.get_tfidf_mat()


def unit_test():
    feature_test()


if __name__ == '__main__':
    unit_test()
