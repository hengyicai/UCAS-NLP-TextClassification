#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datasource.input_corpus import InputCorpus
from feature.unigram_tfidf import UnigramTfidf

input_corpus = InputCorpus('./corpus', encoding='gb18030')
unigram_tfidf = UnigramTfidf(input_corpus)
unigram_tfidf.set_stopwords('./data/stop_words_zh.utf8.txt')
unigram_tfidf.save_tfidf('./output/output.txt', 10000)
