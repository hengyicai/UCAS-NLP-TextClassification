#!/usr/bin/env python
# -*- coding: utf-8 -*-


def feature_test():
    pass


def lda_model_test():
    from datasource.input_corpus import InputCorpus
    input_corpus = InputCorpus('./corpus_test', encoding='gb18030')
    from feature.ngram_tfidf import NgramTfidf
    unigram_tfidf = NgramTfidf(input_corpus)
    unigram_tfidf.set_stopwords('./data/stop_words_zh.utf8.txt')
    from model.lda_decomposition import LDADec
    lda = LDADec(unigram_tfidf)
    lda.save_doc_topic_mat('./output/test.lda.txt')


def svm_model_test():
    from datasource.input_corpus import InputCorpus
    input_corpus = InputCorpus('./corpus_test', encoding='gb18030')
    from feature.ngram_tfidf import NgramTfidf
    unigram_tfidf = NgramTfidf(input_corpus)
    unigram_tfidf.set_stopwords('./data/stop_words_zh.utf8.txt')
    from model.lda_decomposition import LDADec
    lda = LDADec(unigram_tfidf)
    from model.svm_classifier import SVMClassifier
    svm = SVMClassifier(lda.get_doc_topic_mat(), input_corpus.get_filenames_and_targets()[1])
    svm.test()


def test():
    feature_test()
    # lda_model_test()
    svm_model_test()


if __name__ == '__main__':
    test()
