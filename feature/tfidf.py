#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Tfidf(object):

    STOP_WORDS = set((
        "the", "of", "is", "and", "to", "in", "that", "we", "for", "an", "are",
        "by", "be", "as", "on", "with", "can", "if", "from", "which", "you", "it",
        "this", "then", "at", "have", "all", "not", "one", "has", "or", "that"
    ))

    def __init__(self, input_corpus):
        self.input_files = input_corpus.get_files()
        self.stop_words = self.STOP_WORDS.copy()
        self.vocab = list()
        self.weight = [[]]
        self.tf = [[]]
        self.documents = []

    def get_tfidf_mat(self):
        pass

    def get_tf_mat(self):
        pass

    def __get_docs(self):
        pass

    def set_stopwords(self, stopwords_path):
        from os import path
        from os import getcwd
        _get_abs_path = lambda xpath: path.normpath(path.join(getcwd(), xpath))
        abs_path = _get_abs_path(stopwords_path)
        if not path.isfile(abs_path):
            raise Exception("tfidf: file does not exist: " + abs_path)
        content = open(abs_path, 'rb').read().decode('utf-8')
        for line in content.splitlines():
            self.stop_words.add(line)
