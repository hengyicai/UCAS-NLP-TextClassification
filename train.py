#!/usr/bin/env python
# -*- coding: utf-8 -*-

from options.train_options import TrainOptions

LogPrefix = "Document classification: "


def train(opt):
    # Prepare the training corpus
    print(LogPrefix + "Prepare the training corpus begin!")
    from datasource.input_corpus import InputCorpus
    input_corpus = InputCorpus(opt.corpus_root, encoding=opt.encoding)
    print(LogPrefix + "Prepare the training corpus end!")

    # Get the basic tfidf features
    print(LogPrefix + "Get the basic tfidf features begin!")
    from feature.ngram_tfidf import NgramTfidf
    ngram_tfidf = NgramTfidf(input_corpus)
    ngram_tfidf.set_stopwords('./data/stop_words_zh.utf8.txt')
    import numpy as np
    tfidf_mat = np.asarray(ngram_tfidf.get_tfidf_mat(top_k=opt.tfidf_top_k))
    targets = np.asarray(input_corpus.get_filenames_and_targets()[1])
    print(LogPrefix + "Get the basic tfidf features end!")

    # Do feature selection
    print(LogPrefix + "Do feature selection begin!")
    from feature.feature_selection import PMISelection
    pmi_selector = PMISelection(tfidf_mat, targets)
    filtered_tfidf_mat = tfidf_mat[:, pmi_selector.get_boolean_selection_lst()]
    print(LogPrefix + "Do feature selection end!")

    # Training model
    print(LogPrefix + "Training model begin!")
    from model.classifier import SVMClassifier
    model = SVMClassifier()
    from model.classifier import Scorer
    scorer = Scorer(model.get_model(), filtered_tfidf_mat, targets)
    print(LogPrefix + "Training model end!")
    scorer.show_score()

    # Save the model
    model_save_path = opt.path_to_save_model
    model.dump(model_save_path)
    print(LogPrefix + 'model save to ' + model_save_path)


if __name__ == '__main__':
    # Parse arguments
    opt = TrainOptions().parse_arguments()
    train(opt)
