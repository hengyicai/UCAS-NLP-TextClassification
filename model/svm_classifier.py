#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "HayesTsai"


class SVMClassifier(object):
    train_ratio = 0.7
    test_ratio = 1 - train_ratio

    def __init__(self, X, target):
        '''
        :param X: shape([n_samples, n_features])
        :param target: shape([n_samples])
        '''
        self.X = X
        self.target = target
        self.train_X = self.X[0:int(len(self.X) * self.train_ratio)]
        self.train_targets = self.target[0:int(len(self.target) * self.train_ratio)]
        self.test_X = self.X[int(len(self.X) * self.train_ratio):]
        self.test_targets = self.target[int(len(self.target) * self.train_ratio):]
        self.model = None

    def train(self):
        if not self.model:
            from sklearn import svm
            clf = svm.SVC()
            clf.fit(self.train_X, self.train_targets)
            self.model = clf

    def test(self):
        if not self.model:
            self.train()
        predict_targets = self.model.predict(self.test_X)
        from sklearn.metrics import precision_recall_fscore_support as score

        precision, recall, fscore, support = score(self.test_targets, predict_targets)

        import numpy as np
        print('precision: {}'.format(precision))
        print('avg of precision: {}'.format(np.average(precision)))
        print('recall: {}'.format(recall))
        print('avg of recall: {}'.format(np.average(recall)))
        print('fscore: {}'.format(fscore))
        print('avg of fscore: {}'.format(np.average(fscore)))
        print('support: {}'.format(support))
        print('total of support: {}'.format(np.sum(support)))
