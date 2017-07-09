#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "HayesTsai"

import jieba.posseg as pseg
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import tfidf


class UnigramTfidf(tfidf.Tfidf):
    def get_tfidf_mat(self):
        documents = list()

        for input_file in self.input_files:
            content_text = input_file.get_content()
            words = self.__cut(content_text)
            documents.append(words)
            # print(words)

        vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
        transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
        tfidf = transformer.fit_transform(
            vectorizer.fit_transform(documents))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
        self.vocab = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
        self.weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
        '''
        for i in range(len(self.weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for遍历某一类文本下的词语权重
            print(u"-------这里输出第", i, u"类文本的词语tf-idf权重------")
            for j in range(len(self.vocab)):
                print(self.vocab[j])
                print('-->')
                print(str(self.weight[i][j]))
        '''

    def __cut(self, text):
        words = pseg.cut(text)
        tags = []
        for item in words:
            if item.word in self.stop_words:
                continue
            if item.word.isdigit():
                continue
            tags.append(item.word)
        return " ".join(tags)

    def save_tfidf(self, save_to_path, top_k=20):
        if self.weight == [[]]:
            self.get_tfidf_mat()

        # select top_k features
        feature_tfidf_sum_vec = sum(self.weight)

        import numpy as np
        sorted_index = list(np.argsort(feature_tfidf_sum_vec))
        sorted_index.reverse()

        top_k = len(sorted_index) if len(sorted_index) <= top_k else top_k
        new_weight = np.zeros((len(self.weight), top_k))
        new_vocab = []

        for i in range(top_k):
            new_vocab.append(self.vocab[sorted_index[i]])

        for index_of_doc in range(len(self.weight)):
            for index_of_feature in range(top_k):
                new_weight[index_of_doc][index_of_feature] = self.weight[index_of_doc][sorted_index[index_of_feature]]

        self.weight = new_weight
        self.vocab = new_vocab
        with open(save_to_path, 'w') as dest_f:
            dest_f.write("file_names,")
            dest_f.write(",".join(self.vocab).encode('utf-8'))
            dest_f.write(",class")
            dest_f.write('\n')
            file_index = 0
            for doc in self.weight:
                dest_f.write(self.input_files[file_index].get_name())
                dest_f.write(',')
                dest_f.write(",".join([str(item) for item in list(doc)]))
                dest_f.write("," + self.input_files[file_index].get_class())
                dest_f.write('\n')
                file_index += 1
