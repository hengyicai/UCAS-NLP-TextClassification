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
        input_files = self.input_corpus.get_files()
        for input_file in input_files:
            content_text = input_file.get_content()
            if len(content_text) > 0:
                words = self.cut(content_text)
                documents.append(words)
        vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
        transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
        tfidf = transformer.fit_transform(
            vectorizer.fit_transform(documents))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
        word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
        weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
        for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for遍历某一类文本下的词语权重
            print(u"-------这里输出第", i, u"类文本的词语tf-idf权重------")
            for j in range(len(word)):
                print(word[j])
                print('-->')
                print(str(weight[i][j]))

    def cut(self, text):
        # TODO: 去除分词结果中的停用词，数字
        words = pseg.cut(text)
        return " ".join([item.word for item in words])
