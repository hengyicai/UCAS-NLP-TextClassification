#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


class InputCorpus(object):
    def __init__(self, m_dir, encoding):
        if os.path.isdir(m_dir):
            self.input_path = m_dir
            self.encoding = encoding
        else:
            raise Exception(str(m_dir) + " does not exist!")

    def get_files(self):
        ret_files = []
        for root, _, files in os.walk(self.input_path, topdown=False):
            for name in files:
                if '-' in name:
                    clazz = name.split('-')[0]
                    file_path = os.path.abspath(os.path.join(root, name))
                    ret_files.append(
                        InputFile(
                            file_path,
                            clazz,
                            from_encoding=self.encoding,
                            to_encoding='utf-8'
                        )
                    )
        return ret_files


class InputFile(object):
    def __init__(self, file_path, clazz, from_encoding, to_encoding):
        self.path = file_path
        self.encoding = to_encoding
        self.clazz = clazz
        import codecs
        with codecs.open(file_path, encoding=from_encoding, errors='ignore') as m_f:
            self.content = "".join(m_f.readlines())

    def get_content(self):
        return self.content


if __name__ == '__main__':
    input_corpus = InputCorpus('../corpus', 'gb18030')
    files = input_corpus.get_files()
    print(files[0].get_content())
