#!/usr/bin/env python
# -*- coding: utf-8 -*-

from options.test_options import TestOptions


def check_args(opt):
    model_path = opt.model_path
    test_dir = opt.test_dir
    suffix_accepted = opt.suffix_accepted
    import os
    assert os.path.exists(model_path), model_path + ' does not exist!'
    assert os.path.isdir(test_dir), test_dir + ' does not exist!'
    assert isinstance(type(suffix_accepted.split(',')), list), suffix_accepted + 'should be comma splited!'


def test(opt):
    check_args(opt)
    model_path = opt.model_path
    test_dir = opt.test_dir
    suffix_accepted = opt.suffix_accepted



if __name__ == '__main__':
    opt = TestOptions().parse_arguments()
    test(opt)
