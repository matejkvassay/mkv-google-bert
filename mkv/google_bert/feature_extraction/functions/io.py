from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mkv.google_bert.tokenization.functions import convert_to_unicode
import re
import tensorflow as tf


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


def convert_to_input_examples(data_iterable, tuples=False):
    """
    :param data_iterable: Iterable of data samples or tuples of samples.
    :param tuples: True if items of input iterable are tuples of 2 sentences, False if only 1.
    :return:
    """
    unique_id = 0
    if tuples:
        for item in data_iterable:
            yield InputExample(unique_id=unique_id, text_a=item[0], text_b=item[1])
            unique_id += 1
    else:
        for item in data_iterable:
            yield InputExample(unique_id=unique_id, text_a=item)
            unique_id += 1
