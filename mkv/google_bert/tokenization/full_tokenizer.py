from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import unicodedata
import six
import tensorflow as tf

from .functions import *
from .basic_tokenizer import BasicTokenizer
from .wordpiece_tokenizer import WordpieceTokenizer

from . import functions


class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = functions.load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return functions.convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return functions.convert_by_vocab(self.inv_vocab, ids)
