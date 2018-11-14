from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections
import codecs
import json

from mkv.google_bert.model.bert_config import BertConfig
from mkv.google_bert.tokenization import FullTokenizer
from mkv.google_bert.feature_extraction.functions.features import convert_examples_to_features
from mkv.google_bert.feature_extraction.functions.io import read_examples
from mkv.google_bert.feature_extraction.functions.builders import input_fn_builder
from mkv.google_bert.feature_extraction.functions.builders import model_fn_builder


class BertFeatureExtractor(object):
    def __init__(self, bert_config_file, init_checkpoint, vocab_file, layers="-1,-2,-3,-4",
                 max_seq_length=128, do_lower_case=True, batch_size=32, use_tpu=False, master=False,
                 num_tpu_cores=8, use_one_hot_embeddings=False, log_verbosity=tf.logging.INFO):
        """
        :param bert_config_file: The config json file corresponding to the pre-trained BERT model.
                                 This specifies the model architecture.
        :param init_checkpoint: Initial checkpoint (usually from a pre-trained BERT model).
        :param vocab_file: The vocabulary file that the BERT model was trained on.
        :param layers:
        :param max_seq_length: The maximum total input sequence length after WordPiece tokenization. Sequences longer
                               than this will be truncated, and sequences shorter than this will be padded.
        :param do_lower_case: Whether to lower case the input text. Should be True for uncased models and False
                              for cased models.
        :param batch_size: Batch size for predictions.
        :param use_tpu: Whether to use TPU or GPU/CPU.
        :param master: Only used if `use_tpu` is True. Address of the master.
        :param num_tpu_cores: Only used if `use_tpu` is True. Total number of TPU cores to use.
        :param use_one_hot_embeddings: If True, tf.one_hot will be used for embedding lookups, otherwise
                                       tf.nn.embedding_lookup will be used. On TPUs, this should be True,
                                       since it is much faster.
        """
        tf.logging.set_verbosity(log_verbosity)

        self.tokenizer = FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)
        self.max_seq_length = max_seq_length

        # model builder
        self.layer_indexes = [int(x) for x in layers.split(",")]
        bert_config = BertConfig.from_json_file(bert_config_file)
        model_fn = model_fn_builder(
            bert_config=bert_config,
            init_checkpoint=init_checkpoint,
            layer_indexes=self.layer_indexes,
            use_tpu=use_tpu,
            use_one_hot_embeddings=use_one_hot_embeddings)

        # estimator (if TPU is not available, this will fall back to normal Estimator on CPU or GPU)
        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(
            master=master,
            tpu_config=tf.contrib.tpu.TPUConfig(
                num_shards=num_tpu_cores,
                per_host_input_for_training=is_per_host))
        self.estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=use_tpu,
            model_fn=model_fn,
            config=run_config,
            predict_batch_size=batch_size)

    def extract_features(self, input_file, output_file):
        examples = read_examples(input_file)
        features = convert_examples_to_features(
            examples=examples, seq_length=self.max_seq_length, tokenizer=self.tokenizer)

        unique_id_to_feature = {}
        for feature in features:
            unique_id_to_feature[feature.unique_id] = feature

        for feature in features:
            unique_id_to_feature[feature.unique_id] = feature

        input_fn = input_fn_builder(
            features=features, seq_length=self.max_seq_length)

        with codecs.getwriter("utf-8")(tf.gfile.Open(output_file, "w")) as writer:
            for result in self.estimator.predict(input_fn, yield_single_examples=True):
                unique_id = int(result["unique_id"])
                feature = unique_id_to_feature[unique_id]
                output_json = collections.OrderedDict()
                output_json["linex_index"] = unique_id
                all_features = []
                for (i, token) in enumerate(feature.tokens):
                    all_layers = []
                    for (j, layer_index) in enumerate(self.layer_indexes):
                        layer_output = result["layer_output_%d" % j]
                        layers = collections.OrderedDict()
                        layers["index"] = layer_index
                        layers["values"] = [
                            round(float(x), 6) for x in layer_output[i:(i + 1)].flat
                        ]
                        all_layers.append(layers)
                    features = collections.OrderedDict()
                    features["token"] = token
                    features["layers"] = all_layers
                    all_features.append(features)
                output_json["features"] = all_features
                writer.write(json.dumps(output_json) + "\n")
