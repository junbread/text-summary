import tensorflow as tf
import glob
import os
import ntpath

import data_helper
from vocab import Vocab


def _parse_function(example_proto):
    # Create a description of the features.
    feature_description = {
        'article': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'abstract': tf.io.FixedLenFeature([], tf.string, default_value='')
    }
    # Parse the input `tf.Example` proto using the dictionary above.
    parsed_example = tf.io.parse_single_example(
        example_proto, feature_description)
    return parsed_example


def single_example_for_decode(article, vocab, max_enc_len, max_dec_len):
    """이미 전처리가 끝난 텍스트를 입력으로 받아 Single Exmample을 반환"""

    def single_example_generator(article, vocab, max_enc_len, max_dec_len):
        start_decoding = vocab.word_to_id(vocab.START_DECODING)
        stop_decoding = vocab.word_to_id(vocab.STOP_DECODING)

        article_words = article.split()[: max_enc_len]
        enc_len = len(article_words)
        enc_input = [vocab.word_to_id(w) for w in article_words]
        enc_input_extend_vocab, article_oovs = data_helper.article_to_ids(
            article_words, vocab)

        abstract = "<s> 예시 내용 . </s>"
        abstract_sentences = [sent.strip() for sent in data_helper.abstract_to_sents(abstract)]
        abstract = ' '.join(abstract_sentences)
        abstract_words = abstract.split()

        abs_ids = [vocab.word_to_id(w) for w in abstract_words]
        abs_ids_extend_vocab = data_helper.abstract_to_ids(
            abstract_words, vocab, article_oovs)
        dec_input, target = data_helper.get_dec_inp_targ_seqs(
            abs_ids, max_dec_len, start_decoding, stop_decoding)
        _, target = data_helper.get_dec_inp_targ_seqs(
            abs_ids_extend_vocab, max_dec_len, start_decoding, stop_decoding)
        dec_len = len(dec_input)

        output = {
            "enc_len": enc_len,
            "enc_input": enc_input,
            "enc_input_extend_vocab": enc_input_extend_vocab,
            "article_oovs": article_oovs,
            "dec_input": dec_input,
            "target": target,
            "dec_len": dec_len,
            "article": article,
            "abstract": abstract,
            "abstract_sents": abstract_sentences
        }

        for _ in range(4):
            yield output

    dataset = tf.data.Dataset.from_generator(
        lambda: single_example_generator(article, vocab, max_enc_len, max_dec_len),
        output_types={
            "enc_len": tf.int32,
            "enc_input": tf.int32,
            "enc_input_extend_vocab": tf.int32,
            "article_oovs": tf.string,
            "dec_input": tf.int32,
            "target": tf.int32,
            "dec_len": tf.int32,
            "article": tf.string,
            "abstract": tf.string,
            "abstract_sents": tf.string
        }, output_shapes={
            "enc_len": [],
            "enc_input": [None],
            "enc_input_extend_vocab": [None],
            "article_oovs": [None],
            "dec_input": [None],
            "target": [None],
            "dec_len": [],
            "article": [],
            "abstract": [],
            "abstract_sents": [None]
        })
    dataset = dataset.padded_batch(4, padded_shapes=({"enc_len": [],
                                                        "enc_input": [None],
                                                        "enc_input_extend_vocab": [None],
                                                        "article_oovs": [None],
                                                        "dec_input": [max_dec_len],
                                                        "target": [max_dec_len],
                                                        "dec_len": [],
                                                        "article": [],
                                                        "abstract": [],
                                                        "abstract_sents": [None]}),
                                    padding_values={"enc_len": -1,
                                                    "enc_input": 1,
                                                    "enc_input_extend_vocab": 1,
                                                    "article_oovs": b"",
                                                    "dec_input": 1,
                                                    "target": 1,
                                                    "dec_len": -1,
                                                    "article": b"",
                                                    "abstract": b"",
                                                    "abstract_sents": b""},
                                    drop_remainder=True)

    def update(entry):
        return ({"enc_input": entry["enc_input"],
                 "extended_enc_input": entry["enc_input_extend_vocab"],
                 "article_oovs": entry["article_oovs"],
                 "enc_len": entry["enc_len"],
                 "article": entry["article"],
                 "max_oov_len": tf.shape(entry["article_oovs"])[1]},

                {"dec_input": entry["dec_input"],
                 "dec_target": entry["target"],
                 "dec_len": entry["dec_len"],
                 "abstract": entry["abstract"]})

    dataset = dataset.map(update)
    return dataset


def example_generator(filenames, vocab, max_enc_len, max_dec_len, mode, batch_size):

    raw_dataset = tf.data.TFRecordDataset(filenames)
    parsed_dataset = raw_dataset.map(_parse_function)
    if mode == "train":
        parsed_dataset = parsed_dataset.shuffle(
            1000, reshuffle_each_iteration=True).repeat()

    for raw_record in parsed_dataset:

        article = raw_record["article"].numpy().decode()
        abstract = raw_record["abstract"].numpy().decode()

        start_decoding = vocab.word_to_id(vocab.START_DECODING)
        stop_decoding = vocab.word_to_id(vocab.STOP_DECODING)

        article_words = article.split()[: max_enc_len]
        enc_len = len(article_words)
        enc_input = [vocab.word_to_id(w) for w in article_words]
        enc_input_extend_vocab, article_oovs = data_helper.article_to_ids(
            article_words, vocab)

        abstract_sentences = [sent.strip()
                              for sent in data_helper.abstract_to_sents(abstract)]
        abstract = ' '.join(abstract_sentences)
        abstract_words = abstract.split()
        abs_ids = [vocab.word_to_id(w) for w in abstract_words]
        abs_ids_extend_vocab = data_helper.abstract_to_ids(
            abstract_words, vocab, article_oovs)
        dec_input, target = data_helper.get_dec_inp_targ_seqs(
            abs_ids, max_dec_len, start_decoding, stop_decoding)
        _, target = data_helper.get_dec_inp_targ_seqs(
            abs_ids_extend_vocab, max_dec_len, start_decoding, stop_decoding)
        dec_len = len(dec_input)

        output = {
            "enc_len": enc_len,
            "enc_input": enc_input,
            "enc_input_extend_vocab": enc_input_extend_vocab,
            "article_oovs": article_oovs,
            "dec_input": dec_input,
            "target": target,
            "dec_len": dec_len,
            "article": article,
            "abstract": abstract,
            "abstract_sents": abstract_sentences
        }
        if mode == "test" or mode == "eval":
            for _ in range(batch_size):
                yield output
        else:
            yield output


def batch_generator(generator, filenames, vocab, max_enc_len, max_dec_len, batch_size, mode):

    dataset = tf.data.Dataset.from_generator(
        lambda: generator(filenames, vocab, max_enc_len,
                          max_dec_len, mode, batch_size),
        output_types={
            "enc_len": tf.int32,
            "enc_input": tf.int32,
            "enc_input_extend_vocab": tf.int32,
            "article_oovs": tf.string,
            "dec_input": tf.int32,
            "target": tf.int32,
            "dec_len": tf.int32,
            "article": tf.string,
            "abstract": tf.string,
            "abstract_sents": tf.string
        }, output_shapes={
            "enc_len": [],
            "enc_input": [None],
            "enc_input_extend_vocab": [None],
            "article_oovs": [None],
            "dec_input": [None],
            "target": [None],
            "dec_len": [],
            "article": [],
            "abstract": [],
            "abstract_sents": [None]
        })
    dataset = dataset.padded_batch(batch_size, padded_shapes=({"enc_len": [],
                                                               "enc_input": [None],
                                                               "enc_input_extend_vocab": [None],
                                                               "article_oovs": [None],
                                                               "dec_input": [max_dec_len],
                                                               "target": [max_dec_len],
                                                               "dec_len": [],
                                                               "article": [],
                                                               "abstract": [],
                                                               "abstract_sents": [None]}),
                                   padding_values={"enc_len": -1,
                                                   "enc_input": 1,
                                                   "enc_input_extend_vocab": 1,
                                                   "article_oovs": b'',
                                                   "dec_input": 1,
                                                   "target": 1,
                                                   "dec_len": -1,
                                                   "article": b"",
                                                   "abstract": b"",
                                                   "abstract_sents": b''},
                                   drop_remainder=True)

    def update(entry):
        return ({"enc_input": entry["enc_input"],
                 "extended_enc_input": entry["enc_input_extend_vocab"],
                 "article_oovs": entry["article_oovs"],
                 "enc_len": entry["enc_len"],
                 "article": entry["article"],
                 "max_oov_len": tf.shape(entry["article_oovs"])[1]},

                {"dec_input": entry["dec_input"],
                 "dec_target": entry["target"],
                 "dec_len": entry["dec_len"],
                 "abstract": entry["abstract"]})

    dataset = dataset.map(update)
    return dataset


def batcher(data_path, vocab, hpm):

    filenames = glob.glob("{}/*.tfrecords".format(data_path))
    dataset = batch_generator(example_generator, filenames, vocab,
                              hpm["max_enc_len"], hpm["max_dec_len"], hpm["batch_size"], hpm["mode"])

    return dataset
