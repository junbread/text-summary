import os
import time
import tensorflow as tf
import numpy as np
import json

from collections import namedtuple
from pathlib import Path

from .data import Vocab
from .model import SummarizationModel
from .batcher import Example, Batch
from . import beam_search
from . import data
from . import util


class Decoder(object):

    def __init__(self):
        # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
        hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm',
                       'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'min_dec_steps', 'max_enc_steps', 'coverage', 'cov_loss_wt', 'pointer_gen', 'log_root', 'vocab_path', 'vocab_size', 'beam_size']
        hps_dict = {}
        for key, val in default_params().items():  # for each flag
            if key in hparam_list:  # if it's in the list
                hps_dict[key] = val  # add it to the dict
        hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)
        self._hps = hps

        self._vocab = Vocab(hps.vocab_path, hps.vocab_size)

        decode_hps = hps._replace(max_dec_steps=1)
        self._model = SummarizationModel(decode_hps, self._vocab)
        self._model.build_graph()

        self._saver = tf.compat.v1.train.Saver()  # we use this to load checkpoints for decoding
        self._sess = tf.compat.v1.Session(config=util.get_config())

        # Load an initial checkpoint to use for decoding
        ckpt_path = util.load_ckpt(self._saver, self._sess, hps)

    def decode(self, text):

        example = Example(text, ["샘플 텍스트입니다.","중얼중얼 중얼중얼"], self._vocab, self._hps)
        batch = Batch([example for _ in range(self._hps.batch_size)], self._hps, self._vocab)

        # Run beam search to get best Hypothesis
        best_hyp = beam_search.run_beam_search(self._sess, self._model, self._vocab, batch, self._hps)

        # Extract the output ids from the hypothesis and convert back to words
        output_ids = [int(t) for t in best_hyp.tokens[1:]]
        decoded_words = data.outputids2words(
            output_ids, self._vocab, (batch.art_oovs[0] if self._hps.pointer_gen else None))

        # Remove the [STOP] token from decoded_words, if necessary
        try:
            fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
            decoded_words = decoded_words[:fst_stop_idx]
        except ValueError:
            decoded_words = decoded_words
        decoded_output = ' '.join(decoded_words)  # single string

        return decoded_output


def default_params():

    project_root = Path(__file__).parent.parent.parent.parent

    params = {}
    params["data_path"] = project_root.joinpath("data", "preprocessed-tf1", "chunked", "train_*").as_posix()
    params["vocab_path"] = project_root.joinpath("data", "preprocessed-tf1", "vocab").as_posix()
    params["mode"] = "decode"
    params["exp_name"] = "exp-textrank"
    params["log_root"] = project_root.joinpath("src", "summarize", "baseline", params["exp_name"]).as_posix()

    params["hidden_dim"] = 256
    params["emb_dim"] = 128
    params["batch_size"] = 4
    params["max_enc_steps"] = 400
    params["max_dec_steps"] = 200
    params["beam_size"] = 4
    params["min_dec_steps"] = 100
    params["vocab_size"] = 50000
    params["lr"] = 0.15
    params["adagrad_init_acc"] = 0.1
    params["rand_unif_init_mag"] = 0.02
    params["trunc_norm_init_std"] = 1e-4
    params["max_grad_norm"] = 2.0
    params["pointer_gen"] = True
    params["coverage"] = False
    params["cov_loss_wt"] = 1.0

    return params