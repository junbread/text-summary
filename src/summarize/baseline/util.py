"""This file contains some utility functions"""

import tensorflow as tf
import os
import time
import numpy as np
import json

from collections import namedtuple
from pathlib import Path

from .data import Vocab
from .model import SummarizationModel
from .batcher import Example, Batch
from . import beam_search
from . import data


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
        self._sess = tf.compat.v1.Session(config=get_config())

        # Load an initial checkpoint to use for decoding
        ckpt_path = load_ckpt(self._saver, self._sess, hps)

    def decode(self, text):

        example = Example(text, [], self._vocab, self._hps)
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
    params["vocab_path"] = project_root.joinpath("data", "preprocessed", "vocab").as_posix()
    params["mode"] = "decode"
    params["exp_name"] = "model"
    params["log_root"] = project_root.joinpath("src", "summarize", "baseline", params["exp_name"]).as_posix()

    params["hidden_dim"] = 256
    params["emb_dim"] = 128
    params["batch_size"] = 4
    params["max_enc_steps"] = 2000
    params["max_dec_steps"] = 300
    params["beam_size"] = 4
    params["min_dec_steps"] = 35
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


def get_config():
    """Returns config for tf.session"""
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    return config


def load_ckpt(saver, sess, hps, ckpt_dir="train"):
    """Load checkpoint from the ckpt_dir (if unspecified, this is train dir) and restore it to saver and sess, waiting 10 secs in the case of failure. Also returns checkpoint name."""
    while True:
        try:
            latest_filename = "checkpoint_best" if ckpt_dir == "eval" else None
            ckpt_dir = os.path.join(hps.log_root, ckpt_dir)
            ckpt_state = tf.train.get_checkpoint_state(ckpt_dir, latest_filename=latest_filename)
            tf.compat.v1.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            return ckpt_state.model_checkpoint_path
        except:
            tf.compat.v1.logging.info("Failed to load checkpoint from %s. Sleeping for %i secs...", ckpt_dir, 10)
            time.sleep(10)
