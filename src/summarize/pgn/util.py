import tensorflow as tf
import os
from absl import logging
from tqdm import tqdm
from ..vocab import Vocab
from ..batcher import batcher, single_example_for_decode
from .model import PGNModel
from .training_helper import train_model
from .decoder import beam_decode


def train():
    # set default params
    params = default_params()
    params["mode"] = "train"

    logging.info("Building the model ...")
    model = PGNModel(params)

    logging.info("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])

    logging.info("Creating the batcher ...")
    b = batcher(params["data_dir"], vocab, params)

    logging.info("Creating the checkpoint manager")
    checkpoint_dir = "{}".format(params["checkpoint_dir"])
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), PGN=model)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_dir, max_to_keep=20)

    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        logging.info("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        logging.info("Initializing from scratch.")

    logging.info("Starting the training ...")
    train_model(model, b, params, ckpt, ckpt_manager, "output.txt")


class Decoder(object):
    def __init__(self):
        super().__init__()
        logging.set_verbosity(logging.INFO)

        # set default params
        self.params = default_params()
        self.params["mode"] = "test"
        self.params["beam_size"] = self.params["batch_size"] = 4

        logging.info("Building the model ...")
        self.model = PGNModel(self.params)

        logging.info("Creating the vocab ...")
        self.vocab = Vocab(self.params["vocab_path"], self.params["vocab_size"])

        print("Creating the checkpoint manager")
        checkpoint_dir = "{}".format(self.params["checkpoint_dir"])
        ckpt = tf.train.Checkpoint(step=tf.Variable(0), PGN=self.model)
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, checkpoint_dir, max_to_keep=11)

        path = self.params["model_path"] if self.params["model_path"] else ckpt_manager.latest_checkpoint
        ckpt.restore(path)
        print("Model restored")

    def decode(self, text):
        logging.info("Making single example with given text ...")
        batch = single_example_for_decode(text, self.vocab, self.params["max_enc_len"], self.params["max_dec_len"])

        for b in batch:
            result = beam_decode(self.model, b, self.vocab, self.params)
            return result.abstract


def default_params():
    params = {}
    params["max_enc_len"] = 1000
    params["max_dec_len"] = 400
    params["max_dec_steps"] = 120
    params["min_dec_steps"] = 30
    params["batch_size"] = 16
    params["beam_size"] = 4
    params["vocab_size"] = 50000
    params["embed_size"] = 128
    params["enc_units"] = 256
    params["dec_units"] = 256
    params["attn_units"] = 512
    params["learning_rate"] = 0.15
    params["adagrad_init_acc"] = 0.1
    params["max_grad_norm"] = 0.8
    params["checkpoints_save_steps"] = 100
    params["max_steps"] = 10000
    params["num_to_test"] = 5
    params["max_num_to_eval"] = 5
    params["mode"] = ""
    params["model_path"] = ""
    params["checkpoint_dir"] = os.path.join(os.path.dirname(__file__), "ckpt")
    params["test_save_dir"] = os.path.join(os.path.dirname(__file__), "test")
    params["data_dir"] = os.path.join(os.path.dirname(__file__), "data", "chunked")
    params["vocab_path"] = os.path.join(os.path.dirname(__file__), "data", "vocab")
    params["log_file"] = os.path.dirname(__file__)

    return params
