from .preprocess import *
from .pgn.util import Decoder as PGNDecoder
from .textrank.util import Decoder as TextrankDecoder

#from .baseline import BaselineDecoder as BaselineDecoder


class Summarizer(object):
    def __init__(self):
        init_libraries()
        self.pgn_decoder = PGNDecoder()
        self.textrank_decoder = TextrankDecoder()

    def summarize(self, text):
        """return summary of given text"""

        preprocessed_text = process(text)

        result = {
            'pgn': self.pgn_decoder.decode(preprocessed_text),
            'textrank': self.textrank_decoder.decode(preprocessed_text)
        }

        return result
