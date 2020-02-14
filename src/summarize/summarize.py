from . import preprocess

from .pgn.util import Decoder as PGNDecoder
from .textrank.util import Decoder as TextrankDecoder
#from .baseline import BaselineDecoder as BaselineDecoder


class Summarizer(object):
    def __init__(self):
        preprocess.init_libraries()

        self.pgn_decoder = PGNDecoder()
        self.textrank_decoder = TextrankDecoder()
        # self.baseline_decoder = BaselineDecoder()

    def summarize(self, text, options=['pgn', 'textrank']):
        """return summary of given text"""

        preprocessed_text = preprocess.process(text)

        pgn_result = None
        textrank_result = None

        if 'pgn' in options:
            pgn_result = self.pgn_decoder.decode(preprocessed_text)

        if 'textrank' in options:
            textrank_result = self.textrank_decoder.decode(preprocessed_text)
        
        return {
            'pgn': pgn_result,
            'textrank': textrank_result
        }