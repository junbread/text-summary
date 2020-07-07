from preprocess import Preprocessor

from textrank.util import Decoder as TextrankDecoder
from pgn.util import Decoder as PgnDecoder


class Summarizer(object):
    def __init__(self):
        self.proc = Preprocessor()
        self.textrank_decoder = TextrankDecoder()
        self.pgn_decoder = PgnDecoder()

    def summarize(self, text, options=['pgn', 'textrank']):
        """return summary of given text"""

        processed_text = self.proc.process(text)

        pgn_result = None
        textrank_result = None

        if 'pgn' in options:
            pgn_result = self.pgn_decoder.decode(processed_text)

        if 'textrank' in options:
            textrank_result = self.textrank_decoder.decode(processed_text)
        
        return {
            'pgn': pgn_result,
            'textrank': textrank_result
        }