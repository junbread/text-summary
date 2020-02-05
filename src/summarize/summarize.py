from . import preprocess
from .pgn import util as util_pgn

def train_model():
    util_pgn.train()

def summarize(text):
    """return summary of given text"""

    preprocessed_text = preprocess.process(text)

    result = {
        'pgn': util_pgn.decode(preprocessed_text),
        'tpgn': '',
        'bert': ''
    }

    return result
    
if __name__ == "__main__":
    test()