from textrankr import TextRank

class Decoder(object):
    def decode(self, text, count=3):
        return TextRank(text, count)