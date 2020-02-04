from koalanlp.Util import initialize, finalize
from koalanlp import API
from koalanlp.proc import SentenceSplitter, Tagger

import subprocess
import os
import time
import re
import hanja

def process(article):
    try:
        initialize(HNN='LATEST', KHAIII='LATEST')
    except Exception:
        finalize()    
    # remove bylines
    article = re.sub(r'\. *\S+ +\S+ +\w+@(\w+\.)+\w+', '.', article)
    article = re.sub(r'\S+ +\S+ +\w+@(\w+\.)+\w+', '.', article)

    # remove parentheses
    article = re.sub(r'\([^)]+\)', ' ', article)
    article = re.sub(r'\[[^)]+\]', ' ', article)
    article = re.sub(r'\<[^)]+\>', ' ', article)
    article = re.sub(r'\【[^)]+\】', ' ', article)

    # replace hanja to hangul
    hanja.translate(article, 'substitution')

    # remove special characters except necessary punctuations
    article = re.sub(r'[^A-Za-zㄱ-ㅎㅏ-ㅣ가-힣0-9\%\-\_\.\,\?\!\/\"\'ㆍ·。、“”‘’『』《》〈〉「」\~○×□…\ ]', ' ', article)

    # initialize korean language analyzers
    splitter = SentenceSplitter(API.HNN)
    tagger = Tagger(API.KHAIII, kha_resource="/usr/local/share/khaiii")

    # split text into sentences
    sentences = splitter(article)    

    # regularize sentences and split into POS
    article_regularized = ''
    for sent in sentences:
        sent = tagger.tagSentence(sent)
        sent_regularized = []
        for word in sent[0].words:
            sent_regularized.append(' '.join([m.surface for m in word.morphemes]))
        article_regularized += '\n' + ' '.join(sent_regularized)
    
    # regularize whitespaces
    article_regularized = re.sub(r' +', ' ', article_regularized)

    finalize()

    command = ["java", "edu.stanford.nlp.process.PTBTokenizer", "-preserveLines", "-lowerCase"]

    result = ''
    echo = subprocess.Popen(["echo", "'{}'".format(article_regularized)], stdout=subprocess.PIPE)
    result = subprocess.check_output(command, stdin=echo.stdout)
    echo.wait()

    return result.decode("utf-8")