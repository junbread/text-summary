class Vocab:
  
  SENTENCE_START  = '<s>'
  SENTENCE_END = '</s>'

  PAD_TOKEN = '[PAD]'
  UNKNOWN_TOKEN = '[UNK]'
  START_DECODING = '[START]'
  STOP_DECODING = '[STOP]'
  
  def __init__(self, vocab_file, max_size):
    
    self.word2id = {Vocab.UNKNOWN_TOKEN : 0, Vocab.PAD_TOKEN : 1,
     Vocab.START_DECODING : 2, Vocab.STOP_DECODING : 3}
    self.id2word = {0 : Vocab.UNKNOWN_TOKEN, 1 : Vocab.PAD_TOKEN, 2 : Vocab.START_DECODING, 3 : Vocab.STOP_DECODING}
    self.count = 4
    
    with open(vocab_file, 'r', encoding='utf-8') as f:
      for line in f:
        pieces = line.split()
        if len(pieces) != 2 :
          print('Warning : incorrectly formatted line in vocabulary file : %s\n' % line)
          continue
          
        w = pieces[0]
        if w in [Vocab.SENTENCE_START, Vocab.SENTENCE_END, Vocab.UNKNOWN_TOKEN, Vocab.PAD_TOKEN, Vocab.START_DECODING, Vocab.STOP_DECODING]:
          raise Exception('<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
        
        if w in self.word2id:
          raise Exception('Duplicated word in vocabulary file: %s' % w)
        
        self.word2id[w] = self.count
        self.id2word[self.count] = w
        self.count += 1
        if max_size != 0 and self.count >= max_size:
          print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self.count))
          break

    print("Finished constructing vocabulary of %i total words. Last word added: %s" % (self.count, self.id2word[self.count-1]))

      
  def word_to_id(self, word):
    if word not in self.word2id:
      return self.word2id[Vocab.UNKNOWN_TOKEN]
    return self.word2id[word]
  
  def id_to_word(self, word_id):
    if word_id not in self.id2word:
      raise ValueError('Id not found in vocab: %d' % word_id)
    return self.id2word[word_id]
  
  def size(self):
    return self.count