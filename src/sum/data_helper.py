import tensorflow as tf
from .vocab import Vocab

def article_to_ids(article_words, vocab):
    ids = []
    oovs = []
    unk_id = vocab.word_to_id(vocab.UNKNOWN_TOKEN)
    for w in article_words:
        i = vocab.word_to_id(w)
        if i == unk_id:  # If w is OOV
            if w not in oovs:  # Add to list of OOVs
                oovs.append(w)
            # This is 0 for the first article OOV, 1 for the second article OOV...
            oov_num = oovs.index(w)
            # This is e.g. 50000 for the first article OOV, 50001 for the second...
            ids.append(vocab.size() + oov_num)
        else:
            ids.append(i)
    return ids, oovs


def abstract_to_ids(abstract_words, vocab, article_oovs):
    ids = []
    unk_id = vocab.word_to_id(vocab.UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word_to_id(w)
        if i == unk_id:  # If w is an OOV word
            if w in article_oovs:  # If w is an in-article OOV
                # Map to its temporary article OOV number
                vocab_idx = vocab.size() + article_oovs.index(w)
                ids.append(vocab_idx)
            else:  # If w is an out-of-article OOV
                ids.append(unk_id)  # Map to the UNK token id
        else:
            ids.append(i)
    return ids


def output_to_words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id_to_word(i)  # might be [UNK]
        except ValueError as e:  # w is OOV
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = tf.compat.as_str_any(article_oovs.numpy()[article_oov_idx])
            except ValueError as e:  # i doesn't correspond to an article oov
                raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (
                    i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words


def abstract_to_sents(abstract):
    """Splits abstract text from datafile into list of sentences.
    Args:
        abstract: string containing <s> and </s> tags for starts and ends of sentences
    Returns:
        sents: List of sentence strings (no tags)"""
    cur = 0
    sents = []
    while True:
        try:
            start_p = abstract.index(Vocab.SENTENCE_START, cur)
            end_p = abstract.index(Vocab.SENTENCE_END, start_p + 1)
            cur = end_p + len(Vocab.SENTENCE_END)
            sents.append(abstract[start_p+len(Vocab.SENTENCE_START):end_p])
        except ValueError as e:  # no more sentences
            return sents


def get_dec_inp_targ_seqs(sequence, max_len, start_id, stop_id):
    """Given the reference summary as a sequence of tokens, return the input sequence for the decoder, and the target sequence which we will use to calculate loss. The sequence will be truncated if it is longer than max_len. The input sequence must start with the start_id and the target sequence must end with the stop_id (but not if it's been truncated).
    Args:
        sequence: List of ids (integers)
        max_len: integer
        start_id: integer
        stop_id: integer
    Returns:
        inp: sequence length <=max_len starting with start_id
        target: sequence same length as input, ending with stop_id only if there was no truncation
    """
    inp = [start_id] + sequence[:]
    target = sequence[:]
    if len(inp) > max_len:  # truncate
        inp = inp[:max_len]
        target = target[:max_len]  # no end_token
    else:  # no truncation
        target.append(stop_id)  # end token
    assert len(inp) == len(target)
    return inp, target
