import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

stremmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stremmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_word):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_word), dtype=np.float32)
    for idx, w in enumerate(all_word):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag
