import string
import pandas as pd
from nltk import word_tokenize
import spacy
import numpy as np


nlp = spacy.load("pl_core_news_sm")
ctx = 7


def word_to_ctx(word, context, vector):
    context_str = str([x for x in context])
    if context_str in vector:
        if word in vector[context_str]:
            vector[context_str][word] += 1
        else:
            vector[context_str][word] = 0
    else:
        vector[context_str] = {}


class CreateModel:
    def __init__(self, data, vocabulary):
        self.vec = {}
        self.vocab = vocabulary
        self.data = data

    def fit_data(self):
        vector = {}
        for row in self.data:
            tokenized_row = word_tokenize(row)
            for i in range(0, len(tokenized_row) - ctx - 1):
                word = tokenized_row[i + ctx + 1]
                set_ctx = tokenized_row[i: i + ctx]
                word_to_ctx(word, set_ctx, vector)

    def predict_data(self, data):
        data = str([x for x in data])
        if data in self.vec:
            values_as_arr = np.asarray(list(self.vec[data].values()))
            nonzero = np.count_nonzero(values_as_arr)
            if values_as_arr.size == 0 or nonzero == 0:
                return ''
            dr = self.get_dr(values_as_arr, data)
            return dr
        else:
            return ''


    def get_keys(self, data):
        return np.asarray(list(self.vec[data].keys()))


    def get_dr(self, values_as_arr, data):
        thresh_value = 0.2 * np.max(values_as_arr)
        filtr = self.np.asarray(values_as_arr >= thresh_value)
        values_as_arr = values_as_arr[filtr]
        probs = self.values_as_arr / np.sum(values_as_arr)
        keys = self.get_keys(data)[filtr]
        return np.random.choice(keys, size=4, p=probs)


class Vocabulary:
    def __init__(self):
        self.words = {}

    def to_number(self, word):
        if word.lower() in self.words:
            return self.words[word]
        return -1

    def generate_voc(self, data):
        c = 0
        words = {}
        for row in data:
            for tokenized_word in word_tokenize(row):
                e = tokenized_word.lower()
                if e not in words:
                    words[e] = c
                    c += 1
        self.words = words


def slv(list_with_data):
    an = list_with_data[0]
    se = list_with_data[0]
    for a in list_with_data[1:]:
        if a != se:
            se = a
            an += a
    return an


def set_array(el):
    return el[0] if type(el) == list else el


def set_model(vocabulary, expected):
    model = CreateModel(expected['FixedOutput'], vocabulary)
    model.fit_data()
    return model


def transform_data(data, model):
    transformed = []
    for row in data:
        words_tokenized = word_tokenize(row)
        n_r = []
        for i in range(0, len(words_tokenized) - ctx - 1):
            context = words_tokenized[i: i + ctx]
            dr = model.predict_data(context)
            n_r.append(context)
            for word in dr:
                if word in string.punctuation:
                    n_r.append(word)
                    break
        list_as_string = ' '.join(map(set_array, n_r))
        transformed.append(slv(list_as_string))
    return transformed


def load_data(path, columns):
    data = pd.read_csv(path, sep="\t")
    data.columns = columns
    return data


def main():
    train_data = load_data('train/in.tsv', ['FileI', 'ASROutput'])
    expected_data = load_data('train/expected.tsv', ['FixedOutput'])
    vocabulary = Vocabulary()
    vocabulary.generate_voc(expected_data['FixedOutput'])
    model = set_model(vocabulary, expected_data)
    transformed_data = transform_data(train_data['ASROutput'], model)
    frame = pd.DataFrame(transformed_data)
    frame.to_csv('train/out.tsv', sep='\t')


if __name__ == "__main__":
    main()
