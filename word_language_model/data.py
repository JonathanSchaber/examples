import os
import csv
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
        	csv_reader = csv.reader(f)
            for row in csv_reader:
            	chars = [char for char in row[1]]
                # words = line.split() + ['<eos>']
                for char in chars:
                    self.dictionary.add_word(char)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
        	csv_reader = csv.reader(f)
            idss = []
            for row in csv_reader:
            	chars = [char for char in row[1]]
                # words = line.split() + ['<eos>']
                ids = []
                for char in chars:
                    ids.append(self.dictionary.word2idx[char])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids
