import torch
import warnings


class CTCLabelConverter(object):
    def __init__(self, character):
        self.character = ["[blank]"] + list(character)
        self.dict = {char: i for i, char in enumerate(self.character)}

    def encode(self, text, batch_max_length=25):
        length = [len(s) for s in text]
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            tmp = [
                self.dict[char] if char in self.dict else self.dict["[blank]"]
                for char in t
            ]
            tmp = tmp[:batch_max_length]
            batch_text[i][: len(tmp)] = torch.LongTensor(tmp)
        return (batch_text, torch.IntTensor(length))

    def decode(self, text_index, length):
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :l]
            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.character[t[i]])
            text = "".join(char_list)
            texts.append(text)
        return texts


class AttnLabelConverter(object):
    def __init__(self, character):
        self.character = ["[GO]", "[s]"] + list(character)
        self.dict = {char: i for i, char in enumerate(self.character)}

    def encode(self, text, batch_max_length=25):
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            t = ["[GO]"] + list(t) + ["[s]"]
            t = [
                self.dict[char] if char in self.dict else self.dict["[s]"] for char in t
            ]
            batch_text[i][: len(t)] = torch.LongTensor(t)
        return (batch_text, torch.IntTensor(length))

    def decode(self, text_index, length):
        texts = []
        for index, l in enumerate(length):
            text = "".join([self.character[i] for i in text_index[index, :] if i > 1])
            texts.append(text)
        return texts


class Averager(object):
    def __init__(self):
        self.reset()

    def add(self, v):
        self.n_count += 1
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res
