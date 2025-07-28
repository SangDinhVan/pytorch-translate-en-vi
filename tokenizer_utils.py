import re
from collections import Counter
from pyvi import ViTokenizer

class TokenizerEnVi:
    def __init__(self, language, min_freq=2):
        self.language = language
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]

    def tokenize(self, text):
        if self.language == "vi":
            tokens = ViTokenizer.tokenize(text).split()
        else:
            tokens = re.findall(r"\b\w+\b", text)

        return tokens

    def build_vocab(self, sentences):
        tokenized_sentences = [self.tokenize(sentence) for sentence in sentences]
        counter = Counter(token for sent in tokenized_sentences for token in sent)

        vocab = self.special_tokens + [word for word, freq in counter.items() if freq >= self.min_freq]

        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def encode(self, sentence, add_special_tokens=True, max_len=None):
        tokens = self.tokenize(sentence)
        if add_special_tokens:
            tokens = ["<sos>"] + tokens + ["<eos>"]
        if max_len:
            tokens = tokens[:max_len]
        ids = [self.word2idx.get(token, self.word2idx["<unk>"]) for token in tokens]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        tokens = []
        for idx in ids:
            token = self.idx2word.get(idx, "<unk>")
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def vocab_size(self):
        return len(self.word2idx)



