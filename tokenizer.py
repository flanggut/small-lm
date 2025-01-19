from typing import List


class Tokenizer:
    def __init__(self, vocabulary: str):
        self.vocabulary = vocabulary
        self.stoi = {ch: i for i, ch in enumerate(vocabulary)}
        self.itos = {i: ch for i, ch in enumerate(vocabulary)}
        pass

    def encode(self, string: str):
        return [self.stoi[char] for char in string]

    def decode(self, code: List[int]):
        return "".join([self.itos[i] for i in code])
