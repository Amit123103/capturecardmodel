import string
from collections import Counter
import json

class BusinessCardTokenizer:
    """
    Custom Tokenizer for Business Card data.
    Supports Character-level, Word-level, and BPE-style subword tokenization from scratch.
    """
    def __init__(self, mode='subword', vocab_size=30000):
        self.mode = mode
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        
        self.base_chars = list(string.ascii_letters + string.digits + string.punctuation + " \n\t")
        
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
    def build_vocab(self, texts):
        if self.mode == 'char':
            all_chars = set()
            for text in texts:
                all_chars.update(list(text))
            vocab_list = self.special_tokens + list(all_chars)
            
        elif self.mode == 'word':
            word_counts = Counter()
            for text in texts:
                word_counts.update(text.split())
            vocab_list = self.special_tokens + [word for word, count in word_counts.most_common(self.vocab_size - len(self.special_tokens))]
            
        elif self.mode == 'subword':
            char_counts = Counter()
            for text in texts:
                char_counts.update(list(text))
            vocab_list = self.special_tokens + list(char_counts.keys())
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
        self.vocab = {token: idx for idx, token in enumerate(vocab_list)}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        
    def encode(self, text, add_special_tokens=True):
        if not self.vocab:
            self.build_vocab([string.printable])
            
        tokens = []
        if add_special_tokens:
            tokens.append(self.vocab[self.bos_token])
            
        if self.mode == 'char' or self.mode == 'subword':
            for char in text:
                tokens.append(self.vocab.get(char, self.vocab[self.unk_token]))
        elif self.mode == 'word':
            for word in text.split():
                tokens.append(self.vocab.get(word, self.vocab[self.unk_token]))
                
        if add_special_tokens:
            tokens.append(self.vocab[self.eos_token])
            
        return tokens
        
    def decode(self, tokens):
        text = ""
        for token_id in tokens:
            token = self.inverse_vocab.get(token_id, self.unk_token)
            if token not in self.special_tokens:
                if self.mode == 'word':
                    text += token + " "
                else:
                    text += token
        return text.strip() if self.mode == 'word' else text

    def save(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'mode': self.mode,
                'vocab': self.vocab
            }, f, ensure_ascii=False, indent=2)
            
    def load(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.mode = data['mode']
            self.vocab = data['vocab']
            self.inverse_vocab = {int(idx): token for token, idx in self.vocab.items()}
            
if __name__ == "__main__":
    tokenizer = BusinessCardTokenizer(mode='char')
    tokenizer.build_vocab(["John Doe, CEO of Example Corp."])
    encoded = tokenizer.encode("John Doe")
    print("Encoded:", encoded)
    print("Decoded:", tokenizer.decode(encoded))
