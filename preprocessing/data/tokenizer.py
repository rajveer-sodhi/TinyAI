import json
import numpy as np

class Tokenizer:
    """
    Simple tokenizer that wraps existing vocabulary files.
    Supports character-level (tokenizer.json) or word-level (vocab.json) tokenization.
    """
    
    def __init__(self, vocab_path=None):
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_size = 0
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.use_char_level = False
        
        if vocab_path:
            self.load(vocab_path)
    
    def load(self, vocab_path):
        """Load vocabulary from JSON file."""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different vocab formats
        if 'token_to_id' in data:
            # Character-level tokenizer format (output/tokenizer.json)
            self.token_to_id = data['token_to_id']
            self.use_char_level = True
            # Map special tokens
            self.pad_token_id = self.token_to_id.get('[PAD]', 0)
            self.unk_token_id = self.token_to_id.get('[UNK]', 1)
            self.bos_token_id = self.token_to_id.get('[BOS]', 2)
            self.eos_token_id = self.token_to_id.get('[EOS]', 3)
        else:
            # Word-level vocab format (preprocessing/data/vocab.json)
            self.token_to_id = data
            self.use_char_level = False
            # Map special tokens
            self.pad_token_id = self.token_to_id.get('<PAD>', 0)
            self.unk_token_id = self.token_to_id.get('<UNK>', 1)
            self.eos_token_id = self.token_to_id.get('<EOS>', 2)
            # BOS might be 'bos' in word-level vocab
            self.bos_token_id = self.token_to_id.get('bos', self.token_to_id.get('[BOS]', 15))
        
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)
        
        print(f"Loaded vocabulary with {self.vocab_size} tokens")
        print(f"  PAD={self.pad_token_id}, UNK={self.unk_token_id}, BOS={self.bos_token_id}, EOS={self.eos_token_id}")
    
    def _tokenize_char(self, text):
        """Character-level tokenization."""
        tokens = []
        for char in text:
            tokens.append(self.token_to_id.get(char, self.unk_token_id))
        return tokens
    
    def _tokenize_word(self, text):
        """Word-level tokenization."""
        import re
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        tokens = []
        for word in words:
            tokens.append(self.token_to_id.get(word, self.unk_token_id))
        return tokens
    
    def encode(self, text, max_length=512, padding=True):
        """Encode text to token IDs."""
        if self.use_char_level:
            tokens = self._tokenize_char(text)
        else:
            tokens = self._tokenize_word(text)
        
        # Truncate if too long
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        # Pad if needed
        if padding and len(tokens) < max_length:
            tokens = tokens + [self.pad_token_id] * (max_length - len(tokens))
        
        return tokens
    
    def decode(self, ids):
        """Decode token IDs back to text."""
        if isinstance(ids, (list, np.ndarray)):
            tokens = [self.id_to_token.get(int(i), '<UNK>') for i in ids]
        else:
            tokens = [self.id_to_token.get(int(ids), '<UNK>')]
        
        if self.use_char_level:
            return ''.join(tokens)
        else:
            return ' '.join(tokens)
