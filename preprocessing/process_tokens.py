import os
import re
import json
import torch
from collections import Counter


INPUT_FILE = "preprocessing/data/final_train_data.txt" 
OUTPUT_DIR = "preprocessing/data"
VOCAB_SIZE = 20000
TRAIN_FRACTION = 0.9

os.makedirs(OUTPUT_DIR, exist_ok=True)

def tokenize_data(text_file_path, vocab_size=10000):
    print(f"--- Reading {text_file_path} ---")
    
    with open(text_file_path, 'r', encoding="utf-8") as f:
        text = f.read()

    print("Tokenizing that jawn")
    
    words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())

    print(f"Total raw tokens found: {len(words)}")

    word_counts = Counter(words)
    
    most_common = word_counts.most_common(vocab_size - 3)

    vocab = ['<PAD>', '<UNK>', '<EOS>'] + [word for word, _ in most_common]
    
    word_to_idx = {word: i for i, word in enumerate(vocab)}

    print("Convreting text to ints")
    tokens = []
    for word in words:
        tokens.append(word_to_idx.get(word, 1))

    return tokens, word_to_idx, vocab

def train_test_split(data, train_fraction):
    split_idx = int(train_fraction * len(data))
    return data[:split_idx], data[split_idx:]

def save_artifacts(tokens, word_to_idx, vocab):
    print(f"Saving artifacts to {OUTPUT_DIR}")
    
    vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(word_to_idx, f, indent=2)
        
    train_tokens, test_tokens = train_test_split(tokens, TRAIN_FRACTION)
    
    print("Converting to PyTorch tensors ")
    train_tensor = torch.tensor(train_tokens, dtype=torch.long)
    test_tensor = torch.tensor(test_tokens, dtype=torch.long)
    
    torch.save(train_tensor, os.path.join(OUTPUT_DIR, "train.pt"))
    torch.save(test_tensor, os.path.join(OUTPUT_DIR, "test.pt"))
    
    print("\nSUCCESS!")
    print(f"Vocab Size: {len(vocab)}")
    print(f"Train Tensor Shape: {train_tensor.shape}")
    print(f"Tet Tensor Shape:  {test_tensor.shape}")
    print(f"Output diectory:   {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        if os.path.exists(os.path.join("data", INPUT_FILE)):
            INPUT_FILE = os.path.join("data", INPUT_FILE)
        else:
            print(f"ERROR: Could not find {INPUT_FILE}")
            exit(1)

    tokens, word_to_idx, vocab = tokenize_data(INPUT_FILE, VOCAB_SIZE)
    save_artifacts(tokens, word_to_idx, vocab)