import os
from cs336_basics.tokenizer import Tokenizer, train_bpe, save_bpe_vocab_and_merges
from cs336_basics.trainer import PretrainedConfig, train_model

if __name__ == "__main__":
    vocab, merges = train_bpe(f'/mnt/d/cs336/cs336-a1-basics/cs336_basics/data/TinyStories-valid.txt', 10000, ['<|endoftext|>'])
    save_bpe_vocab_and_merges(vocab, merges, f'/mnt/d/cs336/cs336-a1-basics/cs336_basics/data/tokenizer_vocab.json', f'/mnt/d/cs336/cs336-a1-basics/cs336_basics/data/tokenizer_merges.txt')