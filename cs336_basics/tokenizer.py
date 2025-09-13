import os
import regex as re
from typing import Dict, Type, List, Tuple, Optional, Iterable, Iterator
from collections import defaultdict, Counter
import json
from .utils import bytes_to_unicode

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    PAT =  r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    vocab = {}
    special_tokens_id = {}
    merges = []
    next_id = 0
    for token in special_tokens: #将special tokens加入vocab表
        token_bytes = token.encode('utf-8')
        vocab[next_id] = token_bytes
        special_tokens_id[token] = next_id
        next_id += 1
    for i in range(256):
        vocab[next_id] = bytes([i])
        next_id += 1

    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    escaped_special_tokens = [re.escape(token) for token in special_tokens] #根据特殊令牌进行分割
    if escaped_special_tokens:
        pattern = '|'.join(escaped_special_tokens)
        segments = re.split(f'({pattern})', text)
    else:
        segments = [text]
    word_freq = defaultdict(int)
    for segment in segments:
        if segment in special_tokens:
            continue  # 跳过特殊令牌本身
        
        for match in re.finditer(PAT, segment):
            token_text = match.group()
            byte_token = token_text.encode('utf-8')
            word_freq[byte_token] += 1
    
    current_vocab = {}  #初始化单词表
    for word_bytes, freq in word_freq.items():
        tokens = [bytes([b]) for b in word_bytes]
        current_vocab[tuple(tokens)] = freq
    
    print(f"current_vocab: {len(current_vocab)}")

    merge_count = 0
    while len(vocab) < vocab_size:
        pair_freq = defaultdict(int)
        for tokens, freq in current_vocab.items():
            for i in range(len(tokens)-1):
                pair = (tokens[i], tokens[i+1])
                pair_freq[pair] += freq
        
        if not pair_freq:
            #print(f"merge_count: {merge_count}")
            break
                
        #best_pair, best_freq = max(pair_freq.items(), key=lambda x: x[1])
        #这个位置没有判断词频相同时pair的字典序，需要改进
        best_pair = max(pair_freq.items(), key=lambda x: (x[1], x[0]))[0]

        new_token = best_pair[0] + best_pair[1]
        vocab[next_id] = new_token
        merges.append(best_pair)
        next_id += 1
        merge_count += 1

        new_current_vocab = {}
        for tokens, freq in current_vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens)-1 and tokens[i]==best_pair[0] and tokens[i+1]==best_pair[1]:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_current_vocab[tuple(new_tokens)] = freq
        current_vocab = new_current_vocab

    print(f"vocab_size: {len(vocab)}")
    print(f"merges_size: {len(merges)}")

    return vocab, merges

class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], 
                 special_tokens: list[str] = None):
        self.vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = special_tokens or []

        #vocab反向映射
        self.vocab_inv = {v:k for k, v in self.vocab.items()}

        # 初始化预分词正则模式
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.pattern = re.compile(self.PAT)

    #根据给定的vocab、merges路径创建一个tokenizer实体
    @classmethod
    def from_files(cls, vocab_filepath: str,merges_filepath: str,special_tokens: list[str] = None):
        byte_decoder = {v:k for k, v in bytes_to_unicode().items()}
        with open(vocab_filepath, encoding="utf-8") as f:
            json_vocab = json.load(f)
            vocab = {
                vocab_index : bytes([byte_decoder[token] for token in vocab_item])
                for vocab_item, vocab_index in json_vocab.items()
            }
        with open(merges_filepath, encoding="utf-8") as f:
            origin_merges = [tuple(line.rstrip().split(" ")) for line in f]
            merges = [
                (
                    bytes([byte_decoder[token] for token in merge_token_1]),
                    bytes([byte_decoder[token] for token in merge_token_2]),
                )
                for merge_token_1, merge_token_2 in origin_merges
            ]
        return cls(vocab, merges, special_tokens)
    
    def _pre_tokenize(self, text: str) -> List[str]:
        tokens = []
        if self.special_tokens:
            self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped_special_tokens = [re.escape(token) for token in self.special_tokens]
            pattern = '|'.join(escaped_special_tokens)
            segments = re.split(f'({pattern})', text)
        else:
            segments = [text]
        for segment in segments:
            if segment in self.special_tokens:
                tokens.append(segment)
            else:
                for match in self.pattern.finditer(segment):
                    tokens.append(match.group())
        return tokens
    
    def _apply_merge_to_token(self, token_bytes: bytes) -> List[int]:
        cur_tokens = [bytes([b]) for b in token_bytes]

        for merge_pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(cur_tokens):
                if i < len(cur_tokens)-1 and cur_tokens[i]==merge_pair[0] and cur_tokens[i+1]==merge_pair[1]:
                    new_token = merge_pair[0] + merge_pair[1]
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(cur_tokens[i])
                    i += 1
            cur_tokens = new_tokens

        token_ids = []
        for token in cur_tokens:
            if token in self.vocab_inv:
                token_ids.append(self.vocab_inv[token])
            else:
                print(f"Merge token error, the bytes: {token}")
                for b in token:
                    token_ids.append(self.vocab_inv[bytes([b])])
        
        return token_ids
    
    def decode(self, ids: List[int]) -> str:
        seq = b''
        for token_id in ids:
            if token_id in self.vocab:
                seq += self.vocab[token_id]
            else:
                # print(f"Unknown token ids: {token_id}")
                # return None
                byte_sequence += b'\xef\xbf\xbd' # 处理未知ID（使用Unicode替换字符）
        return seq.decode("utf-8", errors='replace')
    
    def encode(self, text: str) -> List[int]:
        token_ids = []
        pre_tokens = self._pre_tokenize(text)

        for token in pre_tokens:
            if token in self.special_tokens:
                token_bytes = token.encode("utf-8")
                token_ids.append(self.vocab_inv[token_bytes])
            else:
                token_bytes = token.encode("utf-8")
                token_ids.extend(self._apply_merge_to_token(token_bytes))
        
        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for chunk in iterable:
            buffer += chunk

            last_boundary = 0
            if self.special_tokens:
                self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
                escaped_special_tokens = [re.escape(token) for token in self.special_tokens]
                pattern = '|'.join(escaped_special_tokens)
                matches = list(re.finditer(pattern, buffer))
                if matches:
                    last_boundary = matches[-1].start()
            
            if last_boundary == 0:
                matches = list(self.pattern.finditer(buffer))
                if matches:
                    last_boundary = matches[-1].start()
            
            if last_boundary > 0:
                text = buffer[:last_boundary]
                for token_id in self.encode(text):
                    yield token_id
                buffer = buffer[last_boundary:]
            
        if buffer:
            for token_id in self.encode(buffer):
                yield token_id
        
    