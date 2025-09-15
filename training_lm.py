import os
#from transformers import AutoTokenizer
from cs336_basics.trainer import PretrainedConfig, train_model
from tqdm import tqdm
import os

# def encode_file_with_progress(tokenizer_name, file_path, output_path=None):
#     """带进度条的文件编码"""
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_name) #自己写的太慢了，不得已用现成的
#     file_size = os.path.getsize(file_path)
    
#     all_tokens = []
    
#     with open(file_path, 'r', encoding='utf-8') as f:
#         with tqdm(total=file_size, unit='B', unit_scale=True, desc="Encoding") as pbar:
#             for line in f:
#                 line = line.strip()
#                 if line:
#                     tokens = tokenizer.encode(line, add_special_tokens=True)
#                     all_tokens.extend(tokens)
                
#                 pbar.update(len(line.encode('utf-8')))
    
#     if output_path:
#         with open(output_path, 'w') as f:
#             for token in all_tokens:
#                 f.write(f"{token}\n")
    
#     return all_tokens


if __name__ == "__main__":
    #对数据集进行tokenize
    # encode_file_with_progress("gpt2", f'/mnt/d/cs336/cs336-a1-basics/cs336_basics/data/TinyStories-valid.txt', f'/mnt/d/cs336/cs336-a1-basics/cs336_basics/data/TinyStories-valid.npy')
    # encode_file_with_progress("gpt2", f'/mnt/d/cs336/cs336-a1-basics/cs336_basics/data/TinyStories-train.txt', f'/mnt/d/cs336/cs336-a1-basics/cs336_basics/data/TinyStories-train.npy') 

    checkpoint_folder = "/mnt/d/cs336/cs336-a1-basics/cs336_basics/checkpoint"
    config = PretrainedConfig(
        project_name="tinystories",
        vocab_path=f"/mnt/d/cs336/cs336-a1-basics/cs336_basics/data/tokenizer_vocab.json",
        merges_path=f"cs336_basics/data/tokenizer_merges.txt",
        special_tokens=["<|endoftext|>"],
        train_path=f"/mnt/d/cs336/cs336-a1-basics/cs336_basics/data/TinyStories-train.npy",
        valid_path=f"/mnt/d/cs336/cs336-a1-basics/cs336_basics/data/TinyStories-valid.npy",
        checkpoint_dir=checkpoint_folder,
    )
    train_model(config)
    print("Finish training process.")

