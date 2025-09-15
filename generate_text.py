from cs336_basics.inference import generate_text
from cs336_basics.model import Transformer
from cs336_basics.trainer import load_checkpoint
from cs336_basics.optimizer import AdamW
from transformers import AutoTokenizer

if __name__ == "__main__":
    model = Transformer(vocab_size=50257, 
                          context_length=128,
                          d_model=512,
                          num_layers=4,
                          num_heads=16,
                          d_ff=1344,
                          rope_theta=10000)
    model.to("cuda")
    optmizer = AdamW(model.parameters())
    load_checkpoint(f"/mnt/d/cs336/cs336-a1-basics/cs336_basics/checkpoint/checkpoint_300.pt", model, optmizer)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    text = generate_text(model, tokenizer, " Once upon a time there was a little girl named YangJie.", 32)
    print(text)
