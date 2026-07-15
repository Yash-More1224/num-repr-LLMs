import yaml
import torch
import json
from tokenizers import Tokenizer
from data.dataset import ToTToDataset
from eval.eval_bpe import build_prompt

def test():
    with open("configs/train_config.yaml") as f:
        train_cfg = yaml.safe_load(f)
    
    tokenizer = Tokenizer.from_file(train_cfg["tokenizer"])
    sep_id = tokenizer.token_to_id("[SEP]")

    print("Loading dev dataset...")
    dataset = ToTToDataset(train_cfg["dev_data"], tokenizer, train_cfg["max_seq_len"])
    first_record = dataset[0]

    print("\n--- dataset.py ---")
    input_ids_full = first_record["input_ids"]
    ids_list = input_ids_full.tolist()
    sep_pos1 = ids_list.index(sep_id) if sep_id in ids_list else len(ids_list) - 1
    prompt1 = input_ids_full[:sep_pos1 + 1].unsqueeze(0)
    print("prompt1 shape:", prompt1.shape)
    print("prompt1 tokens:", prompt1.tolist())

    print("\n--- eval_bpe.py ---")
    with open(train_cfg["dev_data"]) as f:
        examples = [json.loads(line) for line in f]
    example = examples[0]
    prompt2 = build_prompt(example, sep_id, tokenizer, train_cfg["max_seq_len"])
    print("prompt2 shape:", prompt2.shape)
    print("prompt2 tokens:", prompt2.tolist())

    print("\nAre they exactly equal?", torch.equal(prompt1, prompt2))

if __name__ == "__main__":
    test()
