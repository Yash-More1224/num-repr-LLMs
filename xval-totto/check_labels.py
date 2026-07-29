import json
from tokenizers import Tokenizer
from data.dataset_bigbench import BigBenchArithmeticDataset

tokenizer = Tokenizer.from_file("tokenizer/totto_bpe/tokenizer.json")
dataset = BigBenchArithmeticDataset(
    data_path="../data/bigbench_arithmetic/train.jsonl",
    tokenizer=tokenizer,
    model_type="fone",
    max_seq_len=128
)

eos_id = tokenizer.token_to_id("[EOS]")
num_id = tokenizer.token_to_id("[NUM]")

num_count = 0
eos_count = 0
other_count = 0

for i in range(len(dataset)):
    item = dataset[i]
    labels = item["labels"].tolist()
    
    for l in labels:
        if l == -100: continue
        if l == eos_id: eos_count += 1
        elif l == num_id: num_count += 1
        else: other_count += 1

print(f"Total non-ignored labels:")
print(f"  [EOS] tokens: {eos_count}")
print(f"  [NUM] tokens: {num_count}")
print(f"  Other tokens: {other_count}")
