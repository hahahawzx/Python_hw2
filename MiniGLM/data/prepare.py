import os
import sys
import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")

names = sys.argv[1:]

### TODO: read data from ([name]/input.txt for name in names)
### TODO: combine multiple books into one single data file
combined_data = []
for name in names:
    data_address = os.path.join(name, "input.txt")
    if os.path.exists(data_address):
        with open(data_address, "r", encoding="utf-8") as file:
            data = file.read()
            combined_data.append(data)
    else:
        print(f"error，there is no book {name}")
merged_data = "\n".join(combined_data)
### TODO: split data for train(0.9) and valid (0.1)
train_data, val_data = None, None
data_lines = merged_data.split("\n")
total_lines = len(data_lines)
train_size = int(total_lines * 0.9)
val_size = int(total_lines * 0.1)
np.random.shuffle(data_lines)
train_data = "\n".join(data_lines[:train_size])
val_data = "\n".join(data_lines[train_size:train_size + val_size])
###

### TODO: tokenize raw data with tiktoken encoder
train_code = enc.encode_ordinary(train_data)
val_code = enc.encode_ordinary(val_data)
### TODO: transform to numpy array
train_ids, val_ids = None, None
train_ids = np.array(train_code)
val_ids = np.array(val_code)
###

# save numpy array to file [name]/train.bin and [name]/val.bin
os.makedirs("processed_pretrain", exist_ok=True)

train_ids.tofile(os.path.join("processed_pretrain", "train.bin"))
val_ids.tofile(os.path.join("processed_pretrain", 'val.bin'))
