     

import os
import requests
from bpe import BPETokenizerSimple

def download_file_if_absent(url, filename, search_dirs):
    for directory in search_dirs:
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            print(f"{filename} already exists in {file_path}")
            return file_path

    target_path = os.path.join(search_dirs[0], filename)
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(target_path, "wb") as out_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    out_file.write(chunk)
        print(f"Downloaded {filename} to {target_path}")
    except Exception as e:
        print(f"Failed to download {filename}. Error: {e}")

    return target_path


verdict_path = download_file_if_absent(
    url=(
         "https://raw.githubusercontent.com/rasbt/"
         "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
         "the-verdict.txt"
    ),
    filename="the-verdict.txt",
    search_dirs=["ch02/01_main-chapter-code/", "../01_main-chapter-code/", "."]
)

with open(verdict_path, "r", encoding="utf-8") as f: # added ../01_main-chapter-code/
    text = f.read()



tokenizer = BPETokenizerSimple()
tokenizer.train(text, vocab_size=1000, allowed_special={"<|endoftext|>"})
print(len(tokenizer.vocab))
print(len(tokenizer.bpe_merges))

input_text = "Jack embraced beauty through art and life.<|endoftext|> "
token_ids = tokenizer.encode(input_text, allowed_special={"<|endoftext|>"})
print(token_ids)

print("Number of characters:", len(input_text))
print("Number of token IDs:", len(token_ids))

for token_id in token_ids:
    print(f"{token_id} -> {tokenizer.decode([token_id])}")

print(tokenizer.decode(token_ids))

tokenizer.save_vocab_and_merges(vocab_path="vocab.json", bpe_merges_path="bpe_merges.txt")

tokenizer2 = BPETokenizerSimple()
tokenizer2.load_vocab_and_merges(vocab_path="vocab.json", bpe_merges_path="bpe_merges.txt")

print(tokenizer2.decode(token_ids))



# Download files if not already present in this directory

# Define the directories to search and the files to download
search_directories = ["./","."]

files_to_download = {
    "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe": "vocab.bpe",
    "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json": "encoder.json"
}

# Ensure directories exist and download files if needed
paths = {}
for url, filename in files_to_download.items():
    paths[filename] = download_file_if_absent(url, filename, search_directories)

tokenizer_gpt2 = BPETokenizerSimple()
tokenizer_gpt2.load_vocab_and_merges_from_openai(
    vocab_path=paths["encoder.json"], bpe_merges_path=paths["vocab.bpe"]
)

len(tokenizer_gpt2.vocab)

input_text = "This is some text"
token_ids = tokenizer_gpt2.encode(input_text)
print(token_ids)

print(tokenizer_gpt2.decode(token_ids))

import tiktoken

gpt2_tokenizer = tiktoken.get_encoding("gpt2")
tokens = gpt2_tokenizer.encode("This is some text")
print(tokens)