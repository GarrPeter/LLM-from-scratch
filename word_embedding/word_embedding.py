import re;
import tokenizer;

#region Using the busted manual tokenizer
# Import the file
with open("the-verdict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

# Set up vocab
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>","<|unk|>"])
vocab = {token:integer for integer, token in enumerate(all_tokens)}

# Create tokenizer
simple_tokenizer = tokenizer.SimpleTokenizerV2(vocab)
text = """"It's the last he painted, you know,"
        Mrs. Gisburn said with pardonable pride."""
ids = simple_tokenizer.encode(text)
print(ids)
print(simple_tokenizer.decode(ids))

text2 = "Hello, do you like tea?"
text3 = "In the sunlit terraces of the palace"
text = " <|endoftext|> ".join((text2, text3))
print(text)
ids = simple_tokenizer.encode(text)
print(ids)
print(simple_tokenizer.decode(ids))
#endregion

#region Experimenting with tiktoken tokenizer

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.")
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
strings = tokenizer.decode(integers)
print(strings)

#endregion