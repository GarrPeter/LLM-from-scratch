import torch;
import dataset as GPTDataloader;

#----- HYPERPARAMS -----
vocab_size = 50257
output_dim = 256
max_length = 4

#Load training data
with open("the-verdict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

#Setup training set
dataloader = GPTDataloader.create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

#Set up token embeddings
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
token_embeddings = token_embedding_layer(inputs)

#Set up positional embeddings
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))

#Combine token embeddings and positional embeddings to create input embeddings
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)