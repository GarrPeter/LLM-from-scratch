import torch
import tiktoken
from model import GPTModel
from util import text_to_token_ids, token_ids_to_text, generate_text_simple, calc_loss_loader, generate, assign
from dataset import create_dataloader_v1
from training import train_model_simple, evaluate_model
from plotter import plot_losses
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from gpt_download import download_and_load_gpt2
from load_weights import load_weights_into_gpt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

print("Settings:",settings)
print("Parameter dictionary keys:", params.keys())

model_configs = {
    "gpt2-small (124M)": {"emb_dim":768, "n_layers": 12, "n_heads":12},
    "gpt2-medium (355M)": {"emb_dim":1024, "n_layers": 24, "n_heads":16},
    "gpt2-large (774M)": {"emb_dim":1280, "n_layers": 36, "n_heads":20},
    "gpt2-xl (1558M)": {"emb_dim":1600, "n_layers": 48, "n_heads":25},
}

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv_bias":True})

gpt = GPTModel(NEW_CONFIG)
gpt.eval()

load_weights_into_gpt(gpt, params)
gpt.to(device)
tokenizer = tiktoken.get_encoding("gpt2")
torch.manual_seed(123)
token_ids = generate(model=gpt, idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),max_new_tokens=25,context_size=NEW_CONFIG["context_length"],top_k=50,temperature=1.5)
print("Output text:\n",token_ids_to_text(token_ids, tokenizer))

file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

tokenizer = tiktoken.get_encoding("gpt2")

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:",total_tokens)

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)
train_loader = create_dataloader_v1(train_data, batch_size=2, max_length=GPT_CONFIG_124M['context_length'],stride=GPT_CONFIG_124M["context_length"],drop_last=True,shuffle=True,num_workers=0)
val_loader = create_dataloader_v1(val_data, batch_size=2, max_length=GPT_CONFIG_124M['context_length'],stride=GPT_CONFIG_124M["context_length"],drop_last=False,shuffle=False,num_workers=0)



train_loss, val_loss = evaluate_model(gpt, train_loader, val_loader, device, 5)
print(f"Train Loss {train_loss:.3f} Val Loss { val_loss:.3f}")

# vocab = {
#     "closer":0,
#     "every":1,
#     "effort":2,
#     "forward":3,
#     "inches":4,
#     "moves":5,
#     "pizza":6,
#     "toward":7,
#     "you":8,
# }

# inverse_vocab = {v:k for k,v in vocab.items()}

# next_token_logits = torch.tensor([4.51,0.89,-1.90,6.75,1.63,-1.62,-1.89,6.28,1.79])

# def print_sampled_tokens(probas):
#     torch.manual_seed(123)
#     sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
#     sampled_ids = torch.bincount(torch.tensor(sample))
#     for i, freq in enumerate(sampled_ids):
#         print(f"{freq} x {inverse_vocab[i]}")

# probas = torch.softmax(next_token_logits, dim=0)
# next_token_id = torch.argmax(probas).item()
# print(inverse_vocab[next_token_id])

# torch.manual_seed(123)
# next_token_id = torch.multinomial(probas, num_samples=1).item()
# print(inverse_vocab[next_token_id])

# print_sampled_tokens(probas)

# def softmax_with_temperature(logits, temperature):
#     scaled_logits = logits / temperature
#     return torch.softmax(scaled_logits, dim=0)

# temperatures = [1,0.1,5]
# scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
# x = torch.arange(len(vocab))
# bar_width = 0.15
# fig, ax = plt.subplots(figsize=(5,3))
# for i, T in enumerate(temperatures):
#     rects = ax.bar(x+i * bar_width, scaled_probas[i], bar_width, label=f"Temperature = {T}")
#     print(f"Temperature: {T}\n")
#     print_sampled_tokens(scaled_probas[i])
# print(scaled_probas[2])
# ax.set_ylabel("Probability")
# ax.set_xticks(x)
# ax.set_xticklabels(vocab.keys(), rotation=90)
# ax.legend()
# plt.tight_layout()
# plt.show()

# top_k=3
# top_logits, top_pos = torch.topk(next_token_logits, top_k)
# print("Top logits:", top_logits)
# print("Top positions:", top_pos)

# new_logits = torch.where(condition=next_token_logits < top_logits[-1],input=torch.tensor(float('-inf')), other=next_token_logits)
# print(new_logits)

# topk_probas = torch.softmax(new_logits, dim=0)
# print(topk_probas)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def reload_and_continue_training():
#     checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
#     model = GPTModel(GPT_CONFIG_124M)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
#     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#     model.train()

#     file_path = "the-verdict.txt"
#     with open(file_path, "r", encoding="utf-8") as file:
#         text_data = file.read()

#     tokenizer = tiktoken.get_encoding("gpt2")

#     total_characters = len(text_data)
#     total_tokens = len(tokenizer.encode(text_data))
#     print("Characters:", total_characters)
#     print("Tokens:",total_tokens)

#     train_ratio = 0.90
#     split_idx = int(train_ratio * len(text_data))
#     train_data = text_data[:split_idx]
#     val_data = text_data[split_idx:]

#     torch.manual_seed(123)
#     train_loader = create_dataloader_v1(train_data, batch_size=2, max_length=GPT_CONFIG_124M['context_length'],stride=GPT_CONFIG_124M["context_length"],drop_last=True,shuffle=True,num_workers=0)
#     val_loader = create_dataloader_v1(val_data, batch_size=2, max_length=GPT_CONFIG_124M['context_length'],stride=GPT_CONFIG_124M["context_length"],drop_last=False,shuffle=False,num_workers=0)

#     train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs = 1, eval_freq=5, eval_iter=5,start_context="Every effort moves you", tokenizer=tokenizer)

# reload_and_continue_training()

# model = GPTModel(GPT_CONFIG_124M)
# model.load_state_dict(torch.load("model.pth", map_location=device))

# checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
# model = GPTModel(GPT_CONFIG_124M)
# model.load_state_dict(checkpoint["model_state_dict"])
# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
# optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# model.eval()

# tokenizer = tiktoken.get_encoding("gpt2")
# torch.manual_seed(123)
# token_ids = generate(model=model, idx=text_to_token_ids("Every effort moves you", tokenizer), max_new_tokens=25, context_size=GPT_CONFIG_124M["context_length"], top_k=25, temperature=1.4)
# print("output text:\n", token_ids_to_text(token_ids, tokenizer))


#region Model Definition and Training + BETTER GENERATE
# def train_and_save():
#     file_path = "the-verdict.txt"
#     with open(file_path, "r", encoding="utf-8") as file:
#         text_data = file.read()

#     tokenizer = tiktoken.get_encoding("gpt2")

#     total_characters = len(text_data)
#     total_tokens = len(tokenizer.encode(text_data))
#     print("Characters:", total_characters)
#     print("Tokens:",total_tokens)

#     train_ratio = 0.90
#     split_idx = int(train_ratio * len(text_data))
#     train_data = text_data[:split_idx]
#     val_data = text_data[split_idx:]

#     torch.manual_seed(123)
#     model = GPTModel(GPT_CONFIG_124M)
#     model.eval()
#     train_loader = create_dataloader_v1(train_data, batch_size=2, max_length=GPT_CONFIG_124M['context_length'],stride=GPT_CONFIG_124M["context_length"],drop_last=True,shuffle=True,num_workers=0)
#     val_loader = create_dataloader_v1(val_data, batch_size=2, max_length=GPT_CONFIG_124M['context_length'],stride=GPT_CONFIG_124M["context_length"],drop_last=False,shuffle=False,num_workers=0)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
#     num_epochs=10
#     train_losses, val_losses, tokens_seen = train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq=5, eval_iter=5,start_context="Every effort moves you", tokenizer=tokenizer)

#     epochs_tensor = torch.linspace(0,num_epochs, len(train_losses))
#     plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

#     model.to("cpu")
#     model.eval()

#     tokenizer = tiktoken.get_encoding("gpt2")
#     torch.manual_seed(123)
#     token_ids = generate(model=model, idx=text_to_token_ids("Every effort moves you", tokenizer), max_new_tokens=25, context_size=GPT_CONFIG_124M["context_length"], top_k=25, temperature=1.4)
#     print("output text:\n", token_ids_to_text(token_ids, tokenizer))

#     torch.save({"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict()}, "model_and_optimizer.pth")

#endregion

# train_and_save()


#region Model Definition and Training

# file_path = "the-verdict.txt"
# with open(file_path, "r", encoding="utf-8") as file:
#     text_data = file.read()

# tokenizer = tiktoken.get_encoding("gpt2")

# total_characters = len(text_data)
# total_tokens = len(tokenizer.encode(text_data))
# print("Characters:", total_characters)
# print("Tokens:",total_tokens)

# train_ratio = 0.90
# split_idx = int(train_ratio * len(text_data))
# train_data = text_data[:split_idx]
# val_data = text_data[split_idx:]

# torch.manual_seed(123)
# model = GPTModel(GPT_CONFIG_124M)
# model.eval()
# train_loader = create_dataloader_v1(train_data, batch_size=2, max_length=GPT_CONFIG_124M['context_length'],stride=GPT_CONFIG_124M["context_length"],drop_last=True,shuffle=True,num_workers=0)
# val_loader = create_dataloader_v1(val_data, batch_size=2, max_length=GPT_CONFIG_124M['context_length'],stride=GPT_CONFIG_124M["context_length"],drop_last=False,shuffle=False,num_workers=0)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
# num_epochs=10
# train_losses, val_losses, tokens_seen = train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq=5, eval_iter=5,start_context="Every effort moves you", tokenizer=tokenizer)

# epochs_tensor = torch.linspace(0,num_epochs, len(train_losses))
# plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# model.to("cpu")
# model.eval()

# tokenizer = tiktoken.get_encoding("gpt2")
# token_ids = generate_text_simple(model=model, idx=text_to_token_ids("Every effort moves you", tokenizer), max_new_tokens=25, context_size=GPT_CONFIG_124M["context_length"])
# print("output text:\n", token_ids_to_text(token_ids, tokenizer))

#endregion



# with torch.no_grad():
#     train_loss = calc_loss_loader(train_loader, model, device)
#     val_loss = calc_loss_loader(val_loader, model, device)
# print("Training Loss:", train_loss)
# print("Validation Loss:", val_loss)

# print("Train Loader:")
# for x,y in train_loader:
#     print(x.shape, y.shape)
# print("Val Loader")
# for x,y in val_loader:
#     print(x.shape, y.shape)

# torch.manual_seed(123)
# model = GPTModel(GPT_CONFIG_124M)
# model.eval()

# start_context = "Every effort moves you"
# tokenizer = tiktoken.get_encoding("gpt2")

# token_ids = generate_text_simple(model= model, idx=text_to_token_ids(start_context, tokenizer), max_new_tokens=10, context_size = GPT_CONFIG_124M['context_length'])
# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))



# inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
#                        [40,    1107, 588]])   #  "I really like"]

# targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
#                         [1107,  588, 11311]]) #  " really like chocolate"]


# with torch.no_grad():
#     logits = model(inputs)

# print("Logits shape:",logits.shape)
# print("Targets shape:",targets.shape)

# logits_flat = logits.flatten(0,1)
# targets_flat = targets.flatten()
# print("Flattened logits:", logits_flat.shape)
# print("Flattened targets:",targets_flat.shape)

# loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
# print(loss)

# probas = torch.softmax(logits, dim=-1)
# print(probas.shape)

# token_ids = torch.argmax(probas, dim=-1, keepdim=True)
# print("Token IDs:\n", token_ids)
# print(f"Targets batch 1: {token_ids_to_text(targets[0],tokenizer)}")
# print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

# text_idx = 0
# target_probas_1 = probas[text_idx, [0,1,2],targets[text_idx]]
# print("Text 1:",target_probas_1)

# text_idx = 1
# target_probas_2 = probas[text_idx, [0,1,2],targets[text_idx]]
# print("Text 2:",target_probas_2)

# log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
# print(log_probas)

# avg_log_probas = torch.mean(log_probas)
# print(avg_log_probas)

# neg_avg_log_probas = -1 * avg_log_probas
# print(neg_avg_log_probas)

