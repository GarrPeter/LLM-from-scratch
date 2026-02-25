import torch
from torch.utils.data import DataLoader
from dataset import process_spam_data, SpamDataset
import tiktoken
from gpt_download import download_and_load_gpt2
from model import GPTModel
from load_weights import load_weights_into_gpt
from util import generate_text_simple, text_to_token_ids, token_ids_to_text, calc_accuracy_loader, calc_loss_loader_classifier, calc_loss_batch_classifier
from training import train_classifier_simple
from plotter import plot_values
import time

# process_spam_data()

tokenizer = tiktoken.get_encoding("gpt2")

train_dataset = SpamDataset(csv_file="train.csv", max_length=None, tokenizer=tokenizer)
print(train_dataset.max_length)
val_dataset = SpamDataset(csv_file="validation.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)
test_dataset = SpamDataset(csv_file="test.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)

num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)

for input_batch, target_batch in train_loader:
    pass
print("Input Batch dimensions: ", input_batch.shape)
print("Label Batch dimensions:", target_batch.shape)

print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} testing batches")

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim":768, "n_layers": 12, "n_heads":12},
    "gpt2-medium (355M)": {"emb_dim":1024, "n_layers": 24, "n_heads":16},
    "gpt2-large (774M)": {"emb_dim":1280, "n_layers": 36, "n_heads":20},
    "gpt2-xl (1558M)": {"emb_dim":1600, "n_layers": 48, "n_heads":25},
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
setttings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

# text_1 = "Every effort moves you"
# token_ids = generate_text_simple(model=model, idx=text_to_token_ids(text_1, tokenizer), max_new_tokens=15, context_size=BASE_CONFIG["context_length"])
# print(token_ids_to_text(token_ids, tokenizer))

# text_2 = "Is the following text 'spam'? Answer with 'yes' or 'no': 'You are a winner you have been specially selected to receive $1000 cash or a $2000 award.'"

# token_ids = generate_text_simple(model=model, idx=text_to_token_ids(text_2, tokenizer), max_new_tokens=23, context_size=BASE_CONFIG["context_length"])
# print(token_ids_to_text(token_ids, tokenizer))

for param in model.parameters():
    param.requires_grad = False

torch.manual_seed(123)
num_classes = 2
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True

# inputs = tokenizer.encode("Do you have the time")
# inputs = torch.tensor(inputs).unsqueeze(0)
# with torch.no_grad():
#     outputs = model(inputs)
# print("Last output token", outputs[:,-1,:])
# probas = torch.softmax(outputs[:,-1,:], dim=-1)
# label = torch.argmax(probas)
# print("Class label:", label.item())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# torch.manual_seed(123)
# train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
# val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
# test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)

# print(f"Training accuracy: {train_accuracy*100:.2f}%")
# print(f"Validation accuracy: {val_accuracy*100:.2f}%")
# print(f"Test accuracy {test_accuracy*100:.2f}%")

# with torch.no_grad():
#     train_loss = calc_loss_loader_classifier(train_loader, model, device, num_batches=5)
#     val_loss = calc_loss_loader_classifier(val_loader, model, device, num_batches=5)
#     test_loss = calc_loss_loader_classifier(test_loader, model, device, num_batches=5)

# print(f"Training loss: {train_loss:.3f}%")
# print(f"Validation loss: {val_loss:.3f}%")
# print(f"Test loss {test_loss:.3f}%")

start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs=5

train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq=50, eval_iter=5)
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes")

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses, show=False)

epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))

plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, show=False)

train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)

print(f"Training accuracy {train_accuracy*100:.2f}%")
print(f"Validation accuracy {val_accuracy*100:.2f}&")
print(f"Test accuracy {test_accuracy*100:.2f}%")