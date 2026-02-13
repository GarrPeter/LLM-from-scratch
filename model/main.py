import tiktoken
import torch
import torch.nn as nn
from model import DummyGPTModel
from model import LayerNorm
from model import FeedForward
from model import GELU
from model import TransformerBlock
from model import GPTModel
from model import generate_text_simple


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate_emb": 0.1,
    "drop_rate_short": 0.1,
    "drop_rate_attn": 0.1,
    "qkv_bias": False
}

tokenizer = tiktoken.get_encoding("gpt2")

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded: ", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded tensor.shape:", encoded_tensor.shape)

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()
out = generate_text_simple(model=model, idx=encoded_tensor, max_new_tokens=6, context_size=GPT_CONFIG_124M["context_length"])
print("output:",out)
print("output length:",len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)

# batch = []
# txt1 = "Every effort moves you"
# txt2 = "Every day holds a"

# batch.append(torch.tensor(tokenizer.encode(txt1)))
# batch.append(torch.tensor(tokenizer.encode(txt2)))
# batch = torch.stack(batch, dim=0)

# torch.manual_seed(123)
# model = GPTModel(GPT_CONFIG_124M)
# out = model(batch)
# print("Input Batch:\n",batch)
# print("\nOutput Shape:",out.shape)
# print(out)

# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params:,}")

# total_ff_params = 0
# total_attn_params = 0;
# for trf in model.trf_blocks:
#     total_ff_params += sum(p.numel() for p in trf.ff.parameters())
#     total_attn_params += sum(p.numel() for p in trf.att.parameters())

# print(f"Total number of Feed Forward parameters: {total_ff_params:,}")
# print(f"Total number of Attention parameters: {total_attn_params:,}")


# total_size_bytes = total_params * 4
# total_size_mb = total_size_bytes / (1024* 1024)
# print(f"Total size of the model: { total_size_mb:.2f} MB")
# torch.manual_seed(123)
# x = torch.rand(2,4,768)
# block = TransformerBlock(GPT_CONFIG_124M)
# output = block(x)
# print("Input shape:", x.shape)
# print("Output shape:", output.shape)

# torch.manual_seed(123)
# model = DummyGPTModel(GPT_CONFIG_124M)
# logits = model(batch)

# torch.manual_seed(123)
# batch_example = torch.randn(2,5)
# layer = nn.Sequential(nn.Linear(5,6), nn.ReLU())
# out = layer(batch_example)

# ln = LayerNorm(emb_dim=5)
# out_ln = ln(batch_example)
# mean = out_ln.mean(dim=-1, keepdim=True)
# var = out_ln.var(dim=-1, keepdim=True, unbiased=False)

# ffn = FeedForward(GPT_CONFIG_124M)
# x= torch.rand(2,3,768)
# out = ffn(x)


# mean = out.mean(dim=-1, keepdim=True)
# var = out.var(dim=-1, keepdim=True)
# print("Mean:\n",mean)
# print("Variance:\n", var)

# out_norm = (out-mean) / torch.sqrt(var)
# mean = out_norm.mean(dim=-1, keepdim=True)
# var = out_norm.var(dim=-1, keepdim=True)
# print("Normalized Layer outputs:\n", out_norm)
# print("Mean:\n", mean)
# print("Variance:\n", var)

# class ExampleDeepNeuralNetwork(nn.Module):
#     def __init__(self, layer_sizes, use_shortcut):
#         super().__init__()
#         self.use_shortcut = use_shortcut
#         self.layers = nn.ModuleList([
#             nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
#             nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
#             nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
#             nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
#             nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
#         ])

#     def forward(self, x):
#         for layer in self.layers:
#             # Compute the output of the current layer
#             layer_output = layer(x)
#             # Check if shortcut can be applied
#             if self.use_shortcut and x.shape == layer_output.shape:
#                 x = x + layer_output
#             else:
#                 x = layer_output
#         return x


# def print_gradients(model, x):
#     # Forward pass
#     output = model(x)
#     target = torch.tensor([[0.]])

#     # Calculate loss based on how close the target
#     # and output are
#     loss = nn.MSELoss()
#     loss = loss(output, target)
    
#     # Backward pass to calculate the gradients
#     loss.backward()

#     for name, param in model.named_parameters():
#         if 'weight' in name:
#             # Print the mean absolute gradient of the weights
#             print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

# layer_sizes = [3,3,3,3,3,1]
# sample_input = torch.tensor([[1.,0.,-1.]])
# torch.manual_seed(123)
# model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
# print_gradients(model_without_shortcut, sample_input)

# torch.manual_seed(123)
# model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
# print_gradients(model_with_shortcut, sample_input)