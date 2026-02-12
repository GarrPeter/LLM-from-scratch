import torch
import attention;

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

inputs = torch.tensor(torch.rand(6,768))


batch = torch.stack((inputs, inputs), dim=0)
# print(batch.shape)

# d_in = inputs.shape[1]
# d_out = 2

# torch.manual_seed(123)
# context_length = batch.shape[1]
# ca = attention.CausalAttention(d_in, d_out, context_length, 0.0)
# context_vecs = ca(batch)
# print("context_vecs.shape: ",context_vecs.shape)

torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 768
mha = attention.MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=12)
context_vecs = mha(batch)
print(context_vecs)
print("Shape: ", context_vecs.shape)
# print(context_vecs)
# print("context_vecs.shape: ", context_vecs.shape)

# torch.manual_seed(789)
# sa_v2 = attention.SelfAttention_v2(d_in, d_out)

# queries = sa_v2.W_query(inputs)
# keys = sa_v2.W_key(inputs)
# attn_scores = queries @ keys.T
# attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

# context_length = attn_scores.shape[0]
# mask_simple = torch.tril(torch.ones(context_length, context_length))

# masked_simple = attn_weights*mask_simple


# rows_sums = masked_simple.sum(dim=-1, keepdim=True)
# masked_simple_norm = masked_simple / rows_sums


# mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
# masked = attn_scores.masked_fill(mask.bool(), -torch.inf)


# attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
# print(attn_weights)

# torch.manual_seed(123)
# dropout = torch.nn.Dropout(0.5)
# example = torch.ones(6,6)

# torch.manual_seed(123)
# print(dropout(attn_weights))

# torch.manual_seed(789)
# sa_v1 = attention.SelfAttention_v1(d_in, d_out)
# sa_v1.W_query = torch.nn.Parameter(sa_v2.W_query.weight.T)
# sa_v1.W_key = torch.nn.Parameter(sa_v2.W_key.weight.T)
# sa_v1.W_value = torch.nn.Parameter(sa_v2.W_value.weight.T)
# print(sa_v1(inputs))

# attn_scores = inputs @ inputs.T

# attn_weights = torch.softmax(attn_scores, dim=-1)

# all_context_vecs = attn_weights @ inputs


# x_2 = inputs[1]
# d_in = inputs.shape[1]
# d_out = 2

# torch.manual_seed(123)
# W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
# W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
# W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# querys = inputs @ W_query
# keys = inputs @ W_key
# values = inputs @ W_value

# attn_scores_2 = querys[1] @ keys.T

# d_k = keys.shape[-1]
# attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)

# context_vec_2 = attn_weights_2 @ values
# print(context_vec_2)