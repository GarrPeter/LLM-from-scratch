import torch
import tiktoken
import time
import argparse
from model import GPTModel
from generation import generate_text_simple_cached

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--emb_dim", type=int, default=768, help="Model embedding dimension.")
    # parser.add_argument("--hidden_dim", type=int, default=768*4, help="Intermediate FFN or MoE size.")
    # parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads.")
    # parser.add_argument("--n_layers", type=int, default=12, help="Number of transformer blocks.")
    # parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate.")
    # parser.add_argument(
    #     "--no_kv_cache",
    #     action="store_true",
    #     help="Disable KV caching during generation.",
    # )

    # parser.add_argument(
    #     "--num_experts",
    #     type=int,
    #     default=0,
    #     help="Number of experts. If 0, use dense FFN. If >0, use MoE.",
    # )
    # parser.add_argument(
    #     "--num_experts_per_tok",
    #     type=int,
    #     default=2,
    #     help="Top-k experts per token when using MoE (ignored if num_experts=0).",
    # )

    # args = parser.parse_args()

    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "hidden_dim": 768*4,
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False,       # Query-Key-Value bias
        "kv_window_size": 1024,  # NEW: KV cache window size
        "num_experts": 32,
        "num_experts_per_tok": 2
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # disable dropout

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    token_ids = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=200,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start

    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", token_ids)
    print("Output length:", len(token_ids[0]))
    print("Output text:", decoded_text)

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0])/total_time)} tokens/sec")
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")


if __name__ == "__main__":
    main()