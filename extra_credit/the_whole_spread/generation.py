import torch

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

def generate_text_simple_cached(model, idx, max_new_tokens, context_size=None, use_cache=True):
    model.eval()

    ctx_len = context_size or model.pos_emb.num_embeddings
    kv_window_size = model.kv_window_size

    with torch.no_grad():
        if use_cache:
            model.reset_kv_cache()

            input_tokens = idx[:, -ctx_len:]
            input_tokens_length = input_tokens.size(1)

            # prefill to handle input_tokens_length > kv_window_size
            for i in range(0, input_tokens_length, kv_window_size):
                chunk = input_tokens[:, i:i+kv_window_size]
                logits = model(chunk, use_cache=True)

            # can't generate more than ctx_len of result
            # due to the limitation of position embedding
            max_generable = ctx_len - input_tokens_length
            max_new_tokens = min(max_new_tokens, max_generable)

            for _ in range(max_new_tokens):
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx
