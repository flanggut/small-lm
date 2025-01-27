import math
import time
from functools import partial

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import data as Data


class BigramLanguageModel(nn.Module):
    """The dumbest model, just a simple distribution of tokens, given the previous token"""

    def __init__(self, vocabulary_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocabulary_size, vocabulary_size)

    def __call__(self, idx):
        logits = self.token_embedding_table(idx)
        return logits

    def generate(self, idx: mx.array):
        logits = self(idx)
        logits = logits[:, -1, :]
        sample = mx.random.categorical(logits, num_samples=1)
        yield sample

        while True:
            logits = self(sample)
            logits = logits[:, -1, :]
            sample = mx.random.categorical(logits, num_samples=1)
            yield sample


class LlamaAttention(nn.Module):
    """Proper multihead attention module based on Llama."""

    def __init__(self, dims: int, num_heads: int):
        super().__init__()

        self.num_heads = num_heads

        # Rotaty position embedding
        self.rope = nn.RoPE(dims // num_heads)
        self.query_projection = nn.Linear(dims, dims, bias=False)
        self.key_projection = nn.Linear(dims, dims, bias=False)
        self.value_projection = nn.Linear(dims, dims, bias=False)
        self.out_projection = nn.Linear(dims, dims, bias=False)

    def __call__(self, queries, keys, values, mask, cache=None):
        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

        num_heads = self.num_heads
        B, L, D = queries.shape

        # Reshape to broadcast across heads.
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)

        # Apply PE based on cache, update cache.
        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Perform actual attention compute.
        scale = math.sqrt(1 / queries.shape[-1])
        # Transpose keys for dot products.
        scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)
        # Apply mask.
        if mask is not None:
            scores = scores + mask
        scores = mx.softmax(scores, axis=-1)

        # Apply scores to values, and reshape back to input.
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        # Project to output and return cache.
        return self.out_projection(values_hat), (keys, values)


class LlamaEncoderLayer(nn.Module):
    """Llamaa encoder layer with RMS norm and SwiGlu activation."""

    def __init__(self, dims: int, mlp_dims: int, num_heads: int):
        super().__init__()

        self.attention = LlamaAttention(dims, num_heads)

        self.norm1 = nn.RMSNorm(dims)
        self.norm2 = nn.RMSNorm(dims)

        self.linear1 = nn.Linear(dims, mlp_dims, bias=False)
        self.linear2 = nn.Linear(dims, mlp_dims, bias=False)
        self.linear3 = nn.Linear(mlp_dims, dims, bias=False)

    def __call__(self, x, mask=None, cache=None):
        y = self.norm1(x)
        y, cache = self.attention(y, y, y, mask, cache)
        x = x + y

        y = self.norm2(x)
        a = self.linear1(y)
        b = self.linear2(y)
        # Activation SwiGLU
        y = a * mx.sigmoid(a) * b
        y = self.linear3(y)
        x = x + y
        return x, cache


class Llama(nn.Module):
    """Full Llama Model."""

    def __init__(
        self,
        num_layers: int,
        vocab_size: int,
        dims: int,
        mlp_dims: int,
        num_heads: int,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dims)
        self.layers = [
            LlamaEncoderLayer(dims, mlp_dims, num_heads) for _ in range(num_layers)
        ]
        self.norm = nn.RMSNorm(dims)
        self.out_proj = nn.Linear(dims, vocab_size, bias=False)

    def __call__(self, x):
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.embedding.weight.dtype)
        x = self.embedding(x)
        for layer in self.layers:
            x, _ = layer(x, mask=mask)
        x = self.norm(x)
        return self.out_proj(x)

    def generate(self, x, temp=1.0):
        cache = []

        # Make an additive causal mask. We will need that to process the prompt.
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.embedding.weight.dtype)

        # First we process the prompt x the same way as in __call__ but
        # save the caches in cache
        x = self.embedding(x)
        for layer in self.layers:
            x, c = layer(x, mask=mask)
            cache.append(c)  # <--- we store the per layer cache in a simple python list
        x = self.norm(x)
        logits = self.out_proj(
            x[:, -1]
        )  # <--- we only care about the last logits that generate the next token
        y = mx.random.categorical(logits * (1 / temp))

        # y now has size [1]
        # Since MLX is lazily evaluated nothing is computed yet.
        # Calling y.item() would force the computation to happen at
        # this point but we can also choose not to do that and let the
        # user choose when to start the computation.
        yield y

        # Now we parsed the prompt and generated the first token we
        # need to feed it back into the model and loop to generate the
        # rest.
        while True:
            # Unsqueezing the last dimension to add a sequence length
            # dimension of 1
            x = y[:, None]

            x = self.embedding(x)
            for i in range(len(cache)):
                # We are overwriting the arrays in the cache list. When
                # the computation will happen, MLX will be discarding the
                # old cache the moment it is not needed anymore.
                x, cache[i] = self.layers[i](x, mask=None, cache=cache[i])
            x = self.norm(x)
            y = self.out_proj(x[:, -1])
            y = mx.random.categorical(y * (1 / temp))

            yield y


def loss_fn(model: nn.Module, x: mx.array, y: mx.array, reduce=True):
    logits = model(x)
    losses = nn.losses.cross_entropy(logits, y)
    return mx.mean(losses) if reduce else mx.mean(losses, axis=(-1, -2))


def eval_fn(model: nn.Module, inputs: mx.array, targets: mx.array, batch_size: int):
    loss = 0
    for s in range(0, targets.shape[0], batch_size):
        bx, by = inputs[s : s + batch_size], targets[s : s + batch_size]
        losses = loss_fn(model, bx, by, reduce=False)
        loss += mx.sum(losses)

    return loss / (len(targets) // batch_size)


def main():
    mx.set_default_device(mx.gpu)
    # init data
    np.random.seed(1)
    mx.random.seed(1)
    train_data, validation_data, tokenizer = Data.load_data()
    vocab_size = len(tokenizer.vocabulary)
    context_size = 128
    generate_length = 100

    # model = BigramLanguageModel(len(tokenizer.vocabulary))
    model = Llama(
        num_layers=4, vocab_size=vocab_size, dims=64, mlp_dims=128, num_heads=4
    )

    batch_size = 32

    test_idx = mx.zeros((1, 1), dtype=mx.uint32)

    generator = model.generate(test_idx)
    gen = []
    for _ in range(generate_length):
        gen.append(next(generator).item())

    print("Initial Sample:")
    result = tokenizer.decode(gen)
    print(result)

    optimizer = optim.AdamW(learning_rate=1e-3)

    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(inputs, targets):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, inputs, targets)
        optimizer.update(model, grads)
        return loss

    @partial(mx.compile, inputs=model.state)
    def eval_full(inputs, targets):
        return eval_fn(model, inputs, targets, batch_size=4096)

    validation_inputs, validation_targets = map(
        mx.array, Data.to_samples(context_size, validation_data)
    )

    tic = time.perf_counter()
    validation_loss = eval_full(validation_inputs, validation_targets)
    print(f"{validation_loss=}")
    toc = time.perf_counter()
    print(f"Eval took: {(toc-tic):.3f}s")

    # Main training loop
    train_losses = []
    tic = time.perf_counter()
    train_iterator = Data.iterate_batches(batch_size, context_size, train_data)
    for train_iter, (inputs, targets) in zip(range(5000), train_iterator):
        xb, yb = map(mx.array, (inputs, targets))
        loss = step(xb, yb)
        train_losses.append(loss.item())
        if train_iter % 500 == 0:
            train_loss = np.mean(train_losses)
            print(f"Iter {train_iter}, train loss: {train_loss}")

    toc = time.perf_counter()
    print(f"Optimization took: {(toc-tic):.3f}s")

    validation_loss = eval_full(validation_inputs, validation_targets)
    print(f"{validation_loss=}")

    print("\nFinal Sample:")
    generator = model.generate(test_idx)
    gen = []
    for _ in range(generate_length):
        gen.append(next(generator).item())
    result = tokenizer.decode(gen)
    print(result)


if __name__ == "__main__":
    main()
