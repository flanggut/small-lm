import time
from functools import partial

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import data as Data


# The dumbest model, just predict the next letter, with only the previous letter as context.
class BigramLanguageModel(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocabulary_size, vocabulary_size)
        self.rng = np.random.default_rng()

    def __call__(self, idx):
        logits = self.token_embedding_table(idx)
        return logits

    def generate(self, idx: mx.array, max_new_tokens: int):
        for _ in range(max_new_tokens):
            logits = self(idx)
            logits = logits[:, -1, :]
            sample = mx.random.categorical(logits, num_samples=1)
            idx = mx.concatenate([idx, sample], axis=1)
        return idx


def loss_fn(model: nn.Module, x: mx.array, y: mx.array, reduce=True):
    logits = model(x)
    losses = nn.losses.cross_entropy(logits, y)
    return mx.mean(losses) if reduce else mx.mean(losses, axis=(-1, -2))


def eval_fn(model: nn.Module, dataset: np.array, context_size: int, batch_size: int):
    inputs, targets = map(mx.array, Data.to_samples(context_size, dataset))
    loss = 0
    for s in range(0, targets.shape[0], batch_size):
        bx, by = inputs[s : s + batch_size], targets[s : s + batch_size]
        bx, by = map(mx.array, (bx, by))
        losses = loss_fn(model, bx, by, reduce=False)
        loss += mx.sum(losses).item()

    return loss / (len(targets) // batch_size)


def main():
    mx.set_default_device(mx.gpu)
    # init data
    np.random.seed(1)
    mx.random.seed(1)
    train_data, validation_data, tokenizer = Data.load_data()

    model = BigramLanguageModel(len(tokenizer.vocabulary))
    context_size = 8

    idx = mx.zeros((1, 1), dtype=mx.uint32)
    gen = model.generate(idx, 100)[0].tolist()

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

    validation_loss = eval_fn(model, validation_data, context_size, 32)
    print(f"{validation_loss=}")

    tic = time.perf_counter()
    train_iterator = Data.iterate_batches(32, context_size, train_data)
    for train_iter, (inputs, targets) in zip(range(10000), train_iterator):
        xb, yb = map(mx.array, (inputs, targets))
        step(xb, yb)

    toc = time.perf_counter()
    print(f"Optimization took: {(toc-tic):.3f}s")

    validation_loss = eval_fn(model, validation_data, context_size, 32)
    print(f"{validation_loss=}")

    print("\nFinal Sample:")
    idx = mx.zeros((1, 1), dtype=mx.uint32)
    gen = model.generate(idx, 100)[0].tolist()
    result = tokenizer.decode(gen)
    print(result)


if __name__ == "__main__":
    main()
