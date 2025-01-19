import numpy as np

import tokenizer as tk


def load_data() -> (np.array, np.array, tk.Tokenizer):
    with open("data.txt", "r") as f:
        text = f.read()

    print("Length of dataset in chars: ", len(text))

    chars = sorted(list(set(text)))
    tokenizer = tk.Tokenizer(chars)

    data = np.array(tokenizer.encode(text), dtype=np.int64)

    n = int(0.9 * len(data))
    train_data = data[:n]
    validation_data = data[n:]
    return (train_data, validation_data, tokenizer)


def to_samples(context_size: int, dataset: np.array) -> (np.array, np.array):
    tokens = dataset.size
    window_size = context_size + 1  # include target token
    samples = tokens - window_size + 1
    X = np.lib.stride_tricks.as_strided(
        dataset,
        shape=(samples, window_size),
        strides=(dataset.itemsize, dataset.itemsize),
    )
    return X[:, :-1], X[:, 1:]


def iterate_batches(batch_size: int, context_size: int, dataset: np.array):
    inputs, targets = to_samples(context_size, dataset)
    s = 0
    while True:
        if s == 0:
            # Reset permutation:
            perm = np.random.permutation(inputs.shape[0])
        ids = perm[s : s + batch_size]
        yield inputs[ids], targets[ids]
        s += batch_size
        if s >= inputs.shape[0]:
            s = 0
