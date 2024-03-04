import torch
import torch.nn as nn
from torch.nn import functional as F

DATA_PATH = "/Users/mohamedadelabdelhady/workspace/kaggle-sandbox/nano-gpt/input.txt"
torch.manual_seed(1337)

# The simplest model will be a bigram model


# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        # channels are the size of the embedding table 65 in this case
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def main():
    text = get_input_data()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[x] for x in l])

    print(encode("hi there"))
    print(decode(encode("hi there")))

    data = torch.tensor(encode(text), dtype=torch.long)
    print(data.shape, data.dtype)
    print(data[:100])

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # context_size or the block size is the amount of data we feed into the transformer at once
    # the input to the transformer will be a tensor with shape ranging from 1 -> context_length
    block_size = 8
    # x = train_data[:block_size]
    # y = train_data[1:block_size+1]
    # print(x)
    # print(y)
    # for t in range(block_size):
    #     context = x[:t+1]
    #     target = y[t]
    #     print(f"when context is {context} target is {target}")

    batch_size = 4

    def get_batch(split):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y
    xb, yb = get_batch("train")
    print("inputs")
    print(xb.shape)
    print(xb)
    print("target")
    print(yb.shape)
    print(yb)

    print("-"*80)
    for b in range(batch_size):
        for t in range(block_size):
            context = xb[b, :t + 1]
            target = yb[b, t]
            print(f"when context is {context} target is {target}")

    print(vocab_size)
    model = BigramLanguageModel(vocab_size)
    pred, loss = model(xb, yb)
    # initial loss should be -ln(1/65)
    print(pred)
    print(loss)

    idx = torch.zeros((1,1), dtype=torch.long)
    print(decode(model.generate(idx, 100)[0].tolist()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch_size = 32
    for step in range(10_000):
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print(loss)
    idx = torch.zeros((1, 1), dtype=torch.long)
    print(decode(model.generate(idx, 100)[0].tolist()))

def get_input_data():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


if __name__ == "__main__":
    main()
