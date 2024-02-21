import torch

DATA_PATH = "/Users/mohamedadelabdelhady/workspace/kaggle-sandbox/nano-gpt/input.txt"
torch.manual_seed(1337)

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
    print(xb)
    print("target")
    print(yb)


    print("-"*80)
    for b in range(batch_size):
        for t in range(block_size):
            context = xb[b, :t + 1]
            target = yb[b, t]
            print(f"when context is {context} target is {target}")


def get_input_data():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


if __name__ == "__main__":
    main()
