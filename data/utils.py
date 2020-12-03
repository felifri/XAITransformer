import pandas as pd
import os


def parse_prompts_and_continuation(tag, discrete=True):
    dataset_file = "./realtoxicityprompts/prompts.jsonl"
    assert os.path.isfile(dataset_file)
    dataset = pd.read_json(dataset_file, lines=True)
    prompts = pd.json_normalize(dataset['prompt'])
    assert tag in list(prompts.keys())
    x_prompts = prompts['text'].tolist()
    y_prompts = prompts[tag].tolist()

    continuation = pd.json_normalize(dataset['continuation'])
    x_continuation = continuation['text'].tolist()
    y_continuation = continuation[tag].tolist()

    x = x_continuation + x_prompts
    y = y_continuation + y_prompts

    if discrete:
        y = list([0 if e < 0.5 else 1 for e in y])

    return x, y


def parse_full(tag, discrete=True):
    dataset_file = "./realtoxicityprompts/full data.jsonl"
    assert os.path.isfile(dataset_file)
    dataset = pd.read_json(dataset_file, lines=True)
    data = [x[0] for x in dataset['generations'].tolist()]
    assert tag in list(data[0].keys())

    x = list([e['text'] for e in data])
    y = list([e[tag] for e in data])
    assert len(x) == len(y)

    idx = []
    for i in range(len(x)):
        if y[i] is None:
            idx.append(i)

    x = [e for i, e in enumerate(x) if i not in idx]
    y = [e for i, e in enumerate(y) if i not in idx]

    assert len(x) == len(y)
    if discrete:
        y = [0 if e < 0.5 else 1 for e in y]

    return x, y


def parse_all(tag):
    x, y = [], []
    x_, y_ = parse_prompts_and_continuation(tag)
    x += x_
    y += y_
    x_, y_ = parse_full(tag)
    x += x_
    y += y_
    return x, y


if __name__ == '__main__':
    parse_all(tag='toxicity')
