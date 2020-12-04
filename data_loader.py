import os
import pandas as pd
import pickle

####################################################
###### load toxicity data ##########################
####################################################

def parse_prompts_and_continuation(tag, discrete=True):
    dataset_file = "./data/realtoxicityprompts/prompts.jsonl"
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
    dataset_file = "./data/realtoxicityprompts/full data.jsonl"
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

# get toxicity data, x is text as list of strings, y is list of ints (0,1)
def parse_all(tag):
    x, y = [], []
    x_, y_ = parse_prompts_and_continuation(tag)
    x += x_
    y += y_
    x_, y_ = parse_full(tag)
    x += x_
    y += y_
    return x, y


####################################################
###### load movie review data ######################
####################################################

def get_reviews(args):
    data_dir = args.data_dir
    set_list = ['train', 'dev', 'test']
    text, label = [], []
    # join train, dev, test; shuffle and split later
    for set_name in set_list:
        set_dir = os.path.join(data_dir, set_name)
        text_tmp = pickle.load(open(os.path.join(set_dir, 'word_sequences') + '.pkl', 'rb'))
        # join tokenized sentences back to full sentences for sentenceBert
        text_tmp = [' '.join(sub_list) for sub_list in text_tmp]
        text.extend(text_tmp)
        label_tmp = pickle.load(open(os.path.join(set_dir, 'labels') + '.pkl', 'rb'))
        # convert 'pos' & 'neg' to 1 & 0
        label_tmp = convert_label(label_tmp)
        label.extend(label_tmp)
    return text, label

def convert_label(labels):
    converted_labels = []
    for i,label in enumerate(labels):
        if label=='pos':
            converted_labels.append(1)
        elif label=='neg':
            converted_labels.append(0)
    return converted_labels


####################################################
###### main loading file ###########################
####################################################

def load_data(args):
    tag = args.data_name
    texts, labels = [], []
    if tag=='toxicity':
        texts, labels = parse_all(tag)
    elif tag=='reviews':
        texts, labels = get_reviews(args)
    return texts, labels