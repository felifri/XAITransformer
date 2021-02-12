import os
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
import string
words = set(nltk.corpus.words.words())
cos = nn.CosineSimilarity()


# __import__("pdb").set_trace()

def dist2similarity(distance):
    # turn distance to similarity. if distance is small we have value 1, if distance is infinity we have 0
    return torch.exp(-distance)

def finetune_prototypes(args, protos2finetune, model):
    # # if mode == 'weights':
    # # set hook to ignore weights of prototypes to keep when computing gradient, to learn the other weights
    # gradient_mask_fc = torch.zeros(model.fc.weight.size()).to(f'cuda:{args.gpu[0]}')
    # gradient_mask_fc[:,protos2finetune] = 1
    # model.fc.weight.register_hook((lambda grad: grad.mul_(gradient_mask_fc)))
    # if mode == 'prototypes':
    # also, do not update the selected prototypes which should be kept
    gradient_mask_proto = torch.zeros(model.protolayer.size()).to(f'cuda:{args.gpu[0]}')
    gradient_mask_proto[protos2finetune, :, :] = 1
    model.protolayer.register_hook(lambda grad: grad.mul_(gradient_mask_proto))
    return model

def reinit_prototypes(args, protos2reinit, model):
    # reinitialize selected prototypes
    model.protolayer.data[protos2reinit,:] = nn.init.uniform_(torch.empty(model.protolayer.size()))[protos2reinit,:]
    # onyl retrain reinitialized protos and set gradients of other prototypes to zero
    model = finetune_prototypes(args, protos2reinit, model)
    return model

def add_prototypes(args, protos2add, model):
    # reassign protolayer, add new ones
    protos2add = model.compute_emb(protos2add, args)
    protos2add = protos2add.view(-1, model.enc_size, 1)
    new_protos = torch.cat((model.protolayer.data,protos2add))
    model.protolayer = nn.Parameter(new_protos,requires_grad=False)
    # reassign last layer, add new ones
    weights2keep = model.fc.data.detach().clone()
    model.fc = nn.Linear((args.num_prototypes+len(protos2add)), args.num_classes, bias=False)
    new_weights = nn.init.uniform_(torch.empty((len(protos2add), args.num_classes)))
    model.fc.data = torch.cat((weights2keep, new_weights))
    args.num_prototypes = args.num_prototypes + len(protos2add)
    args, model = update_params(args, model)
    return args, model

def remove_prototypes(args, protos2remove, model, use_cos=False):
    if use_cos:
        comb = torch.combinations(torch.arange(args.num_prototypes), r=2)
        for k, l in comb:
            # only increase penalty until distance reaches 19, above constant penalty, since we don't want prototypes to be
            # too far spread
            similarity = cos(model.protolayer[k, :, :], model.protolayer[l, :, :])
            similarity = torch.sum(similarity) / args.proto_size
            if similarity>0.9:
                protos2remove.append(int(k))

    # make list items unique
    protos2remove = list(set(protos2remove))
    protos2keep = [p for p in list(range(args.num_prototypes)) if p not in protos2remove]
    # reassign protolayer, remove unneeded ones and only keep useful ones
    model.protolayer = nn.Parameter(model.protolayer.data[protos2keep,:,:],requires_grad=False)
    weights2keep = model.fc.weight.data.detach().clone()[:,protos2keep]
    model.fc = nn.Linear(len(protos2keep), args.num_classes, bias=False)
    model.fc.weight.data = weights2keep
    args.num_prototypes = args.num_prototypes - len(protos2remove)
    args, model = update_params(args, model)
    return args, model

def update_params(args, model):
    model.num_prototypes = args.num_prototypes
    if args.dilated:
        model.num_filters = [model.num_prototypes // len(args.dilated)] * len(args.dilated)
        model.num_filters[0] += model.num_prototypes % len(args.dilated)
    args.prototype_class_identity =  args.prototype_class_identity[:args.num_prototypes,:]
    return args, model

def get_nearest(args, model, train_batches, text_train, labels_train):
    model.eval()
    dist = []
    for batch, _ in train_batches:
        batch = batch.to(f'cuda:{args.gpu[0]}')
        _, distances, _ = model.forward(batch)
        dist.append(distances)
    proto_ids, proto_texts = model.nearest_neighbors(dist, text_train, labels_train)
    return proto_ids, proto_texts

def prune_prototypes(args, proto_texts, model):
    # define stop words, but e.g. 'not' should not be removed
    stop_words = set(stopwords.words('english')) - {'not', 'no'}
    pruned_protos = []

    # remove stop words, non alpha words, single character words and punctuation
    for p in proto_texts:
        p = p.translate(str.maketrans('', '', string.punctuation+'“”—'))
        pruned_protos.append(" ".join([w for w in p.split() if w.isalpha() and not w in stop_words and len(w)>1]))
    new_prototypes = model.compute_embedding(pruned_protos, args).to(f'cuda:{args.gpu[0]}')
    if len(new_prototypes.size())<3: new_prototypes.unsqueeze_(-1)
    # only assign new prototypes if cosine similarity to old one is close
    angle = cos(model.protolayer, new_prototypes)
    mask = (angle>0.85).squeeze()

    # further reduce length if possible
    for i in range(args.num_prototypes):
        if len(pruned_protos[i])>4:
            new_prototxts_ = " ".join(pruned_protos[i].split()[0:4])
        else:
            new_prototxts_ = pruned_protos[i]
        new_prototypes_ = model.compute_embedding(new_prototxts_, args).to(f'cuda:{args.gpu[0]}')
        angle_ = cos(model.protolayer[i].T, new_prototypes_)
        if angle_>0.85:
            pruned_protos[i] = new_prototxts_
            new_prototypes[i] = new_prototypes_.T

    # mask: replace only words with high cos sim
    for i,m in enumerate(mask):
        if m: proto_texts[i] = pruned_protos[i]
    # assign new prototypes and don't update them when retraining
    model.protolayer.data[mask] = new_prototypes[mask]
    model.protolayer.requires_grad = False
    return model, proto_texts

def visualize_protos(args, embedding, labels, prototypes, n_components=2):
        # visualize prototypes
        if args.trans_type == 'PCA':
            pca = PCA(n_components=n_components)
            pca.fit(embedding)
            print("Explained variance ratio of components after transform: ", pca.explained_variance_ratio_)
            embed_trans = pca.transform(embedding)
            proto_trans = pca.transform(prototypes)
        elif args.trans_type == 'TSNE':
            tsne = TSNE(n_jobs=8,n_components=n_components).fit_transform(np.vstack((embedding,prototypes)))
            [embed_trans, proto_trans] = [tsne[:len(embedding)],tsne[len(embedding):]]
        rnd_samples = np.random.randint(embed_trans.shape[0], size=1000)
        rnd_labels = [labels[i] for i in rnd_samples]
        cdict_d = {0:'red', 1:'green'}
        ldict_d = {0:'data_neg', 1:'data_pos'}
        cdict_p = {0:'blue', 1:'orange'}
        ldict_p = {0:'proto_neg', 1:'proto_pos'}
        fig = plt.figure()
        if n_components==2:
            ax = fig.add_subplot(111)
            for cl in range(args.num_classes):
                ix = np.where(np.array(rnd_labels) == cl)
                ax.scatter(embed_trans[rnd_samples[ix],0],embed_trans[rnd_samples[ix],1],c=cdict_d[cl],marker='x', label=ldict_d[cl],alpha=0.6)
                ix = np.where(args.prototype_class_identity[:,cl].cpu().numpy() == 1)
                ax.scatter(proto_trans[ix,0],proto_trans[ix,1],c=cdict_p[cl],marker='o',label=ldict_p[cl], s=70)
            for i in range(args.num_prototypes):
                txt = "P" + str(i + 1)
                ax.annotate(txt, (proto_trans[i, 0], proto_trans[i, 1]), color='black')
        elif n_components==3:
            ax = fig.add_subplot(111, projection='3d')
            for cl in range(args.num_classes):
                ix = np.where(np.array(rnd_labels) == cl)
                ax.scatter(embed_trans[rnd_samples[ix],0],embed_trans[rnd_samples[ix],1],embed_trans[rnd_samples[ix],2],c=cdict_d[cl],marker='x', label=ldict_d[cl],alpha=0.6)
                ix = np.where(args.prototype_class_identity[:,cl].cpu().numpy() == 1)
                ax.scatter(proto_trans[ix,0],proto_trans[ix,1],proto_trans[ix,2],c=cdict_p[cl],marker='o',label=ldict_p[cl], s=70)
            for i in range(args.num_prototypes):
                txt = "P" + str(i+1)
                ax.annotate(txt, (proto_trans[i,0],proto_trans[i,1],proto_trans[i,2]), color='black')
        ax.legend()
        fig.savefig(os.path.join(os.path.dirname(args.model_path), args.trans_type+'proto_vis'+str(n_components)+'d.png'))

def proto_loss(prototype_distances, label, model, args):
    if args.class_specific:
        max_dist = torch.prod(torch.tensor(model.protolayer.size())) # proxy variable, could be any high value

        # prototypes_of_correct_class is tensor of shape  batch_size * num_prototypes
        # calculate cluster cost, high cost if same class protos are far distant
        prototypes_of_correct_class = torch.t(args.prototype_class_identity[:, label])
        inverted_distances, _ = torch.max((max_dist - prototype_distances) * prototypes_of_correct_class, dim=1)
        clust_loss = torch.mean(max_dist - inverted_distances)
        # assures that each sample is not too far distant form a prototype of its class
        inverted_distances, _ = torch.max((max_dist - prototype_distances) * prototypes_of_correct_class, dim=0)
        distr_loss = torch.mean(max_dist - inverted_distances)

        # calculate separation cost, low (highly negative) cost if other class protos are far distant
        prototypes_of_wrong_class = 1 - prototypes_of_correct_class
        inverted_distances_to_nontarget_prototypes, _ = \
            torch.max((max_dist - prototype_distances) * prototypes_of_wrong_class, dim=1)
        sep_loss = - torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)
    else:
        # Computes the interpretability losses (R1 and R2 from the paper (Li et al. 2018)) for the prototype nets.
        # r1: compute min distance from each prototype to an example, i.e batch_size * num_prototypes -> num_prototypes.
        # this assures that each prototype is as close as possible to at least one of the examples
        distr_loss = torch.mean(torch.min(prototype_distances, dim=0)[0])
        # r2: compute min distance from each example to prototype, i.e batch_size * num_prototypes -> batch_size.
        # this assures that each example is as close as possible to one of the prototypes, which is something like a
        # cluster cost
        clust_loss = torch.mean(torch.min(prototype_distances, dim=1)[0])
        sep_loss = 0

    # diversity loss, assures that prototypes are not too close
    dist_sum = 0
    comb = torch.combinations(torch.arange(args.num_prototypes), r=2)
    for k, l in comb:
        # only increase penalty until distance reaches 19, above constant penalty, since we don't want prototypes to be
        # too far spread
        dist = torch.dist(model.protolayer[k, :, :], model.protolayer[l, :, :], p=2)
        limit = torch.tensor(19).to(f'cuda:{args.gpu[0]}')
        dist_sum += torch.maximum(limit, dist)
        # dist_sum += cos(model.protolayer[k, :, :], model.protolayer[l, :, :])
    # if distance small -> higher penalty
    divers_loss = - dist_sum / comb.size(0)

    if args.use_l1_mask:
        l1_mask = 1 - torch.t(args.prototype_class_identity).to(f'cuda:{args.gpu[0]}')
        l1_loss = (model.fc.weight.data * l1_mask).norm(p=1)
    else:
        l1_loss = model.fc.weight.data.norm(p=1)

    return distr_loss, clust_loss, sep_loss, divers_loss, l1_loss

####################################################
###### load toxicity data ##########################
####################################################

def parse_prompts_and_continuation(tag, discrete=True, discard=False, file_dir=None):
    if file_dir is None:
        dataset_file = "./data/realtoxicityprompts/prompts.jsonl"
    else:
        dataset_file = file_dir + "/prompts.jsonl"

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

    if discard:
        x = list([a for a, e in zip(x, y) if (e < 0.3 or e > 0.7)])
        y = list([e for e in y if (e < 0.3 or e > 0.7)])

    if discrete:
        y = list([0 if e < 0.5 else 1 for e in y])

    return x, y


def parse_full(tag, discrete=True, discard=False, file_dir=None):
    if file_dir is None:
        dataset_file = "./data/realtoxicityprompts/full data.jsonl"
    else:
        dataset_file = file_dir + "/full data.jsonl"

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

    if discard:
        x = list([a for a, e in zip(x, y) if (e < 0.3 or e > 0.7)])
        y = list([e for e in y if (e < 0.3 or e > 0.7)])

    if discrete:
        y = [0 if e < 0.5 else 1 for e in y]

    return x, y

# get toxicity data, x is text as list of strings, y is list of ints (0,1)
def parse_all(tag, args, file_dir=None):
    x, y = [], []
    x_, y_ = parse_prompts_and_continuation(tag, discard=args.discard, file_dir=file_dir)
    x += x_
    y += y_
    x_, y_ = parse_full(tag, discard=args.discard, file_dir=file_dir)
    x += x_
    y += y_
    return x, y


####################################################
###### load movie review data ######################
####################################################

def get_reviews(args):
    set_list = ['train', 'dev', 'test']
    text, label = [], []
    # join train, dev, test; shuffle and split later
    for set_name in set_list:
        set_dir = os.path.join(args.data_dir, args.data_name, set_name)
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
###### load restaurant review data #################
####################################################

def preprocess_restaurant(args, binary=True, file_dir=None, remove_long=True):
    set_dir = os.path.join(args.data_dir, args.data_name)
    if file_dir is None:
        dataset_file = os.path.join(set_dir, "yelp_academic_dataset_review.json")
    else:
        dataset_file = file_dir + "/yelp_academic_dataset_review.json"
    assert os.path.isfile(dataset_file)
    dataset = pd.read_json(dataset_file, lines=True)
    text = dataset['text'].tolist()
    labels = dataset['stars'].tolist()
    assert len(text) == len(labels)

    if remove_long:
        labels = list([l for t, l in zip(text, labels) if len(t.split())<=25])
        text = list([t for t in text if len(t.split())<=25])

    if args.discard:
        text = list([t for t, l in zip(text, labels) if (l <= 1.0 or l >= 5.0)])
        labels = list([l for l in labels if (l <= 1.0 or l >= 5.0)])

    if binary:
        labels = list([0 if l < 2.5 else 1 for l in labels])

    # remove non english words (some reviews in Chinese, etc.)
    for i,t in enumerate(text):
        text[i] = convert_language(t)
        if not text[i]:
            del text[i], labels[i]

    assert len(text) == len(labels)
    max_len = 250_000
    if len(text)> max_len:
        text = [text[i] for i in range(max_len)]
        labels = [labels[i] for i in range(max_len)]
    pickle.dump(text, open(set_dir + '/text.pkl', 'wb'))
    pickle.dump(labels, open(set_dir + '/labels.pkl', 'wb'))
    return text, labels

def convert_language(seq):
    return " ".join(w for w in nltk.wordpunct_tokenize(seq) if w.lower() in words and w.isalpha())

def get_restaurant(args):
    set_dir = os.path.join(args.data_dir, args.data_name)
    text = pickle.load(open(set_dir + '/text.pkl', 'rb'))
    labels = pickle.load(open(set_dir + '/labels.pkl', 'rb'))
    return text, labels

####################################################
###### main loading function #######################
####################################################

def load_data(args, file_dir=None):
    tag = args.data_name
    texts, labels = [], []
    if tag == 'toxicity':
        texts, labels = parse_all(tag, args, file_dir)
    elif tag == 'toxicity_full':
        texts, labels = parse_full('toxicity', discard=args.discard, file_dir=file_dir)
    elif tag == 'rt-polarity':
        texts, labels = get_reviews(args)
    elif tag == 'restaurant':
        texts, labels = get_restaurant(args)
    return texts, labels


###### load/ store embedding to not compute it every single run again ######

def load_embedding(args, fname, set_name):
    path = os.path.join('data/embedding', args.data_name, fname+'_'+set_name+'.pt')
    assert os.path.isfile(path)
    return torch.load(path, map_location=torch.device('cpu'))

def save_embedding(embedding, args, fname, set_name):
    path = os.path.join('data/embedding', args.data_name)
    name = fname + '_' + set_name + '.pt'
    os.makedirs(path, exist_ok=True, mode=0o777)
    path = os.path.join(path, name)
    torch.save(embedding, path)