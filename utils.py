import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.tokenize.treebank import TreebankWordDetokenizer
import gpumap

words = set(nltk.corpus.words.words())


# __import__("pdb").set_trace()

def dist2similarity(distance):
    # turn distance into similarity. if distance is very small we have value ~1, if distance is infinite we have 0.
    # something like cosine. Also scale distance by 0.01 to not get too small values.
    return torch.exp(-distance * 0.05)

def adjust_cl_ids(args, ids):
    for i in ids:
        if i == 0:
            cl = torch.Tensor([1, 0]).view(1,2).to(f'cuda:{args.gpu[0]}')
        elif i == 1:
            cl = torch.Tensor([0, 1]).view(1, 2).to(f'cuda:{args.gpu[0]}')
        args.prototype_class_identity = torch.cat((args.prototype_class_identity, cl))
    return args

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
    args = adjust_cl_ids(args, protos2add[1])
    protos2add = model.compute_embedding(protos2add[0], args)
    protos2add = protos2add.view(-1, model.enc_size, args.proto_size).to(f'cuda:{args.gpu[0]}')
    new_protos = torch.cat((model.protolayer.data,protos2add))
    model.protolayer = nn.Parameter(new_protos,requires_grad=True)
    # reassign last layer, add new ones
    weights2keep = model.fc.weight.data.detach().clone()
    model.fc = nn.Linear((args.num_prototypes+len(protos2add)), args.num_classes, bias=False).to(f'cuda:{args.gpu[0]}')
    new_weights = nn.init.uniform_(torch.empty(args.num_classes, len(protos2add))).to(f'cuda:{args.gpu[0]}')
    model.fc.data = torch.cat((weights2keep, new_weights), dim=1)
    args.num_prototypes = args.num_prototypes + len(protos2add)
    args, model = update_params(args, model)
    return args, model

def remove_prototypes(args, protos2remove, model, use_cos=False):
    mask = [True if p not in protos2remove else False for p in range(args.num_prototypes)]
    args.prototype_class_identity =  args.prototype_class_identity[mask,:]
    if use_cos:
        comb = torch.combinations(torch.arange(args.num_prototypes), r=2)
        for k, l in comb:
            similarity = F.cosine_similarity(model.protolayer[k, :, :], model.protolayer[l, :, :], dim=0)
            similarity = torch.sum(similarity) / args.proto_size
            if similarity>0.9:
                protos2remove.append(int(k))

    # make list items unique
    protos2remove = list(set(protos2remove))
    protos2keep = [p for p in list(range(args.num_prototypes)) if p not in protos2remove]
    # reassign protolayer, remove unneeded ones and only keep useful ones
    model.protolayer = nn.Parameter(model.protolayer.data[protos2keep,:,:],requires_grad=True)
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
    return args, model

def get_nearest(args, model, train_batches_unshuffled, text_train, labels_train):
    model.eval()
    dist = []
    for batch, _ in train_batches_unshuffled:
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
    if len(new_prototypes.size()) < 3: new_prototypes.unsqueeze_(-1)
    # only assign new prototypes if cosine similarity to old one is close
    angle =  F.cosine_similarity(model.protolayer, new_prototypes)
    mask = (angle>0.85).squeeze()

    # further reduce length if possible
    for i in range(args.num_prototypes):
        if len(pruned_protos[i])>4:
            new_prototxts_ = " ".join(pruned_protos[i].split()[0:4])
        else:
            new_prototxts_ = pruned_protos[i]
        new_prototypes_ = model.compute_embedding(new_prototxts_, args).to(f'cuda:{args.gpu[0]}')
        angle_ = F.cosine_similarity(model.protolayer[i].T, new_prototypes_)
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

def visualize_protos(args, embedding, labels, prototypes, model, proto_labels):
    # sample from data set for plot
    sample_size = 1000
    rnd_samples = np.random.randint(embedding.shape[0], size=sample_size)
    rnd_labels = [labels[i] for i in rnd_samples]
    cdict_d = {0: 'red', 1: 'green'}
    ldict_d = {0: 'data_neg', 1: 'data_pos'}
    cdict_p = {0: 'blue', 1: 'orange'}
    ldict_p = {0: 'proto_neg', 1: 'proto_pos'}
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # sample data again to be able to visualize word embedding data, otherwise transformation too hard to compute
    if len(embedding.shape) == 3:
        sample_size = 10000
        seq_length = embedding.shape[1]
        rnd_samples = np.random.randint(embedding.shape[0], size=sample_size)
        rnd_labels = [[labels[i]]*embedding.shape[1] for i in rnd_samples]
        rnd_labels = [label for sent in rnd_labels for label in sent]
        # flatten tensors and reduce size since otherwise not feasible for PCA
        embedding = embedding[rnd_samples]
        embedding = embedding.reshape(-1,model.enc_size)
        prototypes = prototypes.reshape(-1,model.enc_size)
        # subsample again for plot
        rnd_samples = ((np.random.randint(sample_size, size=35) * seq_length).reshape(-1,1) + np.arange(seq_length)).reshape(-1)
        rnd_labels = np.array(rnd_labels)[rnd_samples].tolist()

    if args.trans_type == 'PCA':
        pca = PCA(n_components=2)
        pca.fit(embedding)
        print("Explained variance ratio of components after transform: ", pca.explained_variance_ratio_)
        embed_trans = pca.transform(embedding)
        proto_trans = pca.transform(prototypes)
    elif args.trans_type == 'TSNE':
        tsne = TSNE(n_jobs=8,n_components=2).fit_transform(np.vstack((embedding,prototypes)))
        [embed_trans, proto_trans] = [tsne[:len(embedding)],tsne[len(embedding):]]
    elif args.trans_type == 'UMAP':
        umapped = gpumap.GPUMAP().fit_transform(np.vstack((embedding,prototypes)))
        [embed_trans, proto_trans] = [umapped[:len(embedding)], umapped[len(embedding):]]

    for cl in range(args.num_classes):
        ix = np.where(np.array(rnd_labels) == cl)[0]
        ax.scatter(embed_trans[rnd_samples[ix],0],embed_trans[rnd_samples[ix],1],c=cdict_d[cl],marker='x', label=ldict_d[cl],alpha=0.3)
        ix = np.where(proto_labels[:,cl].cpu().numpy() == 1)[0]
        if args.proto_size > 1:
            ix = [args.proto_size * i + x for i in ix.tolist() for x in range(args.proto_size)]
        ax.scatter(proto_trans[ix,0],proto_trans[ix,1],c=cdict_p[cl],marker='o',label=ldict_p[cl], s=80)

    n = 0
    for i in range(args.num_prototypes):
        txt = "P" + str(i + 1)
        if args.proto_size == 1:
            ax.annotate(txt, (proto_trans[i, 0], proto_trans[i, 1]), color='black')
        elif args.proto_size > 1:
            for j in range(args.proto_size):
                ax.annotate(txt, (proto_trans[n, 0], proto_trans[n, 1]), color='black')
                n += 1

    ax.legend()
    prefix = 'interacted_' if 'interacted' in args.model_path else ''
    fig.savefig(os.path.join(os.path.dirname(args.model_path), prefix+args.trans_type+'proto_vis2D.png'))


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
        # dist = torch.dist(model.protolayer[k, :, :], model.protolayer[l, :, :], p=2)
        # limit = torch.tensor(19).to(f'cuda:{args.gpu[0]}')
        # dist_sum += torch.maximum(limit, dist)
        dist_sum -= torch.mean(F.cosine_similarity(model.protolayer[k, :, :], model.protolayer[l, :, :],dim=0))
    # if distance small -> higher penalty
    divers_loss = - dist_sum / comb.size(0)

    l1_loss = model.fc.weight.data.norm(p=1)

    return distr_loss, clust_loss, sep_loss, divers_loss, l1_loss

####################################################
###### load toxicity data ##########################
####################################################

def parse_prompts_and_continuation(tag, discrete=True, discard=False, remove_long=True, file_dir=None):
    if file_dir is None:
        dataset_file = "./data/realtoxicityprompts/prompts.jsonl"
    else:
        dataset_file = file_dir + "/prompts.jsonl"

    assert os.path.isfile(dataset_file)
    dataset = pd.read_json(dataset_file, lines=True)
    prompts = pd.json_normalize(dataset['prompt'])
    assert tag in list(prompts.keys())
    text_prompts = prompts['text'].tolist()
    labels_prompts = prompts[tag].tolist()

    continuation = pd.json_normalize(dataset['continuation'])
    text_continuation = continuation['text'].tolist()
    labels_continuation = continuation[tag].tolist()

    text = text_continuation + text_prompts
    labels = labels_continuation + labels_prompts

    text, labels = preprocessor_toxic(text, labels, discrete, discard, remove_long)
    return text, labels

def parse_full(tag, args, discrete=True, discard=False, remove_long=True, file_dir=None):
    if file_dir is None:
        dataset_file = "./data/realtoxicityprompts/full data.jsonl"
        file_dir = "./data/realtoxicityprompts"
    else:
        dataset_file = file_dir + "/full data.jsonl"

    assert os.path.isfile(dataset_file)
    dataset = pd.read_json(dataset_file, lines=True)
    data = [x[0] for x in dataset['generations'].tolist()]
    assert tag in list(data[0].keys())

    text = list([e['text'] for e in data])
    labels = list([e[tag] for e in data])
    assert len(text) == len(labels)

    idx = []
    for i in range(len(text)):
        if labels[i] is None:
            idx.append(i)

    text = [t for i, t in enumerate(text) if i not in idx]
    labels = [l for i, l in enumerate(labels) if i not in idx]

    assert len(text) == len(labels)
    text, labels = preprocessor_toxic(text, labels, discrete, discard, remove_long)

    if args.data_name == 'toxicity_full':
        pickle.dump(text, open(file_dir + '/text_full.pkl', 'wb'))
        pickle.dump(labels, open(file_dir + '/labels_full.pkl', 'wb'))
    return text, labels

def parse_all(tag, args, file_dir=None):
    text, labels = [], []
    text_, labels_ = parse_prompts_and_continuation(tag, discard=args.discard, file_dir=file_dir)
    text += text_
    labels += labels_
    text_, labels_ = parse_full(tag, args, discard=args.discard, file_dir=file_dir)
    text += text_
    labels += labels_
    file_dir = os.path.join(args.data_dir, "realtoxicityprompts")
    pickle.dump(text, open(file_dir + '/text.pkl', 'wb'))
    pickle.dump(labels, open(file_dir + '/labels.pkl', 'wb'))
    return text, labels

def preprocessor_toxic(text, labels, discrete, discard, remove_long):
    if remove_long:
        labels = list([l for t, l in zip(text, labels) if len(t.split())<=25])
        text = list([t for t in text if len(t.split())<=25])

    if discard:
        text = list([t for t, l in zip(text, labels) if (l < 0.3 or l > 0.7)])
        labels = list([l for l in labels if (l < 0.3 or l > 0.7)])

    if discrete:
        labels = [0 if l < 0.5 else 1 for l in labels]

    # remove non english words (some reviews in Chinese, etc.), but keep digits and punctuation
    for i,t in enumerate(text):
        text[i] = convert_language(t)
        if not text[i]:
            del text[i], labels[i]

    assert len(text) == len(labels)
    max_len = 200_000
    if len(text)> max_len:
        text = [text[i] for i in range(max_len)]
        labels = [labels[i] for i in range(max_len)]
    return text, labels

def get_toxicity(args):
    f = '_full' if args.data_name == 'toxicity_full' else ''
    data_name = "realtoxicityprompts"
    set_dir = os.path.join(args.data_dir, data_name)
    text = pickle.load(open(set_dir + '/text' + f + '.pkl', 'rb'))
    labels = pickle.load(open(set_dir + '/labels' + f + '.pkl', 'rb'))
    return text, labels
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
        text_tmp = [TreebankWordDetokenizer().detokenize(sub_list) for sub_list in text_tmp]
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

    # remove non english words (some reviews in Chinese, etc.), but keep digits and punctuation
    for i,t in enumerate(text):
        text[i] = convert_language(t)
        if not text[i]:
            del text[i], labels[i]

    assert len(text) == len(labels)
    max_len = 200_000
    if len(text)> max_len:
        text = [text[i] for i in range(max_len)]
        labels = [labels[i] for i in range(max_len)]
    pickle.dump(text, open(set_dir + '/text.pkl', 'wb'))
    pickle.dump(labels, open(set_dir + '/labels.pkl', 'wb'))
    return text, labels

def convert_language(seq):
    return TreebankWordDetokenizer().detokenize(w for w in nltk.wordpunct_tokenize(seq) if (w.lower() in words) or
                                                (w.lower() in string.punctuation) or (w.lower().isdigit()))

def get_restaurant(args):
    set_dir = os.path.join(args.data_dir, args.data_name)
    text = pickle.load(open(set_dir + '/text.pkl', 'rb'))
    labels = pickle.load(open(set_dir + '/labels.pkl', 'rb'))
    return text, labels

####################################################
###### main loading function #######################
####################################################

def load_data(args):
    texts, labels = [], []
    if args.data_name == 'toxicity' or args.data_name == 'toxicity_full':
        texts, labels = get_toxicity(args)
    elif args.data_name == 'rt-polarity':
        texts, labels = get_reviews(args)
    elif args.data_name == 'restaurant':
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