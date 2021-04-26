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
from torchvision.utils import save_image

words = set(nltk.corpus.words.words())


# __import__("pdb").set_trace()

def dist2similarity(distance):
    # turn distance into similarity. if distance is very small we have value ~1, if distance is infinite we have 0.
    # something like cosine. Also scale distance by 0.05 to not get too small values.
    # return torch.exp(-distance * 0.05)
    return 10/distance

def ned_torch(x1: torch.Tensor, x2: torch.Tensor, dim=1, eps=1e-8) -> torch.Tensor:
    # to compute ned for two individual vectors e.g to compute a loss (NOT BATCHES/COLLECTIONS of vectorsc)
    if len(x1.size()) == 1:
        # [K] -> [1]
        ned_2 = 0.5 * ((x1 - x2).var() / (x1.var() + x2.var() + eps))
    # if the input is a (row) vector e.g. when comparing two batches of acts of D=1 like with scores right before sf
    elif x1.size() == torch.Size([x1.size(0), 1]):  # note this special case is needed since var over dim=1 is nan (1 value has no variance).
        # [B, 1] -> [B]
        ned_2 = 0.5 * ((x1 - x2)**2 / (x1**2 + x2**2 + eps)).squeeze()  # Squeeze important to be consistent with .var, otherwise tensors of different sizes come out without the user expecting it
    # common case is if input is a batch
    else:
        # e.g. [B, D] -> [B]
        ned_2 = 0.5 * ((x1 - x2).var(dim=dim) / (x1.var(dim=dim) + x2.var(dim=dim) + eps))
    return ned_2 ** 0.5

def nes_torch(x1, x2, dim=1, eps=1e-8):
    return 1 - ned_torch(x1, x2, dim, eps)

def adjust_cl_ids(args, ids):
    for i in ids:
        if i == 0:
            cl = torch.Tensor([1, 0]).view(1, 2).to(f'cuda:{args.gpu[0]}')
        elif i == 1:
            cl = torch.Tensor([0, 1]).view(1, 2).to(f'cuda:{args.gpu[0]}')
        args.prototype_class_identity = torch.cat((args.prototype_class_identity, cl))
    return args

def update_params(args, model):
    model.num_prototypes = args.num_prototypes
    if args.dilated:
        model.num_filters = [model.num_prototypes // len(args.dilated)] * len(args.dilated)
        model.num_filters[0] += model.num_prototypes % len(args.dilated)
    return args, model

def get_nearest(args, model, train_batches_unshuffled, text_train, labels_train):
    model.eval()
    dist, w = [], []
    with torch.no_grad():
        for batch, mask, _ in train_batches_unshuffled:
            batch = batch.to(f'cuda:{args.gpu[0]}')
            mask = mask.to(f'cuda:{args.gpu[0]}')
            distances, top_w = model.get_dist(batch, mask)
            dist.append(distances)
            w.append(top_w)
        proto_ids, proto_texts = model.nearest_neighbors(dist, w, text_train, labels_train)
    return proto_ids, proto_texts

def visualize_protos(args, embedding, mask, labels, prototypes, model, proto_labels):
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
        sample_size = 7000
        seq_length = embedding.shape[1]
        rnd_samples = np.random.randint(embedding.shape[0], size=sample_size)
        # flatten tensors and reduce size since otherwise not feasible for PCA
        embedding = embedding[rnd_samples]
        embedding = embedding.reshape(-1,model.enc_size)
        prototypes = prototypes.reshape(-1,model.enc_size)
        mask = (mask[rnd_samples] > 0).reshape(-1)
        rnd_labels = [[labels[i]] * seq_length for i in rnd_samples]
        rnd_labels = [label for sent in rnd_labels for label in sent]
        # subsample again for plot
        rnd_samples = ((np.random.randint(sample_size, size=50) * seq_length).reshape(-1,1) + np.arange(seq_length)).reshape(-1)
        rnd_labels = np.array(rnd_labels)[rnd_samples]
        rnd_labels = rnd_labels[mask[rnd_samples]].tolist()
        rnd_samples = rnd_samples[mask[rnd_samples]]

    if args.trans_type == 'PCA':
        pca = PCA(n_components=2)
        pca.fit(embedding)
        print('Explained variance ratio of components after transform: ', pca.explained_variance_ratio_)
        embed_trans = pca.transform(embedding)
        proto_trans = pca.transform(prototypes)
    elif args.trans_type == 'TSNE':
        tsne = TSNE(n_jobs=8, n_components=2, metric=args.metric).fit_transform(np.vstack((embedding,prototypes)))
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
        txt = 'P' + str(i + 1)
        if args.proto_size == 1:
            ax.annotate(txt, (proto_trans[i, 0], proto_trans[i, 1]), color='black')
        elif args.proto_size > 1:
            for j in range(args.proto_size):
                ax.annotate(txt, (proto_trans[n, 0], proto_trans[n, 1]), color='black')
                n += 1

    ax.legend()
    prefix = 'interacted_' if 'interacted' in args.model_path else ''
    fig.savefig(os.path.join(os.path.dirname(args.model_path), prefix+args.trans_type+'proto_vis2D.png'))

def bubble(args, train_batches_unshuffled, model, text_train):
    # describe prototype by local context

    attn_mask, argmin_dist, prototype_distances = [], [], []
    for batch, mask, _ in train_batches_unshuffled:
        batch = batch.to(f'cuda:{args.gpu[0]}')
        mask = mask.to(f'cuda:{args.gpu[0]}')
        distances, attn_w = model.get_dist(batch, mask)
        # argmin_dist.append(torch.cat([torch.argmin(d, dim=2) for d in distances], dim=1))
        prototype_distances.append(torch.cat([torch.min(d, dim=2)[0] for d in distances], dim=1))
        attn_mask.append(attn_w)
    attn_mask = torch.cat(attn_mask, dim=0)
    prototype_distances = torch.cat(prototype_distances, dim=0)
    # argmin_dist = torch.cat(argmin_dist, dim=0)
    # for i, n in enumerate(nearest_sent):
    #     nearest_conv.append(argmin_dist[n, i].cpu().detach().numpy())

    k=8 # top neighbors to be regarded
    nearest_similarity, nearest_sent = prototype_distances.topk(k=k, largest=False, dim=0)
    text_tknzd = model.tokenizer(text_train, return_tensors='pt', padding=True, add_special_tokens=False).input_ids
    proto_texts = []
    for i in range(args.num_prototypes):
        for j in range(args.proto_size):
            proto_texts.append(f'P{i + 1}.{j + 1} |')
            for l in range(k):
                if nearest_similarity[l,i] < -0.5:
                    nearest_word = attn_mask[nearest_sent[l,i], j]
                    token2text = model.tokenizer.decode(text_tknzd[nearest_sent[l,i],nearest_word].tolist())
                    proto_texts[-1] += f' {token2text}'
    # remove stopwords? only add "meaningful" words
    # remove redundant words
    proto_texts = [' '.join(sorted(set(t.split()), key=lambda x: t.index(x))) for t in proto_texts]
    return proto_texts

def proto_loss(prototype_distances, label, model, args):
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

    # diversity loss, assures that prototypes are not too close
    # put penalty only only on prototypes of same class
    comb = torch.combinations(torch.arange(0, args.num_prototypes), r=2)
    # and only softly encourage to be different not opposite, hence max(0, .)
    # divers_loss = torch.mean(F.cosine_similarity(model.protolayer[:,comb][:,:,0], model.protolayer[:,comb][:,:,1]).clamp(min=0))
    if args.metric == 'cosine':
        divers_loss = torch.mean(F.cosine_similarity(model.protolayer[:,comb][:,:,0],
                                                     model.protolayer[:,comb][:,:,1]).squeeze().clamp(min=0.8))
    elif args.metric == 'L2':
        divers_loss = torch.mean(nes_torch(model.protolayer[:,comb][:,:,0],
                                           model.protolayer[:,comb][:,:,1], dim=2).squeeze().clamp(min=0.8))

    # l1 loss on classification layer weights, scaled by number of prototypes
    l1_loss = model.fc.weight.norm(p=1) / args.num_prototypes

    return distr_loss, clust_loss, sep_loss, divers_loss, l1_loss

#### interaction methods ######

def finetune_prototypes(args, protos2finetune, model):
    with torch.no_grad():
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
    with torch.no_grad():
        # reinitialize selected prototypes
        model.protolayer[:, protos2reinit] = nn.Parameter(nn.init.uniform_(torch.empty(model.protolayer.size()))[:,
                                                  protos2reinit].to(f'cuda:{args.gpu[0]}'), requires_grad=True)
        # onyl retrain reinitialized protos and set gradients of other prototypes to zero
        model = finetune_prototypes(args, protos2reinit, model)
    return model

def replace_prototypes(args, protos2add, model):
    with torch.no_grad():
        # reassign protolayer, add new ones
        idx = protos2add[1]
        protos2add, _ = model.compute_embedding(protos2add[0], args)
        if args.level == 'sentence':
            protos2add = protos2add.view(1, -1, model.enc_size).to(f'cuda:{args.gpu[0]}')
        elif args.level == 'word':
            protos2add = protos2add.view(1, -1, model.enc_size, args.proto_size).to(f'cuda:{args.gpu[0]}')

        model.protolayer[:, idx] = protos2add
        model.protolayer.requires_grad = False
    return args, model

def add_prototypes(args, protos2add, model):
    with torch.no_grad():
        # reassign protolayer, add new ones
        args = adjust_cl_ids(args, protos2add[1])
        protos2add, mask2add = model.compute_embedding(protos2add[0], args)
        if args.level == 'sentence':
            protos2add = protos2add.view(1, -1, model.enc_size).to(f'cuda:{args.gpu[0]}')
        elif args.level == 'word':
            protos2add = protos2add.view(1, -1, model.enc_size, args.proto_size).to(f'cuda:{args.gpu[0]}')

        new_protos = torch.cat((model.protolayer.clone(), protos2add), dim=1)
        model.protolayer = nn.Parameter(new_protos, requires_grad=False)
        # reassign last layer, add new ones
        weights2keep = model.fc.weight.detach().clone()
        model.fc = nn.Linear((args.num_prototypes+len(protos2add)), args.num_classes, bias=False).to(f'cuda:{args.gpu[0]}')
        new_weights = nn.init.uniform_(torch.empty(args.num_classes, len(protos2add))).to(f'cuda:{args.gpu[0]}')
        model.fc.weight.data = torch.cat((weights2keep, new_weights), dim=1)
        args.num_prototypes = args.num_prototypes + len(protos2add)
        args, model = update_params(args, model)
    return args, model

def remove_prototypes(args, protos2remove, model, use_cos=False, use_weight=False):
    with torch.no_grad():
        # prune number of prototypes, the ones that do not have high weight wrt max weight are discarded
        if use_weight:
            w_limit = torch.max(abs(model.fc.weight)) * 0.3
            for i in range(args.num_prototypes):
                if (abs(model.fc.weight[0,i]) < w_limit) and (abs(model.fc.weight[1,i]) < w_limit):
                    protos2remove.append(i)
        # if prototypes are too close/ similar throw away
        if use_cos:
            comb = torch.combinations(torch.arange(args.num_prototypes), r=2)
            for k, l in comb:
                similarity = F.cosine_similarity(model.protolayer[:, k], model.protolayer[:, l], dim=1)
                similarity = torch.sum(similarity) / args.proto_size
                if similarity>0.9:
                    protos2remove.append(int(k))

        # make list items unique
        protos2remove = list(set(protos2remove))
        protos2keep = [p for p in list(range(args.num_prototypes)) if p not in protos2remove]
        args.prototype_class_identity =  args.prototype_class_identity[protos2keep,:]
        # reassign protolayer, remove unneeded ones and only keep useful ones
        model.protolayer = nn.Parameter(model.protolayer[:, protos2keep], requires_grad=True)
        weights2keep = model.fc.weight.detach().clone()[:, protos2keep]
        model.fc.weight.copy_(weights2keep)
        args.num_prototypes = args.num_prototypes - len(protos2remove)
        args, model = update_params(args, model)
    return args, model

def prune_prototypes(args, proto_texts, model):
    with torch.no_grad():
        sim = 0.9
        # define stop words, but e.g. 'not' should not be removed
        stop_words = set(stopwords.words('english')) - {'not', 'no'}
        pruned_protos = []

        # remove stop words, non alpha words, single character words and punctuation
        for p in proto_texts:
            p = p.translate(str.maketrans('', '', string.punctuation+'“”—'))
            pruned_protos.append(TreebankWordDetokenizer().detokenize([w for w in p.split() if w.isalpha()
                                                                       and not w in stop_words and len(w)>1]))
        new_prototypes = model.compute_embedding(pruned_protos, args)[0].to(f'cuda:{args.gpu[0]}')
        if len(new_prototypes.size()) < 3: new_prototypes.unsqueeze_(0)
        # only assign new prototypes if cosine similarity to old one is close
        angle =  F.cosine_similarity(model.protolayer, new_prototypes, dim=2)
        mask = (angle > sim).squeeze()

        l = 12
        # further reduce length if possible
        for i in range(args.num_prototypes):
            if len(pruned_protos[i]) > l:
                if mask[i]:
                    new_prototxt = TreebankWordDetokenizer().detokenize(pruned_protos[i].split()[0:l])
                else:
                    new_prototxt = TreebankWordDetokenizer().detokenize(proto_texts[i].split()[0:l])
            else:
                new_prototxt = pruned_protos[i]
            new_prototype, _ = model.compute_embedding(new_prototxt, args)
            new_prototype = new_prototype.to(f'cuda:{args.gpu[0]}')
            angle_ = F.cosine_similarity(model.protolayer[:,i], new_prototype.squeeze(0), dim=1)
            if angle_ > sim:
                pruned_protos[i] = new_prototxt
                new_prototypes[0,i,:] = new_prototype

        # mask: replace only words with high cos sim
        for i,m in enumerate(mask):
            if m: proto_texts[i] = pruned_protos[i]
        # assign new prototypes and don't update them when retraining
        model.protolayer[:, mask] = new_prototypes[:, mask].float()
        model.protolayer.requires_grad = False
    return model, proto_texts

####################################################
###### load toxicity data ##########################
####################################################

def parse_prompts_and_continuation(tag, discrete=True, discard=False, remove_long=True, file_dir=None):
    if file_dir is None:
        dataset_file = './data/realtoxicityprompts/prompts.jsonl'
    else:
        dataset_file = file_dir + '/prompts.jsonl'

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
        dataset_file = './data/realtoxicityprompts/full data.jsonl'
        file_dir = './data/realtoxicityprompts'
    else:
        dataset_file = file_dir + '/full data.jsonl'

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
    file_dir = os.path.join(args.data_dir, 'realtoxicityprompts')
    pickle.dump(text, open(file_dir + '/text.pkl', 'wb'))
    pickle.dump(labels, open(file_dir + '/labels.pkl', 'wb'))
    return text, labels

def preprocessor_toxic(text, labels, discrete, discard, remove_long):
    if remove_long:
        txt = []
        lbl = []
        # assures that not too long sequences are used especially required for Clip model
        import clip
        for t, l in zip(text, labels):
            try:
                clip.tokenize(t)
                if len(nltk.word_tokenize(t)) <= 30:
                    txt.append(t)
                    lbl.append(l)
            except:
                pass
        text = txt
        labels = lbl

    if discard:
        text = list([t for t, l in zip(text, labels) if (l < 0.3 or l > 0.7)])
        labels = list([l for l in labels if (l < 0.3 or l > 0.7)])

    if discrete:
        labels = [0 if l < 0.5 else 1 for l in labels]

    # remove non english words (some reviews in Chinese, etc.), but keep digits and punctuation
    # for i,t in enumerate(text):
    #     text[i] = convert_language(t)
    #     if not text[i]:
    #         del text[i], labels[i]

    assert len(text) == len(labels)
    max_len = 200_000
    if len(text)> max_len:
        text = [text[i] for i in range(max_len)]
        labels = [labels[i] for i in range(max_len)]
    return text, labels

def get_toxicity(args):
    f = '_full' if args.data_name == 'toxicity_full' else ''
    data_name = 'realtoxicityprompts'
    set_dir = os.path.join(args.data_dir, data_name)
    text = pickle.load(open(set_dir + '/text' + f + '.pkl', 'rb'))
    labels = pickle.load(open(set_dir + '/labels' + f + '.pkl', 'rb'))
    return text, labels

####################################################
###### load ethics data ############################
####################################################

def preprocess_ethics(args):
    set_dir = os.path.join(args.data_dir, args.data_name, 'commonsense')
    set_names = ['/cm_train.csv', '/cm_test.csv']#, '/cm_test_hard.csv'
    df = pd.concat((pd.read_csv(set_dir+set_name) for set_name in set_names))
    sub = df.loc[df["is_short"]==True]
    text = sub["input"].tolist()
    labels = sub["label"].tolist()

    pickle.dump(text, open(set_dir + '/text.pkl', 'wb'))
    pickle.dump(labels, open(set_dir + '/labels.pkl', 'wb'))

def get_ethics(args):
    set_dir = os.path.join(args.data_dir, args.data_name, 'commonsense')
    text = pickle.load(open(set_dir + '/text.pkl', 'rb'))
    labels = pickle.load(open(set_dir + '/labels.pkl', 'rb'))
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
        dataset_file = os.path.join(set_dir, 'yelp_academic_dataset_review.json')
    else:
        dataset_file = file_dir + '/yelp_academic_dataset_review.json'
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
    if args.data_name == 'toxicity' or args.data_name == 'toxicity_full':
        texts, labels = get_toxicity(args)
    elif args.data_name == 'rt-polarity':
        texts, labels = get_reviews(args)
    elif args.data_name == 'ethics':
        texts, labels = get_ethics(args)
    elif args.data_name == 'restaurant':
        texts, labels = get_restaurant(args)
    return texts, labels


###### load/ store embedding to not compute it every single run again ######

def load_embedding(args, fname, set_name):
    path = os.path.join('data/embedding', args.data_name)
    name = fname + '_' + set_name
    path_e = os.path.join(path, name + '.pt')
    assert os.path.isfile(path_e)
    path_m = os.path.join(path, name + '_mask.pt')
    assert os.path.isfile(path_m)
    embedding = torch.load(path_e, map_location=torch.device('cpu'))
    mask = torch.load(path_m, map_location=torch.device('cpu'))
    return embedding, mask

def save_embedding(embedding, mask, args, fname, set_name):
    path = os.path.join('data/embedding', args.data_name)
    os.makedirs(path, exist_ok=True, mode=0o777)
    name = fname + '_' + set_name
    path_e = os.path.join(path, name + '.pt')
    torch.save(embedding, path_e)
    path_m = os.path.join(path, name + '_mask.pt')
    torch.save(mask, path_m)


#############################################################
#############################################################
#############################################################


def load_image_features():
    image_dir = 'restaurant'
    path = '/workspace/repositories/datasets'
    if image_dir == 'imagenet':
        path_c = os.path.join(path, 'clip', 'imagenet_emb')
    elif image_dir == 'restaurant':
        path_c = os.path.join(path, 'clip', 'yelp_restaurant_emb')
    files_c = [f for f in os.listdir(path_c) if os.path.isfile(os.path.join(path_c, f))]
    image_features, fname = [], []
    for file in files_c:
        with open(os.path.join(path_c, file), 'rb') as f:
            f, i = pickle.load(f)
            image_features.append(i)
            fname.append(f)
    image_features = np.concatenate(image_features)
    if image_dir == 'imagenet':
        fname = [st[22:] for sublist in fname for st in sublist]
    elif image_dir == 'restaurant':
        fname = [st[33:] for sublist in fname for st in sublist]
    return fname, image_features

def path2img(fname):
    path = '/workspace/repositories/datasets'
    with open(os.path.join(path, fname), 'rb') as f:
        im = torch.from_numpy(np.array(plt.imread(f, format='jpeg')))
        if len(im.shape) < 3:
            im = im.unsqueeze(2).repeat(1,1,3)
        image = im.float().permute(2,0,1) / 255
    return image

def nearest_image(args, model, proto_texts):
    import sentence_transformers
    fname, image_features = load_image_features()
    # query = model.protolayer.detach().clone().squeeze()
    query, _ = model.compute_embedding(proto_texts, args)
    query = query.squeeze()
    topk = 3
    nearest_img = sentence_transformers.util.semantic_search(query, image_features, top_k=topk)
    nearest_img = [k['corpus_id'] for topk_img in nearest_img for k in topk_img]

    n = 0
    for i in range(args.num_prototypes):
        # proto_images = []
        for k in range(topk):
            img = path2img(fname[nearest_img[n]]).float()
            # proto_images.append(img)
            save_image(img, os.path.dirname(args.model_path) + f'/proto{i+1}.{k+1}.png')
            n += 1
