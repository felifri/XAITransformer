import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
#from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import nltk
import string
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
# import gpumap
from torchvision.utils import save_image
import clip
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse


words = set(nltk.corpus.words.words())
tok = TreebankWordTokenizer()
detok = TreebankWordDetokenizer()


# __import__("pdb").set_trace()


def ned_torch(x1, x2, dim=1, eps=1e-8):
    ned_2 = 0.5 * ((x1 - x2).var(dim=dim) / (x1.var(dim=dim) + x2.var(dim=dim) + eps))
    return ned_2 ** 0.5


def nes_torch(x1, x2, dim=1, eps=1e-8):
    return 1 - ned_torch(x1, x2, dim, eps)


def adjust_cl_ids(args, idx):
    if idx == 0:
        cl = torch.Tensor([1, 0]).view(1, 2).to(f'cuda:{args.gpu[0]}')
    elif idx == 1:
        cl = torch.Tensor([0, 1]).view(1, 2).to(f'cuda:{args.gpu[0]}')
    return cl


def update_params(args, model):
    model.num_prototypes = args.num_prototypes
    if args.dilated:
        model.num_filters = [model.num_prototypes // len(args.dilated)] * len(args.dilated)
        model.num_filters[0] += model.num_prototypes % len(args.dilated)
    return args, model


def extent_data(args, embedding_train, mask_train, text_train, labels_train, embedding, mask, text, label):
    embedding_train = torch.cat((embedding_train, embedding.cpu().squeeze(0)))
    mask_train = torch.cat((mask_train, mask.squeeze(0)))
    text_train.append(text)
    labels_train.append(int(label))
    train_batches_unshuffled = torch.utils.data.DataLoader(list(zip(embedding_train, mask_train, labels_train)),
                                                           batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                                           num_workers=0)
    return embedding_train, mask_train, text_train, labels_train, train_batches_unshuffled


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
        proto_ids, proto_texts, [nearest_sent, nearest_words] = model.nearest_neighbors(dist, w, text_train,
                                                                                        labels_train)
    return proto_ids, proto_texts, [nearest_sent, nearest_words]


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
        embedding = embedding.reshape(-1, model.enc_size)
        prototypes = prototypes.reshape(-1, model.enc_size)
        mask = (mask[rnd_samples] > 0).reshape(-1)
        rnd_labels = [[labels[i]] * seq_length for i in rnd_samples]
        rnd_labels = [label for sent in rnd_labels for label in sent]
        # subsample again for plot
        rnd_samples = ((np.random.randint(sample_size, size=50) * seq_length).reshape(-1, 1) + np.arange(
            seq_length)).reshape(-1)
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
        tsne = TSNE(n_jobs=8, n_components=2, metric=args.metric).fit_transform(np.vstack((embedding, prototypes)))
        [embed_trans, proto_trans] = [tsne[:len(embedding)], tsne[len(embedding):]]
    # elif args.trans_type == 'UMAP':
    #     umapped = gpumap.GPUMAP().fit_transform(np.vstack((embedding, prototypes)))
    #     [embed_trans, proto_trans] = [umapped[:len(embedding)], umapped[len(embedding):]]

    for cl in range(args.num_classes):
        ix = np.where(np.array(rnd_labels) == cl)[0]
        ax.scatter(embed_trans[rnd_samples[ix], 0], embed_trans[rnd_samples[ix], 1], c=cdict_d[cl], marker='x',
                   label=ldict_d[cl], alpha=0.3)
        ix = np.where(proto_labels[:, cl].cpu().numpy() == 1)[0]
        if args.proto_size > 1:
            ix = [args.proto_size * i + x for i in ix.tolist() for x in range(args.proto_size)]
        ax.scatter(proto_trans[ix, 0], proto_trans[ix, 1], c=cdict_p[cl], marker='o', label=ldict_p[cl], s=80)

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
    fig.savefig(os.path.join(os.path.dirname(args.model_path), prefix + args.trans_type + 'proto_vis2D.png'))


def proto_loss(prototype_distances, label, model, args):
    # model_weights = model.get_proto_weights()
    # min_indices = np.argmax(model_weights, axis=0)
    # results = np.zeros((model_weights.shape))
    # results[min_indices, np.arange(model_weights.shape[1])] = 1
    # args.prototype_class_identity = torch.tensor(results).to(f'cuda:{args.gpu[0]}')
    max_dist = torch.prod(torch.tensor(model.protolayer.size()))  # proxy variable, could be any high value

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
    # put penalty only onlyon prototypes of same class
    comb = torch.combinations(torch.arange(0, args.num_prototypes), r=2)
    if args.metric == 'cosine':
        divers_loss = torch.mean(F.cosine_similarity(model.protolayer[:, comb][:, :, 0],  model.protolayer[:, comb][:, :, 1]).squeeze()) #.clamp(min=0.8)

    elif args.metric == 'L2':
        divers_loss = torch.mean(nes_torch(model.protolayer[:, comb][:, :, 0], model.protolayer[:, comb][:, :, 1], dim=2).squeeze()) #.clamp(min=0.8)

    if args.soft:
        soft_loss = - torch.mean(F.cosine_similarity(model.protolayer[:, args.soft[1]], args.soft[4].squeeze(0),
                                                     dim=1).squeeze().clamp(max=args.soft[3]))
    else:
        soft_loss = 0
    divers_loss += soft_loss * 0.5

    # l1 loss on classification layer weights, scaled by number of prototypes
    l1_loss = model.fc.weight.norm(p=1) / args.num_prototypes
    
    
    
    return distr_loss, clust_loss, sep_loss, divers_loss, l1_loss


def project(args, embedding_train, model, train_batches_unshuffled, text_train, labels_train):
    # project prototypes
    if args.level == 'sentence':
        proto_ids, _, [nearest_sent, _] = get_nearest(args, model, train_batches_unshuffled, text_train,
                                                      labels_train)
        new_proto = embedding_train[nearest_sent, :]
    elif args.level == 'word':
        proto_ids, _, [nearest_sent, nearest_words] = get_nearest(args, model, train_batches_unshuffled, text_train,
                                                                  labels_train)
        new_proto = embedding_train[nearest_sent[:, np.newaxis].repeat(args.proto_size, axis=1), nearest_words, :]

    new_proto = new_proto.view(model.protolayer.shape)
    model.protolayer.copy_(new_proto)
    # give prototypes their "true" label
    s = 'label'
    proto_labels = torch.tensor([int(p[p.index(s) + len(s) + 1]) for p in proto_ids])
    args.prototype_class_identity.copy_(torch.stack((1 - proto_labels, proto_labels), dim=1))
    return model, args


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
        model.protolayer.requires_grad = True
        gradient_mask_proto = torch.zeros(model.protolayer.size()).to(f'cuda:{args.gpu[0]}')
        gradient_mask_proto[:, protos2finetune] = 1
        model.protolayer.register_hook(lambda grad: grad.mul_(gradient_mask_proto))
    return model


def reinit_prototypes(args, protos2reinit, model):
    with torch.no_grad():
        # reinitialize selected prototypes
        model.protolayer[:, protos2reinit] = nn.Parameter(nn.init.uniform_(torch.empty(model.protolayer.size()))[:,
                                                          protos2reinit], requires_grad=True).to(f'cuda:{args.gpu[0]}')
        weights = model.fc.weight.detach().clone()
        weights[:, protos2reinit] = nn.init.uniform_(torch.empty(args.num_classes, len(protos2reinit))
                                                     ).to(f'cuda:{args.gpu[0]}')
        model.fc.weight.copy_(weights)
        # onyl retrain reinitialized protos and set gradients of other prototypes to zero
        model = finetune_prototypes(args, protos2reinit, model)
    return model


def replace_prototypes(args, protos2replace, model, embedding_train, mask_train, text_train, labels_train):
    with torch.no_grad():
        # reassign protolayer, add new ones
        idx = protos2replace[1]
        args.prototype_class_identity[idx, :] = adjust_cl_ids(args, protos2replace[2])
        max_l = args.proto_size if args.level == 'word' else False
        protos2replace_e, mask_e = model.compute_embedding([protos2replace[0]], args, max_l)
        if args.level == 'sentence':
            protos2replace_e = protos2replace_e.view(1, -1, model.enc_size).to(f'cuda:{args.gpu[0]}')
            mask_e = mask_e.view(1, -1, model.enc_size)
        elif args.level == 'word':
            protos2replace_e = protos2replace_e.view(1, -1, model.enc_size, args.proto_size).to(f'cuda:{args.gpu[0]}')

        model.protolayer[:, idx] = nn.Parameter(protos2replace_e, requires_grad=False)
        # weights = model.fc.weight.detach().clone()
        # weights[:, idx] = nn.init.uniform_(torch.empty(args.num_classes)).to(f'cuda:{args.gpu[0]}')
        # model.fc.weight.copy_(weights)
        embedding_train, mask_train, text_train, labels_train, train_batches_unshuffled = extent_data(args,
                                                                                                      embedding_train,
                                                                                                      mask_train,
                                                                                                      text_train,
                                                                                                      labels_train,
                                                                                                      protos2replace_e,
                                                                                                      mask_e,
                                                                                                      protos2replace[0],
                                                                                                      protos2replace[2])
    return args, model, embedding_train, mask_train, text_train, labels_train, train_batches_unshuffled


def robustness(args, model, embedding_train, mask_train, text_train, labels_train, train_batches_unshuffled):
    # 4 modes of replacements: 1) replace with facts, 2) replace positive explanations with optimal ones 3) replace negative explanations with optimal ones 4) replace with optimal explanations
    # Steps: 1. get prototypes and their classes 2. get % of prototypes to replace 3. specify what to replace with 4. iterate over array to replace one by one
    proto_info, proto_texts, _ = get_nearest(args, model, train_batches_unshuffled, text_train, labels_train)
    s = 'label'
    proto_labels = torch.tensor([int(p[p.index(s) + len(s) + 1]) for p in proto_info])
    proto_labels = torch.stack((1 - proto_labels, proto_labels), dim=1)
    class_distribs = torch.sum(proto_labels, dim=0).float() / torch.sum(proto_labels)
    protos_to_replace = []
    
    if args.robustness == 'facts':
        num_protos_to_replace = int(args.robustness_percentage / 100 * args.num_prototypes)
        if num_protos_to_replace > 20:
            print(f'Only 20 facts available, replacing {num_protos_to_replace} with facts')
            num_protos_to_replace = 20
        # get unique random facts
        facts = np.random.choice(FACTS, num_protos_to_replace, replace=False)
        # get random prototypes of pos and neg to replace according to class distribution
        pos_indices = np.random.choice(np.where(proto_labels[:, 1] == 1)[0], int(num_protos_to_replace * class_distribs[1]), replace=False)
        neg_indices = np.random.choice(np.where(proto_labels[:, 1] == 0)[0], int(num_protos_to_replace * class_distribs[0]), replace=False)
        # add protos to protos to replace array [[text, index, class id]]
        k = 0
        for i in range(len(pos_indices)):
            protos_to_replace.append([facts[k], pos_indices[i], 1])
            k = k + 1
        for i in range(len(neg_indices)):
            protos_to_replace.append([facts[k], neg_indices[i], 0])
            k = k + 1
        
    if args.robustness == 'positive':
        num_protos_to_replace = int(args.robustness_percentage / 100 * len(np.where(proto_labels[:, 1] == 1)[0]))
        if num_protos_to_replace > 11:
            print(f'Only 11 positive reviews available, replacing {num_protos_to_replace} with positive reviews')
            num_protos_to_replace = 11
        texts = np.random.choice(POSITIVE, num_protos_to_replace, replace=False)
        indices = np.random.choice(np.where(proto_labels[:, 1] == 1)[0], num_protos_to_replace, replace=False)
        for i in range(len(indices)):
            protos_to_replace.append([texts[i], indices[i], 1])
    
    elif args.robustness == 'negative':
        num_protos_to_replace = int(args.robustness_percentage / 100 * len(np.where(proto_labels[:, 1] == 0)[0]))
        if num_protos_to_replace > 11:
            print(f'Only 11 negative reviews available, replacing {num_protos_to_replace} with negative reviews')
            num_protos_to_replace = 11
        texts = np.random.choice(NEGATIVE, num_protos_to_replace, replace=False)
        indices = np.random.choice(np.where(proto_labels[:, 1] == 0)[0], num_protos_to_replace, replace=False)
        for i in range(len(indices)):
            protos_to_replace.append([texts[i], indices[i], 0])
            
    elif args.robustness == "pos_neg":
        pos_protos_to_replace = int(args.robustness_percentage / 100 * len(np.where(proto_labels[:, 1] == 1)[0]))
        neg_proto_to_replace = int(args.robustness_percentage / 100 * len(np.where(proto_labels[:, 1] == 0)[0]))
        pos_texts = np.random.choice(POSITIVE, pos_protos_to_replace, replace=False)
        neg_texts = np.random.choice(NEGATIVE, neg_proto_to_replace, replace=False)
        pos_indices = np.random.choice(np.where(proto_labels[:, 1] == 1)[0], pos_protos_to_replace, replace=False)
        neg_indices = np.random.choice(np.where(proto_labels[:, 1] == 0)[0], neg_proto_to_replace, replace=False)
        for i in range(len(pos_indices)):
            protos_to_replace.append([pos_texts[i], pos_indices[i], 1])
        for i in range(len(neg_indices)):
            protos_to_replace.append([neg_texts[i], neg_indices[i], 0])
            
    return protos_to_replace

def replace_sentence_prototypes(args, protos2replace, model, embedding_train, mask_train, text_train, labels_train):
    with torch.no_grad():
        # reassign protolayer, add new ones
        idx = protos2replace[1]
        args.prototype_class_identity[idx, :] = adjust_cl_ids(args, protos2replace[2])
        if args.robustness == 'facts':
            index = FACTS.index(protos2replace[0])
            path = os.path.join('data/embedding', 'robustness', 'facts')
            path_e = os.path.join(path, f'fact_{index}.pt')
            path_m = os.path.join(path, f'fact_{index}_mask.pt')
            if os.path.isfile(path_e) and os.path.isfile(path_m):
                protos2replace_e = torch.load(path_e, map_location=torch.device('cpu'))
                mask_e = torch.load(path_m, map_location=torch.device('cpu'))
            else:
                protos2replace_e, mask_e = model.compute_embedding([protos2replace[0]], args, max_l=False) 
                os.makedirs(path, exist_ok=True, mode=0o777)
                torch.save(protos2replace_e, path_e)
                torch.save(mask_e, path_m)
        elif args.robustness == 'positive':
            index = POSITIVE.index(protos2replace[0])
            path = os.path.join('data/embedding', 'robustness', 'positive')
            path_e = os.path.join(path, f'positive_{index}.pt')
            path_m = os.path.join(path, f'positive_{index}_mask.pt')
            if os.path.isfile(path_e) and os.path.isfile(path_m):
                protos2replace_e = torch.load(path_e, map_location=torch.device('cpu'))
                mask_e = torch.load(path_m, map_location=torch.device('cpu'))
            else:
                protos2replace_e, mask_e = model.compute_embedding([protos2replace[0]], args, max_l=False) 
                os.makedirs(path, exist_ok=True, mode=0o777)
                torch.save(protos2replace_e, path_e)
                torch.save(mask_e, path_m)
        elif args.robustness == 'negative':
            index = NEGATIVE.index(protos2replace[0])
            path = os.path.join('data/embedding', 'robustness', 'negative')
            path_e = os.path.join(path, f'negative_{index}.pt')
            path_m = os.path.join(path, f'negative_{index}_mask.pt')
            if os.path.isfile(path_e) and os.path.isfile(path_m):
                protos2replace_e = torch.load(path_e, map_location=torch.device('cpu'))
                mask_e = torch.load(path_m, map_location=torch.device('cpu'))
            else:
                protos2replace_e, mask_e = model.compute_embedding([protos2replace[0]], args, max_l=False) 
                os.makedirs(path, exist_ok=True, mode=0o777)
                torch.save(protos2replace_e, path_e)
                torch.save(mask_e, path_m)
        elif args.robustness == 'pos_neg':
            index = POSITIVE.index(protos2replace[0]) if protos2replace[2] == 1 else NEGATIVE.index(protos2replace[0])
            if protos2replace[2] == 1:
                path = os.path.join('data/embedding', 'robustness', 'positive')
                path_e = os.path.join(path, f'positive_{index}.pt')
                path_m = os.path.join(path, f'positive_{index}_mask.pt')
            else:
                path = os.path.join('data/embedding', 'robustness', 'negative')
                path_e = os.path.join(path, f'negative_{index}.pt')
                path_m = os.path.join(path, f'negative_{index}_mask.pt')
            if os.path.isfile(path_e) and os.path.isfile(path_m):
                protos2replace_e = torch.load(path_e, map_location=torch.device('cpu'))
                mask_e = torch.load(path_m, map_location=torch.device('cpu'))
            else:
                protos2replace_e, mask_e = model.compute_embedding([protos2replace[0]], args, max_l=False) 
                os.makedirs(path, exist_ok=True, mode=0o777)
                torch.save(protos2replace_e, path_e)
                torch.save(mask_e, path_m)        
                
        protos2replace_e = protos2replace_e.view(1, -1, model.enc_size).to(f'cuda:{args.gpu[0]}')
        mask_e = mask_e.view(1, -1, model.enc_size)

        model.protolayer[:, idx] = nn.Parameter(protos2replace_e, requires_grad=False)
        if(args.robustness_reinit):
            weights = model.fc.weight.detach().clone()
            weights[:, idx] = nn.init.uniform_(torch.empty(args.num_classes)).to(f'cuda:{args.gpu[0]}')
            model.fc.weight.copy_(weights)
        embedding_train, mask_train, text_train, labels_train, train_batches_unshuffled = extent_data(args,
                                                                                                    embedding_train,
                                                                                                    mask_train,
                                                                                                    text_train,
                                                                                                    labels_train,
                                                                                                    protos2replace_e,
                                                                                                    mask_e,
                                                                                                    protos2replace[0],
                                                                                                    protos2replace[2])
    return args, model, embedding_train, mask_train, text_train, labels_train, train_batches_unshuffled
        
    
def soft_rplc_prototypes(args, protos2replace, model, embedding_train, mask_train, text_train, labels_train):
    with torch.no_grad():
        # reassign protolayer, add new ones
        idx = protos2replace[1]
        args.prototype_class_identity[idx, :] = adjust_cl_ids(args, protos2replace[2])
        max_l = embedding_train.size() if args.level == 'word' else False
        protos2replace_e, mask_e = model.compute_embedding(protos2replace[0], args, max_l)
        if args.level == 'sentence':
            protos2replace_e = protos2replace_e.view(1, -1, model.enc_size).to(f'cuda:{args.gpu[0]}')
        elif args.level == 'word':
            protos2replace_e = protos2replace_e.view(1, -1, model.enc_size, args.proto_size).to(f'cuda:{args.gpu[0]}')

        args.soft.append(protos2replace_e)
        model = finetune_prototypes(args, idx, model)
        weights = model.fc.weight.detach().clone()
        weights[:, idx] = - args.prototype_class_identity[idx, :]
        model.fc.weight.copy_(weights)
        embedding_train, mask_train, text_train, labels_train, train_batches_unshuffled = extent_data(args,
                                                                                                      embedding_train,
                                                                                                      mask_train,
                                                                                                      text_train,
                                                                                                      labels_train,
                                                                                                      protos2replace_e,
                                                                                                      mask_e,
                                                                                                      protos2replace[0],
                                                                                                      protos2replace[2])
    return args, model, embedding_train, mask_train, text_train, labels_train, train_batches_unshuffled


def add_prototypes(args, protos2add, model, embedding_train, mask_train, text_train, labels_train):
    with torch.no_grad():
        # reassign protolayer, add new ones
        cl = adjust_cl_ids(args, protos2add[1])
        args.prototype_class_identity = torch.cat((args.prototype_class_identity, cl))
        max_l = args.proto_size if args.level == 'word' else False
        protos2add_e, mask_e = model.compute_embedding(protos2add[0], args, max_l)
        if args.level == 'sentence':
            protos2add_e = protos2add_e.view(1, -1, model.enc_size).to(f'cuda:{args.gpu[0]}')
        elif args.level == 'word':
            protos2add_e = protos2add_e.view(1, -1, model.enc_size, args.proto_size).to(f'cuda:{args.gpu[0]}')
            protos2add_e = protos2add_e[:, args.proto_size]  # cut off if words2add are longer than proto size

        new_protos = torch.cat((model.protolayer.clone(), protos2add_e), dim=1)
        model.protolayer = nn.Parameter(new_protos, requires_grad=False)
        # reassign last layer, add new ones
        weights2keep = model.fc.weight.detach().clone()
        weights_new = nn.init.uniform_(torch.empty(args.num_classes, len(protos2add_e))).to(f'cuda:{args.gpu[0]}')
        model.fc = nn.Linear((args.num_prototypes + len(protos2add_e)), args.num_classes, bias=False).to(
            f'cuda:{args.gpu[0]}')
        model.fc.weight.copy_(torch.cat((weights2keep, weights_new), dim=1))
        args.num_prototypes = args.num_prototypes + len(protos2add_e)
        args, model = update_params(args, model)
        embedding_train, mask_train, text_train, labels_train, train_batches_unshuffled = extent_data(args,
                                                                                                      embedding_train,
                                                                                                      mask_train,
                                                                                                      text_train,
                                                                                                      labels_train,
                                                                                                      protos2add_e,
                                                                                                      mask_e,
                                                                                                      protos2add[0],
                                                                                                      protos2add[1])
    return args, model, embedding_train, mask_train, text_train, labels_train, train_batches_unshuffled


def remove_prototypes(args, protos2remove, model, use_cos=False, use_weight=False):
    with torch.no_grad():
        # prune number of prototypes, the ones that do not have high weight wrt max weight are discarded
        if use_weight:
            w_limit = torch.max(abs(model.fc.weight)) * 0.3
            for i in range(args.num_prototypes):
                if (abs(model.fc.weight[0, i]) < w_limit) and (abs(model.fc.weight[1, i]) < w_limit):
                    protos2remove.append(i)
        # if prototypes are too close/ similar throw away
        if use_cos:
            comb = torch.combinations(torch.arange(args.num_prototypes), r=2)
            for k, l in comb:
                similarity = F.cosine_similarity(model.protolayer[:, k], model.protolayer[:, l], dim=1)
                similarity = torch.sum(similarity) / args.proto_size
                if similarity > 0.9:
                    protos2remove.append(int(k))

        # make list items unique
        protos2remove = list(set(protos2remove))
        protos2keep = [p for p in list(range(args.num_prototypes)) if p not in protos2remove]
        args.prototype_class_identity = args.prototype_class_identity[protos2keep, :]
        # reassign protolayer, remove unneeded ones and only keep useful ones
        model.protolayer = nn.Parameter(model.protolayer[:, protos2keep], requires_grad=False).to(f'cuda:{args.gpu[0]}')
        weights2keep = model.fc.weight.detach().clone()[:, protos2keep]
        model.fc = nn.Linear(len(protos2keep), args.num_classes, bias=False).to(f'cuda:{args.gpu[0]}')
        model.fc.weight.copy_(weights2keep)
        args.num_prototypes = args.num_prototypes - len(protos2remove)
        args, model = update_params(args, model)
    return args, model


def prune_prototypes(args, proto_texts, model, embedding_train, mask_train, text_train, labels_train):
    with torch.no_grad():
        sim = 0.75
        # keep only first two sentences of each sequence and no longer than 12 tokens.
        pruned_protos = []
        for i in range(len(proto_texts)):
            pt, k = [], 0
            for j, p in enumerate(tok.tokenize(proto_texts[i])):
                pt.append(p)
                if p in '.!?':
                    k += 1
                if k == 2 or j == 15:
                    break
            pruned_protos.append(detok.detokenize(pt))
        new_prototypes, new_mask = model.compute_embedding(pruned_protos, args)
        new_prototypes = new_prototypes.to(f'cuda:{args.gpu[0]}')
        if len(new_prototypes.size()) < 3:
            new_prototypes.unsqueeze_(0)
            new_mask.unsqueeze_(0)
        # only assign new prototypes if cosine similarity to old one is high
        angle = F.cosine_similarity(model.protolayer, new_prototypes, dim=2)
        mask = (angle > sim).squeeze()

        # assign new prototypes and don't update them when retraining
        model.protolayer[:, mask] = new_prototypes[:, mask].float()
        model.protolayer.requires_grad = False
        # mask: replace only words with high cos sim
        for i, m in enumerate(mask):
            if m:
                embedding_train, mask_train, text_train, labels_train, train_batches_unshuffled = \
                    extent_data(args, embedding_train, mask_train, text_train, labels_train,
                                new_prototypes[:, i].unsqueeze(0), new_mask[:, i].unsqueeze(0), pruned_protos[i],
                                args.prototype_class_identity[i, 1])
    return args, model, embedding_train, mask_train, text_train, labels_train, train_batches_unshuffled


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

    assert len(text) == len(labels)
    max_len = 200_000
    if len(text) > max_len:
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
###### load jigsaw data ############################
####################################################
def preprocess_jigsaw(args):
    set_dir = os.path.join(args.data_dir, args.data_name)
    set_names = ['/train.csv', '/test.csv', '/test_labels.csv']
    df_train = pd.read_csv(set_dir + set_names[0])
    #df_test = pd.read_csv(set_dir + set_names[1])
    #df_test_labels = pd.read_csv(set_dir + set_names[2])

    text_train = df_train["comment_text"].tolist()
    labels_train = df_train["toxic"].tolist()
    #text_test = df_test["comment_text"].tolist()
    #labels_test = df_test_labels["toxic"].tolist()
    #split train set into train val 20:80
    text_train, text_val, labels_train, labels_val = train_test_split(text_train, labels_train, test_size=0.8, random_state=42)
    #split val into val test 7:1
    text_val, text_test, labels_val, labels_test = train_test_split(text_train, labels_train, test_size=1/8, random_state=42)
    pickle.dump(text_train, open(set_dir + '/text_train.pkl', 'wb'))
    pickle.dump(labels_train, open(set_dir + '/labels_train.pkl', 'wb'))
    pickle.dump(text_test, open(set_dir + '/text_test.pkl', 'wb'))
    pickle.dump(labels_test, open(set_dir + '/labels_test.pkl', 'wb'))
    pickle.dump(text_val, open(set_dir + '/text_val.pkl', 'wb'))
    pickle.dump(labels_val, open(set_dir + '/labels_val.pkl', 'wb'))

def get_jigsaw(args):
    set_dir = os.path.join(args.data_dir, args.data_name)
    if not os.path.exists(set_dir + '/text_train.pkl'):
        preprocess_jigsaw(args)
    text_train = pickle.load(open(set_dir + '/text_train.pkl', 'rb'))
    labels_train = pickle.load(open(set_dir + '/labels_train.pkl', 'rb'))
    text_val = pickle.load(open(set_dir + '/text_val.pkl', 'rb'))
    labels_val = pickle.load(open(set_dir + '/labels_val.pkl', 'rb'))
    text_test = pickle.load(open(set_dir + '/text_test.pkl', 'rb'))
    labels_test = pickle.load(open(set_dir + '/labels_test.pkl', 'rb'))
    return text_train, text_val, text_test, labels_train, labels_val, labels_test


####################################################
###### load ethics data ############################
####################################################

def preprocess_ethics(args):
    set_dir = os.path.join(args.data_dir, args.data_name, 'commonsense')
    set_names = ['/cm_train.csv', '/cm_test.csv']  # , '/cm_test_hard.csv'
    df = pd.concat((pd.read_csv(set_dir + set_name) for set_name in set_names))
    sub = df.loc[df["is_short"] == True]
    text = sub["input"].tolist()
    labels = sub["label"].tolist()

    pickle.dump(text, open(set_dir + '/text.pkl', 'wb'))
    pickle.dump(labels, open(set_dir + '/labels.pkl', 'wb'))


def get_ethics(args):
    set_dir = os.path.join(args.data_dir, args.data_name, 'commonsense')
    text_train = pickle.load(open(set_dir + '/text_train.pkl', 'rb'))
    labels_train = pickle.load(open(set_dir + '/labels_train.pkl', 'rb'))
    text_val = pickle.load(open(set_dir + '/text_val.pkl', 'rb'))
    labels_val = pickle.load(open(set_dir + '/labels_val.pkl', 'rb'))
    text_test = pickle.load(open(set_dir + '/text_test.pkl', 'rb'))
    labels_test = pickle.load(open(set_dir + '/labels_test.pkl', 'rb'))
    return text_train, text_val, text_test, labels_train, labels_val, labels_test


####################################################
###### load movie review data ######################
####################################################

def get_reviews(args):
    set_list = ['train', 'dev', 'test']
    text, labels = [], []
    # join train, dev, test; shuffle and split later
    for set_name in set_list:
        set_dir = os.path.join(args.data_dir, args.data_name, set_name)
        text_tmp = pickle.load(open(os.path.join(set_dir, 'word_sequences') + '.pkl', 'rb'))
        # join tokenized sentences back to full sentences for sentenceBert
        text_tmp = [detok.detokenize(sub_list) for sub_list in text_tmp]
        text.append(text_tmp)
        label_tmp = pickle.load(open(os.path.join(set_dir, 'labels') + '.pkl', 'rb'))
        # convert 'pos' & 'neg' to 1 & 0
        label_tmp = convert_label(label_tmp)
        labels.append(label_tmp)
    return text[0], text[1], text[2], labels[0], labels[1], labels[2]


def convert_label(labels):
    converted_labels = []
    for i, label in enumerate(labels):
        if label == 'pos':
            converted_labels.append(1)
        elif label == 'neg':
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
        labels = list([l for t, l in zip(text, labels) if len(tok.tokenize(t)) <= 25])
        text = list([t for t in text if len(tok.tokenize(t)) <= 25])

    if args.discard:
        text = list([t for t, l in zip(text, labels) if (l <= 1.0 or l >= 5.0)])
        labels = list([l for l in labels if (l <= 1.0 or l >= 5.0)])

    if binary:
        labels = list([0 if l < 2.5 else 1 for l in labels])

    # remove non english words (some reviews in Chinese, etc.), but keep digits and punctuation
    for i, t in enumerate(text):
        text[i] = convert_language(t)
        if not text[i]:
            del text[i], labels[i]

    assert len(text) == len(labels)
    max_len = 200_000
    if len(text) > max_len:
        text = [text[i] for i in range(max_len)]
        labels = [labels[i] for i in range(max_len)]
    #split dataset into train and test 70:30
    text_train, text_test, labels_train, labels_test = train_test_split(text, labels, test_size=0.3, random_state=42)
    #then split test val 50:50 for 70:15:15
    text_test, text_val, labels_test, labels_val = train_test_split(text_test, labels_test, test_size=0.5, random_state=42)
    pickle.dump(text_train, open(set_dir + '/text_train.pkl', 'wb'))
    pickle.dump(labels_train, open(set_dir + '/labels_train.pkl', 'wb'))
    pickle.dump(text_test, open(set_dir + '/text_test.pkl', 'wb'))
    pickle.dump(labels_test, open(set_dir + '/labels_test.pkl', 'wb'))
    pickle.dump(text_val, open(set_dir + '/text_val.pkl', 'wb'))
    pickle.dump(labels_val, open(set_dir + '/labels_val.pkl', 'wb'))
    return text, labels


def convert_language(seq):
    return detok.detokenize(w for w in nltk.wordpunct_tokenize(seq) if (w.lower() in words) or
                            (w.lower() in string.punctuation) or (w.lower().isdigit()))


def get_restaurant(args):
    set_dir = os.path.join(args.data_dir, args.data_name)
    if not os.path.exists(set_dir + '/text_train.pkl'):
        preprocess_restaurant(args)
    text_train = pickle.load(open(set_dir + '/text_train.pkl', 'rb'))
    labels_train = pickle.load(open(set_dir + '/labels_train.pkl', 'rb'))
    text_val = pickle.load(open(set_dir + '/text_val.pkl', 'rb'))
    labels_val = pickle.load(open(set_dir + '/labels_val.pkl', 'rb'))
    text_test = pickle.load(open(set_dir + '/text_test.pkl', 'rb'))
    labels_test = pickle.load(open(set_dir + '/labels_test.pkl', 'rb'))
    return text_train, text_val, text_test, labels_train, labels_val, labels_test

####################################################
###### load propaganda data #################
####################################################

def preprocess_propaganda(args):
    set_dir = os.path.join(args.data_dir, args.data_name)
    assert os.path.isfile(set_dir + '/proppy_1.0.dev.tsv')
    assert os.path.isfile(set_dir + '/proppy_1.0.test.tsv')
    assert os.path.isfile(set_dir + '/proppy_1.0.train.tsv')
    column_names = ['article_text', 'event_location', 'average_tone', 'article_date', 'article_ID', 'article_URL1', 'MBFC_factuality_label1',\
        'article_URL2', 'MBFC_factuality_label2', 'URL_to_MBFC_page', 'source_name', 'MBFC_notes_about_source', 'MBFC_bias_label', 'source_URL', 'propaganda_label']
    train = pd.read_csv(set_dir + '/proppy_1.0.train.tsv', sep='\t', names=column_names)
    test = pd.read_csv(set_dir + '/proppy_1.0.test.tsv', sep='\t', names=column_names)
    val = pd.read_csv(set_dir + '/proppy_1.0.dev.tsv', sep='\t', names=column_names)
    text_train = train['article_text'].tolist()
    labels_train = train['propaganda_label'].tolist()
    #labels are -1 and 1 convert to 0 and 1
    labels_train = [int((label+1)/2) for label in labels_train]
    text_test = test['article_text'].tolist()
    labels_test = test['propaganda_label'].tolist()
    labels_test = [int((label+1)/2) for label in labels_test]
    text_val = val['article_text'].tolist()
    labels_val = val['propaganda_label'].tolist()
    labels_val = [int((label+1)/2) for label in labels_val]
    pickle.dump(text_train, open(set_dir + '/text_train.pkl', 'wb'))
    pickle.dump(labels_train, open(set_dir + '/labels_train.pkl', 'wb'))
    pickle.dump(text_test, open(set_dir + '/text_test.pkl', 'wb'))
    pickle.dump(labels_test, open(set_dir + '/labels_test.pkl', 'wb'))
    pickle.dump(text_val, open(set_dir + '/text_val.pkl', 'wb'))
    pickle.dump(labels_val, open(set_dir + '/labels_val.pkl', 'wb'))

def get_propaganda(args):
    set_dir = os.path.join(args.data_dir, args.data_name)
    if not os.path.exists(set_dir + '/text_train.pkl'):
        preprocess_propaganda(args)
    text_train = pickle.load(open(set_dir + '/text_train.pkl', 'rb'))
    labels_train = pickle.load(open(set_dir + '/labels_train.pkl', 'rb'))
    text_val = pickle.load(open(set_dir + '/text_val.pkl', 'rb'))
    labels_val = pickle.load(open(set_dir + '/labels_val.pkl', 'rb'))
    text_test = pickle.load(open(set_dir + '/text_test.pkl', 'rb'))
    labels_test = pickle.load(open(set_dir + '/labels_test.pkl', 'rb'))
    return text_train, text_val, text_test, labels_train, labels_val, labels_test

####################################################
###### main loading function #######################
####################################################

def get_data(args):
    set_dir = os.path.join(args.data_dir, args.data_name)
    text_train = pickle.load(open(set_dir + '/text_train.pkl', 'rb'))
    labels_train = pickle.load(open(set_dir + '/labels_train.pkl', 'rb'))
    text_val = pickle.load(open(set_dir + '/text_val.pkl', 'rb'))
    labels_val = pickle.load(open(set_dir + '/labels_val.pkl', 'rb'))
    text_test = pickle.load(open(set_dir + '/text_test.pkl', 'rb'))
    labels_test = pickle.load(open(set_dir + '/labels_test.pkl', 'rb'))
    return text_train, text_val, text_test, labels_train, labels_val, labels_test


def load_data(args):
    if args.data_name == 'toxicity' or args.data_name == 'toxicity_full':
        text_train, text_val, text_test, labels_train, labels_val, labels_test = get_toxicity(args)
    elif args.data_name == 'rt-polarity':
        text_train, text_val, text_test, labels_train, labels_val, labels_test = get_reviews(args)
    elif args.data_name == 'ethics':
        text_train, text_val, text_test, labels_train, labels_val, labels_test = get_ethics(args)
    elif args.data_name == 'restaurant':
        text_train, text_val, text_test, labels_train, labels_val, labels_test = get_restaurant(args)
    elif args.data_name == 'jigsaw':
        text_train, text_val, text_test, labels_train, labels_val, labels_test = get_jigsaw(args)
    elif args.data_name == 'propaganda':
        text_train, text_val, text_test, labels_train, labels_val, labels_test = get_propaganda(args)
    return text_train, text_val, text_test, labels_train, labels_val, labels_test


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


def compute_image_features(image_dir='SMID_images_400px/img'):
    # image_dir = 'YelpOpenReviews/photos/photos'
    model, preprocess = clip.load('ViT-B/16', f'cuda:0')
    for param in model.parameters():
        param.requires_grad = False
    path = '/workspace/repositories/datasets/' + image_dir
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    image_features, fname = [], []
    for file in tqdm(files):
        with open(os.path.join(path, file), 'rb') as f:
            image = preprocess(Image.open(f)).unsqueeze(0).to(f'cuda:0')
            with torch.no_grad():
                image_features.append(model.encode_image(image).cpu())
            fname.append(os.path.join(image_dir, file))
    image_features = torch.cat(image_features)

    pickle.dump([fname, image_features], open(os.path.dirname(path) + '/image_features.pkl', 'wb'))
    return fname, image_features


def load_image_features(image_dir='imagenet'):
    image_dir = 'YelpOpenReviews/clip'
    # image_dir = 'SMID_images_400px/clip'
    path = '/workspace/repositories/datasets'
    if image_dir == 'imagenet':
        path_ = os.path.join(path, 'clip', 'imagenet_emb')
    else:
        path_ = os.path.join(path, image_dir)
    files = [f for f in os.listdir(path_) if os.path.isfile(os.path.join(path_, f))]
    image_features, fname = [], []
    for file in files:
        with open(os.path.join(path_, file), 'rb') as f:
            f, i = pickle.load(f)
            image_features.append(i)
            fname.append(f)
    image_features = np.concatenate(image_features)
    if image_dir == 'imagenet':
        fname = [st[22:] for sublist in fname for st in sublist]
    else:
        fname = [st for sublist in fname for st in sublist]
    return fname, image_features


def path2img(fname):
    path = '/workspace/repositories/datasets'
    with open(os.path.join(path, fname), 'rb') as f:
        im = torch.from_numpy(np.array(plt.imread(f, format='jpeg')))
        if len(im.shape) < 3:
            im = im.unsqueeze(2).repeat(1, 1, 3)
        image = im.float().permute(2, 0, 1) / 255
    return image


def nearest_image(args, model, proto_texts):
    import sentence_transformers
    fname, image_features = load_image_features()
    # query = model.protolayer.detach().clone().squeeze()
    query, _ = model.compute_embedding(proto_texts, args)
    query = query.squeeze()
    topk = 3
    nearest_img = sentence_transformers.util.semantic_search(query, torch.tensor(image_features).float(), top_k=topk)
    nearest_img = [k['corpus_id'] for topk_img in nearest_img for k in topk_img]

    n = 0
    for i in range(args.num_prototypes):
        # proto_images = []
        for k in range(topk):
            img = path2img(fname[nearest_img[n]]).float()
            # proto_images.append(img)
            save_image(img, os.path.dirname(args.model_path) + f'/proto{i + 1}.{k + 1}.png')
            n += 1


def load_images(args, image_dir='SMID_images_400px/img'):
    image_dir = 'YelpOpenReviews/photos/photos'
    _, preprocess = clip.load('ViT-B/16', f'cuda:{args.gpu}')
    path = '/workspace/repositories/datasets/' + image_dir
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    images, fname = [], []
    __import__("pdb").set_trace()

    for file in tqdm(files):
        with open(os.path.join(path, file), 'rb') as f:
            images.append(preprocess(Image.open(f)).unsqueeze(0))
            fname.append(os.path.join(image_dir, file))
    images = torch.cat(images)
    pickle.dump([fname, images], open(os.path.dirname(path) + '/images.pkl', 'wb'))
    return fname, images

#############################################################
#############################################################
#############################################################
#Functions for automatic evaluation
def parse_results():
    import glob
    import csv
    result_dir = "experiments/train_results"
    
    results_dict = {}
    
    for subdir in os.listdir(result_dir):
        if subdir == "archived":
            continue
        
        parts = subdir.split("_")
        num_prototypes = int(parts[1])
        model_name = parts[2]
        dataset = parts[3]
        similarity = parts[6]
        
        # Initialize the dictionary for this dataset if it doesn't exist yet
        if dataset not in results_dict:
            results_dict[dataset] = {}
        
        # Initialize the dictionary for this model if it doesn't exist yet
        if model_name not in results_dict[dataset]:
            results_dict[dataset][model_name] = {}
        
        # Initialize the dictionary for this number of prototypes if it doesn't exist yet
        if num_prototypes not in results_dict[dataset][model_name]:
            results_dict[dataset][model_name][num_prototypes] = {}
        
        
        results_files = glob.glob(os.path.join('experiments/train_results', subdir, '*prototypes.txt'))
        for results_file in results_files:
            with open(results_file, 'r') as f:
                for line in f:
                    if line.startswith("num_prototypes"):
                        practical_prototypes = int(line.split(": ")[1])
                    if line.startswith("test acc"):
                        acc = float(line.split(": ")[1])
        
        interacted_files = glob.glob(os.path.join('experiments/train_results', subdir, 'interacted_*prototypes.txt'))
        for interacted_file in interacted_files:
            with open(interacted_file, 'r') as f:
                for line in f:
                    if line.startswith("num_prototypes"):
                        unique_practical_prototypes = int(line.split(": ")[1])
                    if line.startswith("test acc"):
                        unique_acc = float(line.split(": ")[1])
        
        if similarity not in results_dict[dataset][model_name][num_prototypes]:
            results_dict[dataset][model_name][num_prototypes][similarity] = {
                "practical_prototypes": [practical_prototypes],
                "acc": [acc],
                "unique_practical_prototypes": [unique_practical_prototypes],
                "unique_acc": [unique_acc]
                }
        else:
            results_dict[dataset][model_name][num_prototypes][similarity]["practical_prototypes"].append(practical_prototypes)
            results_dict[dataset][model_name][num_prototypes][similarity]["acc"].append(acc)
            results_dict[dataset][model_name][num_prototypes][similarity]["unique_practical_prototypes"].append(unique_practical_prototypes)
            results_dict[dataset][model_name][num_prototypes][similarity]["unique_acc"].append(unique_acc)
    
    for dataset in results_dict:
        with open(f"experiments/{dataset}_results.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(["model", "num_prototypes", "similarity", "practical_prototypes", "acc", "unique_practical_prototypes", "unique_acc"])
            for model_name in results_dict[dataset]:
                for num_prototypes in results_dict[dataset][model_name]:
                    for similarity in results_dict[dataset][model_name][num_prototypes]:
                        writer.writerow([model_name, num_prototypes, similarity, results_dict[dataset][model_name][num_prototypes][similarity]["practical_prototypes"], results_dict[dataset][model_name][num_prototypes][similarity]["acc"], results_dict[dataset][model_name][num_prototypes][similarity]["unique_practical_prototypes"], results_dict[dataset][model_name][num_prototypes][similarity]["unique_acc"]])
        
    with open('experiments/results_dict.pkl', 'wb') as f:
        pickle.dump(results_dict, f)
        
    return results_dict

            
def compute_averages(results_dict):
    averages_dict = {}
    for dataset in results_dict:
        for model_name in results_dict[dataset]:
            for num_prototypes in results_dict[dataset][model_name]:
                for similarity in results_dict[dataset][model_name][num_prototypes]:
                    # Initialize averages_dict for this combination
                    if dataset not in averages_dict:
                        averages_dict[dataset] = {}
                    if model_name not in averages_dict[dataset]:
                        averages_dict[dataset][model_name] = {}
                    if num_prototypes not in averages_dict[dataset][model_name]:
                        averages_dict[dataset][model_name][num_prototypes] = {}
                    if similarity not in averages_dict[dataset][model_name][num_prototypes]:
                        averages_dict[dataset][model_name][num_prototypes][similarity] = {}

                    # Compute averages
                    results = results_dict[dataset][model_name][num_prototypes][similarity]
                    avg_practical_prototypes = np.mean(results["practical_prototypes"])
                    avg_acc = np.mean(results["acc"])
                    avg_unique_practical_prototypes = np.mean(results["unique_practical_prototypes"])
                    avg_unique_acc = np.mean(results["unique_acc"])
                    std_practical_prototypes = np.std(results["practical_prototypes"])
                    std_acc = np.std(results["acc"])
                    std_unique_practical_prototypes = np.std(results["unique_practical_prototypes"])
                    std_unique_acc = np.std(results["unique_acc"])
                    
                    # Save averages to averages_dict
                    averages_dict[dataset][model_name][num_prototypes][similarity]["avg_practical_prototypes"] = avg_practical_prototypes
                    averages_dict[dataset][model_name][num_prototypes][similarity]["avg_acc"] = avg_acc
                    averages_dict[dataset][model_name][num_prototypes][similarity]["avg_unique_practical_prototypes"] = avg_unique_practical_prototypes
                    averages_dict[dataset][model_name][num_prototypes][similarity]["avg_unique_acc"] = avg_unique_acc
                    averages_dict[dataset][model_name][num_prototypes][similarity]["std_practical_prototypes"] = std_practical_prototypes
                    averages_dict[dataset][model_name][num_prototypes][similarity]["std_acc"] = std_acc
                    averages_dict[dataset][model_name][num_prototypes][similarity]["std_unique_practical_prototypes"] = std_unique_practical_prototypes
                    averages_dict[dataset][model_name][num_prototypes][similarity]["std_unique_acc"] = std_unique_acc
                    
                    # Save averages and stds to CSV
                    # with open(f'experiments/{dataset}_avg_results.csv', 'w', newline='') as csvfile:
                    #     writer = csv.writer(csvfile, delimiter=',')
                    #     writer.writerow(["model", "num_protots", "smilarity", "avg_practical_prototypes", "std_practical_prototypes", "avg_acc", "std_acc", "avg_unique_practical_prototypes", "std_unique_practical_prototypes", "avg_unique_acc", "std_unique_acc"])
                    #     writer.writerow([model_name, num_prototypes, similarity, avg_practical_prototypes, std_practical_prototypes, avg_acc, std_acc, avg_unique_practical_prototypes, std_unique_practical_prototypes, avg_unique_acc, std_unique_acc])
                    
    return averages_dict

def write_averages_to_csv(averages_dict):
    for dataset in averages_dict:
        with open(f"experiments/{dataset}_results.csv", 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for model_name in averages_dict[dataset]:
                for num_prototypes in averages_dict[dataset][model_name]:
                    for similarity in averages_dict[dataset][model_name][num_prototypes]:
                        row = [model_name, num_prototypes, similarity]
                        results = averages_dict[dataset][model_name][num_prototypes][similarity]
                        row.extend([
                            results["avg_practical_prototypes"], 
                            results["avg_acc"], 
                            results["avg_unique_practical_prototypes"], 
                            results["avg_unique_acc"],
                            results["std_practical_prototypes"], 
                            results["std_acc"], 
                            results["std_unique_practical_prototypes"], 
                            results["std_unique_acc"]
                        ])
                        writer.writerow(row)

def transform_explain(args, path):
    df = pd.read_csv(path)
    df = df[df["true label \n"] == df["predicted label \n"]]
    
    cols = [f"score_{i} \n" for i in range(1, args.num_prototypes+1)]
    cols = df[cols].idxmax(axis=1)
    
    #get index of explanation
    index = cols.str.replace("score_", "").str.replace(" \n", "")
    index = [int(i) for i in index.to_list()]
    print(index)
    #get two different numbers that are different from index
    rand1 = []
    rand2 = []
    for i in range(len(index)):
        num1 = np.random.randint(1, args.num_prototypes+1)
        while num1 == index[i]:
            num1 = np.random.randint(1, args.num_prototypes+1)
        
        num2 = np.random.randint(1, args.num_prototypes+1)
        while num2 == index[i] or num2 == num1:
            num2 = np.random.randint(1, args.num_prototypes+1)
        
        rand1.append(num1)
        rand2.append(num2)
        

    expl = cols.str.replace("score", "explanation")
    naming = {}
    for i in range(1, args.num_prototypes+1):
        naming[f"explanation_{i} \n"] = i * 5
    strings = [df.iloc[i, naming[expl.iloc[i]]] for i in range(len(expl))]
    result = df[["test sample \n", "true label \n"]]
    result["expl \n"] = strings
    
    randstr1 = [df.iloc[i, naming[f"explanation_{rand1[i]} \n"]] for i in range(len(rand1))]
    randstr2 = [df.iloc[i, naming[f"explanation_{rand2[i]} \n"]] for i in range(len(rand2))]
    
    dit = {}
    i = 0
    for sen in strings:
        if sen not in dit:
            dit[sen] = i
            i = i + 1
    indices = [dit[sen] for sen in strings]
    result["indices \n"] = indices
    result["random expl1 \n"] = randstr1
    result["random expl2 \n"] = randstr2
    npath = os.path.join(os.path.dirname(path), "explained_modded.csv")
    result.to_csv(npath)
    
    # get least count of any group, if value counts of groups are less than 5, drop them from results
    groups = result["indices \n"].value_counts().sort_values()
    least_count = groups[groups > 5].min()
    result = result[result["indices \n"].isin(groups[groups > 5].index)]
    
    
    
    # create disproportionate sample from result using least_count
    survey = result.groupby("indices \n").apply(lambda x: x.sample(least_count)).reset_index(drop=True)
    # shuffle the rows randomly
    survey = survey.sample(frac=1).reset_index(drop=True)
    # add a random explanation column that randomly sampled from survey["expl \n"] and different from survey["expl \n"]

   
    

    survey_path = os.path.join(os.path.dirname(path), "survey.csv")
    survey.to_csv(survey_path)
    

def create_html_survey(survey_path):
    import random
    #set start and end row to read questions from
    start_row = 0
    end_row = 17

    #set number of questions to duplicate for quality assurance testing
    num_duplicates = 2
    duplicate_indices = sorted(random.sample(range(start_row, end_row), num_duplicates))

    #load dataframe
    df = pd.read_csv(survey_path)
    base_path = os.path.dirname(survey_path)
    
    #"golden standard" question
    golden_rule_question = 'What is the color of the sky?'
    golden_rule_answer = 'blue'
    golden_rule_explanation = 'The color of the sky is blue.'
    golden_rule_choices = ['black', 'green']

    # Randomly shuffle the answer choices
    random.shuffle(golden_rule_choices)

    # Add the correct answer to the list of choices
    golden_rule_choices.append(golden_rule_answer)
    
    # Randomly shuffle the choices again
    random.shuffle(golden_rule_choices)
    golden_rule_choices.append('none of the above')
    # Create the HTML for the "golden rule" question
    golden_rule_html = f'<div>\n<p>Question: {golden_rule_question}</p>\n'
    for i, choice in enumerate(golden_rule_choices):
        golden_rule_html += f'<input type="radio" name="q00_{i}" value="{choice}" id="q00_{i}_choice{i+1}" required>\n'
        golden_rule_html += f'<label for="q00_{i}_choice{i+1}">{choice}</label><br>\n'
    golden_rule_html += f'<p>{golden_rule_explanation}</p>\n</div>\n'

    # Create the HTML for the questions
    html = golden_rule_html
    html += '|'

    for index, row in df[start_row:end_row].iterrows():
        if index in duplicate_indices:
            for i in range(num_duplicates):
                # Extract the question, answer, and explanations from the row
                question = row['test sample \n']
                answer = row['expl \n']
                explanations = [row['random expl1 \n'], row['random expl2 \n']]
                
                # Randomly shuffle the answer and explanation list
                choices = explanations[:2] + [answer]
                random.shuffle(choices)
                choices.append('none of the above')
                
                # Create the HTML for the multiple-choice question with a unique ID
                html += f'<div>\n<p>Question: {question}</p>\n'
                for j, choice in enumerate(choices):
                    html += f'<input type="radio" name="q{index}_{i}" value="{choice}" id="q{index}_{i}_choice{j+1}" required>\n'
                    html += f'<label for="q{index}_{i}_choice{j+1}">{choice}</label><br>\n'
                html += '</div>|'
        else:
            # Extract the question, answer, and explanations from the row
            question = row['test sample \n']
            answer = row['expl \n']
            explanations = [row['random expl1 \n'], row['random expl2 \n']]
            
            # Randomly shuffle the answer and explanation list
            choices = explanations[:2] + [answer]
            random.shuffle(choices)
            choices.append('none of the above')
            
            # Create the HTML for the multiple-choice question with a unique ID
            html += f'<div>\n<p>Question: {question}</p>\n'
            for j, choice in enumerate(choices):
                html += f'<input type="radio" name="q{index}" value="{choice}" id="q{index}_choice{j+1}" required>\n'
                html += f'<label for="q{index}_choice{j+1}">{choice}</label><br>\n'
            html += '</div>|'
    
    #randomly shuffle the questions
    html_list = html.split('|')
    random.shuffle(html_list)
    shuffled_html = '\n\n'.join(html_list)
    
    with open(os.path.join(base_path, 'survey.html'), 'w') as f:
        f.write(shuffled_html)

def parse_robustness(path):
    import glob
    pattern = path + "/robustness_*_*.txt"
    file_list = glob.glob(pattern)
    
    df_list = []
    for file_name in file_list:
        with open(file_name, 'r') as f:
            for line in f:
                if line.startswith('test acc:'):
                    acc = float(line.split(': ')[-1])
        df = pd.DataFrame()
        if 'pos_neg' in file_name:
            df["type"] = ["Dual-Polarity"]
            df["percentage"] = [file_name.split('/')[-1].split('_')[3]]
            df["reinit"] = [file_name.split('/')[-1].split('_')[4]]
            df["epochs"] = [file_name.split('/')[-1].split('_')[5]]
        else:
            df["type"] = [file_name.split('/')[-1].split('_')[1]]
            df["percentage"] = [file_name.split('/')[-1].split('_')[2]]
            df["reinit"] = [file_name.split('/')[-1].split('_')[3]]
            df["epochs"] = [file_name.split('/')[-1].split('_')[4]]
        df["accuracy"] = [acc]
        df_list.append(df)
    df = pd.concat(df_list)
    df.to_csv(path + "/robustness.csv", index=False)
    
    for type in df["type"].unique():
        df_type = df[df["type"] == type]
        df_type.to_csv(path + f"/robustness_{type}.csv", index=False)

        grouped_df = df_type.groupby(['reinit', 'epochs', 'percentage']).mean().reset_index()
        grouped_df.to_csv(path + f"/robustness_{type}_average.csv", index=False)

    return df


def plot_robustness(path):
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.read_csv(path)
    palette = sns.color_palette("husl", n_colors=3)
    sns.set_style("whitegrid")
    sns.set_palette(palette)
    type = df["type"].unique()[0]
    if type == "pos_neg":
        type = "Dual-Polarity"
    
    grouped = df.groupby(["reinit", "epochs"])
    for i, (name, group) in enumerate(grouped):
        reinit = name[0]
        epochs = name[1]
        label = f"{reinit}, {epochs}"
        sns.lineplot(x="percentage", y="accuracy", data=group, label=label, color=palette[i])
    plt.title(f"{type.capitalize()} Replacement")
    plt.xlabel("Percentage of Replacements")
    plt.ylabel("Accuracy")
    plt.legend(title="Reinit, Epochs")
    basedir = os.path.dirname(path)
    plt.savefig(basedir + f"/robustness_{type}.png")


POSITIVE = ["Showing filmmaking, photography, and performance skills, this is a story.",
            "With a good message and actors, this movie is funny, happy, and positive.",
            "Action, suspense, comedy, plot, and effects make this movie exciting and fun.",
            "This drama shows life and its challenges realistically and emotionally, with good acting and direction.",
            "With stunning storytelling, cinematography, and acting, this movie is an emotional masterpiece.",
            "You will love this movie for its humor, heart, inspiration, and the amazing performances of its actors. It has a great message too.",
            "The plot and the effects of this movie are amazing. It is full of action, suspense, and humor that will keep you entertained.",
            "A beautiful and moving drama with a realistic and poignant portrayal of life and superb performances.",
            "This comedy is a breath of fresh air. The script is witty and clever, and the cast is hilarious."
            "The world and the characters of this fantasy are complex and rich. It is a captivating and epic movie with amazing visuals.",
            "A fascinating and insightful documentary with a compelling and important topic and evidence.",
            "With a spooky and unforeseeable plot and atmosphere, this horror movie will scare and thrill you."]

NEGATIVE = ["A boring and dull movie with a weak and cliched plot and characters and poor acting.",
            "It was a boring and dull movie with a plot and characters that lacked originality and acting that was subpar.",
            "Wasting the talent of the actors as this crude and unfunny script results in a stupid and offensive movie",
            "The movie is confusing and disappointing due to a messy and illogical plot and bad visual effects.",
            "Unlikeable characters and a slow, boring pace make this a depressing watch.",
            "A cheesy and predictable movie with a lame and unrealistic story and romance and bad dialogue.",
            "Generic, bland, uninspiring. A copy and paste movie.",
            "Using cherry-picked data and sources the documentary follows a biased and misleading agenda.",
            "A corny and sappy movie with a forced and unrealistic romance and melodrama and cliches.",
            "Just a lame movie due to predictable jump scares and way too much gore.",
            "The poor and exaggerated portrayal of a historical figure can be seen in this movie.",
            "The characters are poorly developed, and the plot is predictable."]

FACTS = ["Animals are multicellular, eukaryotic organisms that belong to the kingdom Animalia.", 
                "About 15,000-20,000 new animal species are discovered every year.", 
                "A butterfly has about 12,000 eyes.", 
                "Tigers have striped skin, not just striped fur."
                "Jellyfish are made up of 95% water.",
                "The loudest animal in the world is a 2cm long prawn called Pistol Shrimp.",
                "Flamingos are not pink; they get their color from their diet of brine shrimp and algae.",
                "Otters hold hands while sleeping to keep from drifting apart.",
                "Elephants can recognize themselves in a mirror.",
                "Sloths can take up to a month to digest a single leaf.",
                "The blue whale is the largest animal that ever lived.",
                "The cheetah is the fastest land animal, reaching speeds of up to 120 km/h.",
                "The great white shark can detect a drop of blood in 25 liters of water.",
            	"The hummingbird is the only bird that can fly backwards.",
                "The platypus is one of the few mammals that lay eggs instead of giving birth.",
                "The octopus has three hearts, nine brains and blue blood.",
                "The giraffe has a tongue that is about 50 cm long and can clean its ears with it.",
                "The kangaroo can jump up to 9 meters in a single leap.",
                "The koala sleeps for up to 22 hours a day.",
                "The axolotl can regenerate its limbs, tail, heart and even parts of its brain",
                "Cats have a free-floating collarbone that enables them to fit through tight spaces and land safely after falls."]

if __name__ == '__main__':
    result_dict = parse_results()
    averages_dict = compute_averages(result_dict)
    write_averages_to_csv(averages_dict)