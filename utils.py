import os
import torch
from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

# __import__("pdb").set_trace()

def save_checkpoint(save_dir, state, time_stmp, filename='best_model.pth.tar'):
    save_path_checkpoint = os.path.join(save_dir, time_stmp, filename)
    os.makedirs(os.path.dirname(save_path_checkpoint), exist_ok=True)
    torch.save(state, save_path_checkpoint)

def visualize_protos(embedding, labels, prototypes, n_components, trans_type, save_path):
        # visualize prototypes
        if trans_type == 'PCA':
            pca = PCA(n_components=n_components)
            pca.fit(embedding)
            print("Explained variance ratio of components after transform: ", pca.explained_variance_ratio_)
            embed_trans = pca.transform(embedding)
            proto_trans = pca.transform(prototypes)
        elif trans_type == 'TSNE':
            tsne = TSNE(n_jobs=8,n_components=n_components).fit_transform(np.vstack((embedding,prototypes)))
            [embed_trans, proto_trans] = [tsne[:len(embedding)],tsne[len(embedding):]]

        rnd_samples = np.random.randint(embed_trans.shape[0], size=500)
        rnd_labels = [labels[i] for i in rnd_samples]
        rnd_labels = ['green' if x == 1 else 'red' for x in rnd_labels]
        fig = plt.figure()
        if n_components==2:
            ax = fig.add_subplot(111)
            ax.scatter(embed_trans[rnd_samples,0],embed_trans[rnd_samples,1],c=rnd_labels,marker='x', label='data')
            ax.scatter(proto_trans[:,0],proto_trans[:,1],c='blue',marker='o',label='prototypes')
        elif n_components==3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(embed_trans[rnd_samples,0],embed_trans[rnd_samples,1],embed_trans[rnd_samples,2],c=rnd_labels,marker='x', label='data')
            ax.scatter(proto_trans[:,0],proto_trans[:,1],proto_trans[:,2],c='blue',marker='o',label='prototypes')
        ax.legend()
        fig.savefig(os.path.join(save_path, trans_type+'proto_vis'+str(n_components)+'d.png'))

def proto_loss(prototype_distances, label, model, args):
    if model.module.class_specific:
        max_dist = torch.prod(torch.tensor(model.module.protolayer.size())) # proxy variable, could be any high value

        # prototypes_of_correct_class is tensor of shape  batch_size * num_prototypes
        # calculate cluster cost, high cost if same class protos are far distant
        prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:, label]).to(f'cuda:{args.gpu[0]}')
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

    # assures that each prototype itself does not consist out of the exact same input entry multiple times
    dist_sum = 0
    if args.modeltype == 'dist' and args.proto_size > 1:
        comb = torch.combinations(torch.arange(args.proto_size), r=2)
        for k, l in comb:
            dist_sum += torch.dist(model.module.protolayer[:, k, :], model.module.protolayer[:, l, :], p=2)
        # if distance small -> higher penalty
        divers1_loss = - dist_sum / comb.size(0)
    else:
        divers1_loss = 0

    # diversity loss, assures that prototypes are not too close
    dist_sum = 0
    comb = torch.combinations(torch.arange(args.num_prototypes), r=2)
    for k, l in comb:
        dist_sum += torch.dist(model.module.protolayer[k, :, :], model.module.protolayer[l, :, :], p=2)
    # if distance small -> higher penalty
    divers2_loss = - dist_sum / comb.size(0)

    if model.module.use_l1_mask:
        l1_mask = 1 - torch.t(model.module.prototype_class_identity).to(f'cuda:{args.gpu[0]}')
        l1_loss = (model.module.fc.weight * l1_mask).norm(p=1)
    else:
        l1_loss = model.module.fc.weight.norm(p=1)

    return distr_loss, clust_loss, sep_loss, divers1_loss, divers2_loss, l1_loss

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