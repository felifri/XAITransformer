"""
Script for training a protopnet. (Li et al. 2018)
https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/17082/16552

CUDA_VISIBLE_DEVICES=0 python main_attention.py --mode train --data standard --batch_size 128 --lr 0.0001 --num_epochs 2 --n_splits 5 --split 0 --fp_data /home/ml-wstammer/WS/datasets/plantpheno_berry/t4/whiteref_norm/mean/parsed/ --perc_pxl_per_sample 10
"""

import pickle
import numpy as np
import os
import torch
# import uuid
import argparse
import matplotlib as mpl
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score # balanced_accuracy_score
from matplotlib import rc
from setproctitle import setproctitle
import matplotlib.pyplot as plt
import random

from models import ProtopNetNLP

sns.set(style='ticks', palette='Set2')
sns.despine()
rc('text', usetex=True)

mpl.rcParams['savefig.pad_inches'] = 0

parser = argparse.ArgumentParser(description='Crazy Stuff')
parser.add_argument('-m', '--mode', default="train", type=str,
                    help='What do you want to do? Select either train, test, full_image_test, attention')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate')
parser.add_argument('--cpu', action='store_true', default=False,
                    help='whether to use cpu')
parser.add_argument('-e', '--num_epochs', default=100, type=int,
                    help='How many epochs?')
parser.add_argument('-bs', '--batch_size', default=20, type=int,
                    help='Batch size')
# parser.add_argument('--test_epoch', default=10, type=int,
#                     help='After how many epochs should the model be evaluated on the test data?')
parser.add_argument('--data-dir', default='data/rt-polarity',
                    help='Train data in format defined by --data-io param.')
parser.add_argument('--num-prototypes', default=80,
                    help='total number of prototypes')
parser.add_argument('--lambda2', default=0.1,
                    help='weight for prototype loss computation')
parser.add_argument('--lambda3', default=0.1,
                    help='weight for prototype loss computation')
parser.add_argument('--num-classes', default=2,
                    help='How many classes are to be classified?')
parser.add_argument('--class_weights', default=0.5,
                    help='Class weight for cross entropy loss')
parser.add_argument('--enc_size', default=768,
                    help='embedding size of sentence/ word encoding')
parser.add_argument('--gpu', type=int, default=0, help='GPU device number, -1  means CPU.')

def get_args(args):
    if args.cpu:
        args.device = "cpu"
    else:
        args.device = "cuda"
    return args

def get_data(args):
    set = args.mode
    set_dir = []
    # f_names = ['rt-polarity.neg', 'rt-polarity.pos']

    if set=='train':
        set_dir = os.path.join(args.data_dir, 'train')
    elif set=='val':
        set_dir = os.path.join(args.data_dir, 'dev')
    elif set=='test':
        set_dir = os.path.join(args.data_dir, 'test')

    text = pickle.load( open(os.path.join(set_dir, 'word_sequences') + '.pkl', 'rb'))
    text = [' '.join(sub_list) for sub_list in text]    #join tokenized text back for sentenceBert
    labels = pickle.load( open(os.path.join(set_dir, 'labels') + '.pkl', 'rb'))
    return text, labels

def get_train_batches(text, labels, batch_size=10):
    def divide_chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]
    tmp = list(zip(text, labels))
    random.shuffle(tmp)
    text, labels = zip(*tmp)
    text_batches = list(divide_chunks(text, batch_size))
    label_batches = list(divide_chunks(labels, batch_size))
    return text_batches, label_batches

class ProtoLoss:
    def __init__(self):
        pass

    def __call__(self, feature_vector_distances, prototype_distances):
        """
        Computes the interpretability losses (R1 and R2 from the paper (Li et al. 2018)) for the prototype nets.

        :param feature_vector_distances: tensor of size [n_prototypes, n_batches], distance between the data encodings
                                          of the autoencoder and the prototypes
        :param prototype_distances: tensor of size [n_batches, n_prototypes], distance between the prototypes and
                                    data encodings of the autoencoder
        :return:
        """
        #assert prototype_distances.shape == feature_vector_distances.T.shape
        r1_loss = torch.mean(torch.min(feature_vector_distances, dim=1)[0])
        r2_loss = torch.mean(torch.min(prototype_distances, dim=1)[0])
        return r1_loss, r2_loss


def train(args):
    global proctitle

    text, labels = get_data(args)
    model = ProtopNetNLP(args)
    # init = model.init_protos(self, args, text_train, labels_train)

    num_params = sum(p.numel() for p in model.parameters())
    print("Number of parameters {}".format(num_params))
    print("Running on {}".format(args.gpu))
    model.cuda(args.gpu)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce_crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(args.class_weights).float().cuda(args.gpu))
    interp_criteria = ProtoLoss()

    model.train()
    num_epochs = args.num_epochs
    for epoch in tqdm(range(num_epochs)):
        setproctitle(proctitle + args.mode + " | epoch {} of {}".format(epoch + 1, num_epochs))
        all_preds = []
        losses_per_batch = []
        ce_loss_per_batch = []
        r1_loss_per_batch = []
        r2_loss_per_batch = []
        text_batches, label_batches = get_train_batches(text, labels, args.batch_size)

        for i,(text_batch,label_batch) in enumerate(zip(text_batches,label_batches)):
            optimizer.zero_grad()

            #text_batch = text_batch.cuda(args.gpu)
            label_batch = label_batch.cuda(args.gpu)

            outputs = model.forward(text_batch, args.gpu)
            prototype_distances, feature_vector_distances, predicted_label, _ = outputs

            # compute individual losses and backward step
            ce_loss = ce_crit(predicted_label, label_batch)
            r1_loss, r2_loss = interp_criteria(feature_vector_distances, prototype_distances)
            loss = ce_loss + \
                   args.lambda2 * r1_loss + \
                   args.lambda3 * r2_loss

            _, predicted = torch.max(predicted_label.data, 1)
            all_preds += predicted.cpu().numpy().tolist()

            loss.backward(retain_graph=True)
            # nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            # store losses
            losses_per_batch.append(float(loss))
            ce_loss_per_batch.append(float(ce_loss))
            r1_loss_per_batch.append(float(r1_loss))
            r2_loss_per_batch.append(float(r2_loss))

        mean_loss = np.mean(losses_per_batch)
        acc = accuracy_score(labels, all_preds)
        print("Epoch {}, mean loss per batch {:.4f}, train acc {:.4f}".format(epoch, mean_loss, 100 * acc))

def transform_space(X):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # from sklearn.manifold import TSNE
    # X_trans = TSNE(n_components=2).fit_transform(X)
    from sklearn.decomposition import PCA
    X_trans = PCA(n_components=3).fit(X)
    ax.scatter(X_trans[:,0],X_trans[:,1],X_trans[:,2])
    plt.show()

def nearest_neighbors(text_embedded, prototypes):
    distances = torch.cdist(text_embedded, prototypes, p=2) # shape, num_samples x num_prototypes
    nearest_ids = torch.argmin(distances, dim=0)
    return nearest_ids # text[nearest_ids]


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    args = parser.parse_args()
    #args = get_args(args)
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
    global proctitle
    proctitle = "Prototype learning"
    setproctitle(proctitle + args.mode + " | warming up")
    if args.mode == 'train':
        train(args)
    # elif args.mode == 'test':
    #     test(args)
    # elif args.mode == 'adapt':
    #     adapt_prototypes(args)
    # else:
    #     print("Nothing to do here")
    #     exit()