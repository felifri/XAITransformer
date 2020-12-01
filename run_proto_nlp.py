"""
Script for training a protopnet. (Li et al. 2018)
https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/17082/16552

CUDA_VISIBLE_DEVICES=0 python main_attention.py --mode train --data standard --batch_size 128 --lr 0.0001 --num_epochs 2 --n_splits 5 --split 0 --fp_data /home/ml-wstammer/WS/datasets/plantpheno_berry/t4/whiteref_norm/mean/parsed/ --perc_pxl_per_sample 10
"""

import pickle
import numpy as np
import os
import torch
import argparse
import matplotlib as mpl
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score # balanced_accuracy_score
from matplotlib import rc
import matplotlib.pyplot as plt
import random
import uuid

from rtpt.rtpt import RTPT
from models import ProtopNetNLP

sns.set(style='ticks', palette='Set2')
sns.despine()
rc('text', usetex=True)

mpl.rcParams['savefig.pad_inches'] = 0

# Create RTPT object
rtpt = RTPT(name_initials='FF', experiment_name='Transformer_Prototype', max_iterations=100)
# Start the RTPT tracking
rtpt.start()

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
parser.add_argument('--test_epoch', default=10, type=int,
                    help='After how many epochs should the model be evaluated on the test data?')
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
parser.add_argument('--class_weights', default=[0.5,0.5],
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

def get_data(args, set='train'):
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

def convert_label(labels, gpu):
    converted_labels = torch.empty(len(labels), dtype=torch.long)
    for i,label in enumerate(labels):
        if label=='pos':
            converted_labels[i] = 1
        elif label=='neg':
            converted_labels[i] = 0
    return converted_labels.cuda(gpu)

def save_checkpoint(save_dir, state, best, id, filename='best_model.pth.tar'):
    if best:
        save_path_checkpoint = os.path.join(os.path.join(save_dir, id), filename)
        os.makedirs(os.path.dirname(save_path_checkpoint), exist_ok=True)
        torch.save(state, save_path_checkpoint.replace('best_model.pth.tar', 'best_model.pth.tar'))

def train(args):
    run_id = args.mode + '_' + str(uuid.uuid1())
    save_dir = "./runs/results/"

    text, labels = get_data(args, args.mode)
    text_val, labels_val = get_data(args, 'val')
    model = ProtopNetNLP(args)
    # init = model.init_protos(self, args, text_train, labels_train)

    num_params = sum(p.numel() for p in model.parameters())
    print("Number of parameters {}".format(num_params))
    print("Running on gpu {}".format(args.gpu))
    model.cuda(args.gpu)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce_crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(args.class_weights).float().cuda(args.gpu))
    interp_criteria = ProtoLoss()
    text_val_batches, label_val_batches = get_train_batches(text_val, labels_val, args.batch_size)

    model.train()
    num_epochs = args.num_epochs
    print("\nStarting training for {} epochs\n".format(num_epochs))
    best_acc = 0
    for epoch in tqdm(range(num_epochs)):
        #setproctitle(proctitle + args.mode + " | epoch {} of {}".format(epoch + 1, num_epochs))
        all_preds = []
        all_labels = []
        losses_per_batch = []
        ce_loss_per_batch = []
        r1_loss_per_batch = []
        r2_loss_per_batch = []
        text_batches, label_batches = get_train_batches(text, labels, args.batch_size)

        # Update the RTPT
        rtpt.step(subtitle=f"epoch={epoch}")

        for i,(text_batch,label_batch) in enumerate(zip(text_batches,label_batches)):
            optimizer.zero_grad()

            outputs = model.forward(text_batch, args.gpu)
            prototype_distances, feature_vector_distances, predicted_label, _ = outputs

            # compute individual losses and backward step
            label_batch = convert_label(label_batch, args.gpu)
            ce_loss = ce_crit(predicted_label, label_batch)
            r1_loss, r2_loss = interp_criteria(feature_vector_distances, prototype_distances)
            loss = ce_loss + \
                   args.lambda2 * r1_loss + \
                   args.lambda3 * r2_loss

            _, predicted = torch.max(predicted_label.data, 1)
            all_preds += predicted.cpu().numpy().tolist()
            all_labels += label_batch.cpu().numpy().tolist()

            loss.backward(retain_graph=True)
            # nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            # store losses
            losses_per_batch.append(float(loss))
            ce_loss_per_batch.append(float(ce_loss))
            r1_loss_per_batch.append(float(r1_loss))
            r2_loss_per_batch.append(float(r2_loss))

            if (epoch + 1) % args.test_epoch == 0 or epoch + 1 == num_epochs:
                model.eval()
                losses_per_batch = []
                ce_loss_per_batch = []
                r1_loss_per_batch = []
                r2_loss_per_batch = []
                all_labels_val = []
                all_preds_val = []
                with torch.no_grad():
                    for i, (text_val_batch, label_val_batch) in enumerate(zip(text_val_batches, label_val_batches)):

                        outputs = model.forward(text_val_batch, args.gpu)
                        prototype_distances, feature_vector_distances, predicted_label, _ = outputs

                        # compute individual losses and backward step
                        label_val_batch = convert_label(label_val_batch, args.gpu)
                        ce_loss = ce_crit(predicted_label, label_val_batch)
                        r1_loss, r2_loss = interp_criteria(feature_vector_distances, prototype_distances)
                        loss = ce_loss + \
                               args.lambda2 * r1_loss + \
                               args.lambda3 * r2_loss

                        _, predicted = torch.max(predicted_label.data, 1)
                        all_preds_val += predicted.cpu().numpy().tolist()
                        all_labels_val += label_val_batch.cpu().numpy().tolist()

                        # store losses
                        losses_per_batch.append(float(loss))
                        ce_loss_per_batch.append(float(ce_loss))
                        r1_loss_per_batch.append(float(r1_loss))
                        r2_loss_per_batch.append(float(r2_loss))

                    mean_loss = np.mean(losses_per_batch)
                    acc_val = accuracy_score(all_labels_val, all_preds_val)
                    print("Validation: mean loss {:.4f}, acc_val {:.4f}".format(mean_loss, 100 * acc_val))

                save_checkpoint(save_dir, {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'hyper_params': args,
                    # 'eval_acc': 100 * correct / total,
                    'acc_val': acc_val,
                }, best=acc_val >= best_acc, run_id=run_id)
                if acc_val >= best_acc:
                    best_acc = acc_val

        mean_loss = np.mean(losses_per_batch)
        acc = accuracy_score(all_labels, all_preds)
        print("Epoch {}, mean loss {:.4f}, train acc {:.4f}".format(epoch, mean_loss, 100 * acc))

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

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    if args.mode == 'train':
        train(args)
    # elif args.mode == 'test':
    #     test(args)
    # elif args.mode == 'adapt':
    #     adapt_prototypes(args)
    # else:
    #     print("Nothing to do here")
    #     exit()