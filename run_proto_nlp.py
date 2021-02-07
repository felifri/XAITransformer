import argparse
import datetime
import glob
import os
import random

import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from rtpt import RTPT

from models import ProtoPNetConv, ProtoNet
from utils import save_embedding, load_embedding, load_data, visualize_protos, proto_loss, prune_prototypes, get_nearest

parser = argparse.ArgumentParser(description='Crazy Stuff')
parser.add_argument('-m', '--mode', default="train test", type=str, nargs='+',
                    help='What do you want to do? Select either any combination of train, test, query, finetune, prune')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Select learning rate')
parser.add_argument('-e', '--num_epochs', default=200, type=int,
                    help='How many epochs?')
parser.add_argument('-bs', '--batch_size', default=650, type=int,
                    help='Select batch size')
parser.add_argument('--val_epoch', default=10, type=int,
                    help='After how many epochs should the model be evaluated on the validation data?')
parser.add_argument('--data_dir', default='./data',
                    help='Select data path')
parser.add_argument('--data_name', default='rt-polarity', type=str, choices=['rt-polarity', 'toxicity', 'toxicity_full', 'restaurant'],
                    help='Select name of data set')
parser.add_argument('--num_prototypes', default=10, type=int,
                    help='Total number of prototypes')
parser.add_argument('-l1','--lambda1', default=0.02, type=float,
                    help='Weight for prototype distribution loss')
parser.add_argument('-l2','--lambda2', default=0.1, type=float,
                    help='Weight for prototype cluster loss')
parser.add_argument('-l3','--lambda3', default=0.01, type=float,
                    help='Weight for prototype separation loss')
parser.add_argument('-l4','--lambda4', default=0.01, type=float,
                    help='Weight for wihtin prototype diversity loss')
parser.add_argument('-l5','--lambda5', default=1e-4, type=float,
                    help='Weight for l1 weight regularization loss')
parser.add_argument('--num_classes', default=2, type=int,
                    help='How many classes are to be classified?')
parser.add_argument('-g','--gpu', type=int, default=[0], nargs='+',
                    help='GPU device number(s)')
parser.add_argument('--one_shot', type=bool, default=False,
                    help='Whether to use one-shot learning or not (i.e. only a few training examples)')
parser.add_argument('--trans_type', type=str, default='PCA', choices=['PCA', 'TSNE'],
                    help='Select transformation to visualize the prototypes')
parser.add_argument('--discard', type=bool, default=False,
                    help='Whether edge cases in the middle between completely toxic(1) and not toxic(0) shall be omitted')
parser.add_argument('--proto_size', type=int, default=1,
                    help='Define how many words should be used to define a prototype')
parser.add_argument('--level', type=str, default='word', choices=['word','sentence'],
                    help='Define whether prototypes are computed on word (Bert/GPT2) or sentence level (SentBert/CLS)')
parser.add_argument('--language_model', type=str, default='Bert', choices=['Bert','SentBert','GPT2'],
                    help='Define which language model to use')
parser.add_argument('--dilated', type=int, default=[1], nargs='+',
                    help='Whether to use dilation in the ProtoP convolution and which step size')
parser.add_argument('--avoid_spec_token', type=bool, default=False,
                    help='Whether to manually set PAD, SEP and CLS token to high value after Bert embedding computation')
parser.add_argument('--compute_emb', type=bool, default=False,
                    help='Whether to recompute (True) the embedding or just load it (False)')
parser.add_argument('--query', type=str, default='you are a faggot', nargs='+',
                    help='Type your query to test the model and get classification explanation')

def train(args, train_batches, val_batches, model):
    num_epochs = args.num_epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce_crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(args.class_weights).float().to(f'cuda:{args.gpu[0]}'))

    print(f"\nStart training for {num_epochs} epochs\n")
    best_acc, state = 0, {}

    for epoch in tqdm(range(num_epochs)):
        model.train()
        all_preds ,all_labels = [], []
        losses_per_batch = []
        ce_loss_per_batch = []
        distr_loss_per_batch = []
        clust_loss_per_batch = []
        sep_loss_per_batch = []
        divers_loss_per_batch = []
        l1_loss_per_batch = []

        # Update the RTPT
        rtpt.step()

        for emb_batch, label_batch in train_batches:
            emb_batch = emb_batch.to(f'cuda:{args.gpu[0]}')
            label_batch = label_batch.to(f'cuda:{args.gpu[0]}')

            optimizer.zero_grad()
            prototype_distances, _, predicted_label = model.forward(emb_batch)

            # compute individual losses and backward step
            ce_loss = ce_crit(predicted_label, label_batch)
            distr_loss, clust_loss, sep_loss, divers_loss, l1_loss = \
                proto_loss(prototype_distances, label_batch, model, args)
            loss = ce_loss + \
                   args.lambda1 * distr_loss + \
                   args.lambda2 * clust_loss + \
                   args.lambda3 * sep_loss + \
                   args.lambda4 * divers_loss + \
                   args.lambda5 * l1_loss

            _, predicted = torch.max(predicted_label.data, 1)
            all_preds += predicted.cpu().numpy().tolist()
            all_labels += label_batch.cpu().numpy().tolist()

            loss.backward()
            optimizer.step()
            # store losses
            losses_per_batch.append(float(loss))
            ce_loss_per_batch.append(float(ce_loss))
            distr_loss_per_batch.append(float(args.lambda1 * distr_loss))
            clust_loss_per_batch.append(float(args.lambda2 * clust_loss))
            sep_loss_per_batch.append(float(args.lambda3 * sep_loss))
            divers_loss_per_batch.append(float(args.lambda5 * divers_loss))
            l1_loss_per_batch.append(float(args.lambda5 * l1_loss))

        mean_loss = np.mean(losses_per_batch)
        ce_mean_loss = np.mean(ce_loss_per_batch)
        distr_mean_loss = np.mean(distr_loss_per_batch)
        clust_mean_loss = np.mean(clust_loss_per_batch)
        sep_mean_loss = np.mean(sep_loss_per_batch)
        divers_mean_loss = np.mean(divers_loss_per_batch)
        l1_mean_loss = np.mean(l1_loss_per_batch)
        acc = balanced_accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}, losses: mean {mean_loss:.3f}, ce {ce_mean_loss:.3f}, distr {distr_mean_loss:.3f}, "
              f"clust {clust_mean_loss:.3f}, sep {sep_mean_loss:.3f}, divers {divers_mean_loss:.3f}, "
              f"l1 {l1_mean_loss:.3f}, train acc {100 * acc:.3f}")

        if (epoch + 1) % args.val_epoch == 0 or epoch + 1 == num_epochs:
            model.eval()
            all_preds = []
            all_labels = []
            losses_per_batch = []
            with torch.no_grad():
                for emb_batch, label_batch in val_batches:
                    emb_batch = emb_batch.to(f'cuda:{args.gpu[0]}')
                    label_batch = label_batch.to(f'cuda:{args.gpu[0]}')
                    prototype_distances, _, predicted_label = model.forward(emb_batch)

                    # compute individual losses and backward step
                    ce_loss = ce_crit(predicted_label, label_batch)
                    distr_loss, clust_loss, sep_loss, divers_loss, l1_loss = \
                        proto_loss(prototype_distances, label_batch, model, args)
                    loss = ce_loss + \
                           args.lambda1 * distr_loss + \
                           args.lambda2 * clust_loss + \
                           args.lambda3 * sep_loss + \
                           args.lambda4 * divers_loss + \
                           args.lambda5 * l1_loss

                    losses_per_batch.append(float(loss))
                    _, predicted = torch.max(predicted_label.data, 1)
                    all_preds += predicted.cpu().numpy().tolist()
                    all_labels += label_batch.cpu().numpy().tolist()

                loss_val = np.mean(losses_per_batch)
                acc_val = balanced_accuracy_score(all_labels, all_preds)
                print(f"Validation: mean loss {loss_val:.3f}, acc_val {100 * acc_val:.3f}")

            if acc_val > best_acc:
                best_acc = acc_val
                state = { 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'hyper_params': args,
                    'acc_val': acc_val,
                }
    torch.save(state, args.model_path)


def test(args, embedding_train, train_batches, test_batches, labels_train, text_train, model):
    print("\nStart evaluation, loading model:", args.model_path)
    model.eval()
    ce_crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(args.class_weights).float().to(f'cuda:{args.gpu[0]}'))

    all_preds = []
    all_labels = []
    losses_per_batch = []

    with torch.no_grad():
        for emb_batch, label_batch in test_batches:
            emb_batch = emb_batch.to(f'cuda:{args.gpu[0]}')
            label_batch = label_batch.to(f'cuda:{args.gpu[0]}')
            prototype_distances, _, predicted_label = model.forward(emb_batch)

            # compute individual losses
            ce_loss = ce_crit(predicted_label, label_batch)
            distr_loss, clust_loss, sep_loss, divers_loss, l1_loss = \
                proto_loss(prototype_distances, label_batch, model, args)
            loss = ce_loss + \
                   args.lambda1 * distr_loss + \
                   args.lambda2 * clust_loss + \
                   args.lambda3 * sep_loss + \
                   args.lambda4 * divers_loss + \
                   args.lambda5 * l1_loss

            losses_per_batch.append(float(loss))
            _, predicted = torch.max(predicted_label.data, 1)
            all_preds += predicted.cpu().numpy().tolist()
            all_labels += label_batch.cpu().numpy().tolist()

        loss = np.mean(losses_per_batch)
        acc_test = balanced_accuracy_score(all_labels, all_preds)
        print(f"test evaluation on best model: loss {loss:.3f}, acc_test {100 * acc_test:.3f}")

        # "convert" prototype embedding to text (take text of nearest training sample)
        proto_ids, proto_texts = get_nearest(args, model, train_batches, text_train, labels_train)
        weights = model.get_proto_weights()

        if os.path.basename(args.model_path).startswith('finetuned'):
            fname = "finetuned_prototypes.txt"
        elif os.path.basename(args.model_path).startswith('pruned'):
            fname = "pruned_prototypes.txt"
            _, new_proto_texts, mask = prune_prototypes(proto_texts, model, args)
            for m in mask: proto_texts[m] = new_proto_texts[m]
        else:
            fname = "prototypes.txt"
        proto_texts = [id + txt for id, txt in zip(proto_ids, proto_texts)]
        save_path = os.path.join(os.path.dirname(args.model_path), fname)
        txt_file = open(save_path, "w+")
        for arg in vars(args):
            txt_file.write(f"{arg}: {vars(args)[arg]}\n")
        txt_file.write(f"test loss: {loss:.3f}\n")
        txt_file.write(f"test acc: {100*acc_test:.2f}\n")
        for line in proto_texts:
            txt_file.write(line + "\n")
        for line in weights:
            txt_file.write(str(line) + "\n")
        txt_file.close()

        # plot prototypes
        prototypes = model.get_protos().cpu().numpy()
        if len(embedding_train.shape) == 2:
            visualize_protos(embedding_train.cpu().numpy(), labels_train, prototypes, n_components=2, trans_type=args.trans_type, save_path=os.path.dirname(save_path))
            # visualize_protos(embedding_train.cpu().numpy(), labels_train, prototypes, n_components=3, trans_type=args.trans_type, save_path=os.path.dirname(save_path))


def query(args, train_batches, labels_train, text_train, model):
    print("\nEvaluate query, loading model:", args.model_path)
    model.eval()

    embedding_query = model.compute_embedding(args.query, args)
    torch.cuda.empty_cache()  # is required since BERT encoding is only possible on 1 GPU (memory limitation)

    # "convert" prototype embedding to text (take text of nearest training sample)
    proto_ids, proto_texts = get_nearest(args, model, train_batches, text_train, labels_train)
    proto_texts = [id+txt for id, txt in zip(proto_ids,proto_texts)]
    distances, _, predicted_label = model.forward(embedding_query.to(f'cuda:{args.gpu[0]}'))
    predicted = torch.argmax(predicted_label).cpu()

    query2proto = torch.argmin(distances)
    nearest_proto = proto_texts[query2proto]
    weights = model.get_proto_weights()

    save_path = os.path.join(os.path.dirname(args.model_path), "query.txt")
    txt_file = open(save_path, "w+")
    txt_file.write(''.join(args.query) + "\n")
    txt_file.write(f"nearest prototype id: {query2proto+1}\n")
    txt_file.write(f"weights: {weights[query2proto]}\n")
    txt_file.write(nearest_proto + "\n")
    txt_file.write(f"predicted: {predicted}\n")
    for line in weights:
        txt_file.write(f"{line} \n")
    for arg in vars(args):
        txt_file.write(f"{arg}: {vars(args)[arg]}\n")
    txt_file.close()


def finetune(args, train_batches, val_batches, embedding_train, test_batches, labels_train, text_train, model):
    print("\nRetrain, loading model:", args.model_path)

    # protos2keep = list(map(int,input("Select prototypes to keep: ").split()))
    protos2keep = [0,1,2,3,7,9]
    if not protos2keep:
        protos2keep = list(range(args.num_prototypes))

    # # if mode == 'weights':
    # # set hook to ignore weights of prototypes to keep when computing gradient, to learn the other weights
    # gradient_mask_fc = torch.ones(model.fc.weight.size()).to(f'cuda:{args.gpu[0]}')
    # gradient_mask_fc[:,protos2keep] = 0
    # model.fc.weight.register_hook((lambda grad: grad.mul_(gradient_mask_fc)))
    # if mode == 'prototypes':
    # also, do not update the selected prototypes that should be kept
    gradient_mask_proto = torch.ones(model.protolayer.size()).to(f'cuda:{args.gpu[0]}')
    gradient_mask_proto[protos2keep, :, :] = 0
    model.protolayer.register_hook(lambda grad: grad.mul_(gradient_mask_proto))
    args.model_path = os.path.join(os.path.dirname(args.model_path), 'finetuned_best_model.pth.tar')
    train(args, train_batches, val_batches, model)
    test(args, embedding_train, train_batches, test_batches, labels_train, text_train, model)


def prune(args, train_batches, val_batches, embedding_train, test_batches, labels_train, text_train, model):
    print("\nPrune, loading model:", args.model_path)
    _, proto_texts = get_nearest(args, model, train_batches, text_train, labels_train)

    # prune prototypes by removing stop words
    new_prototypes, _, mask = prune_prototypes(proto_texts, model, args)
    # assign new prototypes and don't update them when retraining
    model.protolayer.data[mask] = new_prototypes[mask]
    model.protolayer.requires_grad = False

    # update model with new parameters (prototypes)
    args.model_path = os.path.join(os.path.dirname(args.model_path), 'pruned_best_model.pth.tar')
    state = { 'state_dict': model.state_dict(), 'hyper_params': args}
    torch.save(state, args.model_path)
    # retrain network, only retrain last layer (fc)
    train(args, train_batches, val_batches, model)
    test(args, embedding_train, train_batches, test_batches, labels_train, text_train, model)


# def adjust_prototypes():
#     print("\nAdjust, loading model:", args.model_path)
#     # remove prototypes
#     protos2remove = [1,5,6]
#     model.fc.weight.data[:, protos2remove] = 0
#     gradient_mask_fc = torch.ones(model.fc.weight.size()).to(f'cuda:{args.gpu[0]}')
#     gradient_mask_fc[:, protos2remove] = 0
#     model.fc.weight.register_hook((lambda grad: grad.mul_(gradient_mask_fc)))
#     gradient_mask_proto = torch.ones(model.protolayer.size()).to(f'cuda:{args.gpu[0]}')
#     gradient_mask_proto[protos2remove, :, :] = 0
#     model.protolayer.register_hook(lambda grad: grad.mul_(gradient_mask_proto))
#     # add prototypes


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_num_threads(6)
    args = parser.parse_args()

    # Create RTPT object
    rtpt = RTPT(name_initials='FelFri', experiment_name='TransfProto', max_iterations=args.num_epochs)
    # Start the RTPT tracking
    rtpt.start()

    text, labels = load_data(args)
    # split data, and split test set again to get validation and test set
    text_train, text_test, labels_train, labels_test = train_test_split(text, labels, test_size=0.3, stratify=labels)
    text_val, text_test, labels_val, labels_test = train_test_split(text_test, labels_test, test_size=0.5,
                                                                         stratify=labels_test)

    # set class weights for balanced cross entropy computation
    balance = labels.count(0) / len(labels)
    args.class_weights = [1-balance, balance]

    # define which prototype belongs to which class (onehot encoded matrix)
    args.prototype_class_identity = torch.zeros(args.num_prototypes, args.num_classes).to(f'cuda:{args.gpu[0]}')
    args.prototype_class_identity[::2,0] = 1
    args.prototype_class_identity[1::2,1] = 1
    # protos_per_class = round(balance*args.num_prototypes)
    # args.prototype_class_identity[:protos_per_class, 0] = 1
    # args.prototype_class_identity[protos_per_class:, 1] = 1
    args.class_specific = True
    args.use_l1_mask = True

    if args.one_shot:
        idx = random.sample(range(len(text_train)), 100)
        text_train = list(text_train[i] for i in idx)
        args.compute_emb = True

    model = []
    if args.level == 'word':
        model = ProtoPNetConv(args)
    elif args.level == 'sentence':
        model = ProtoNet(args)

    print(f"Running on gpu {args.gpu}")
    model.to(f'cuda:{args.gpu[0]}')

    avoid = ''
    if args.avoid_spec_token:
        avoid = '_avoid'

    fname = args.language_model + args.level
    if not args.compute_emb:
        embedding_train = load_embedding(args, fname, 'train' + avoid)
        embedding_val = load_embedding(args, fname, 'val' + avoid)
        embedding_test = load_embedding(args, fname, 'test' + avoid)
    else:
        embedding_train = model.compute_embedding(text_train, args)
        embedding_val = model.compute_embedding(text_val, args)
        embedding_test = model.compute_embedding(text_test, args)
        save_embedding(embedding_train, args, fname, 'train' + avoid)
        save_embedding(embedding_val, args, fname, 'val' + avoid)
        save_embedding(embedding_test, args, fname, 'test' + avoid)
        torch.cuda.empty_cache()  # is required since BERT encoding is only possible on 1 GPU (memory limitation)

    train_batches = torch.utils.data.DataLoader(list(zip(embedding_train, labels_train)), batch_size=args.batch_size,
                                                shuffle=False, pin_memory=True, num_workers=0)#, drop_last=True)
    val_batches = torch.utils.data.DataLoader(list(zip(embedding_val, labels_val)), batch_size=args.batch_size,
                                              shuffle=False, pin_memory=True, num_workers=0)#, drop_last=True)
    test_batches = torch.utils.data.DataLoader(list(zip(embedding_test, labels_test)), batch_size=args.batch_size,
                                               shuffle=False, pin_memory=True, num_workers=0)#, drop_last=True)

    time_stmp = datetime.datetime.now().strftime(f"%d-%m %H:%M_{args.num_prototypes}{fname}{args.proto_size}")
    args.model_path = os.path.join("./experiments/train_results/", time_stmp, 'best_model.pth.tar')

    if 'train' in args.mode:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        train(args, train_batches, val_batches, model)
    if not os.path.exists(args.model_path):
        load_path = "./experiments/train_results/*"
        model_paths = glob.glob(os.path.join(load_path, 'best_model.pth.tar'))
        model_paths.sort()
        args.model_path = model_paths[-1]
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    if 'test' in args.mode:
        test(args, embedding_train, train_batches, test_batches, labels_train, text_train, model)
    if 'query' in args.mode:
        query(args, train_batches, labels_train, text_train, model)
    if 'finetune' in args.mode:
        finetune(args, train_batches, val_batches, embedding_train, test_batches, labels_train, text_train, model)
    if 'prune' in args.mode:
        prune(args, train_batches, val_batches, embedding_train, test_batches, labels_train, text_train, model)
