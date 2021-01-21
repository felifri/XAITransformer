import argparse
import datetime
import glob
import os
import random

import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
# from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from rtpt import RTPT

from models import ProtoPNetConv, ProtoPNetDist, ProtoNet
from utils import save_embedding, load_embedding, save_checkpoint, load_data, visualize_protos

parser = argparse.ArgumentParser(description='Crazy Stuff')
parser.add_argument('-m', '--mode', default="all", type=str, choices=['train','test','query','all'],
                    help='What do you want to do? Select either only train, test or both')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Select learning rate')
parser.add_argument('-e', '--num_epochs', default=200, type=int,
                    help='How many epochs?')
parser.add_argument('-bs', '--batch_size', default=650, type=int,
                    help='Select batch size')
parser.add_argument('--val_epoch', default=10, type=int,
                    help='After how many epochs should the model be evaluated on the validation data?')
parser.add_argument('--data_dir', default='./data/rt-polarity',
                    help='Select data path')
parser.add_argument('--data_name', default='rt-polarity', type=str, choices=['rt-polarity', 'toxicity', 'toxicity_full'],
                    help='Select name of data set')
parser.add_argument('--num_prototypes', default=10, type=int,
                    help='Total number of prototypes')
parser.add_argument('-l1','--lambda1', default=0.1, type=float,
                    help='Weight for prototype loss computation')
parser.add_argument('-l2','--lambda2', default=0.1, type=float,
                    help='Weight for prototype loss computation')
parser.add_argument('-l3','--lambda3', default=0.5, type=float,
                    help='Weight for padding loss computation')
parser.add_argument('--num_classes', default=2, type=int,
                    help='How many classes are to be classified?')
parser.add_argument('--class_weights', default=[0.5,0.5],
                    help='Class weight for cross entropy loss')
parser.add_argument('-g','--gpu', type=int, default=[0], nargs='+',
                    help='GPU device number(s)')
parser.add_argument('--one_shot', type=bool, default=False,
                    help='Whether to use one-shot learning or not (i.e. only a few training examples)')
parser.add_argument('--trans_type', type=str, default='PCA', choices=['PCA', 'TSNE'],
                    help='Which transformation should be used to visualize the prototypes')
parser.add_argument('--discard', type=bool, default=False,
                    help='Whether edge cases in the middle between completely toxic(1) and not toxic(0) shall be omitted')
parser.add_argument('--proto_size', type=int, default=4,
                    help='Define how many words should be used to define a prototype')
parser.add_argument('--modeltype', type=str, default='dist', choices=['conv','dist'],
                    help='Define which similarity computation to use')
parser.add_argument('--language_model', type=str, default='Bert', choices=['Bert','SentBert','GPT2'],
                    help='Define which language model to use')
parser.add_argument('--dilated', type=int, default=[1], nargs='+',
                    help='Whether to use dilation in the convolution ProtoP and if with which step size')
parser.add_argument('--avoid_spec_token', type=bool, default=False,
                    help='Whether to manually set PAD, SEP and CLS token to high value after Bert embedding computation')
parser.add_argument('--compute_emb', type=bool, default=False,
                    help='Whether to recompute (True) the embedding or just load it (False)')
parser.add_argument('--query', type=str, default='this is a very interesting movie', nargs='+',
                    help='Type your query to test the model and get classification explanation')

def train(args, embedding_train, labels_train, embedding_val, labels_val, model):
    save_dir = "./experiments/train_results/"
    time_stmp = datetime.datetime.now().strftime(f"%m-%d %H:%M_{args.num_prototypes}{args.modeltype}{args.proto_size}")

    num_epochs = args.num_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_epochs // 10, num_epochs)
    ce_crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(args.class_weights).float().to(f'cuda:{args.gpu[0]}'))

    model.train()
    print(f"\nStarting training for {num_epochs} epochs\n")
    best_acc = 0
    train_batches = torch.utils.data.DataLoader(list(zip(embedding_train, labels_train)), batch_size=args.batch_size,
                                                shuffle=True, pin_memory=True, num_workers=0)#, drop_last=True)
    val_batches = torch.utils.data.DataLoader(list(zip(embedding_val, labels_val)), batch_size=args.batch_size,
                                              shuffle=False, pin_memory=True, num_workers=0)#, drop_last=True)
    for epoch in tqdm(range(num_epochs)):
        all_preds = []
        all_labels = []
        losses_per_batch = []
        ce_loss_per_batch = []
        r1_loss_per_batch = []
        r2_loss_per_batch = []
        p1_loss_per_batch = []

        # Update the RTPT
        rtpt.step()

        for emb_batch, label_batch in train_batches:
            emb_batch = emb_batch.to(f'cuda:{args.gpu[0]}')
            label_batch = label_batch.to(f'cuda:{args.gpu[0]}')

            optimizer.zero_grad()
            prototype_distances, _, predicted_label = model.forward(emb_batch)

            # compute individual losses and backward step
            ce_loss = ce_crit(predicted_label, label_batch)
            r1_loss, r2_loss, p1_loss = model.module.proto_loss(prototype_distances)
            loss = ce_loss + \
                   args.lambda1 * r1_loss + \
                   args.lambda2 * r2_loss + \
                   args.lambda3 * p1_loss

            _, predicted = torch.max(predicted_label.data, 1)
            all_preds += predicted.cpu().numpy().tolist()
            all_labels += label_batch.cpu().numpy().tolist()

            loss.backward()
            optimizer.step()
            # store losses
            losses_per_batch.append(float(loss))
            ce_loss_per_batch.append(float(ce_loss))
            r1_loss_per_batch.append(float(r1_loss))
            r2_loss_per_batch.append(float(r2_loss))
            p1_loss_per_batch.append(float(p1_loss))

        # scheduler.step()
        mean_loss = np.mean(losses_per_batch)
        ce_mean_loss = np.mean(ce_loss_per_batch)
        r1_mean_loss = np.mean(r1_loss_per_batch)
        r2_mean_loss = np.mean(r2_loss_per_batch)
        p1_mean_loss = np.mean(p1_loss_per_batch)
        acc = balanced_accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}, losses: mean {mean_loss:.4f}, ce {ce_mean_loss:.4f}, r1 {r1_mean_loss:.4f}, "
              f"r2 {r2_mean_loss:.4f}, p1 {p1_mean_loss:.4f}, train acc {100 * acc:.4f}")

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
                    r1_loss, r2_loss, p1_loss = model.module.proto_loss(prototype_distances)
                    loss = ce_loss + \
                           args.lambda1 * r1_loss + \
                           args.lambda2 * r2_loss + \
                           args.lambda3 * p1_loss

                    losses_per_batch.append(float(loss))
                    _, predicted = torch.max(predicted_label.data, 1)
                    all_preds += predicted.cpu().numpy().tolist()
                    all_labels += label_batch.cpu().numpy().tolist()

                loss_val = np.mean(losses_per_batch)
                acc_val = balanced_accuracy_score(all_labels, all_preds)
                print(f"Validation: mean loss {loss_val:.4f}, acc_val {100 * acc_val:.4f}")

            save_checkpoint(save_dir, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'hyper_params': args,
                'acc_val': acc_val,
            }, time_stmp, best=acc_val >= best_acc)
            if acc_val >= best_acc:
                best_acc = acc_val
    return os.path.join(save_dir, time_stmp, 'best_model.pth.tar')


def test(args, embedding_train, labels_train, embedding_test, labels_test, text_train, model, model_path):
    if not model_path:
        load_path = "./experiments/train_results/*"
        model_paths = glob.glob(os.path.join(load_path, 'best_model.pth.tar'))
        model_paths.sort()
        model_path = model_paths[-1]
    print("\nStarting evaluation, loading model:", model_path)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    ce_crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(args.class_weights).float().to(f'cuda:{args.gpu[0]}'))

    all_preds = []
    all_labels = []
    losses_per_batch = []
    test_batches = torch.utils.data.DataLoader(list(zip(embedding_test, labels_test)), batch_size=args.batch_size,
                                               shuffle=False, pin_memory=True, num_workers=0)#, drop_last=True)
    with torch.no_grad():
        for emb_batch, label_batch in test_batches:
            emb_batch = emb_batch.to(f'cuda:{args.gpu[0]}')
            label_batch = label_batch.to(f'cuda:{args.gpu[0]}')
            prototype_distances, _, predicted_label = model.forward(emb_batch)

            # compute individual losses and backward step
            ce_loss = ce_crit(predicted_label, label_batch)
            r1_loss, r2_loss, p1_loss = model.module.proto_loss(prototype_distances)
            loss = ce_loss + \
                   args.lambda1 * r1_loss + \
                   args.lambda2 * r2_loss + \
                   args.lambda3 * p1_loss

            losses_per_batch.append(float(loss))
            _, predicted = torch.max(predicted_label.data, 1)
            all_preds += predicted.cpu().numpy().tolist()
            all_labels += label_batch.cpu().numpy().tolist()

        loss = np.mean(losses_per_batch)
        acc_test = balanced_accuracy_score(all_labels, all_preds)
        print(f"test evaluation on best model: loss {loss:.4f}, acc_test {100 * acc_test:.4f}")

        train_batches = torch.utils.data.DataLoader(embedding_train, batch_size=args.batch_size,
                                              shuffle=False, pin_memory=False, num_workers=0)#, drop_last=True)
        dist = []
        for batch in train_batches:
            batch = batch.to(f'cuda:{args.gpu[0]}')
            _, distances, _ = model.forward(batch)
            dist.append(distances)

        # "convert" prototype embedding to text (take text of nearest training sample)
        proto_texts = model.module.nearest_neighbors(dist, text_train, labels_train)

        weights = model.module.get_proto_weights()
        save_path = os.path.join(os.path.dirname(model_path), "prototypes.txt")
        txt_file = open(save_path, "w+")
        for arg in vars(args):
            txt_file.write(f"{arg}: {vars(args)[arg]}\n")
        txt_file.write(f"test loss: {loss:.4f}\n")
        txt_file.write(f"test acc: {100*acc_test:.2f}\n")
        for line in proto_texts:
            txt_file.write(line + "\n")
        for line in weights:
            txt_file.write(str(line) + "\n")
        txt_file.close()

        # get prototypes
        prototypes = model.module.get_protos().cpu().numpy()
        embedding_train = embedding_train.cpu().numpy()
        if len(embedding_train.shape) == 2:
            visualize_protos(embedding_train, labels_train, prototypes, n_components=2, trans_type=args.trans_type, save_path=os.path.dirname(save_path))
            # visualize_protos(embedding_train, labels_train, prototypes, n_components=3, trans_type=args.trans_type, save_path=os.path.dirname(save_path))


def query(args, embedding_train, labels_train, text_train, model, model_path):
    if not model_path:
        load_path = "./experiments/train_results/*"
        model_paths = glob.glob(os.path.join(load_path, 'best_model.pth.tar'))
        model_paths.sort()
        model_path = model_paths[-1]
    print("\nEvaluate query, loading model:", model_path)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    embedding_query = model.module.compute_embedding(args.query, args)
    torch.cuda.empty_cache()  # is required since BERT encoding is only possible on 1 GPU (memory limitation)
    train_batches = torch.utils.data.DataLoader(embedding_train, batch_size=args.batch_size,
                                                shuffle=False, pin_memory=False, num_workers=0)  # , drop_last=True)
    dist = []
    with torch.no_grad():
        for batch in train_batches:
            batch = batch.to(f'cuda:{args.gpu[0]}')
            _, distances, _ = model.forward(batch)
            dist.append(distances)

    # "convert" prototype embedding to text (take text of nearest training sample)
    proto_texts = model.module.nearest_neighbors(dist, text_train, labels_train)
    distances, _, predicted_label = model.forward(embedding_query.to(f'cuda:{args.gpu[0]}'))

    predicted = torch.argmax(predicted_label).cpu()

    query2proto = torch.argmin(distances)
    nearest_proto = proto_texts[query2proto]
    weights = model.module.get_proto_weights()

    save_path = os.path.join(os.path.dirname(model_path), "query.txt")
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


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_num_threads(8)
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

    if args.one_shot:
        idx = random.sample(range(len(text_train)), 100)
        text_train = list(text_train[i] for i in idx)

    model = []
    fname = args.language_model
    if fname == 'Bert' or fname == 'GPT2':
        if args.modeltype == 'dist':
            model = ProtoPNetDist(args)
        elif args.modeltype == 'conv':
            model = ProtoPNetConv(args)
    elif fname == 'SentBert':
        model = ProtoNet(args)

    print(f"Running on gpu {args.gpu}")
    model = torch.nn.DataParallel(model, device_ids=args.gpu)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=args.gpu)
    model.to(f'cuda:{args.gpu[0]}')

    avoid = ''
    if args.avoid_spec_token:
        avoid = '_avoid'

    if not args.compute_emb:
        embedding_train = load_embedding(args, fname, 'train' + avoid)
        embedding_val = load_embedding(args, fname, 'val' + avoid)
        embedding_test = load_embedding(args, fname, 'test' + avoid)
    else:
        embedding_train = model.module.compute_embedding(text_train, args)
        embedding_val = model.module.compute_embedding(text_val, args)
        embedding_test = model.module.compute_embedding(text_test, args)
        save_embedding(embedding_train, args, fname, 'train' + avoid)
        save_embedding(embedding_val, args, fname, 'val' + avoid)
        save_embedding(embedding_test, args, fname, 'test' + avoid)
        torch.cuda.empty_cache()  # is required since BERT encoding is only possible on 1 GPU (memory limitation)

    if args.mode == 'all':
        model_path = train(args, embedding_train, labels_train, embedding_val, labels_val, model)
        test(args, embedding_train, labels_train, embedding_test, labels_test, text_train, model, model_path)
        query(args, embedding_train, labels_train, text_train, model, model_path)
    elif args.mode == 'train':
        model_path = train(args, embedding_train, labels_train, embedding_val, labels_val, model)
    elif args.mode == 'test':
        test(args, embedding_train, labels_train, embedding_test, labels_test, text_train, model, [])
    elif args.mode == 'query':
        query(args, embedding_train, labels_train, text_train, model, [])
