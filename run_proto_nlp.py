import argparse
import datetime
import glob
import sys
import os
import random

import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    from rtpt.rtpt import RTPT
except:
    sys.path.append('../rtpt')
    from rtpt import RTPT

from models import ProtoPNetConv, ProtoPNetDist, ProtoNet
from utils import save_embedding, load_embedding, save_checkpoint, load_data, visualize_protos

# Create RTPT object
rtpt = RTPT(name_initials='FelFri', experiment_name='TransfProto', max_iterations=200)
# Start the RTPT tracking
rtpt.start()

parser = argparse.ArgumentParser(description='Crazy Stuff')
parser.add_argument('-m', '--mode', default="both", type=str, choices=['train','test','both'],
                    help='What do you want to do? Select either only train, test or both')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Select learning rate')
parser.add_argument('-e', '--num_epochs', default=200, type=int,
                    help='How many epochs?')
parser.add_argument('-bs', '--batch_size', default=256, type=int,
                    help='Select batch size')
parser.add_argument('--val_epoch', default=10, type=int,
                    help='After how many epochs should the model be evaluated on the validation data?')
parser.add_argument('--data_dir', default='./data/rt-polarity',
                    help='Select data path')
parser.add_argument('--data_name', default='rt-polarity', type=str, choices=['rt-polarity', 'toxicity'],
                    help='Select data name')
parser.add_argument('--num_prototypes', default=10, type=int,
                    help='Total number of prototypes')
parser.add_argument('-l1','--lambda1', default=0.1, type=float,
                    help='Weight for prototype loss computation')
parser.add_argument('-l2','--lambda2', default=0.1, type=float,
                    help='Weight for prototype loss computation')
parser.add_argument('-l3','--lambda3', default=1, type=float,
                    help='Weight for padding loss computation')
parser.add_argument('--num_classes', default=2, type=int,
                    help='How many classes are to be classified?')
parser.add_argument('--class_weights', default=[0.5,0.5],
                    help='Class weight for cross entropy loss')
parser.add_argument('-g','--gpu', type=int, default=0, nargs='+',
                    help='GPU device number(s)')
parser.add_argument('--one_shot', type=bool, default=False,
                    help='Whether to use one-shot learning or not (i.e. only a few training examples)')
parser.add_argument('--trans_type', type=str, default='PCA', choices=['PCA', 'TSNE'],
                    help='Which transformation should be used to visualize the prototypes')
parser.add_argument('--discard', type=bool, default=False,
                    help='Whether edge cases in the middle between completely toxic(1) and not toxic(0) shall be omitted')
parser.add_argument('--proto_size', type=int, default=4,
                    help='Define how many words should be used to define a prototype')
parser.add_argument('--model', type=str, default='dist', choices=['p_conv','p_dist','dist'],
                    help='Define which model to use')
parser.add_argument('--dilated', type=int, default=[3,4,5,6], nargs='+',
                    help='Whether to use dilation in the convolution ProtoP and if with which step size')

def train(args, text_train, labels_train, text_val, labels_val):
    save_dir = "./experiments/train_results/"
    time_stmp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    model = []
    if args.model == 'p_dist':
        model = ProtoPNetDist(args)
    elif args.model == 'p_conv':
        model = ProtoPNetConv(args)
    elif args.model == 'dist':
        model = ProtoNet(args)

    print("Running on gpu {}".format(args.gpu))
    model = torch.nn.DataParallel(model, device_ids=args.gpu)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=args.gpu)
    model.to(f'cuda:{args.gpu[0]}')

    fname = 'Bert' if args.model.startswith('p') else 'SentBert'
    try:
        embedding_train = load_embedding(args, fname, 'train')
        embedding_val = load_embedding(args, fname, 'val')
    except:
        embedding_train = model.module.compute_embedding(text_train, args.gpu[0])
        embedding_val = model.module.compute_embedding(text_val, args.gpu[0])
        save_embedding(embedding_train, args, fname, 'train')
        save_embedding(embedding_val, args, fname, 'val')
        torch.cuda.empty_cache()  # is required since BERT encoding is only possible on 1 GPU (memory limitation)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce_crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(args.class_weights).float().cuda(args.gpu[0]))

    model.train()
    num_epochs = args.num_epochs
    print("\nStarting training for {} epochs\n".format(num_epochs))
    best_acc = 0
    for epoch in tqdm(range(num_epochs)):
        all_preds = []
        all_labels = []
        losses_per_batch = []
        ce_loss_per_batch = []
        r1_loss_per_batch = []
        r2_loss_per_batch = []
        p1_loss_per_batch = []
        train_batches = torch.utils.data.DataLoader(list(zip(embedding_train, labels_train)), batch_size=args.batch_size,
                                                  shuffle=True)#, drop_last=True, num_workers=len(args.gpu))
        # Update the RTPT
        rtpt.step(subtitle=f"epoch={epoch+1}")

        for emb_batch, label_batch in train_batches:
            emb_batch = emb_batch.to(f'cuda:{args.gpu[0]}')
            label_batch = torch.LongTensor(label_batch).to(f'cuda:{args.gpu[0]}')

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

        mean_loss = np.mean(losses_per_batch)
        ce_mean_loss = np.mean(ce_loss_per_batch)
        r1_mean_loss = np.mean(r1_loss_per_batch)
        r2_mean_loss = np.mean(r2_loss_per_batch)
        p1_mean_loss = np.mean(p1_loss_per_batch)
        acc = balanced_accuracy_score(all_labels, all_preds)
        print("Epoch {}, losses: mean {:.4f}, ce {:.4f}, r1 {:.4f}, "
              "r2 {:.4f}, p1 {:.4f}, train acc {:.4f}".format(epoch+1,
                                                        mean_loss,
                                                        ce_mean_loss,
                                                        r1_mean_loss,
                                                        r2_mean_loss,
                                                        p1_mean_loss,
                                                        100 * acc))

        if (epoch + 1) % args.val_epoch == 0 or epoch + 1 == num_epochs:
            model.eval()
            all_preds = []
            all_labels = []
            losses_per_batch = []
            val_batches = torch.utils.data.DataLoader(list(zip(embedding_val, labels_val)), batch_size=args.batch_size,
                                                      shuffle=False)#, drop_last=True, num_workers=len(args.gpu))
            with torch.no_grad():
                for emb_batch, label_batch in val_batches:
                    emb_batch = emb_batch.to(f'cuda:{args.gpu[0]}')
                    label_batch = torch.LongTensor(label_batch).to(f'cuda:{args.gpu[0]}')
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
                print("Validation: mean loss {:.4f}, acc_val {:.4f}".format(loss_val, 100 * acc_val))

            save_checkpoint(save_dir, {
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'hyper_params': args,
                'acc_val': acc_val,
            }, time_stmp, best=acc_val >= best_acc)
            if acc_val >= best_acc:
                best_acc = acc_val


def test(args, text_train, labels_train, text_test, labels_test):
    load_path = "./experiments/train_results/*"
    model_paths = glob.glob(os.path.join(load_path, 'best_model.pth.tar'))
    model_paths.sort()
    model_path = model_paths[-1]
    print("\nStarting evaluation, loading model:", model_path)

    model = []
    if args.model=='p_dist':
        model = ProtoPNetDist(args)
    elif args.model=='p_conv':
        model = ProtoPNetConv(args)
    elif args.model=='dist':
        model = ProtoNet(args)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(f'cuda:{args.gpu[0]}')
    model.eval()
    ce_crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(args.class_weights).float().to(f'cuda:{args.gpu[0]}'))

    fname = 'Bert' if args.model.startswith('p') else 'SentBert'
    try:
        embedding_train = load_embedding(args, fname, 'train')
        embedding_test = load_embedding(args, fname, 'test')
    except:
        embedding_train = model.compute_embedding(text_train, args.gpu[0])
        embedding_test = model.compute_embedding(text_test, args.gpu[0])
        save_embedding(embedding_train, args, fname, 'train')
        save_embedding(embedding_test, args, fname, 'test')
        torch.cuda.empty_cache()  # is required since BERT encoding is only possible on 1 GPU (memory limitation)

    all_preds = []
    all_labels = []
    losses_per_batch = []
    test_batches = torch.utils.data.DataLoader(list(zip(embedding_test, labels_test)), batch_size=args.batch_size,
                                               shuffle=False)#, num_workers=len(args.gpu))
    with torch.no_grad():
        for emb_batch, label_batch in test_batches:
            emb_batch = emb_batch.to(f'cuda:{args.gpu[0]}')
            label_batch = torch.LongTensor(label_batch).to(f'cuda:{args.gpu[0]}')
            prototype_distances, _, predicted_label = model.forward(emb_batch)

            # compute individual losses and backward step
            ce_loss = ce_crit(predicted_label, label_batch)
            r1_loss, r2_loss, p1_loss = model.proto_loss(prototype_distances)
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
                                              shuffle=False)#, num_workers=len(args.gpu))
        dist=[]
        for batch in train_batches:
            batch = batch.to(f'cuda:{args.gpu[0]}')
            _, distances, _ = model.forward(batch)
            dist.append(distances)
        dist = torch.cat(dist)

        # "convert" prototype embedding to text (take text of nearest training sample)
        proto_texts = model.nearest_neighbors(dist, text_train, model)

        weights = model.get_proto_weights()
        save_path = os.path.join(os.path.dirname(model_path), "prototypes.txt")
        #os.makedirs(os.path.dirname(save_path), exist_ok=True)
        txt_file = open(save_path, "w+")
        for line in proto_texts:
            txt_file.write(str(line))
            txt_file.write("\n")
        for line in weights:
            txt_file.write(str(line))
            txt_file.write("\n")
        txt_file.close()

        # get prototypes
        prototypes = model.get_protos().cpu().numpy()
        embedding_train = embedding_train.cpu().numpy()
        visualize_protos(embedding_train, labels_train, prototypes, n_components=2, trans_type=args.trans_type, save_path=os.path.dirname(save_path))
        # visualize_protos(embedding_train, labels_train, prototypes, n_components=3, trans_type=args.trans_type, save_path=os.path.dirname(save_path))


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    args = parser.parse_args()

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

    if args.mode == 'both':
        train(args, text_train, labels_train, text_val, labels_val)
        test(args, text_train, labels_train, text_test, labels_test)
    elif args.mode == 'train':
        train(args, text_train, labels_train, text_val, labels_val)
    elif args.mode == 'test':
        test(args, text_train, labels_train, text_test, labels_test)
