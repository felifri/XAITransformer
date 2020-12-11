import argparse
import sys
import random

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from itertools import product

try:
    from rtpt.rtpt import RTPT
except:
    sys.path.append('../rtpt')
    from rtpt import RTPT

from models import ProtoNetNLP
import utils

# Create RTPT object
rtpt = RTPT(name_initials='FF', experiment_name='Transformer_Prototype', max_iterations=100)
# Start the RTPT tracking
rtpt.start()

parser = argparse.ArgumentParser(description='Crazy Stuff')
parser.add_argument('--lr', type=float, default=[0.01,0.001],
                    help='Learning rate')
parser.add_argument('--cpu', action='store_true', default=False,
                    help='Whether to use cpu')
parser.add_argument('-e', '--num_epochs', default=[100], type=int,
                    help='How many epochs?')
parser.add_argument('-bs', '--batch_size', default=[256], type=int,
                    help='Batch size')
parser.add_argument('--val_epoch', default=10, type=int,
                    help='After how many epochs should the model be evaluated on the validation data?')
parser.add_argument('--data_dir', default='./data/rt-polarity',
                    help='Select data path')
parser.add_argument('--data_name', default='reviews', type=str, choices=['reviews', 'toxicity'],
                    help='Select data name')
parser.add_argument('--num_prototypes', default=[2,4,10], type = int,
                    help='Total number of prototypes')
parser.add_argument('-l2','--lambda2', default=[0.1,0.4,0.9], type=float,
                    help='Weight for prototype loss computation')
parser.add_argument('-l3','--lambda3', default=[0.1,0.4,0.9], type=float,
                    help='Weight for prototype loss computation')
parser.add_argument('--num_classes', default=2, type=int,
                    help='How many classes are to be classified?')
parser.add_argument('--class_weights', default=[0.5,0.5],
                    help='Class weight for cross entropy loss')
parser.add_argument('-g','--gpu', type=int, default=0, help='GPU device number, -1  means CPU.')
parser.add_argument('--one_shot', type=bool, default=False,
                    help='Whether to use one-shot learning or not (i.e. only a few training examples)')
parser.add_argument('--discard', type=bool, default=False, help='Whether edge cases in the middle between completely '
                                                                'toxic (1) and not toxic at all (0) shall be omitted')
parser.add_argument('--model', type=str, default='Proto', choices=['Proto','Baseline'],
                    help='Define which model to use')

def search(args, text_train, labels_train, text_val, labels_val, text_test, labels_test):
    global acc, mean_loss
    parameters = dict(
        num_prototypes = args.num_prototypes,
        lr = args.lr,
        num_epochs = args.num_epochs,
        batch_size = args.batch_size,
        lambda2 = args.lambda2,
        lambda3 = args.lambda3
    )

    param_values = [v for v in parameters.values()]
    print(param_values)

    print("Running on gpu {}".format(args.gpu))

    for run_id, (num_prototypes, lr, num_epochs, batch_size, lambda2, lambda3) in enumerate(product(*param_values)):
        comment = f' num_prototypes = {num_prototypes} batch_size = {batch_size} lr = {lr} ' \
                  f'lambda2 = {lambda2} lambda3 ={lambda3} num_epochs = {num_epochs}'
        tb = SummaryWriter(comment=comment)

        args.num_prototypes = num_prototypes
        model = ProtoNetNLP(args)
        model.cuda(args.gpu)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        ce_crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(args.class_weights).float().cuda(args.gpu))
        interp_criteria = utils.ProtoLoss()

        model.train()
        embedding = model.compute_embedding(text_train, args.gpu)
        embedding_val = model.compute_embedding(text_val, args.gpu)
        print("\nStarting training for {} epochs\n".format(num_epochs))
        best_acc = 0
        best_model = []
        for epoch in tqdm(range(num_epochs)):
            all_preds = []
            all_labels = []
            losses_per_batch = []
            ce_loss_per_batch = []
            r1_loss_per_batch = []
            r2_loss_per_batch = []
            emb_batches, label_batches = utils.get_batches(embedding, labels_train, batch_size)

            # Update the RTPT
            rtpt.step(subtitle=f"epoch={epoch+1}")

            for i,(emb_batch, label_batch) in enumerate(zip(emb_batches, label_batches)):
                optimizer.zero_grad()

                outputs = model.forward(emb_batch)
                prototype_distances, feature_vector_distances, predicted_label = outputs

                # compute individual losses and backward step
                ce_loss = ce_crit(predicted_label, label_batch)
                r1_loss, r2_loss = interp_criteria(feature_vector_distances, prototype_distances)
                loss = ce_loss + \
                       lambda2 * r1_loss + \
                       lambda3 * r2_loss

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

            mean_loss = np.mean(losses_per_batch)
            ce_mean_loss = np.mean(ce_loss_per_batch)
            r1_mean_loss = np.mean(r1_loss_per_batch)
            r2_mean_loss = np.mean(r2_loss_per_batch)
            acc = balanced_accuracy_score(all_labels, all_preds)
            print("Epoch {}, mean loss {:.4f}, ce loss {:.4f}, r1 loss {:.4f}, "
                  "r2 loss {:.4f}, train acc {:.4f}".format(epoch+1,
                                                            mean_loss,
                                                            ce_mean_loss,
                                                            r1_mean_loss,
                                                            r2_mean_loss,
                                                            100 * acc))

            if (epoch + 1) % args.val_epoch == 0 or epoch + 1 == num_epochs:
                model.eval()
                with torch.no_grad():
                    outputs = model.forward(embedding_val)
                    prototype_distances, feature_vector_distances, predicted_label = outputs

                    # compute individual losses and backward step
                    ce_loss = ce_crit(predicted_label, labels_val)
                    r1_loss, r2_loss = interp_criteria(feature_vector_distances, prototype_distances)
                    loss_val = ce_loss + \
                           lambda2 * r1_loss + \
                           lambda3 * r2_loss

                    _, predicted_val = torch.max(predicted_label.data, 1)
                    acc_val = balanced_accuracy_score(labels_val.cpu().numpy(), predicted_val.cpu().numpy())
                    print("Validation: mean loss {:.4f}, acc_val {:.4f}".format(loss_val, 100 * acc_val))

                    tb.add_scalar("Loss_val", loss_val, epoch+1)
                    tb.add_scalar("Accuracy_val", acc_val, epoch+1)
                if acc_val >= best_acc:
                    best_acc = acc_val
                    best_model = model.state_dict()

        model.load_state_dict(best_model)
        model.cuda(args.gpu)
        model.eval()
        embedding_test = model.compute_embedding(text_test, args.gpu)
        with torch.no_grad():
            outputs = model.forward(embedding_test)
            _, _, predicted_label = outputs

            _, predicted = torch.max(predicted_label.data, 1)
            acc_test = balanced_accuracy_score(labels_test.cpu().numpy(), predicted.cpu().numpy())

        tb.add_hparams({"lr": lr, "bsize": batch_size, "lambda2": lambda2, "lambda3": lambda3},
                        dict(best_accuracy=best_acc, test_accuarcy=acc_test))

    tb.close()


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    args = parser.parse_args()

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    text, labels = utils.load_data(args)
    # split data, and split test set again to get validation and test set
    text_train, text_test, labels_train, labels_test = train_test_split(text, labels, test_size=0.3, stratify=labels)
    text_val, text_test, labels_val, labels_test = train_test_split(text_test, labels_test, test_size=0.5,
                                                                         stratify=labels_test)
    labels_train = torch.LongTensor(labels_train).cuda(args.gpu)
    labels_val = torch.LongTensor(labels_val).cuda(args.gpu)
    labels_test = torch.LongTensor(labels_test).cuda(args.gpu)
    # set class weights for balanced cross entropy computation
    balance = labels.count(0) / len(labels)
    args.class_weights = [1-balance, balance]

    if args.one_shot:
        idx = random.sample(range(len(text_train)),100)
        text_train = list(text_train[i] for i in idx)
        labels_train = torch.LongTensor([labels_train[i] for i in idx]).cuda(args.gpu)

    search(args, text_train, labels_train, text_val, labels_val, text_test, labels_test)