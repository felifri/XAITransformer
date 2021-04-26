import argparse
import datetime
import glob
import os
import random

import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from rtpt import RTPT
from transformers import get_linear_schedule_with_warmup

from models import ProtoPNetConv, ProtoNet
from utils import save_embedding, load_embedding, load_data, visualize_protos, proto_loss, prune_prototypes, \
    get_nearest, remove_prototypes, add_prototypes, reinit_prototypes, finetune_prototypes, bubble, nearest_image, \
    replace_prototypes

parser = argparse.ArgumentParser(description='Transformer Prototype Learning')
parser.add_argument('-m', '--mode', default='train test', type=str, nargs='+',
                    help='What do you want to do? Select either any combination of train, test, query, finetune, prune, '
                         'add, remove, reinitialize.')
parser.add_argument('--lr', type=float, default=0.004,
                    help='Select learning rate')
parser.add_argument('-e', '--num_epochs', default=200, type=int,
                    help='How many epochs?')
parser.add_argument('-bs', '--batch_size', default=1024, type=int,
                    help='Select batch size')
parser.add_argument('--val_epoch', default=10, type=int,
                    help='After how many epochs should the model be evaluated on the validation data?')
parser.add_argument('--data_dir', default='./data',
                    help='Select data path')
parser.add_argument('--data_name', default='rt-polarity', type=str, choices=['rt-polarity', 'toxicity',
                                                                             'toxicity_full', 'ethics', 'restaurant'],
                    help='Select name of data set')
parser.add_argument('--num_prototypes', default=10, type=int,
                    help='Total number of prototypes')
parser.add_argument('-l1','--lambda1', default=0.2, type=float,
                    help='Weight for prototype distribution loss')
parser.add_argument('-l2','--lambda2', default=0.2, type=float,
                    help='Weight for prototype cluster loss')
parser.add_argument('-l3','--lambda3', default=0.1, type=float,
                    help='Weight for prototype separation loss')
parser.add_argument('-l4','--lambda4', default=0.1, type=float,
                    help='Weight for prototype diversity loss')
parser.add_argument('-l5','--lambda5', default=1e-3, type=float,
                    help='Weight for l1 weight regularization loss')
parser.add_argument('--num_classes', default=2, type=int,
                    help='How many classes are to be classified?')
parser.add_argument('-g','--gpu', type=int, default=[0], nargs='+',
                    help='GPU device number(s)')
parser.add_argument('--few_shot', type=bool, default=False,
                    help='Whether to use few-shot learning or not (i.e. only a few training examples)')
parser.add_argument('--trans_type', type=str, default='PCA', choices=['PCA', 'TSNE', 'UMAP'],
                    help='Select transformation to visualize the prototypes')
parser.add_argument('--discard', type=bool, default=False,
                    help='Whether edge cases in the middle between completely toxic(1) and not toxic(0) shall be omitted')
parser.add_argument('--proto_size', type=int, default=1,
                    help='Define how many words should be used to define a prototype')
parser.add_argument('--level', type=str, default='word', choices=['word','sentence'],
                    help='Define whether prototypes are computed on word (Bert/GPT2) or sentence level (SentBert/CLS)')
parser.add_argument('--language_model', type=str, default='Bert', choices=['Bert','SentBert','GPT2','TXL','Roberta',
                    'DistilBert','Clip'], help='Define which language model to use')
parser.add_argument('-d','--dilated', type=int, default=[1], nargs='+',
                    help='Whether to use dilation in the ProtoP convolution and which step size')
parser.add_argument('--avoid_spec_token', type=bool, default=False,
                    help='Whether to manually set PAD, SEP and CLS token to high value after Bert embedding computation')
parser.add_argument('--compute_emb', type=bool, default=False,
                    help='Whether to recompute (True) the embedding or just load it (False)')
parser.add_argument('--query', type=str, default='I do not like the food here', nargs='+',
                    help='Type your query to test the model and get classification explanation')
parser.add_argument('--metric', type=str, default='cosine', choices=['cosine','L2'],
                    help='What metric should be used to compute the distance/ similarity?')
parser.add_argument('--attn', type=str, default=False,
                    help='Whether to use self-attention on the word embeddings before distance computation')
parser.add_argument('--auto_prune', type=str, default=False,
                    help='Whether to remove/prune automatically low weight prototypes after 70% of train epochs')

def train(args, train_batches, val_batches, model):
    num_epochs = args.num_epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce_crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(args.class_weights).float().to(f'cuda:{args.gpu[0]}'))
    # scheduler = get_linear_schedule_with_warmup(optimizer, min(10, num_epochs // 20), num_epochs)

    print(f'\nStart training for {num_epochs} epochs\n')
    best_acc = 0

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

        for emb_batch, mask_batch, label_batch in train_batches:
            emb_batch = emb_batch.to(f'cuda:{args.gpu[0]}')
            mask_batch = mask_batch.to(f'cuda:{args.gpu[0]}')
            label_batch = label_batch.to(f'cuda:{args.gpu[0]}')

            optimizer.zero_grad()
            prototype_distances, predicted_label = model.forward(emb_batch, mask_batch)

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

            _, predicted = torch.max(predicted_label, 1)
            all_preds += predicted.cpu().detach().numpy().tolist()
            all_labels += label_batch.cpu().detach().numpy().tolist()

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.fc.weight.copy_(model.fc.weight.clamp(max=0.0))
            # store losses
            losses_per_batch.append(float(loss))
            ce_loss_per_batch.append(float(ce_loss))
            distr_loss_per_batch.append(float(args.lambda1 * distr_loss))
            clust_loss_per_batch.append(float(args.lambda2 * clust_loss))
            sep_loss_per_batch.append(float(args.lambda3 * sep_loss))
            divers_loss_per_batch.append(float(args.lambda4 * divers_loss))
            l1_loss_per_batch.append(float(args.lambda5 * l1_loss))

        # scheduler.step()
        mean_loss = np.mean(losses_per_batch)
        ce_mean_loss = np.mean(ce_loss_per_batch)
        distr_mean_loss = np.mean(distr_loss_per_batch)
        clust_mean_loss = np.mean(clust_loss_per_batch)
        sep_mean_loss = np.mean(sep_loss_per_batch)
        divers_mean_loss = np.mean(divers_loss_per_batch)
        l1_mean_loss = np.mean(l1_loss_per_batch)
        acc = balanced_accuracy_score(all_labels, all_preds)
        print(f'Epoch {epoch+1}, losses: mean {mean_loss:.3f}, ce {ce_mean_loss:.3f}, distr {distr_mean_loss:.3f}, '
              f'clust {clust_mean_loss:.3f}, sep {sep_mean_loss:.3f}, divers {divers_mean_loss:.3f}, '
              f'l1 {l1_mean_loss:.3f}, train acc {100 * acc:.3f}')

        if ((epoch + 1) % args.val_epoch == 0) or (epoch + 1 == num_epochs):
            model.eval()
            all_preds = []
            all_labels = []
            losses_per_batch = []
            with torch.no_grad():
                for emb_batch, mask_batch, label_batch in val_batches:
                    emb_batch = emb_batch.to(f'cuda:{args.gpu[0]}')
                    mask_batch = mask_batch.to(f'cuda:{args.gpu[0]}')
                    label_batch = label_batch.to(f'cuda:{args.gpu[0]}')
                    prototype_distances, predicted_label = model.forward(emb_batch, mask_batch)

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
                    _, predicted = torch.max(predicted_label, 1)
                    all_preds += predicted.detach().cpu().numpy().tolist()
                    all_labels += label_batch.cpu().detach().numpy().tolist()

                loss_val = np.mean(losses_per_batch)
                acc_val = balanced_accuracy_score(all_labels, all_preds)
                print(f'Validation: mean loss {loss_val:.3f}, acc_val {100 * acc_val:.3f}')

                if acc_val > best_acc:
                    best_acc = acc_val
                    state = {'state_dict': model.state_dict(), 'hyper_params': args, 'acc_val': acc_val}

        # if (epoch + 1) == (num_epochs * 8 // 10):
        #     with torch.no_grad():
        #         best_acc = 0
        #         # project prototypes
        #         _, proto_texts = get_nearest(args, model, train_batches_unshuffled, text_train, labels_train)
        #         new_proto, _ = model.compute_embedding(proto_texts, args)
        #         model.protolayer.copy_(new_proto)
        #         model.protolayer.requires_grad = False
        # if (epoch + 1) == (num_epochs * 7 // 10):
        #     with torch.no_grad():
        #         # __import__("pdb").set_trace()
        #         # project prototypes
        #         _, proto_texts = get_nearest(args, model, train_batches_unshuffled, text_train, labels_train)
        #         new_proto, _ = model.compute_embedding(proto_texts, args)
        #         model.protolayer.copy_(new_proto)

    model.load_state_dict(state['state_dict'])
    torch.save(state, args.model_path)
    return model


def test(args, embedding_train, mask_train, train_batches_unshuffled, test_batches, labels_train, text_train, model, pruned_text=None):
    print('\nStart evaluation, loading model:', args.model_path)
    model.eval()
    ce_crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(args.class_weights).float().to(f'cuda:{args.gpu[0]}'))

    all_preds = []
    all_labels = []
    losses_per_batch = []

    with torch.no_grad():
        for emb_batch, mask_batch, label_batch in test_batches:
            emb_batch = emb_batch.to(f'cuda:{args.gpu[0]}')
            mask_batch = mask_batch.to(f'cuda:{args.gpu[0]}')
            label_batch = label_batch.to(f'cuda:{args.gpu[0]}')
            prototype_distances, predicted_label = model.forward(emb_batch, mask_batch)

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
            _, predicted = torch.max(predicted_label, 1)
            all_preds += predicted.cpu().detach().numpy().tolist()
            all_labels += label_batch.cpu().detach().numpy().tolist()

        loss = np.mean(losses_per_batch)
        acc_test = balanced_accuracy_score(all_labels, all_preds)
        print(f'Test evaluation on best model: loss {loss:.3f}, acc_test {100 * acc_test:.3f}')

        # "convert" prototype embedding to text (take text of nearest training sample)
        proto_info, proto_texts = get_nearest(args, model, train_batches_unshuffled, text_train, labels_train)
        weights = model.get_proto_weights()

        if os.path.basename(args.model_path).startswith('interacted'):
            fname = 'interacted_' + str(args.num_prototypes) + 'prototypes.txt'
            if 'prune' in args.mode:
                proto_texts = pruned_text
        else:
            fname = str(args.num_prototypes) + 'prototypes.txt'

        if args.language_model == 'Clip':
            nearest_image(args, model, proto_texts)

        proto_texts = [id_ + txt for id_, txt in zip(proto_info, proto_texts)]
        save_path = os.path.join(os.path.dirname(args.model_path), fname)
        txt_file = open(save_path, 'w+')
        for arg in vars(args):
            txt_file.write(f'{arg}: {vars(args)[arg]}\n')
        txt_file.write(f'test loss: {loss:.3f}\n')
        txt_file.write(f'test acc: {100*acc_test:.2f}\n')
        for line in proto_texts:
            txt_file.write(line + '\n')
        for line in weights:
            x = []
            for l in line: x.append(f'{l:.3f}')
            txt_file.write(str(x) + '\n')
        txt_file.close()

        # give prototype its "true" label after training
        s = 'label'
        proto_labels = torch.tensor([int(p[p.index(s) + len(s)+1]) for p in proto_info])
        proto_labels = torch.stack((1-proto_labels, proto_labels), dim=1)

        # plot prototypes
        prototypes = model.get_protos().cpu().numpy()
        visualize_protos(args, embedding_train.cpu().numpy(), mask_train, labels_train, prototypes, model, proto_labels)

        if len(embedding_train.shape) == 3 and args.attn:
            neighborhood = bubble(args, train_batches_unshuffled, model, text_train)
            fname += '_WordVis'
            save_path = os.path.join(os.path.dirname(args.model_path), fname)
            txt_file = open(save_path, 'w+')
            for line in neighborhood:
                txt_file.write(line + '\n')
            txt_file.close()


def query(args, train_batches_unshuffled, labels_train, text_train, model):
    print('\nEvaluate query, loading model:', args.model_path)
    model.eval()
    args.is_image = False
    if args.is_image:
        from PIL import Image
        pth = 'data/test_images/4.jpg'
        args.query = Image.open(pth)

    embedding_query, mask_query = model.compute_embedding(args.query, args)
    torch.cuda.empty_cache()  # is required since BERT encoding is only possible on 1 GPU (memory limitation)

    # "convert" prototype embedding to text (take text of nearest training sample)
    proto_info, proto_texts = get_nearest(args, model, train_batches_unshuffled, text_train, labels_train)
    prototype_distances, predicted_label = model.forward(embedding_query.to(f'cuda:{args.gpu[0]}'), mask_query)
    predicted = torch.argmax(predicted_label).cpu().detach()

    k = 3
    query2proto = torch.topk(prototype_distances.cpu().detach().squeeze(), k=k, largest=False)
    weights = model.get_proto_weights()
    weight = - weights[query2proto[1], predicted]
    similarity = - query2proto[0]

    save_path = os.path.join(os.path.dirname(args.model_path), 'query.txt')
    sentiment_dict = {1: 'positive', 0: 'negative'}

    txt_file = open(save_path, 'w+')
    txt_file.write('query: ' + ''.join(args.query) + '\n')
    txt_file.write(f'predicted: {sentiment_dict[int(predicted)]}\n\n')
    for i in range(k):
        nearest_proto = proto_texts[query2proto[1][i]]
        txt_file.write(f'Explanation_{i+1}: {nearest_proto}\n')
        txt_file.write(f'similarity: {float(similarity[i]):.3f}\n')
        txt_file.write(f'weight: {weight[i]:.3f}\n\n')
    txt_file.close()


def interact(args, train_batches, mask_train, train_batches_unshuffled, val_batches, embedding_train, test_batches, labels_train, text_train, model):
    print('\nInteract, loading model:', args.model_path)
    if 'remove' in args.mode:
        # protos2remove = list(map(int,input('Select prototypes to remove: ').split()))
        protos2remove = []
        args, model = remove_prototypes(args, protos2remove, model, use_cos=False)

    if 'add' in args.mode:
        protos2add = [['bad service'],[0]]
        # protos2add = list(map(int,input('Select prototypes to add: ').split()))
        args, model = add_prototypes(args, protos2add, model)

    if 'replace' in args.mode:
        protos2add = [['bad service'], [0]]
        args, model = replace_prototypes(args, protos2add, model)
        protos2add = [['great service'], [1]]
        args, model = replace_prototypes(args, protos2add, model)
        protos2add = [['shit location'], [2]]
        args, model = replace_prototypes(args, protos2add, model)
        protos2add = [['awesome food'], [3]]
        args, model = replace_prototypes(args, protos2add, model)

    if 'reinitialize' in args.mode:
        protos2reinit = []
        # protos2retrain = list(map(int,input('Select prototypes to retrain: ').split()))
        model = reinit_prototypes(args, protos2reinit, model)

    if 'finetune' in args.mode:
        protos2finetune = [0]
        # protos2finetune = list(map(int,input('Select prototypes to finetune: ').split()))
        model = finetune_prototypes(args, protos2finetune, model)

    pruned_protos = []
    if 'prune' in args.mode:
        _, protos2prune = get_nearest(args, model, train_batches_unshuffled, text_train, labels_train)
        # protos2prune = list(map(int,input('Select prototypes to prune: ').split()))
        model, pruned_protos = prune_prototypes(args, protos2prune, model)

    # save changed model
    args.model_path = os.path.join(os.path.dirname(args.model_path), 'interacted_best_model.pth.tar')
    # retrain whole network or only retrain last layer (fc)
    # args.num_epochs = 100
    model = train(args, train_batches, val_batches, model)
    test(args, embedding_train, mask_train, train_batches_unshuffled, test_batches, labels_train, text_train, model, pruned_protos)


if __name__ == '__main__':
    # torch.manual_seed(0)
    # np.random.seed(0)
    torch.set_num_threads(6)
    args = parser.parse_args()

    # Create RTPT object and start the RTPT tracking
    rtpt = RTPT(name_initials='FF', experiment_name='TProNet', max_iterations=args.num_epochs)
    rtpt.start()

    text, labels = load_data(args)
    # split data, and split test set again to get validation and test set
    text_train, text_test, labels_train, labels_test = train_test_split(text, labels, test_size=0.3, stratify=labels,
                                                                        random_state=42)
    text_val, text_test, labels_val, labels_test = train_test_split(text_test, labels_test, test_size=0.5,
                                                                    stratify=labels_test, random_state=12)

    # set class weights for balanced loss computation
    args.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)

    if args.num_prototypes % args.num_classes:
        print('number of prototypes should be divisible by number of classes')
        args.num_prototypes -= args.num_prototypes % args.num_classes
    # define which prototype belongs to which class (onehot encoded matrix)
    args.prototype_class_identity = torch.eye(args.num_classes).repeat(args.num_prototypes//args.num_classes, 1
                                                                       ).to(f'cuda:{args.gpu[0]}')
    # protos_per_class = round(balance*args.num_prototypes)
    # args.prototype_class_identity[:protos_per_class, 0] = 1
    # args.prototype_class_identity[protos_per_class:, 1] = 1

    if args.few_shot:
        idx = random.sample(range(len(text_train)), 100)
        text_train = list(text_train[i] for i in idx)
        args.compute_emb = True

    model = []
    if args.level == 'word':
        model = ProtoPNetConv(args)
    elif args.level == 'sentence':
        model = ProtoNet(args)

    print(f'Running on gpu {args.gpu}')
    # model = torch.nn.DataParallel(model, device_ids=args.gpu)
    model.to(f'cuda:{args.gpu[0]}')

    avoid = ''
    if args.avoid_spec_token:
        avoid = '_avoid'

    fname = args.language_model
    if not args.compute_emb:
        embedding_train, mask_train = load_embedding(args, fname, 'train' + avoid)
        embedding_val, mask_val = load_embedding(args, fname, 'val' + avoid)
        embedding_test, mask_test = load_embedding(args, fname, 'test' + avoid)
    else:
        embedding_train, mask_train = model.compute_embedding(text_train, args)
        embedding_val, mask_val = model.compute_embedding(text_val, args)
        embedding_test, mask_test = model.compute_embedding(text_test, args)
        save_embedding(embedding_train, mask_train, args, fname, 'train' + avoid)
        save_embedding(embedding_val, mask_val, args, fname, 'val' + avoid)
        save_embedding(embedding_test, mask_test, args, fname, 'test' + avoid)
        torch.cuda.empty_cache()  # free up language model from GPU

    train_batches = torch.utils.data.DataLoader(list(zip(embedding_train, mask_train, labels_train)),
                                                batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                                num_workers=0)
    train_batches_unshuffled = torch.utils.data.DataLoader(list(zip(embedding_train, mask_train, labels_train)),
                                                           batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                                           num_workers=0)
    val_batches = torch.utils.data.DataLoader(list(zip(embedding_val, mask_val, labels_val)),
                                              batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)
    test_batches = torch.utils.data.DataLoader(list(zip(embedding_test, mask_test, labels_test)),
                                               batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                               num_workers=0)

    time_stmp = datetime.datetime.now().strftime(f'%d-%m %H:%M_{args.num_prototypes}{fname}{args.proto_size}')
    args.model_path = os.path.join('./experiments/train_results/', time_stmp, 'best_model.pth.tar')

    if 'train' in args.mode:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        model = train(args, train_batches, val_batches, model)
    if not os.path.exists(args.model_path):
        load_path = './experiments/train_results/*'
        model_paths = glob.glob(os.path.join(load_path, 'best_model.pth.tar'))
        model_paths.sort()
        args.model_path = model_paths[-1]
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'])
    if 'test' in args.mode:
        test(args, embedding_train, mask_train, train_batches_unshuffled, test_batches, labels_train, text_train, model)
    if 'query' in args.mode:
        query(args, train_batches_unshuffled, labels_train, text_train, model)
    if 'add' in args.mode or 'remove' in args.mode or 'finetune' in args.mode or 'reinitialize' in args.mode \
            or 'prune' in args.mode or 'replace' in args.mode:
        interact(args, train_batches, mask_train, train_batches_unshuffled, val_batches, embedding_train, test_batches,
                 labels_train, text_train, model)