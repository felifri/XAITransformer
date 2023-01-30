import argparse
import datetime
import glob
import os
import random
from rtpt import RTPT
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
# from transformers import get_linear_schedule_with_warmup
from PIL import Image
from collections import Counter
from models import ProtoTrexS, ProtoTrexW
from utils import save_embedding, load_embedding, load_data, visualize_protos, proto_loss, prune_prototypes, \
    get_nearest, remove_prototypes, add_prototypes, reinit_prototypes, finetune_prototypes, nearest_image, \
    replace_prototypes, soft_rplc_prototypes, project, preprocess_restaurant, preprocess_jigsaw

parser = argparse.ArgumentParser(description='Transformer Prototype Learning')
parser.add_argument('-m', '--mode', default='train test', type=str, nargs='+',
                    help='What do you want to do? Select either any combination of train, test, query, finetune, '
                         'prune, add, remove, reinitialize, explain, unique, survey')
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
parser.add_argument('--data_name', default='rt-polarity', type=str, choices=['rt-polarity', 'toxicity', 'jigsaw',
                                                                             'toxicity_full', 'ethics', 'restaurant',
                                                                             'movie_review', 'propaganda'],
                    help='Select name of data set')
parser.add_argument('--num_prototypes', default=10, type=int,
                    help='Total number of prototypes')
parser.add_argument('-l1', '--lambda1', default=0.2, type=float,
                    help='Weight for prototype distribution loss')
parser.add_argument('-l2', '--lambda2', default=0.2, type=float,
                    help='Weight for prototype cluster loss')
parser.add_argument('-l3', '--lambda3', default=0.1, type=float,
                    help='Weight for prototype separation loss')
parser.add_argument('-l4', '--lambda4', default=0.05, type=float,
                    help='Weight for prototype diversity loss')
parser.add_argument('-l5', '--lambda5', default=0.009, type=float,
                    help='Weight for l1 weight regularization loss')
parser.add_argument('-l6', '--lambda6', default=0.1, type=float,
                    help='Weight for kl divergence loss')
parser.add_argument('--num_classes', default=2, type=int,
                    help='How many classes are to be classified?')
parser.add_argument('-g', '--gpu', type=int, default=[0], nargs='+',
                    help='GPU device number(s)')
parser.add_argument('--few_shot', type=bool, default=False,
                    help='Whether to use few-shot learning or not (i.e. only a few training examples)')
parser.add_argument('--trans_type', type=str, default='PCA', choices=['PCA', 'TSNE', 'UMAP'],
                    help='Select transformation to visualize the prototypes')
parser.add_argument('--discard', type=bool, default=False,
                    help='Whether edge cases (~0.5) in the middle between toxic(1) and not toxic(0) shall be omitted')
parser.add_argument('--proto_size', type=int, default=1,
                    help='Define how many words should be used to define a word-level prototype')
parser.add_argument('--level', type=str, default='word', choices=['word', 'sentence'],
                    help='Define whether prototypes are computed on word (Bert/GPT2) or sentence level (SentBert/CLS)')
parser.add_argument('--language_model', type=str, default='Bert', choices=['Bert', 'SentBert', 'GPT2', 'GPTJ', 'TXL',
                                                                           'Roberta', 'DistilBert', 'Clip', 'Sentence-T5', 'all-mpnet', 'SGPT-5.8', 'SGPT-125', 'SGPT-7.1'],
                    help='Define which language model to use')
parser.add_argument('-d', '--dilated', type=int, default=[1], nargs='+',
                    help='Whether to use dilation in the ProtoP convolution and which step size')
parser.add_argument('--compute_emb', type=bool, default=False,
                    help='Whether to recompute (True) the embedding or just load it (False)')
parser.add_argument('--query', type=str, default=['I do not like the food here'], nargs='+',
                    help='Type your query to test the model and get classification explanation')
parser.add_argument('--metric', type=str, default='cosine', choices=['cosine', 'L2'],
                    help='What metric should be used to compute the distance/ similarity?')
parser.add_argument('--attn', type=str, default=False,
                    help='Whether to use self-attention on the word embeddings before distance computation')
parser.add_argument('--project', type=str, default=False,
                    help='Whether to project the prototypes on their nearest neighbor after x epochs')
parser.add_argument('--soft', type=str, default=False, nargs='+',
                    help='Whether to softly apply loss')
parser.add_argument('--pid', type=str, default='',
                    help='Name for process')


def train(args, train_batches, val_batches, model, embedding_train, train_batches_unshuffled, text_train, labels_train):
    num_epochs = args.num_epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce_crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(args.class_weights).float().to(f'cuda:{args.gpu[0]}'))
    # scheduler = get_linear_schedule_with_warmup(optimizer, min(10, num_epochs // 20), num_epochs)

    print(f'\nStart training for {num_epochs} epochs\n')
    best_acc = 0

    for epoch in tqdm(range(num_epochs)):
        model.train()
        all_preds, all_labels = [], []
        losses_per_batch = []
        ce_loss_per_batch = []
        distr_loss_per_batch = []
        clust_loss_per_batch = []
        sep_loss_per_batch = []
        divers_loss_per_batch = []
        l1_loss_per_batch = []
        kl_loss_per_batch = []

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
            distr_loss, clust_loss, sep_loss, divers_loss, l1_loss, kl_loss = \
                proto_loss(prototype_distances, label_batch, model, args)
            loss = ce_loss + \
                   args.lambda1 * distr_loss + \
                   args.lambda2 * clust_loss + \
                   args.lambda3 * sep_loss + \
                   args.lambda4 * divers_loss + \
                   args.lambda5 * l1_loss + \
                   args.lambda6 * kl_loss

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
            kl_loss_per_batch.append(float(args.lambda6 * kl_loss))

        # scheduler.step()
        mean_loss = np.mean(losses_per_batch)
        ce_mean_loss = np.mean(ce_loss_per_batch)
        distr_mean_loss = np.mean(distr_loss_per_batch)
        clust_mean_loss = np.mean(clust_loss_per_batch)
        sep_mean_loss = np.mean(sep_loss_per_batch)
        divers_mean_loss = np.mean(divers_loss_per_batch)
        l1_mean_loss = np.mean(l1_loss_per_batch)
        kl_mean_loss = np.mean(kl_loss_per_batch)
        acc = balanced_accuracy_score(all_labels, all_preds)
        print(f'Epoch {epoch + 1}, losses: mean {mean_loss:.3f}, ce {ce_mean_loss:.3f}, distr {distr_mean_loss:.3f}, '
              f'clust {clust_mean_loss:.3f}, sep {sep_mean_loss:.3f}, divers {divers_mean_loss:.3f}, '
              f'l1 {l1_mean_loss:.3f}, kl {kl_mean_loss}, train acc {100 * acc:.3f}')

        if ((epoch + 1) % args.val_epoch == 0) and ((epoch + 1) > (num_epochs * 2 // 10)) or (epoch + 1 == num_epochs):
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
                    distr_loss, clust_loss, sep_loss, divers_loss, l1_loss, kl_loss = \
                        proto_loss(prototype_distances, label_batch, model, args)
                    loss = ce_loss + \
                           args.lambda1 * distr_loss + \
                           args.lambda2 * clust_loss + \
                           args.lambda3 * sep_loss + \
                           args.lambda4 * divers_loss + \
                           args.lambda5 * l1_loss + \
                           args.lambda6 * kl_loss

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

        # project prototypes
        if (epoch + 1) % 5 == 0 and args.project and (num_epochs * 2 // 10) < (epoch + 1) < (num_epochs * 8 // 10):
            with torch.no_grad():
                model, args = project(args, embedding_train, model, train_batches_unshuffled, text_train, labels_train)
        # final projection, train only classification layer
        if (epoch + 1) == (num_epochs * 8 // 10) and args.project:
            with torch.no_grad():
                best_acc = 0
                model, args = project(args, embedding_train, model, train_batches_unshuffled, text_train, labels_train)
                model.protolayer.requires_grad = False

    model.load_state_dict(state['state_dict'])
    torch.save(state, args.model_path)
    return model


def test(args, embedding_train, mask_train, train_batches_unshuffled, test_batches, labels_train, text_train, model):
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
            distr_loss, clust_loss, sep_loss, divers_loss, l1_loss, kl_loss = \
                proto_loss(prototype_distances, label_batch, model, args)
            loss = ce_loss + \
                   args.lambda1 * distr_loss + \
                   args.lambda2 * clust_loss + \
                   args.lambda3 * sep_loss + \
                   args.lambda4 * divers_loss + \
                   args.lambda5 * l1_loss + \
                   args.lambda6 * kl_loss

            losses_per_batch.append(float(loss))
            _, predicted = torch.max(predicted_label, 1)
            all_preds += predicted.cpu().detach().numpy().tolist()
            all_labels += label_batch.cpu().detach().numpy().tolist()

        loss = np.mean(losses_per_batch)
        acc_test = balanced_accuracy_score(all_labels, all_preds)
        print(f'Test evaluation on best model: loss {loss:.3f}, acc_test {100 * acc_test:.3f}')

        # "convert" prototype embedding to text (take text of nearest training sample)
        proto_info, proto_texts, _ = get_nearest(args, model, train_batches_unshuffled, text_train, labels_train)
        weights = model.get_proto_weights()

        if os.path.basename(args.model_path).startswith('interacted'):
            fname = 'interacted_' + str(args.num_prototypes) + 'prototypes.txt'
        else:
            fname = str(args.num_prototypes) + 'prototypes.txt'

        # if args.language_model == 'Clip':
        #     nearest_image(args, model, proto_texts)

        proto_texts = [id_ + txt for id_, txt in zip(proto_info, proto_texts)]
        save_path = os.path.join(os.path.dirname(args.model_path), fname)
        txt_file = open(save_path, 'w+')
        for arg in vars(args):
            txt_file.write(f'{arg}: {vars(args)[arg]}\n')
        txt_file.write(f'test loss: {loss:.3f}\n')
        txt_file.write(f'test acc: {100 * acc_test:.2f}\n')
        for line in proto_texts:
            txt_file.write(line + '\n')
        for line in weights:
            x = []
            for l in line: x.append(f'{l:.3f}')
            txt_file.write(str(x) + '\n')
        txt_file.close()

        # give prototype its "true" label after training
        s = 'label'
        proto_labels = torch.tensor([int(p[p.index(s) + len(s) + 1]) for p in proto_info])
        proto_labels = torch.stack((1 - proto_labels, proto_labels), dim=1)

        # plot prototypes
        prototypes = model.get_protos().cpu().numpy()
        visualize_protos(args, embedding_train.cpu().numpy(), mask_train, labels_train, prototypes, model, proto_labels)


def query(args, train_batches_unshuffled, labels_train, text_train, model):
    print('\nEvaluate query, loading model:', args.model_path)
    model.eval()
    args.is_image = False
    if args.is_image:
        pth = 'data/test_images/4.jpg'
        args.query = Image.open(pth)

    embedding_query, mask_query = model.compute_embedding(args.query, args)
    embedding_query = embedding_query.to(f'cuda:{args.gpu[0]}')
    mask_query = mask_query.to(f'cuda:{args.gpu[0]}')
    args.query = args.query[0]
    torch.cuda.empty_cache()  # is required since BERT encoding is only possible on 1 GPU (memory limitation)

    # "convert" prototype embedding to text (take text of nearest training sample)
    _, proto_texts, _ = get_nearest(args, model, train_batches_unshuffled, text_train, labels_train)
    prototype_distances, predicted_label = model.forward(embedding_query, mask_query)
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
        txt_file.write(f'Explanation_{i + 1}: {nearest_proto}\n')
        txt_file.write(
            f'score: {float(similarity[i]):.3f} * {weight[i]:.3f} = {float(similarity[i] * weight[i]):.3f}\n')
    txt_file.write(f'\nwith:\nsimilarity * weight = score')
    txt_file.close()

def survey(args, train_batches_unshuffled, labels_train, text_train, text_test, labels_test, model):
    '''
    Function samples 100 datapoints from the test set and creates 4(8) different csv files.
    Each file contains a random permutation of the 100 datapoints then they differ:
    1st csv contains only the datapoints
    2nd csv contains datapoints with random explanations picked from the prototypes of the model
    3rd csv contains datapoints with the best explanation given by inference
    4th csv contains datapoints with the best explanation as well as the prediction of the model
    
    For each of the csvs there is a corresponding csv with the true labels of the datapoints.
    '''
    mode = "Steerable" #set which type of survey you want to create
    print('\nCreating Survey, loading model:', args.model_path)
    model.eval()
    _, proto_texts, _ = get_nearest(args, model, train_batches_unshuffled, text_train, labels_train)
    if mode == "ProtoTex":
        rtpt = RTPT(name_initials='PK', experiment_name='Proto-Trex-Survey', max_iterations=100)
        rtpt.start()
        # load 100 random examples from test set and their corresponding labels
        zipped_elements = list(zip(text_test, labels_test))
        random_elements = random.sample(zipped_elements, 100)
        random_texts, random_labels = zip(*random_elements)
        # create a dictionary that will keep track of all necessary values
        dictionary = {}
        dictionary['Input'] = random_texts
        dictionary['True Label'] = random_labels
        
        # iterate over each entry in the dictionary and add the predicted labels as well as the explanations to the dictionary
        for text in random_texts:
            rtpt.step()
            args.query = text
            embedding_query, mask_query = model.compute_embedding(args.query, args)
            embedding_query = embedding_query.to(f'cuda:{args.gpu[0]}')
            mask_query = mask_query.to(f'cuda:{args.gpu[0]}')
            args.query = args.query[0]
            torch.cuda.empty_cache() # is required since BERT encoding is only possible on 1 GPU (memory limitation)
        
            prototype_distances, predicted_label = model.forward(embedding_query, mask_query)
            predicted = torch.argmax(predicted_label).cpu().detach()
            query2proto = torch.topk(prototype_distances.cpu().detach().squeeze(), k=1, largest=False)
            nearest_proto = proto_texts[query2proto[1]]
        
            #add prediction to dictionary
            if "prediction" not in dictionary:
                dictionary["prediction"] = [int(predicted)]
            else:
                dictionary["prediction"].append(int(predicted))
            #add explanation to dictionary
            if "explanation" not in dictionary:
                dictionary["explanation"] = [nearest_proto]
            else:
                dictionary["explanation"].append(nearest_proto)
            #add random explanation to dictionary
            if "random explanation" not in dictionary:
                dictionary["random explanation"] = [random.choice(proto_texts)]
            else:
                dictionary["random explanation"].append(random.choice(proto_texts))
                
        # create a dataframe from the dictionary
        csv_df4 = pd.DataFrame(dictionary)
        csv_path4 = os.path.join(os.path.dirname(args.model_path), 'survey4.csv')
        with open(csv_path4, 'w') as f:
            csv_df4.to_csv(f, index=False)
        # create a dataframe with only the input and randomize the order of the rows
        csv_df1 = csv_df4.drop(columns=['prediction', 'explanation', 'random explanation'])
        csv_df1 = csv_df1.sample(frac=1).reset_index(drop=True)
        csv_path1 = os.path.join(os.path.dirname(args.model_path), 'survey1.csv')
        with open(csv_path1, 'w') as f:
            csv_df1.to_csv(f, index=False)
        # create a dataframe with only the input and the random explanation and randomize the order of the rows
        csv_df2 = csv_df4.drop(columns=['prediction', 'explanation'])
        csv_df2 = csv_df2.sample(frac=1).reset_index(drop=True)
        csv_path2 = os.path.join(os.path.dirname(args.model_path), 'survey2.csv')
        with open(csv_path2, 'w') as f:
            csv_df2.to_csv(f, index=False)
        # create a dataframe with only the input and the explanation and randomize the order of the rows
        csv_df3 = csv_df4.drop(columns=['prediction', 'random explanation'])
        csv_df3 = csv_df3.sample(frac=1).reset_index(drop=True)
        csv_path3 = os.path.join(os.path.dirname(args.model_path), 'survey3.csv')
        with open(csv_path3, 'w') as f:
            csv_df3.to_csv(f, index=False)
    elif mode == "Steerable":
        rtpt = RTPT(name_initials='PK', experiment_name='Proto-Trex-Survey', max_iterations=100)
        rtpt.start()
        # load 100 random examples from test set and their corresponding labels
        zipped_elements = list(zip(text_test, labels_test))
        random_elements = random.sample(zipped_elements, 100)
        random_texts, random_labels = zip(*random_elements)
        # create a dictionary that will keep track of all necessary values
        dictionary = {}
        dictionary['Input'] = random_texts
        dictionary['True Label'] = random_labels
        for text in random_texts:
            rtpt.step()
            args.query = text
            embedding_query, mask_query = model.compute_embedding(args.query, args)
            embedding_query = embedding_query.to(f'cuda:{args.gpu[0]}')
            mask_query = mask_query.to(f'cuda:{args.gpu[0]}')
            args.query = args.query[0]
            #torch.cuda.empty_cache()  # is required since BERT encoding is only possible on 1 GPU (memory limitation)

            # "convert" prototype embedding to text (take text of nearest training sample)
            _, proto_texts, _ = get_nearest(args, model, train_batches_unshuffled, text_train, labels_train)
            prototype_distances, predicted_label = model.forward(embedding_query, mask_query)
            predicted = torch.argmax(predicted_label).cpu().detach()
            k = 5 #args.num_prototypes
            query2proto = torch.topk(prototype_distances.cpu().detach().squeeze(), k=k, largest=False)
            weights = model.get_proto_weights()
            weight = - weights[query2proto[1], predicted]
            similarity = - query2proto[0]
            scores = []
            #print(len(weight), len(similarity))
            for i in range(k):
                scores.append(float(weight[i] * similarity[i]))
            sorted_scores = sorted(scores, reverse=True)
            index_scores = list(zip(range(len(scores)),sorted_scores))
            
            if "prediction" not in dictionary:
                dictionary["prediction"] = [int(predicted)]
            else:
                dictionary["prediction"].append(int(predicted))
            #add explanation to dictionary
            for i in range(3):
                index, score = index_scores[i]
                nearest_proto = proto_texts[query2proto[1][index]]
                if f"explanation {i}" not in dictionary:
                    dictionary[f"explanation {i}"] = [nearest_proto]
                else:
                    dictionary[f"explanation {i}"].append(nearest_proto)
            #add random explanation to dictionary
            for i in range(2):
                if f"random explanation {i}" not in dictionary:
                    dictionary[f"random explanation {i}"] = [random.choice(proto_texts)]
                else:
                    dictionary[f"random explanation {i}"].append(random.choice(proto_texts))
        csv_df = pd.DataFrame(dictionary)
        csv_path = os.path.join(os.path.dirname(args.model_path), 'steerable_survey.csv')
        with open(csv_path, 'w') as f:
            csv_df.to_csv(f, index=False)
            

def interact(args, train_batches, mask_train, train_batches_unshuffled, val_batches, embedding_train, test_batches,
             labels_train, text_train, model):
    print('\nInteract, loading model:', args.model_path)
    if 'remove' in args.mode:
        protos2remove = [1,9]
        args, model = remove_prototypes(args, protos2remove, model, use_cos=False)

    if 'add' in args.mode:
        protos2add = ['They offer a bad service.', 0]
        args, model, embedding_train, mask_train, text_train, labels_train, train_batches_unshuffled = \
            add_prototypes(args, protos2add, model, embedding_train, mask_train, text_train, labels_train)

    if 'soft' in args.mode:
        # sequence, position, class_id, certainty, embedding
        args.soft = ['They offer a bad service.', 8, 0, 1]
        args, model, embedding_train, mask_train, text_train, labels_train, train_batches_unshuffled = \
            soft_rplc_prototypes(args, args.soft, model, embedding_train, mask_train, text_train, labels_train)

    if 'replace' in args.mode:
        # sequence, position, class_id
        protos2replace = ['Very good written movie, strongly acted', 0, 0]
        args, model, embedding_train, mask_train, text_train, labels_train, train_batches_unshuffled = \
            replace_prototypes(args, protos2replace, model, embedding_train, mask_train, text_train, labels_train)

        model.protolayer.requires_grad = False

    if 'unique' in args.mode:
        
        # load information about all prototypes
        proto_info, proto_texts, _ = get_nearest(args, model, train_batches_unshuffled, text_train, labels_train)
        # sample duplicates in dictionary
        index = {}
        prots_to_remove = []
        for i, x in enumerate(proto_texts):
            if x not in index:
                index[x] = [i]
            else:
                index[x].append(i)
        # find median prototype for each duplicate and remove all but the closest one, if similarity is high enough
        for index_list_names, index_list in index.items():
            selected_tensors = [model.protolayer[:, i] for i in index_list]
            stacked_tensors = torch.stack(selected_tensors, dim=0)
            median_tensor = torch.median(stacked_tensors, dim=0)[0]
            similarites = [F.cosine_similarity(median_tensor, x) for x in selected_tensors]
            closest_index = similarites.index(max(similarites))
            prots_to_remove.extend([i for i in index_list if i != index_list[closest_index]]) #and F.cosine_similarity(median_tensor, model.protolayer[:, i]) > 0.9])
        # remove duplicates from model
        args, model = remove_prototypes(args, prots_to_remove, model, use_cos=False, use_weight=False)

    if 'reinitialize' in args.mode:
        protos2reinit = [0]
        model = reinit_prototypes(args, protos2reinit, model)

    if 'finetune' in args.mode:
        protos2finetune = [0]
        model = finetune_prototypes(args, protos2finetune, model)

    if 'prune' in args.mode:
        _, protos2prune, _ = get_nearest(args, model, train_batches_unshuffled, text_train, labels_train)
        args, model, embedding_train, mask_train, text_train, labels_train, train_batches_unshuffled = \
            prune_prototypes(args, protos2prune, model, embedding_train, mask_train, text_train, labels_train)

    
        
        

    # save changed model
    args.model_path = os.path.join(os.path.dirname(args.model_path), 'interacted_best_model.pth.tar')
    # retrain only retrain last layer (fc)
    args.num_epochs = 100
    args.project = False
    model = train(args, train_batches, val_batches, model, embedding_train, train_batches_unshuffled, text_train,
                  labels_train)
    test(args, embedding_train, mask_train, train_batches_unshuffled, test_batches, labels_train, text_train, model)
    return args, model, embedding_train, mask_train, text_train, labels_train, train_batches_unshuffled


def explain(args, embedding_test, mask_test, text_test, labels_test, model, train_batches_unshuffled, text_train,
            labels_train):
    print('\nExplain each test sample, loading model:', args.model_path)
    model.eval()

    # "convert" prototype embedding to text (take text of nearest training sample)
    _, proto_texts, _ = get_nearest(args, model, train_batches_unshuffled, text_train, labels_train)
    weights = model.get_proto_weights()
    explained_test_samples = []

    with torch.no_grad():
        values = [f'test sample \n', f'true label \n', f'predicted label \n', f'probability class 0 \n',
                  f'probability class 1 \n']
        for j in range(args.num_prototypes):
            values.append(f'explanation_{j + 1} \n')
            values.append(f'id_{j + 1} \n')
            values.append(f'similarity_{j + 1} \n')
            values.append(f'weight_{j + 1} \n')
            values.append(f'score_{j + 1} \n')

        explained_test_samples.append(values)

        for i in range(len(labels_test)):
            emb = embedding_test[i].to(f'cuda:{args.gpu[0]}').unsqueeze(0)
            mask = mask_test[i].to(f'cuda:{args.gpu[0]}').unsqueeze(0)
            prototype_distances, predicted_label = model.forward(emb, mask)
            predicted = torch.argmax(predicted_label).cpu().detach()
            probability = torch.nn.functional.softmax(predicted_label, dim=1).squeeze().tolist()
            similarity_score = prototype_distances.cpu().detach().squeeze() * weights[:, predicted]
            top_scores = similarity_score

            values = [''.join(text_test[i]) + '\n', f'{int(labels_test[i])}\n', f'{int(predicted)}\n',
                      f'{probability[0]:.3f}\n', f'{probability[1]:.3f}\n']
            for j in range(args.num_prototypes):
                idx = j
                nearest_proto = proto_texts[idx]
                values.append(f'{nearest_proto}\n')
                values.append(f'{idx + 1}\n')
                values.append(f'{float(-prototype_distances[:, idx]):.3f}\n')
                values.append(f'{float(-weights[idx, predicted]):.3f}\n')
                values.append(f'{float(top_scores[j]):.3f}\n')
            explained_test_samples.append(values)

    import csv
    save_path = os.path.join(os.path.dirname(args.model_path), 'explained' + args.pid + '.csv')
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(explained_test_samples)


def faithful(args, embedding_test, mask_test, text_test, labels_test, model, k=1):
    import pandas as pd
    data_explained = pd.read_csv(os.path.join(os.path.dirname(args.model_path), 'explained_normal.csv'))

    score = ['score_1 \n', 'score_2 \n', 'score_3 \n', 'score_4 \n', 'score_5 \n', 'score_6 \n', 'score_7 \n',
             'score_8 \n']  # , 'score_9 \n', 'score_10 \n']
    _, top_ids = torch.topk(torch.tensor(data_explained[score].to_numpy()), k=k)

    explained_test_samples = []
    values = []
    all_preds = []
    with torch.no_grad():
        values.append(f'test sample \n')
        values.append(f'true label \n')
        values.append(f'predicted label \n')
        values.append(f'probability class 0 \n')
        values.append(f'probability class 1 \n')
        explained_test_samples.append(values)
        for i in range(len(labels_test)):
            weights = model.fc.weight.detach().clone()
            tmp = weights.clone()
            weights[:, top_ids[i]] = torch.zeros(args.num_classes, 1).to(f'cuda:{args.gpu[0]}')
            model.fc.weight.copy_(weights)
            emb = embedding_test[i].to(f'cuda:{args.gpu[0]}').unsqueeze(0)
            mask = mask_test[i].to(f'cuda:{args.gpu[0]}').unsqueeze(0)
            prototype_distances, predicted_label = model.forward(emb, mask)
            model.fc.weight.copy_(tmp)
            predicted = torch.argmax(predicted_label).cpu().detach()
            probability = torch.nn.functional.softmax(predicted_label, dim=1).squeeze().tolist()
            all_preds += [predicted.cpu().detach().numpy().tolist()]

            values = [''.join(text_test[i]) + '\n', f'{int(labels_test[i])}\n', f'{int(predicted)}\n',
                      f'{probability[0]:.3f}\n', f'{probability[1]:.3f}\n']
            explained_test_samples.append(values)

    acc_test = balanced_accuracy_score(labels_test, all_preds)
    print(acc_test * 100)
    import csv
    save_path = os.path.join(os.path.dirname(args.model_path), 'compr_' + args.pid + '.csv')
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(explained_test_samples)


if __name__ == '__main__':
    # torch.manual_seed(0)
    # np.random.seed(0)
    torch.set_num_threads(6)
    args = parser.parse_args()

    rtpt = RTPT(name_initials='PK', experiment_name='Proto-Trex', max_iterations=args.num_epochs)
    rtpt.start()

    #if args.data_name == 'restaurant':
    #    preprocess_restaurant(args)
    #if args.data_name == 'jigsaw':
    #    preprocess_jigsaw(args)
    text_train, text_val, text_test, labels_train, labels_val, labels_test = load_data(args)

    # set class weights for balanced loss computation
    args.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_train), y=labels_train)

    if args.num_prototypes % args.num_classes:
        print('number of prototypes should be divisible by number of classes')
    #    args.num_prototypes -= args.num_prototypes % args.num_classes
    # define which prototype belongs to which class (onehot encoded matrix)
    args.prototype_class_identity = torch.eye(args.num_classes).repeat(args.num_prototypes // args.num_classes, 1
                                                                       ).to(f'cuda:{args.gpu[0]}')

    # compute class distribution once
    class_counts = Counter(labels_train)
    args.dataset_class_distribution = {i: class_counts[i] / len(labels_train) for i in range(args.num_classes)}
    args.dataset_class_distribution = torch.tensor(list(args.dataset_class_distribution.values()), dtype=torch.float32).unsqueeze(0)
    print("dataset_class_distribution:", args.dataset_class_distribution)

    model = []
    if args.level == 'word':
        model = ProtoTrexW(args)
    elif args.level == 'sentence':
        model = ProtoTrexS(args)

    print(f'Running on gpu {args.gpu}')
    model.to(f'cuda:{args.gpu[0]}')

    fname = args.language_model

    if not args.compute_emb:
        embedding_train, mask_train = load_embedding(args, fname, 'train')
        embedding_val, mask_val = load_embedding(args, fname, 'val')
        embedding_test, mask_test = load_embedding(args, fname, 'test')
    else:
        embedding_train, mask_train = model.compute_embedding(text_train, args)
        embedding_val, mask_val = model.compute_embedding(text_val, args)
        embedding_test, mask_test = model.compute_embedding(text_test, args)
        save_embedding(embedding_train, mask_train, args, fname, 'train')
        save_embedding(embedding_val, mask_val, args, fname, 'val')
        save_embedding(embedding_test, mask_test, args, fname, 'test')
        torch.cuda.empty_cache()  # free up language model from GPU

    if args.few_shot:
        idx = random.sample(range(len(text_train)), 100)
        text_train = list(text_train[i] for i in idx)
        labels_train = list(labels_train[i] for i in idx)
        embedding_train = embedding_train[idx, :]
        mask_train = mask_train[idx, :]

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

    time_stmp = datetime.datetime.now().strftime(f'%m-%d %H:%M:%S_{args.num_prototypes}_{fname}_{args.data_name}_{args.proto_size}_'
                                                 f'{args.attn}_{args.metric}_{args.pid}')
    args.model_path = os.path.join('./experiments/train_results/', time_stmp, 'best_model.pth.tar')

    if 'train' in args.mode:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        model = train(args, train_batches, val_batches, model, embedding_train, train_batches_unshuffled, text_train,
                      labels_train)
    if not os.path.exists(args.model_path):
        #load latest model path with given amount of prototypes if it exists
        model_paths = glob.glob(f'./experiments/train_results/*_{fname}_{args.data_name}_*/*best_model.pth.tar')
        model_paths.sort()
        args.model_path = model_paths[-1]
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'])
    if 'test' in args.mode:
        test(args, embedding_train, mask_train, train_batches_unshuffled, test_batches, labels_train, text_train, model)
    if 'query' in args.mode:
        query(args, train_batches_unshuffled, labels_train, text_train, model)
    if 'add' in args.mode or 'remove' in args.mode or 'finetune' in args.mode or 'reinitialize' in args.mode or 'prune' in args.mode or 'replace' in args.mode or 'soft' in args.mode or 'unique' in args.mode:
        args, model, embedding_train, mask_train, text_train, labels_train, train_batches_unshuffled = interact(args, train_batches, mask_train, train_batches_unshuffled, val_batches, embedding_train, test_batches, labels_train, text_train, model)
    if 'explain' in args.mode:
        explain(args, embedding_test, mask_test, text_test, labels_test, model, train_batches_unshuffled, text_train,
                labels_train)
    if 'faithful' in args.mode:
        faithful(args, embedding_test, mask_test, text_test, labels_test, model)
    if 'survey' in args.mode:
        survey(args, train_batches_unshuffled, labels_train, text_train, text_test, labels_test, model)
        
