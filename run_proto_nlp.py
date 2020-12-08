import numpy as np
import os
import torch
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random
import datetime
import glob
import sys
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

try:
    from rtpt.rtpt import RTPT
except:
    sys.path.append('../rtpt')
    from rtpt import RTPT

from models import ProtopNetNLP
import data_loader

# Create RTPT object
rtpt = RTPT(name_initials='FF', experiment_name='Transformer_Prototype', max_iterations=100)
# Start the RTPT tracking
rtpt.start()

parser = argparse.ArgumentParser(description='Crazy Stuff')
parser.add_argument('-m', '--mode', default="normal", type=str,
                    help='What do you want to do? Select either normal, train, test,')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate')
parser.add_argument('--cpu', action='store_true', default=False,
                    help='Whether to use cpu')
parser.add_argument('-e', '--num_epochs', default=100, type=int,
                    help='How many epochs?')
parser.add_argument('-bs', '--batch_size', default=128, type=int,
                    help='Batch size')
parser.add_argument('--val_epoch', default=10, type=int,
                    help='After how many epochs should the model be evaluated on the validation data?')
parser.add_argument('--data_dir', default='./data/rt-polarity',
                    help='Select data path')
parser.add_argument('--data_name', default='reviews', type=str, choices=['reviews', 'toxicity'],
                    help='Select data name')
parser.add_argument('--num_prototypes', default=10, type = int,
                    help='Total number of prototypes')
parser.add_argument('-l2','--lambda2', default=0.1, type=float,
                    help='Weight for prototype loss computation')
parser.add_argument('-l3','--lambda3', default=0.1, type=float,
                    help='Weight for prototype loss computation')
parser.add_argument('--num_classes', default=2, type=int,
                    help='How many classes are to be classified?')
parser.add_argument('--class_weights', default=[0.5,0.5],
                    help='Class weight for cross entropy loss')
parser.add_argument('-g','--gpu', type=int, default=0, help='GPU device number, -1  means CPU.')
parser.add_argument('--one_shot', type=bool, default=False,
                    help='Whether to use one-shot learning or not (i.e. only a few training examples)')
parser.add_argument('--trans_type', type=str, default='PCA', choices=['PCA', 'TSNE'],
                    help='Which transformation should be used to visualize the prototypes')
parser.add_argument('--discard', type=bool, default=False, help='Whether edge cases in the middle between completely '
                                                                'toxic (1) and not toxic at all (0) shall be omitted')


def get_batches(embedding, labels, batch_size=128):
    def divide_chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]
    tmp = list(zip(embedding, labels))
    random.shuffle(tmp)
    embedding, labels = zip(*tmp)
    embedding_batches = list(divide_chunks(torch.stack(embedding), batch_size))
    label_batches = list(divide_chunks(torch.stack(labels), batch_size))
    return embedding_batches, label_batches

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

def save_checkpoint(save_dir, state, time_stmp, best, filename='best_model.pth.tar'):
    if best:
        save_path_checkpoint = os.path.join(save_dir, time_stmp, filename)
        os.makedirs(os.path.dirname(save_path_checkpoint), exist_ok=True)
        torch.save(state, save_path_checkpoint)

def split_data(text, labels):
    split = [0.7,0.1,0.2]
    idx = [round(elem*len(text)) for elem in split]
    idx = np.cumsum(idx)
    tmp = list(zip(text, labels))
    random.shuffle(tmp)
    text, labels = zip(*tmp)
    labels = torch.stack(labels)
    text_train = text[:idx[0]]
    labels_train = labels[:idx[0]]
    text_val = text[idx[0]:idx[1]]
    labels_val = labels[idx[0]:idx[1]]
    text_test = text[idx[1]:idx[2]]
    labels_test = labels[idx[1]:idx[2]]
    return text_train, labels_train, text_val, labels_val, text_test, labels_test


def train(args, text_train, labels_train, text_val, labels_val):
    save_dir = "./experiments/train_results/"
    time_stmp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    model = ProtopNetNLP(args)

    num_params = sum(p.numel() for p in model.parameters())
    print("Number of parameters {}".format(num_params))
    print("Running on gpu {}".format(args.gpu))
    model.cuda(args.gpu)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce_crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(args.class_weights).float().cuda(args.gpu))
    interp_criteria = ProtoLoss()

    model.train()
    embedding = model.compute_embedding(text_train, args.gpu)
    embedding_val = model.compute_embedding(text_val, args.gpu)
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
        emb_batches, label_batches = get_batches(embedding, labels_train, args.batch_size)

        # Update the RTPT
        rtpt.step(subtitle=f"epoch={epoch}")

        for i,(emb_batch, label_batch) in enumerate(zip(emb_batches, label_batches)):
            optimizer.zero_grad()

            outputs = model.forward(emb_batch)
            prototype_distances, feature_vector_distances, predicted_label = outputs

            # compute individual losses and backward step
            ce_loss = ce_crit(predicted_label, label_batch)
            r1_loss, r2_loss = interp_criteria(feature_vector_distances, prototype_distances)
            loss = ce_loss + \
                   args.lambda2 * r1_loss + \
                   args.lambda3 * r2_loss

            _, predicted = torch.max(predicted_label.data, 1)
            all_preds += predicted.cpu().numpy().tolist()
            all_labels += label_batch.cpu().numpy().tolist()

            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
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
        acc = accuracy_score(all_labels, all_preds)
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
                loss = ce_loss + \
                       args.lambda2 * r1_loss + \
                       args.lambda3 * r2_loss

                _, predicted_val = torch.max(predicted_label.data, 1)
                acc_val = accuracy_score(labels_val.cpu().numpy(), predicted_val.cpu().numpy())
                print("Validation: mean loss {:.4f}, acc_val {:.4f}".format(loss, 100 * acc_val))

            save_checkpoint(save_dir, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
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
    # test_dir = "./experiments/test_results/"

    model = ProtopNetNLP(args)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda(args.gpu)
    print(model.get_proto_weights())
    model.eval()
    ce_crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(args.class_weights).float().cuda(args.gpu))
    interp_criteria = ProtoLoss()

    embedding = model.compute_embedding(text_train, args.gpu)
    embedding_test = model.compute_embedding(text_test, args.gpu)

    with torch.no_grad():
        outputs = model.forward(embedding_test)
        prototype_distances, feature_vector_distances, predicted_label = outputs

        # compute individual losses and backward step
        ce_loss = ce_crit(predicted_label, labels_test)
        r1_loss, r2_loss = interp_criteria(feature_vector_distances, prototype_distances)
        loss = ce_loss + \
               args.lambda2 * r1_loss + \
               args.lambda3 * r2_loss

        _, predicted = torch.max(predicted_label.data, 1)
        acc_test = accuracy_score(labels_test.cpu().numpy(), predicted.cpu().numpy())
        print(f"test evaluation on best model: loss {loss:.4f}, acc_test {100 * acc_test:.4f}")

        # get prototypes
        prototypes = model.get_protos()
        # "convert" prototype embedding to text (take text of nearest training sample)
        nearest_ids = nearest_neighbors(embedding, prototypes)
        proto_texts = [[index, text[index]] for index in nearest_ids]

        save_path = os.path.join(os.path.dirname(model_path), "prototypes.txt")
        #os.makedirs(os.path.dirname(save_path), exist_ok=True)
        txt_file = open(save_path, "w+")
        for line in proto_texts:
            txt_file.write(str(line))
            txt_file.write("\n")
        txt_file.close()

        embedding = embedding.cpu().numpy()
        prototypes = prototypes.cpu().numpy()
        labels_train = labels_train.cpu().numpy()
        visualize_protos(embedding, labels_train, prototypes, n_components=2, type=args.trans_type, save_path=os.path.dirname(save_path))
        # visualize_protos(embedding, labels_train, prototypes, n_components=3, type=args.trans_type, save_path=os.path.dirname(save_path))


def visualize_protos(embedding, labels, prototypes, n_components, type, save_path):
        # visualize prototypes
        if type == 'PCA':
            pca = PCA(n_components=n_components)
            pca.fit(embedding)
            print("Explained variance ratio of components after transform: ", pca.explained_variance_ratio_)
            embed_trans = pca.transform(embedding)
            proto_trans = pca.transform(prototypes)
        elif type == 'TSNE':
            tsne = TSNE(n_components=n_components).fit_transform(np.vstack((embedding,prototypes)))
            [embed_trans, proto_trans] = np.split(tsne, len(embedding))

        rnd_samples = np.random.randint(embed_trans.shape[0], size=500)
        rnd_labels = labels[rnd_samples]
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
        fig.savefig(os.path.join(save_path, 'proto_vis'+str(n_components)+'d.png'))

def nearest_neighbors(text_embedded, prototypes):
    distances = torch.cdist(text_embedded, prototypes, p=2) # shape: num_samples x num_prototypes
    nearest_ids = torch.argmin(distances, dim=0)
    return nearest_ids.cpu().numpy()


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    args = parser.parse_args()

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    text, labels = data_loader.load_data(args)
    labels = torch.LongTensor(labels).cuda(args.gpu)
    text_train, labels_train, text_val, labels_val, text_test, labels_test = split_data(text, labels)

    if args.one_shot:
        idx = random.sample(range(len(text_train)),100)
        text_train = list(text_train[i] for i in idx)
        labels_train = torch.LongTensor([labels_train[i] for i in idx]).cuda(args.gpu)

    if args.mode == 'normal':
        train(args, text_train, labels_train, text_val, labels_val)
        test(args, text_train, labels_train, text_test, labels_test)
    elif args.mode == 'train':
        train(args, text_train, labels_train, text_val, labels_val)
    elif args.mode == 'test':
        test(args, text_train, labels_train, text_test, labels_test)
