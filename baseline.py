import argparse
import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from rtpt import RTPT
from models import BaseNet
from utils import save_embedding, load_embedding, load_data
from transformers import get_linear_schedule_with_warmup

parser = argparse.ArgumentParser(description='Crazy Stuff')
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
parser.add_argument('--num_classes', default=2, type=int,
                    help='How many classes are to be classified?')
parser.add_argument('--class_weights', default=[0.5,0.5],
                    help='Class weight for cross entropy loss')
parser.add_argument('-g','--gpu', type=int, default=[0], nargs='+',
                    help='GPU device number(s)')
parser.add_argument('--one_shot', type=bool, default=False,
                    help='Whether to use one-shot learning or not (i.e. only a few training examples)')
parser.add_argument('--discard', type=bool, default=False,
                    help='Whether edge cases in the middle between completely toxic(1) and not toxic(0) shall be omitted')
parser.add_argument('--proto_size', type=int, default=1,
                    help='Define how many words should be used to define a prototype')
parser.add_argument('--language_model', type=str, default='Bert', choices=['Bert','SentBert','GPT2'],
                    help='Define which language model to use')
parser.add_argument('--avoid_spec_token', type=bool, default=False,
                    help='Whether to manually set PAD, SEP and CLS token to high value after Bert embedding computation')
parser.add_argument('--compute_emb', type=bool, default=False,
                    help='Whether to recompute (True) the embedding or just load it (False)')
parser.add_argument('--metric', type=str, default='L2',
                    help='metric')

def train(args, text_train, labels_train, text_val, labels_val, text_test, labels_test):
    fname = args.language_model
    model = BaseNet(args)

    print("Running on gpu {}".format(args.gpu))
    model.to(f'cuda:{args.gpu[0]}')

    avoid = ''
    if args.avoid_spec_token:
        avoid = '_avoid'

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

    num_epochs = args.num_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce_crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(args.class_weights).float().to(f'cuda:{args.gpu[0]}'))
    scheduler = get_linear_schedule_with_warmup(optimizer, min(10, num_epochs // 20), num_epochs)

    model.train()
    print(f"\nStarting training for {num_epochs} epochs\n")
    best_acc = 0
    train_batches = torch.utils.data.DataLoader(list(zip(embedding_train, labels_train)), batch_size=args.batch_size,
                                                shuffle=True, pin_memory=True, num_workers=0)#, drop_last=True)
    val_batches = torch.utils.data.DataLoader(list(zip(embedding_val, labels_val)), batch_size=args.batch_size,
                                              shuffle=False, pin_memory=True, num_workers=0)#, drop_last=True)
    for epoch in tqdm(range(num_epochs)):
        all_preds ,all_labels = [], []
        losses_per_batch = []
        # Update the RTPT
        rtpt.step()

        for emb_batch, label_batch in train_batches:
            emb_batch = emb_batch.to(f'cuda:{args.gpu[0]}')
            label_batch = label_batch.to(f'cuda:{args.gpu[0]}')

            optimizer.zero_grad()
            predicted_label = model.forward(emb_batch, [])

            # compute individual losses and backward step
            loss = ce_crit(predicted_label, label_batch)
            _, predicted = torch.max(predicted_label.data, 1)
            all_preds += predicted.cpu().numpy().tolist()
            all_labels += label_batch.cpu().numpy().tolist()

            loss.backward()
            optimizer.step()
            # store losses
            losses_per_batch.append(float(loss))

        scheduler.step()
        mean_loss = np.mean(losses_per_batch)
        acc = balanced_accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}, mean loss {mean_loss:.3f}, train acc {100 * acc:.4f}")

        if (epoch + 1) % args.val_epoch == 0 or epoch + 1 == num_epochs:
            model.eval()
            all_preds = []
            all_labels = []
            losses_per_batch = []
            with torch.no_grad():
                for emb_batch, label_batch in val_batches:
                    emb_batch = emb_batch.to(f'cuda:{args.gpu[0]}')
                    label_batch = label_batch.to(f'cuda:{args.gpu[0]}')
                    predicted_label = model.forward(emb_batch, [])

                    # compute individual losses and backward step
                    loss = ce_crit(predicted_label, label_batch)
                    losses_per_batch.append(float(loss))
                    _, predicted = torch.max(predicted_label.data, 1)
                    all_preds += predicted.cpu().numpy().tolist()
                    all_labels += label_batch.cpu().numpy().tolist()

                loss_val = np.mean(losses_per_batch)
                acc_val = balanced_accuracy_score(all_labels, all_preds)
                print(f"Validation: mean loss {loss_val:.3f}, acc_val {100 * acc_val:.3f}")

                if acc_val > best_acc:
                    best_acc = acc_val
                    best_model = model.state_dict()

    model.load_state_dict(best_model)
    model.eval()

    all_preds = []
    all_labels = []
    losses_per_batch = []
    test_batches = torch.utils.data.DataLoader(list(zip(embedding_test, labels_test)), batch_size=args.batch_size,
                                               shuffle=False, pin_memory=True, num_workers=0)#, drop_last=True)
    with torch.no_grad():
        for emb_batch, label_batch in test_batches:
            emb_batch = emb_batch.to(f'cuda:{args.gpu[0]}')
            label_batch = label_batch.to(f'cuda:{args.gpu[0]}')
            predicted_label = model.forward(emb_batch, [])
            loss = ce_crit(predicted_label, label_batch)

            losses_per_batch.append(float(loss))
            _, predicted = torch.max(predicted_label.data, 1)
            all_preds += predicted.cpu().numpy().tolist()
            all_labels += label_batch.cpu().numpy().tolist()

        loss = np.mean(losses_per_batch)
        acc_test = balanced_accuracy_score(all_labels, all_preds)
        print(f"test evaluation on best model: loss {loss:.3f}, acc_test {100 * acc_test:.3f}")

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

    train(args, text_train, labels_train, text_val, labels_val, text_test, labels_test)
