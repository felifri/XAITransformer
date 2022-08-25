import argparse
import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from utils import load_data
import logging
from transformers import AdamW, BertTokenizer, GPT2Tokenizer, DistilBertTokenizer
from transformers import get_linear_schedule_with_warmup
from models import BertForSequenceClassification2Layers, GPT2ForSequenceClassification2Layers, \
    DistilBertForSequenceClassification2Layers
import os

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
parser.add_argument('--class_weights', default=[0.5, 0.5],
                    help='Class weight for cross entropy loss')
parser.add_argument('-g', '--gpu', type=int, default=[0], nargs='+',
                    help='GPU device number(s)')
parser.add_argument('--one_shot', type=bool, default=False,
                    help='Whether to use one-shot learning or not (i.e. only a few training examples)')
parser.add_argument('--discard', type=bool, default=False,
                    help='Whether edge cases (~0.5) in the middle between toxic(1) and not toxic(0) shall be omitted')
parser.add_argument('--language_model', type=str, default='Bert', choices=['Bert', 'SentBert', 'GPT2', 'TXL', 'Roberta',
                                                                           'DistilBert', 'Clip'],
                    help='Define which language model to use')


def _from_pretrained(cls, *args, **kw):
    """Load a transformers model in PyTorch, with fallback to TF2/Keras weights."""
    try:
        return cls.from_pretrained(*args, **kw)
    except OSError as e:
        logging.warning("Caught OSError loading model: %s", e)
        logging.warning(
            "Re-trying to convert from TensorFlow checkpoint (from_tf=True)")
        return cls.from_pretrained(*args, from_tf=True, **kw)


def train(args, text_train, labels_train, text_val, labels_val, text_test, labels_test):
    if args.language_model == 'Bert':
        model_name_or_path = 'bert-large-uncased'
        model = BertForSequenceClassification2Layers.from_pretrained(model_name_or_path, num_labels=args.num_classes)
        tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    elif args.language_model == 'GPT2':
        model_name_or_path = 'gpt2-xl'
        tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
        tokenizer.pad_token = '[PAD]'
        model = GPT2ForSequenceClassification2Layers.from_pretrained(model_name_or_path, num_labels=args.num_classes, pad_token_id=tokenizer.eos_token_id)
    elif args.language_model == 'DistilBert':
        model_name_or_path = 'distilbert-base-uncased'
        model = DistilBertForSequenceClassification2Layers.from_pretrained(model_name_or_path,
                                                                           num_labels=args.num_classes)
        tokenizer = DistilBertTokenizer.from_pretrained(model_name_or_path)
    
    model.train()
    model.to('cuda')
    for param in model.base_model.parameters():
        param.requires_grad = False

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_epochs = args.num_epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_epochs // 10, num_epochs)
    ce_crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(args.class_weights).float().to('cuda'))

    print("\nStarting training for {} epochs\n".format(num_epochs))
    best_acc = 0
    train_batches = torch.utils.data.DataLoader(list(zip(text_train, labels_train)), batch_size=args.batch_size,
                                                shuffle=True, pin_memory=True)
    val_batches = torch.utils.data.DataLoader(list(zip(text_val, labels_val)), batch_size=args.batch_size,
                                              shuffle=False, pin_memory=True)
    for epoch in tqdm(range(num_epochs)):
        losses_per_batch = []
        all_preds = []
        all_labels = []
        for text_batch, label_batch in train_batches:
            encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']

            input_ids = input_ids.to('cuda')
            attention_mask = attention_mask.to('cuda')
            label_batch = label_batch.to('cuda')

            outputs = model(input_ids, attention_mask=attention_mask, return_dict=True)
            loss = ce_crit(outputs.logits, label_batch)

            _, predicted = torch.max(outputs.logits.data, 1)
            all_preds += predicted.cpu().numpy().tolist()
            all_labels += label_batch.cpu().numpy().tolist()

            loss.backward()
            optimizer.step()
            # store losses
            losses_per_batch.append(float(loss))

        scheduler.step()
        mean_loss = np.mean(losses_per_batch)

        acc = balanced_accuracy_score(all_labels, all_preds)
        print("Epoch {}, mean loss {:.4f}, train acc {:.4f}".format(epoch + 1,
                                                                    mean_loss,
                                                                    100 * acc))

        if (epoch + 1) % args.val_epoch == 0 or epoch + 1 == num_epochs:
            model.eval()
            all_preds = []
            all_labels = []
            losses_per_batch = []
            with torch.no_grad():
                for text_batch, label_batch in val_batches:
                    encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
                    input_ids = encoding['input_ids']
                    attention_mask = encoding['attention_mask']

                    input_ids = input_ids.to('cuda')
                    attention_mask = attention_mask.to('cuda')
                    label_batch = label_batch.to('cuda')

                    outputs = model(input_ids, attention_mask=attention_mask, return_dict=True)
                    loss = ce_crit(outputs.logits, label_batch)

                    _, predicted = torch.max(outputs.logits.data, 1)
                    all_preds += predicted.cpu().numpy().tolist()
                    all_labels += label_batch.cpu().numpy().tolist()

                    # store losses
                    losses_per_batch.append(float(loss))

                loss = np.mean(losses_per_batch)
                acc_val = balanced_accuracy_score(all_labels, all_preds)
                print(f"test evaluation on best model: loss {loss:.4f}, acc_val {100 * acc_val:.3f}")

                if acc_val > best_acc:
                    best_acc = acc_val
                    best_model = model.state_dict()

    model.load_state_dict(best_model)
    model.eval()

    all_preds = []
    all_labels = []
    losses_per_batch = []
    test_batches = torch.utils.data.DataLoader(list(zip(text_test, labels_test)), batch_size=args.batch_size,
                                               shuffle=False, pin_memory=True, num_workers=0)  # , drop_last=True)
    with torch.no_grad():
        for text_batch, label_batch in test_batches:
            encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']

            input_ids = input_ids.to('cuda')
            attention_mask = attention_mask.to('cuda')
            label_batch = label_batch.to('cuda')

            outputs = model(input_ids, attention_mask=attention_mask, return_dict=True)
            loss = ce_crit(outputs.logits, label_batch)

            _, predicted = torch.max(outputs.logits.data, 1)
            all_preds += predicted.cpu().numpy().tolist()
            all_labels += label_batch.cpu().numpy().tolist()

            # store losses
            losses_per_batch.append(float(loss))

        loss = np.mean(losses_per_batch)
        acc_test = balanced_accuracy_score(all_labels, all_preds)
        print(f"test evaluation on best model: loss {loss:.3f}, acc_test {100 * acc_test:.3f}")

    save_path = f"./trained_{args.language_model}_BaseClassifier/{args.data_name}/model.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    # torch.manual_seed(0)
    # np.random.seed(0)
    # torch.set_num_threads(6)
    args = parser.parse_args()

    text_train, text_val, text_test, labels_train, labels_val, labels_test = load_data(args)

    # set class weights for balanced loss computation
    args.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_train), y=labels_train)

    train(args, text_train, labels_train, text_val, labels_val, text_test, labels_test)
