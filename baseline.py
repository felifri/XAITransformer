import argparse
import torch
import numpy as np
import os
import datetime
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
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
parser.add_argument('--avoid_pad_token', type=bool, default=False,
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
    if args.avoid_pad_token:
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

    return model


def compute_faithfulness(args, model):
    import pickle
    args.data_name = 'movie_review'
    set_dir = os.path.join(args.data_dir, args.data_name)
    labels = pickle.load(open(set_dir + '/labels.pkl', 'rb'))
    review = pickle.load(open(set_dir + '/reviews.pkl', 'rb'))
    rationale = pickle.load(open(set_dir + '/rationale.pkl', 'rb'))
    review_wo_rationale = pickle.load(open(set_dir + '/reviews_wo_rationale.pkl', 'rb'))

    if not args.compute_emb:
        embedding_review, mask_review = load_embedding(args, fname, 'review' + avoid)
        embedding_rationale, mask_rationale = load_embedding(args, fname, 'rationale' + avoid)
        embedding_review_wo_rationale, mask_review_wo_ratinoale = load_embedding(args, fname, 'review_wo_rationale' + avoid)
    else:
        embedding_review, mask_review = model.compute_embedding(review, args)
        embedding_rationale, mask_rationale = model.compute_embedding(rationale, args)
        embedding_review_wo_rationale, mask_review_wo_ratinoale = model.compute_embedding(review_wo_rationale, args)
        save_embedding(embedding_review, mask_review, args, fname, 'review' + avoid)
        save_embedding(embedding_rationale, mask_rationale, args, fname, 'rationale' + avoid)
        save_embedding(embedding_review_wo_rationale, mask_review_wo_ratinoale, args, fname, 'review_wo_rationale' + avoid)
        torch.cuda.empty_cache()  # free up language model from GPU

    with torch.no_grad():
        evaluated_test_samples = []
        values = []
        values.append(f'review \n')
        values.append(f'rationale \n')
        values.append(f'review_wo_rationale \n')
        values.append(f'true label \n')
        values.append(f'predicted_review \n')
        values.append(f'probability_review0 \n')
        values.append(f'probability_review1 \n')
        values.append(f'predicted_rationale \n')
        values.append(f'probability_rationale0 \n')
        values.append(f'probability_rationale1 \n')
        values.append(f'predicted_review_wo_rat \n')
        values.append(f'probability_review_wo_rat0 \n')
        values.append(f'probability_review_wo_rat1 \n')
        values.append(f'comprehensiveness \n')
        values.append(f'sufficiency \n')
        evaluated_test_samples.append(values)

        comp, suff = 0, 0
        for i in range(len(labels)):
            emb_review = embedding_review[i].to(f'cuda:{args.gpu[0]}').unsqueeze(0)
            emb_rationale = embedding_rationale[i].to(f'cuda:{args.gpu[0]}').unsqueeze(0)
            emb_review_wo_rationale = embedding_review_wo_rationale[i].to(f'cuda:{args.gpu[0]}').unsqueeze(0)
            mask_rev = mask_review[i].to(f'cuda:{args.gpu[0]}').unsqueeze(0)
            mask_rat = mask_rationale[i].to(f'cuda:{args.gpu[0]}').unsqueeze(0)
            mask_review_wo_rat = mask_review_wo_ratinoale[i].to(f'cuda:{args.gpu[0]}').unsqueeze(0)
            predicted_label_review = model.forward(emb_review, mask_rev)
            predicted_label_rationale = model.forward(emb_rationale, mask_rat)
            predicted_label_review_wo_ratinoale = model.forward(emb_review_wo_rationale, mask_review_wo_rat)

            predicted_review = torch.argmax(predicted_label_review).cpu().detach()
            probability_review = torch.nn.functional.softmax(predicted_label_review, dim=1).squeeze().tolist()
            predicted_rationale = torch.argmax(predicted_label_rationale).cpu().detach()
            probability_rationale = torch.nn.functional.softmax(predicted_label_rationale, dim=1).squeeze().tolist()
            predicted_review_wo_ratinoale = torch.argmax(predicted_label_review_wo_ratinoale).cpu().detach()
            probability_review_wo_ratinoale = torch.nn.functional.softmax(predicted_label_review_wo_ratinoale, dim=1).squeeze().tolist()

            comprehensiveness = probability_review[labels[i]] - probability_review_wo_ratinoale[labels[i]]
            sufficiency = probability_review[labels[i]] - probability_rationale[labels[i]]
            comp += comprehensiveness
            suff += sufficiency
            values = []
            values.append(''.join(review[i]) + '\n')
            values.append(''.join(rationale[i]) + '\n')
            values.append(''.join(review_wo_rationale[i]) + '\n')
            values.append(f'{int(labels[i])}\n')
            values.append(f'{int(predicted_review)}\n')
            values.append(f'{probability_review[0]:.3f}\n')
            values.append(f'{probability_review[1]:.3f}\n')
            values.append(f'{int(predicted_rationale)}\n')
            values.append(f'{probability_rationale[0]:.3f}\n')
            values.append(f'{probability_rationale[1]:.3f}\n')
            values.append(f'{int(predicted_review_wo_ratinoale)}\n')
            values.append(f'{probability_review_wo_ratinoale[0]:.3f}\n')
            values.append(f'{probability_review_wo_ratinoale[1]:.3f}\n')
            values.append(f'{comprehensiveness:.3f}\n')
            values.append(f'{sufficiency:.3f}\n')
            evaluated_test_samples.append(values)

    print(f'comp: {comp/len(labels)}')
    print(f'suff: {suff/len(labels)}')
    import csv

    save_path = os.path.join(os.path.dirname(args.model_path), 'faithfulness_baseline.csv')
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(evaluated_test_samples)


if __name__ == '__main__':
    # torch.manual_seed(0)
    # np.random.seed(0)
    torch.set_num_threads(6)
    args = parser.parse_args()

    # Create RTPT object and start the RTPT tracking
    rtpt = RTPT(name_initials='FF', experiment_name='Proto-Trex', max_iterations=args.num_epochs)
    rtpt.start()

    fname = args.language_model
    avoid = ''
    if args.avoid_pad_token:
        avoid = '_avoid'

    time_stmp = datetime.datetime.now().strftime(f'%m-%d %H:%M_{args.num_prototypes}_{fname}_{args.data_name}')
    args.model_path = os.path.join('./experiments/train_results/', time_stmp, 'best_model.pth.tar')
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    text_train, text_val, text_test, labels_train, labels_val, labels_test = load_data(args)

    # set class weights for balanced loss computation
    args.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_train), y=labels_train)

    model = train(args, text_train, labels_train, text_val, labels_val, text_test, labels_test)
    compute_faithfulness(args, model)



def compute_differences(k=1):
    pth = './experiments/train_results/09-13 09:53_10_SentBert_1_False_cosine_mov_1/'
    data_normal = pd.read_csv(pth + 'explained_normal.csv')
    data_rat = pd.read_csv(pth + 'explained_rationale.csv')
    data_wo_rat = pd.read_csv(pth + 'explained_wo_rationale.csv')
    # data_wo_rat = pd.read_csv(pth + 'compr_.csv')
    data_inter = pd.read_csv(pth + 'explained_interacted.csv')

    __import__("pdb").set_trace()
    inter = np.mean(data_normal['probability class 0 \n'].to_numpy() - data_inter['probability class 0 \n'].to_numpy())
    comp = np.mean(data_normal['probability class 0 \n'].to_numpy() - data_wo_rat['probability class 0 \n'].to_numpy())
    suff = np.mean(data_normal['probability class 0 \n'].to_numpy() - data_rat['probability class 0 \n'].to_numpy())

    print(f'norm - inter: {inter:.3f}')
    print(f'comp norm - wo_rat: {comp:.3f}')
    print(f'suff norm - rat: {suff:.3f}')

    sim = ['similarity_1 \n', 'similarity_2 \n', 'similarity_3 \n', 'similarity_4 \n', 'similarity_5 \n',
           'similarity_6 \n', 'similarity_7 \n', 'similarity_8 \n', 'similarity_9 \n', 'similarity_10 \n']
    w = ['weight_1 \n', 'weight_2 \n', 'weight_3 \n', 'weight_4 \n', 'weight_5 \n', 'weight_6 \n', 'weight_7 \n',
         'weight_8 \n', 'weight_9 \n', 'weight_10 \n']
    score = ['score_1 \n', 'score_2 \n', 'score_3 \n', 'score_4 \n', 'score_5 \n', 'score_6 \n', 'score_7 \n',
             'score_8 \n', 'score_9 \n', 'score_10 \n']

    top_scores, top_ids = torch.topk(torch.tensor(data_normal[score].to_numpy()), k=k)

    # needed to allow double indexing
    n = torch.arange(data_normal.shape[0]).unsqueeze(-1)

    diff1 = np.mean(data_normal[sim].to_numpy()[n, top_ids] - data_rat[sim].to_numpy()[n, top_ids])
    diff11 = np.mean(data_normal[w].to_numpy()[n, top_ids] - data_rat[w].to_numpy()[n, top_ids])
    diff12 = np.mean(top_scores.numpy() - data_rat[score].to_numpy()[n, top_ids])

    diff2 = np.mean(data_normal[sim].to_numpy()[n, top_ids] - data_wo_rat[sim].to_numpy()[n, top_ids])
    diff21 = np.mean(data_normal[w].to_numpy()[n, top_ids] - data_wo_rat[w].to_numpy()[n, top_ids])
    diff22 = np.mean(top_scores.numpy() - data_wo_rat[score].to_numpy()[n, top_ids])

    diff3 = np.mean(data_normal[sim].to_numpy()[n, top_ids] - data_inter[sim].to_numpy()[n, top_ids])
    diff31 = np.mean(data_normal[w].to_numpy()[n, top_ids] - data_inter[w].to_numpy()[n, top_ids])
    diff32 = np.mean(top_scores.numpy() - data_inter[score].to_numpy()[n, top_ids])

    print(f'difference distance norm - rat: {diff1:.3f} ({diff1 / np.mean(data_normal[sim].to_numpy()[n, top_ids] * 0.01):.1f})')
    print(f'difference weight norm - rat: {diff11:.3f} ({diff11 / np.mean(data_normal[w].to_numpy()[n, top_ids] * 0.01):.1f})')
    print(f'difference score norm - rat: {diff12:.3f} ({diff12 / np.mean(data_normal[score].to_numpy()[n, top_ids] * 0.01):.1f})')

    print(f'difference distance norm - wo_rat: {diff2:.3f} ({diff2 / np.mean(data_normal[sim].to_numpy()[n, top_ids] * 0.01):.1f})')
    print(f'difference weight norm - wo_rat: {diff21:.3f} ({diff21 / np.mean(data_normal[w].to_numpy()[n, top_ids] * 0.01):.1f})')
    print(f'difference score norm - wo_rat: {diff22:.3f} ({diff22 / np.mean(data_normal[score].to_numpy()[n, top_ids] * 0.01):.1f})')

    print(f'difference distance norm - inter: {diff3:.3f} ({diff3 / np.mean(data_normal[sim].to_numpy()[n, top_ids] * 0.01):.1f})')
    print(f'difference weight norm - inter: {diff31:.3f} ({diff31 / np.mean(data_normal[w].to_numpy()[n, top_ids] * 0.01):.1f})')
    print(f'difference score norm - inter: {diff32:.3f} ({diff32 / np.mean(data_normal[score].to_numpy()[n, top_ids] * 0.01):.1f})')
