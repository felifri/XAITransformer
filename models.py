import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model, DistilBertTokenizer, DistilBertModel
from transformers import BertForSequenceClassification, GPT2ForSequenceClassification, \
    DistilBertForSequenceClassification, \
    RobertaTokenizer, RobertaModel
import transformers
import logging
from utils import nes_torch
import clip


class ProtoTrexS(nn.Module):
    def __init__(self, args):
        super(ProtoTrexS, self).__init__()
        self.num_prototypes = args.num_prototypes
        if args.language_model == 'SentBert':
            self.enc_size = 1024
        elif args.language_model == 'Clip':
            self.enc_size = 512
        self.metric = args.metric
        self.protolayer = nn.Parameter(nn.init.uniform_(torch.empty(1, args.num_prototypes, self.enc_size)),
                                       requires_grad=True)
        self.fc = nn.Linear(args.num_prototypes, args.num_classes, bias=False)

    def forward(self, embedding, _):
        prototype_distances = self.compute_distance(embedding)
        class_out = self.fc(prototype_distances)
        return prototype_distances, class_out

    def compute_distance(self, embedding):
        if self.metric == 'L2':
            # prototype_distances = torch.cdist(embedding.float(), self.protolayer.squeeze(), p=2)
            prototype_distances = - nes_torch(embedding.unsqueeze(1), self.protolayer, dim=-1)
        elif self.metric == 'cosine':
            prototype_distances = - F.cosine_similarity(embedding.unsqueeze(1), self.protolayer, dim=-1)
        return prototype_distances

    def get_dist(self, embedding, _):
        distances = self.compute_distance(embedding)
        return distances, []

    def get_protos(self):
        return self.protolayer.squeeze()

    def get_proto_weights(self):
        return self.fc.weight.T.cpu().detach().numpy()

    @staticmethod
    def compute_embedding(x, args, max_l=False):
        if args.language_model == 'SentBert':
            LM = SentenceTransformer('bert-large-nli-mean-tokens', device=args.gpu[0])
            embedding = LM.encode(x, convert_to_tensor=True, device=args.gpu[0]).cpu().detach()
        elif args.language_model == 'Clip':
            LM, preprocess = clip.load('ViT-B/32', f'cuda:{args.gpu[0]}')
            # x = preprocess(x).unsqueeze(0)  # in case of image as input
            x = clip.tokenize(x)
            batches = torch.utils.data.DataLoader(x, batch_size=200, shuffle=False)
            embedding = []
            for batch in batches:
                batch = batch.to(f'cuda:{args.gpu[0]}')
                output = LM.encode_text(batch)
                # output = LM.encode_image(batch)
                embedding.append(output.cpu().detach().float())
            embedding = torch.cat(embedding, dim=0)

        for param in LM.parameters():
            param.requires_grad = False
        if len(embedding.size()) == 1:
            embedding = embedding.unsqueeze(0).unsqueeze(0)
        mask = torch.ones(embedding.shape)  # required for attention models
        return embedding, mask

    @staticmethod
    def nearest_neighbors(distances, _, text_train, labels_train):
        distances = torch.cat(distances)
        _, nearest_ids = torch.topk(distances, 1, dim=0, largest=False)
        nearest_ids = nearest_ids.cpu().detach().numpy().T
        proto_id = [f'P{proto + 1} | sentence {index} | label {labels_train[index]} | text: ' for proto, sent
                    in enumerate(nearest_ids) for index in sent]
        proto_texts = [f'{text_train[index]}' for sent in nearest_ids for index in sent]
        return proto_id, proto_texts, [nearest_ids, []]


class BaseNet(ProtoTrexS):
    def __init__(self, args):
        super(BaseNet, self).__init__(args)
        self.fc = nn.Sequential(
            nn.Linear(self.enc_size, args.num_prototypes),
            nn.Dropout(p=0.5),
            nn.Linear(args.num_prototypes, args.num_classes, bias=False),
        )
        del self.protolayer

    def forward(self, embedding, _):
        return self.fc(embedding)


class ProtoTrexW(nn.Module):
    def __init__(self, args):
        super(ProtoTrexW, self).__init__()
        if args.language_model == 'Bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
            self.enc_size = 1024
        elif args.language_model == 'GPT2':
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
            self.tokenizer.pad_token = '[PAD]'
            self.enc_size = 1600
        elif args.language_model == 'Roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli')
            self.enc_size = 1024
        elif args.language_model == 'DistilBert':
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.enc_size = 768
        self.proto_size = args.proto_size
        self.num_prototypes = args.num_prototypes
        self.metric = args.metric
        self.fc = nn.Linear(args.num_prototypes, args.num_classes, bias=False)
        self.protolayer = nn.Parameter(
            nn.init.uniform_(torch.empty((1, self.num_prototypes, self.enc_size, self.proto_size))),
            requires_grad=True)
        self.attention = nn.MultiheadAttention(self.enc_size, num_heads=1)
        self.slots = min(9, self.proto_size * 2)
        self.dilated = args.dilated
        self.attn = args.attn
        self.num_filters = [self.num_prototypes // len(self.dilated)] * len(self.dilated)
        self.num_filters[0] += self.num_prototypes % len(self.dilated)

    def forward(self, embedding, mask):
        if self.attn:
            embedding, _ = self.compute_attention(embedding, mask)
        distances = self.compute_distance(embedding, mask)
        prototype_distances = torch.cat([torch.min(dist, dim=2)[0] for dist in distances], dim=1)
        class_out = self.fc(prototype_distances)
        return prototype_distances, class_out

    def get_proto_weights(self):
        return self.fc.weight.T.cpu().detach().numpy()

    def get_protos(self):
        return self.protolayer

    def compute_embedding(self, x, args, max_l=False):
        if args.language_model == 'Bert':
            LM = BertModel.from_pretrained('bert-large-uncased').to(f'cuda:{args.gpu[0]}')
        elif args.language_model == 'GPT2':
            LM = GPT2Model.from_pretrained('gpt2-xl').to(f'cuda:{args.gpu[0]}')
        elif args.language_model == 'Roberta':
            LM = RobertaModel.from_pretrained('roberta-large-mnli').to(f'cuda:{args.gpu[0]}')
        elif args.language_model == 'DistilBert':
            LM = DistilBertModel.from_pretrained('distilbert-base-uncased').to(f'cuda:{args.gpu[0]}')
        self.enc_size = LM.config.hidden_size
        for param in LM.parameters():
            param.requires_grad = False

        bs = 10  # divide data by a batch size if too big for memory to process embedding at once
        word_embedding = []
        if not max_l:
            inputs = self.tokenizer(x, return_tensors='pt', padding=True, add_special_tokens=False)
        elif max_l:
            inputs = self.tokenizer(x, return_tensors='pt', padding='max_length', max_length=max_l,
                                    add_special_tokens=False)
        inputs_ = {'input_ids': [], 'attention_mask': []}
        attn_mask = inputs['attention_mask']
        for i in range(0, len(x), bs):
            inputs_['input_ids'] = inputs['input_ids'][i:i + bs].to(f'cuda:{args.gpu[0]}')
            inputs_['attention_mask'] = inputs['attention_mask'][i:i + bs].to(f'cuda:{args.gpu[0]}')
            outputs = LM(inputs_['input_ids'], attention_mask=inputs_['attention_mask'])
            word_embedding.extend(outputs[0].cpu().detach())
        embedding = torch.stack(word_embedding, dim=0)
        return embedding, attn_mask

    def compute_attention(self, embedding, mask):
        bs = embedding.shape[0]
        embedding = embedding.permute(1, 0, 2)
        mask = (mask < 1)
        _, w_attention = self.attention(embedding, embedding, embedding, key_padding_mask=mask)
        w_attention = w_attention.mean(dim=1)
        _, top_w = w_attention.topk(k=self.slots)
        top_w, _ = top_w.sort(dim=1)  # keep original order
        embedding = embedding.permute(1, 0, 2)
        # reduce each sequence from seq_len to k, pool only top k most attended words
        embedding = embedding[torch.arange(bs).unsqueeze(-1), top_w, :]
        return embedding, top_w

    def compute_distance(self, batch, mask):
        N, S = batch.shape[0:2]  # Batch size, Sequence length
        E = self.enc_size  # Encoding size
        K = self.proto_size  # Patch length
        p = self.protolayer.view(1, self.num_prototypes, 1, K * E)
        distances = []
        if self.attn:
            c = torch.combinations(torch.arange(S), r=K)
            C = c.shape[0]
            b = batch[:, c, :].view(N, 1, C, K * E)
            if self.metric == 'L2':
                dist = - nes_torch(b, p, dim=-1)
            elif self.metric == 'cosine':
                dist = - F.cosine_similarity(b, p, dim=-1)
            distances.append(dist)
        else:
            j = 0
            for d, n in zip(self.dilated, self.num_filters):
                H = S - d * (K - 1)  # Number of patches
                x = batch.unsqueeze(1)
                # use sliding window to get patches
                x = F.unfold(x, kernel_size=(K, 1), dilation=d)
                x = x.view(N, 1, H, K * E)
                p_ = p[:, j:j + n, :]
                p_ = p_.view(1, n, 1, K * E)
                if self.metric == 'L2':
                    dist = - nes_torch(x, p_, dim=-1)
                elif self.metric == 'cosine':
                    dist = - F.cosine_similarity(x, p_, dim=-1)
                # cut off combinations that contain padding, still keep for every example at least one combination, even
                # if it contains padding
                overlap = d * (K - 1)
                m = mask[:, overlap:].unsqueeze(1)
                m[:, :, 0] = 1
                dist = dist * m
                distances.append(dist)
                j += n

        return distances

    def get_dist(self, embedding, mask):
        if self.attn:
            embedding, top_w = self.compute_attention(embedding, mask)
        else:
            top_w = []
        distances = self.compute_distance(embedding, mask)
        return distances, top_w

    def nearest_neighbors(self, distances, top_w, text_train, labels_train):
        argmin_dist, prototype_distances, nearest_patch = [], [], []
        # compute min and argmin value in each sentence for each prototype.
        # remove the dilation dimension, num_dilations x [batch x num_proto x num_convolutions] to [b x #p x #conv]
        for batch in distances:
            # id of conv per sentence that achieves lowest distance
            argmin_dist.append(torch.cat([torch.argmin(dilated, dim=2) for dilated in batch], dim=1))
            # value of conv per sentence that achieves lowest distance
            prototype_distances.append(torch.cat([torch.min(dilated, dim=2)[0] for dilated in batch], dim=1))
        # compute nearest sentence id
        prototype_distances = torch.cat(prototype_distances, dim=0)
        nearest_sent = torch.argmin(prototype_distances, dim=0).cpu().detach().numpy()
        # look up nearest convolution id in nearest sentence for each prototype
        argmin_dist = torch.cat(argmin_dist, dim=0)
        nearest_patch = argmin_dist[nearest_sent, torch.arange(self.num_prototypes)].cpu().detach().numpy()

        # get text for prototypes
        text_nearest, nearest_words, proto_texts, proto_ids = [], [], [], []
        text_tknzd = self.tokenizer(text_train, return_tensors='pt', padding=True, add_special_tokens=False).input_ids
        if self.attn:
            top_w = torch.cat(top_w, dim=0).cpu().detach().numpy()
            c = torch.combinations(torch.arange(top_w.shape[1]), r=self.proto_size)
            word_ids = c[nearest_patch]
            nearest_words = top_w[np.expand_dims(nearest_sent, -1), word_ids]
            text_nearest = text_tknzd[nearest_sent]
        else:
            j = 0
            for d, n in zip(self.dilated, self.num_filters):
                # only finds the beginning word id since we did a convolution. so we have to add the subsequent words,
                # also add padding required by convolution
                nearest_words.extend(
                    [[word_id + x * d for x in range(self.proto_size)] for word_id in nearest_patch[j:j + n]])
                text_nearest.extend(text_tknzd[nearest_sent[j:j + n]])
                j += n

        for i, (s_index, w_indices) in enumerate(zip(nearest_sent, nearest_words)):
            token2text = self.tokenizer.decode(text_nearest[i][w_indices].tolist())
            proto_ids.append(
                f'P{i + 1} | sentence {s_index} | label {labels_train[s_index]} | text: {text_train[s_index]}| proto: ')
            proto_texts.append(f'{token2text}')

        return proto_ids, proto_texts, [nearest_sent, nearest_words]


def _from_pretrained(cls, *args, **kw):
    """Load a transformers model in PyTorch, with fallback to TF2/Keras weights."""
    try:
        return cls.from_pretrained(*args, **kw)
    except OSError as e:
        logging.warning("Caught OSError loading model: %s", e)
        logging.warning(
            "Re-trying to convert from TensorFlow checkpoint (from_tf=True)")
        return cls.from_pretrained(*args, from_tf=True, **kw)


class BertForSequenceClassification2Layers(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 10),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(10, config.num_labels, bias=False),
        )

        self.init_weights()


class GPT2ForSequenceClassification2Layers(GPT2ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 10),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(10, config.num_labels, bias=False),
        )

        self.init_weights()


class DistilBertForSequenceClassification2Layers(DistilBertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 10),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(10, config.num_labels, bias=False),
        )

        self.init_weights()


class BaseNetBERT(nn.Module):
    def __init__(self):
        super(BaseNetBERT, self).__init__()
        self.model_name_or_path = 'bert-large-uncased'
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name_or_path)
        model_config = transformers.AutoConfig.from_pretrained(
            self.model_name_or_path,
            num_labels=2,
            output_hidden_states=True,
            output_attentions=True,
        )
        # This is a just a regular PyTorch model.
        self.model = _from_pretrained(
            transformers.AutoModelForSequenceClassification,
            self.model_name_or_path,
            config=model_config)
