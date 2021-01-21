import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model
from math import floor
import transformers
import logging
from transformers import BertForSequenceClassification

class ProtoNet(nn.Module):
    def __init__(self, args):
        super(ProtoNet, self).__init__()

        self.sentBert = SentenceTransformer('bert-large-nli-mean-tokens', device=args.gpu[0])
        self.enc_size = self.sentBert.get_sentence_embedding_dimension()
        for param in self.sentBert.parameters():
            param.requires_grad = False
        self.protolayer = nn.Parameter(nn.init.uniform_(torch.empty((args.num_prototypes, self.enc_size))),
                                                             requires_grad=True)
        self.fc = nn.Linear(args.num_prototypes, args.num_classes, bias=False)

    def forward(self, embedding):
        prototype_distances = torch.cdist(embedding, self.protolayer, p=2)
        class_out = self.fc(prototype_distances)
        return prototype_distances, prototype_distances, class_out

    def get_protos(self):
        return self.protolayer

    def get_proto_weights(self):
        return self.fc.weight.T.cpu().detach().numpy()

    def compute_embedding(self, x, args):
        embedding = self.sentBert.encode(x, convert_to_tensor=True, device=args.gpu[0])
        if len(embedding.size()) == 1:
            embedding.unsqueeze_(0)
        return embedding

    @staticmethod
    def nearest_neighbors(distances, text_train, labels_train):
        distances = torch.cat(distances)
        _, nearest_ids = torch.topk(distances, 1, dim=0, largest=False)
        proto_texts = [f"P{proto+1} | sentence {index} | label {labels_train[index]} | text: {text_train[index]}"
                       for proto, sent in enumerate(nearest_ids.cpu().numpy().T) for index in sent]
        return proto_texts

    @staticmethod
    def proto_loss(prototype_distances):
        """
        Computes the interpretability losses (R1 and R2 from the paper (Li et al. 2018)) for the prototype nets.
        """
        r1_loss = torch.mean(torch.min(prototype_distances, dim=0)[0])
        r2_loss = torch.mean(torch.min(prototype_distances, dim=1)[0])
        # only sentence level, no word level thus
        p1_loss = 0
        return r1_loss, r2_loss, p1_loss

class BaseNet(ProtoNet):
    def __init__(self, args):
        super(BaseNet, self).__init__(args)
        self.fc = nn.Sequential(
            nn.Linear(self.enc_size, 20),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(20, args.num_classes, bias=False),
        )

    def forward(self, embedding):
        return self.fc(embedding)


class ProtoPNet(nn.Module):
    def __init__(self, args):
        super(ProtoPNet, self).__init__()
        if args.language_model == 'Bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
            self.LM = BertModel.from_pretrained('bert-large-cased')
            self.enc_size = 1024  # TODO: currently static instead of dynamic
        elif 'GPT2':
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.LM = GPT2Model.from_pretrained('gpt2-xl')
            self.enc_size = 1600  # TODO: currently static instead of dynamic
        for param in self.LM.parameters():
            param.requires_grad = False
        self.proto_size = args.proto_size
        self.num_protos = args.num_prototypes
        self.fc = nn.Linear(args.num_prototypes, args.num_classes, bias=False)
        self.emb_trafo = nn.Sequential(
                nn.Linear(in_features=self.enc_size, out_features=self.enc_size),
                nn.ReLU()
                )
        self.protolayer = nn.Parameter(nn.init.uniform_(torch.empty((args.num_prototypes, self.proto_size, self.enc_size))),
                                       requires_grad=True)

    def get_proto_weights(self):
        return self.fc.weight.T.cpu().detach().numpy()

    def get_protos(self):
        return self.protolayer

    def compute_embedding(self, x, args):
        bs = 10 # divide data by a batch size if too big for memory to process embedding at once
        word_embedding = []
        inputs = self.tokenizer(x, return_tensors="pt", padding=True)
        for i in range(0,len(x),bs):
            inputs_ = inputs.copy()
            inputs_['input_ids'] = inputs_['input_ids'][i:i+bs].to(f'cuda:{args.gpu[0]}')
            inputs_['attention_mask'] = inputs_['attention_mask'][i:i+bs].to(f'cuda:{args.gpu[0]}')
            outputs = self.LM(inputs_['input_ids'], attention_mask=inputs_['attention_mask'])
            if args.avoid_spec_token:
                # setting embedding values of PAD, CLS and SEP token to a high number to make them "unlikely regarded"
                # in distance computation
                inputs_['attention_mask'][inputs_['attention_mask']==0] = 1e3
                # for n in range(len(inputs_['attention_mask'])):
                #     inputs_['attention_mask'][n][0] = 1e3
                #     inputs_['attention_mask'][n][inputs_['input_ids'][n]==102] = 1e3
                word_embedding.extend((outputs[0] * inputs_['attention_mask'].unsqueeze(-1)).cpu())
            else:
                word_embedding.extend(outputs[0].cpu())
        embedding = torch.stack(word_embedding, dim=0)
        return embedding

class ProtoPNetConv(ProtoPNet):
    def __init__(self, args):
        super(ProtoPNetConv, self).__init__(args)
        self.ones = nn.Parameter(torch.ones(args.num_prototypes, self.enc_size, args.proto_size), requires_grad=False)
        self.dilated = args.dilated
        self.num_filters = [floor(self.num_protos/len(self.dilated))] * len(self.dilated)
        self.num_filters[0] += self.num_protos % len(self.dilated)
        self.protolayer = nn.Parameter(nn.init.uniform_(torch.empty((args.num_prototypes, self.enc_size, self.proto_size))),
                                       requires_grad=True)

    def forward(self, embedding):
        # embedding = self.emb_trafo(embedding)
        distances = self.l2_convolution(embedding)
        prototype_distances = torch.cat([torch.min(dist, dim=2)[0] for dist in distances], dim=1)
        class_out = self.fc(prototype_distances)
        return prototype_distances, distances, class_out

    def l2_convolution(self, x):
        # l2-convolution filters on input x
        x = x.permute(0,2,1)
        x2 = x ** 2

        p2 = self.protolayer ** 2
        p2_sum = torch.sum(p2, dim=(1, 2)).view(-1,1)

        distances, j = [], 0
        for d, n in zip(self.dilated, self.num_filters):
            x2_patch_sum = F.conv1d(input=x2, weight=self.ones[:n], dilation=d)
            xp = F.conv1d(input=x, weight=self.protolayer[j:j+n], dilation=d)
            # L2-distance aka sqrt(x² - 2xp + p²)
            dist = torch.sqrt(torch.abs(x2_patch_sum - 2 * xp + p2_sum[j:j+n]))
            distances.append(dist)
            j += n

        return distances

    def nearest_neighbors(self, distances, text_train, labels_train):
        argmin_dist, prototype_distances, nearest_conv =  [], [], []
        # compute min and argmin value in each sentence for each prototype
        for d in distances:
            argmin_dist.append(torch.cat([torch.argmin(dist, dim=2) for dist in d], dim=1))
            prototype_distances.append(torch.cat([torch.min(dist, dim=2)[0] for dist in d], dim=1))
        prototype_distances = torch.cat(prototype_distances, dim=0)
        # compute nearest sentence id and look up nearest convolution id in this sentence for each prototype
        nearest_sent = torch.argmin(prototype_distances, dim=0).cpu().numpy()
        argmin_dist = torch.cat(argmin_dist, dim=0)
        for i,n in enumerate(nearest_sent):
            nearest_conv.append(argmin_dist[n,i].cpu().numpy())

        # get text for prototypes
        text_nearest, nearest_words, proto_texts = [], [], []
        text_tknzd = self.tokenizer(text_train, return_tensors="pt", padding=True).input_ids
        j = 0
        for d, n in zip(self.dilated, self.num_filters):
            # only finds the beginning word id since we did a convolution. so we have to add the subsequent words, also
            # add padding required by convolution
            nearest_words.extend([[word_id + x*d for x in range(self.proto_size)] for word_id in nearest_conv[j:j+n]])
            text_nearest.extend(text_tknzd[nearest_sent[j:j+n]])
            j += n

        for i, (s_index, w_indices) in enumerate(zip(nearest_sent, nearest_words)):
            token2text = self.tokenizer.decode(text_nearest[i][w_indices].tolist())
            proto_texts.append(f"P{i+1} | sentence {s_index} | label {labels_train[s_index]} | proto: {token2text} | text: {text_train[s_index]}")

        return proto_texts

    @staticmethod
    def proto_loss(prototype_distances):
        """
        Computes the interpretability losses (R1 and R2 from the paper (Li et al. 2018)) for the prototype nets.
        """
        r1_loss = torch.mean(torch.min(prototype_distances, dim=0)[0])
        r2_loss = torch.mean(torch.min(prototype_distances, dim=1)[0])

        # conv assures itself that same input entries are not possible
        p1_loss = 0
        return r1_loss, r2_loss, p1_loss

class ProtoPNetDist(ProtoPNet):
    def __init__(self, args):
        super(ProtoPNetDist, self).__init__(args)

    def forward(self, embedding):
        # embedding = self.emb_trafo(embedding)
        distances = torch.cdist(embedding.unsqueeze(1), self.protolayer, p=2)
        # pool along words per sentence, to get only #proto_size smallest distances per sentence and prototype
        # and compute sum along this #proto_size to get only one distance value per sentence and prototype
        min_distances = -F.max_pool2d(-distances, kernel_size=(distances.size(2), 1))
        prototype_distances = torch.sum(torch.abs(min_distances), dim=-1).squeeze()
        class_out = self.fc(prototype_distances)
        return prototype_distances, distances, class_out

    def nearest_neighbors(self, distances, text_train, labels_train):
        distances = torch.cat(distances)
        min_distances_per_sentence = -F.max_pool2d(-distances,
                                      kernel_size=(distances.size(2),1))
        min_distances_sent = torch.sum(torch.abs(min_distances_per_sentence), dim=-1)
        # get argmin for number of protos along number of sentence dimension
        nearest_sent = torch.argmin(min_distances_sent, dim=0).squeeze().cpu().numpy()

        min_distances_word = -F.max_pool2d(-distances.permute(1,2,0,3),
                                      kernel_size=(distances.size(0),1))
        nearest_words = torch.argmin(min_distances_word, dim=1).squeeze().cpu().numpy()

        proto_texts = []
        text_tknzd = self.tokenizer(text_train, return_tensors="pt", padding=True).input_ids
        for i, (s_index, w_indices) in enumerate(zip(nearest_sent, nearest_words)):
            token2text = self.tokenizer.decode(text_tknzd[s_index][w_indices].tolist()) # TODO: bug that words are appended together
            proto_texts.append(f"P{i+1} | sentence {s_index} | label {labels_train[s_index]} | proto: {token2text} | text: {text_train[s_index]}")

        return proto_texts

    def proto_loss(self, prototype_distances):
        """
        Computes the interpretability losses (R1 and R2 from the paper (Li et al. 2018)) for the prototype nets.
        """
        r1_loss = torch.mean(torch.min(prototype_distances, dim=0)[0])
        r2_loss = torch.mean(torch.min(prototype_distances, dim=1)[0])

        # assures that each prototype itself does not consist out of the exact same input entry multiple times
        if self.proto_size > 1:
            dist = []
            comb = torch.combinations(torch.arange(self.proto_size), r=2)
            for k, l in comb:
                dist.append(torch.mean(torch.abs((self.protolayer[:, k, :] - self.protolayer[:, l, :])),dim=1))
            # if distance small -> high penalty, check to not divide by 0
            _mean = torch.mean(torch.stack(dist))
            p1_loss = 1 / _mean if _mean else 1000
        else:
            p1_loss = 0
        return r1_loss, r2_loss, p1_loss

class BasePartsNet(ProtoPNet):
    def __init__(self, args):
        super(BasePartsNet, self).__init__(args)
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(self.enc_size, 20),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(20, args.num_classes, bias=False),
        )

    def forward(self, embedding):
        emb = embedding.view(embedding.size(0),-1)
        lin = nn.Linear(emb.size(1),self.enc_size)
        emb = lin(emb)
        return self.fc(emb)


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
            nn.Linear(config.hidden_size, 20),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(20, config.num_labels, bias=False),
        )

        self.init_weights()

class BaseNetBERT(nn.Module):
    def __init__(self):
        super(BaseNetBERT, self).__init__()
        self.model_name_or_path = 'bert-large'
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
