import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
from math import floor


class ProtoNet(nn.Module):
    def __init__(self, args):
        super(ProtoNet, self).__init__()

        self.sentBert = SentenceTransformer('bert-large-nli-mean-tokens', device=args.gpu[0])
        enc_size = self.sentBert.get_sentence_embedding_dimension()
        for param in self.sentBert.parameters():
            param.requires_grad = False
        self.protolayer = nn.Parameter(nn.init.uniform_(torch.empty((args.num_prototypes, enc_size))),
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

    def compute_embedding(self, x, gpu):
        embedding = self.sentBert.encode(x, convert_to_tensor=True, device=gpu)
        return embedding

    @staticmethod
    def nearest_neighbors(distances, text_train, _):
        nearest_ids = torch.argmin(distances, dim=0)
        proto_texts = [[index, text_train[index]] for index in nearest_ids.cpu().numpy()]
        return proto_texts

    @staticmethod
    def proto_loss(prototype_distances):
        """
        Computes the interpretability losses (R1 and R2 from the paper (Li et al. 2018)) for the prototype nets.
        """
        r1_loss = torch.mean(torch.min(prototype_distances, dim=0)[0])
        r2_loss = torch.mean(torch.min(prototype_distances, dim=1)[0])
        return r1_loss, r2_loss, 0


class ProtoPNet(nn.Module):
    def __init__(self, args):
        super(ProtoPNet, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
        self.Bert = BertModel.from_pretrained('bert-large-cased')
        for param in self.Bert.parameters():
            param.requires_grad = False
        self.enc_size = 1024  # TODO: currently static instead of dynamic
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

    def compute_embedding(self, x, gpu):
        bs = 1500 # divide data by a batch size if too big to process embedding at once
        word_embedding = []
        inputs = self.tokenizer(x, return_tensors="pt", padding=True)
        for i in range(0,len(x),bs):
            inputs_ = inputs.copy()
            inputs_['input_ids'] = inputs_['input_ids'][i:i+bs].to(f'cuda:{gpu}')
            inputs_['attention_mask'] = inputs_['attention_mask'][i:i+bs].to(f'cuda:{gpu}')
            outputs = self.Bert(inputs_['input_ids'], attention_mask=inputs_['attention_mask'])
            # setting embedding values of padding to a high number to make them "unlikely regarded" in distance computation
            inputs_['attention_mask'][inputs_['attention_mask']==0] = 1e2
            word_embedding.extend((outputs[0] * inputs_['attention_mask'].unsqueeze(-1)).cpu())
            # cls_embedding,extend(outputs[1])
        embedding = torch.stack(word_embedding, dim=0)
        return embedding

class ProtoPNetConv(ProtoPNet):
    def __init__(self, args):
        super(ProtoPNetConv, self).__init__(args)
        self.ones = nn.Parameter(torch.ones(args.num_prototypes, self.enc_size, args.proto_size), requires_grad=False)
        self.dilated = args.dilated
        self.num_filters = [floor(self.num_protos/len(self.dilated))] * len(self.dilated)
        self.num_filters[0] += self.num_protos % len(self.dilated)

    def forward(self, embedding):
        # embedding = self.emb_trafo(embedding)
        distances = self.l2_convolution(embedding)
        prototype_distances = torch.cat([torch.min(dist, dim=2)[0] for dist in distances], dim=1)
        class_out = self.fc(prototype_distances)
        return prototype_distances, distances, class_out

    def l2_convolution(self, x):
        # l2-convolution filters on input x
        x2 = x ** 2
        x2 = x2.permute(0,2,1)

        protolayer = self.protolayer.view(-1, self.enc_size, self.proto_size)
        p2 = protolayer ** 2
        p2 = torch.sum(p2, dim=(1, 2))
        p2_reshape = p2.view(1, -1, 1)

        distances, j = [], 0
        for d, n in zip(self.dilated, self.num_filters):
            x2_patch_sum = F.conv1d(input=x2, weight=self.ones[:n], dilation=d)
            xp = F.conv1d(input=x.permute(0,2,1), weight=protolayer[j:j+n], dilation=d)
            # L2-distance aka x² - 2xp + p²
            dist = x2_patch_sum - 2 * xp + p2_reshape[:,j:j+n,:]
            distances.append(torch.sqrt(torch.abs(dist)))
            j += n

        return distances

    def nearest_neighbors(self, distances, text_train, model):
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
            nearest_conv.append(torch.argmin(argmin_dist[n,i]).cpu().numpy())

        # get text for prototypes
        text_nearest, nearest_words, proto_texts = [], [], []
        text_tknzd = model.tokenizer(text_train, return_tensors="pt", padding=True).input_ids
        j = 0
        for d, n in zip(self.dilated, self.num_filters):
            # only finds the beginning word id since we did a convolution. so we have to add the subsequent words, also
            # add padding required by convolution
            nearest_words.extend([[word_id + x*d for x in range(self.proto_size)] for word_id in nearest_conv[j:j+n]])
            text_nearest.extend(text_tknzd[nearest_sent[j:j+n]])
            j += n

        for i, (s_index, w_indices) in enumerate(zip(nearest_sent, nearest_words)):
            token2text = model.tokenizer.decode(text_nearest[i][w_indices].tolist())
            proto_texts.append([s_index, token2text, text_train[s_index]])

        return proto_texts

    def proto_loss(self, prototype_distances):
        """
        Computes the interpretability losses (R1 and R2 from the paper (Li et al. 2018)) for the prototype nets.
        """
        r1_loss = torch.mean(torch.min(prototype_distances, dim=0)[0])
        r2_loss = torch.mean(torch.min(prototype_distances, dim=1)[0])
        if torch.isnan(prototype_distances).any():
            __import__("pdb").set_trace()
        # conv assures itself that same tokens are not possible
        p1_loss = 0
        # assures that each prototype is not too close to padding token
        # pad_token = model.tokenizer(['test','test sentence for padding', return_tensors="pt", padding=True)
        # pad_embedding = model.Bert(**pad_token)[0][0][-1]
        # p2_loss = torch.sum(torch.cdist(self.prototypes, pad_embedding, p=2))
        return r1_loss, r2_loss, p1_loss#, p2_loss

class ProtoPNetDist(ProtoPNet):
    def __init__(self, args):
        super(ProtoPNetDist, self).__init__(args)

    def forward(self, embedding):
        # embedding = self.emb_trafo(embedding)
        distances = torch.cdist(embedding.unsqueeze(1), self.protolayer, p=2)
        # pool along words per sentence, to get only #proto_size smallest distances per sentence and prototype
        # and compute mean along this #proto_size to get only one distance value per sentence and prototype
        min_distances = -F.max_pool2d(-distances,
                                      kernel_size=(distances.size(2), 1))
        prototype_distances = torch.mean(min_distances, dim=-1).squeeze()
        class_out = self.fc(prototype_distances)
        return prototype_distances, distances, class_out

    @staticmethod
    def nearest_neighbors(distances, text_train, model):
        min_distances_per_sentence = -F.max_pool2d(-distances,
                                      kernel_size=(distances.size(2),1))
        min_distances_sent = torch.mean(min_distances_per_sentence, dim=-1)
        # get argmin for number of protos along number of sentence dimension
        nearest_sentence = torch.argmin(min_distances_sent, dim=0).squeeze().cpu().numpy()

        min_distances_word = -F.max_pool2d(-distances.permute(1,2,0,3),
                                      kernel_size=(distances.size(0),1))
        nearest_word = torch.argmin(min_distances_word, dim=1).squeeze().cpu().numpy()

        proto_texts = []
        text_tknzd = model.tokenizer(text_train, return_tensors="pt", padding=True).input_ids
        for (s_index, w_index) in zip(nearest_sentence, nearest_word):
            token2text = model.tokenizer.decode(text_tknzd[s_index][w_index].tolist())
            proto_texts.append([s_index, token2text, text_train[s_index]])

        return proto_texts

    def proto_loss(self, prototype_distances):
        """
        Computes the interpretability losses (R1 and R2 from the paper (Li et al. 2018)) for the prototype nets.
        """
        r1_loss = torch.mean(torch.min(prototype_distances, dim=0)[0])
        r2_loss = torch.mean(torch.min(prototype_distances, dim=1)[0])

        # assures that each prototype itself does not consist out of the exact same token multiple times
        dist = []
        comb = torch.combinations(torch.arange(self.proto_size), r=2)
        for k, l in comb:
            dist.append(torch.mean(torch.abs((self.protolayer[:, k, :] - self.protolayer[:, l, :])),dim=1))
        # if distance small -> high penalty, check to not divide by 0
        p1_loss = 1 / torch.mean(torch.stack(dist)) if torch.mean(torch.stack(dist)) else 1000

        # assures that each prototype is not too close to padding token
        # pad_token = model.tokenizer(['test','test sentence for padding', return_tensors="pt", padding=True)
        # pad_embedding = model.Bert(**pad_token)[0][0][-1]
        # p2_loss = torch.sum(torch.cdist(self.prototypes, pad_embedding, p=2))
        return r1_loss, r2_loss, p1_loss#, p2_loss

class BaseNet(nn.Module):
    def __init__(self, args):
        super(BaseNet, self).__init__()

        self.sentBert = SentenceTransformer('bert-large-nli-mean-tokens', device=args.gpu[0])
        enc_size = self.sentBert.get_sentence_embedding_dimension()
        for param in self.sentBert.parameters():
            param.requires_grad = False
        self.fc = nn.Sequential(
            nn.Linear(enc_size, 20),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(20, args.num_classes, bias=False),
        )

    def forward(self, embedding):
        return self.fc(embedding)

    def compute_embedding(self, x, gpu):
        embedding = self.sentBert.encode(x, convert_to_tensor=True, device=gpu)
        return embedding