import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
from math import ceil, floor


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
        inputs = self.tokenizer(x, return_tensors="pt", truncation=True, padding=True)
        for i in range(0,len(x),bs):
            inputs_ = inputs.copy()
            inputs_['input_ids'] = inputs_['input_ids'][i:i+bs].to(f'cuda:{gpu}')
            inputs_['attention_mask'] = inputs_['attention_mask'][i:i+bs].to(f'cuda:{gpu}')
            inputs_['token_type_ids'] = inputs_['token_type_ids'][i:i+bs].to(f'cuda:{gpu}')
            outputs = self.Bert(**inputs_)
            word_embedding.extend(outputs[0].cpu())
            # cls_embedding,extend(outputs[1])
        embedding = torch.stack(word_embedding, dim=0)
        return embedding

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
        # if distance small -> high penalty
        p1_loss = 1 / torch.mean(torch.stack(dist))

        # assures that each prototype is not too close to padding token
        # pad_token = model.tokenizer(['test','test sentence for padding', return_tensors="pt", padding=True, truncation=True)
        # pad_embedding = model.Bert(**pad_token)[0][0][-1]
        # p2_loss = torch.sum(torch.cdist(self.prototypes, pad_embedding, p=2))
        return r1_loss, r2_loss, p1_loss#, p2_loss


class ProtoPNetConv(ProtoPNet):
    def __init__(self, args):
        super(ProtoPNetConv, self).__init__(args)
        self.ones = nn.Parameter(torch.ones(args.num_prototypes, self.enc_size, args.proto_size), requires_grad=False)
        self.dilated = args.dilated

    def forward(self, embedding):
        # embedding = self.emb_trafo(embedding)
        distances = self.l2_convolution(embedding)
        prototype_distances, _ = torch.min(distances, dim=2)
        class_out = self.fc(prototype_distances)
        return prototype_distances, distances, class_out

    def l2_convolution(self, x):
        # l2-convolution filters on input x
        x2 = x ** 2
        x2 = x2.permute(0,2,1)
        x2_patch_sum = self.conv_(x2)

        protolayer = self.protolayer.view(-1, self.enc_size, self.proto_size)
        p2 = protolayer ** 2
        p2 = torch.sum(p2, dim=(1, 2))
        p2_reshape = p2.view(1, -1, 1)

        x = x.permute(0,2,1)
        xp = F.conv1d(input=x, weight=protolayer)
        # L2-distance aka x² - 2xp + p²
        distances = x2_patch_sum - 2 * xp + p2_reshape
        distances = torch.sqrt(torch.abs(distances))
        return distances

    def conv_(self, x2):
        # compute multiple convolutions for different dilations and prototypes
        x2_patch_sum = []
        num_filters = [floor(self.num_protos/len(self.dilated))] * len(self.dilated)
        num_filters[0] += self.num_protos % len(self.dilated)
        kernel_size = self.proto_size
        # cut off last convolution, if padding too long, even/uneven number problem
        cut_off = x2.size(-1) - kernel_size + 1
        for d,n in zip(self.dilated, num_filters):
            # compute padding size such that all dilations yield same sized convs, always round up here
            dil2pad = ceil((kernel_size-1) * (d-1) / 2)
            x2_patch_sum.append(F.conv1d(input=x2, weight=self.ones[:n], padding=dil2pad, dilation=d)[:,:,:cut_off])
        x2_patch_sum = torch.cat(x2_patch_sum,dim=1)
        return x2_patch_sum

    def nearest_neighbors(self, distances, text_train, model):
        text_nearest, nearest_conv, nearest_words, proto_texts = [], [], [], []
        prototype_distances, _ = torch.min(distances, dim=2)
        nearest_sent = torch.argmin(prototype_distances, dim=0).cpu().numpy()
        for i,n in enumerate(nearest_sent):
            nearest_conv.append(torch.argmin(distances[n,i,:]).cpu().numpy())

        text_tknzd = model.tokenizer(text_train, return_tensors="pt", truncation=True, padding=True).input_ids
        num_filters = [floor(self.num_protos/len(self.dilated))] * len(self.dilated)
        num_filters[0] += self.num_protos % len(self.dilated)
        j = 0
        for i,d in enumerate(self.dilated):
            # only finds the beginning word id since we did a convolution. so we have to add the subsequent words, also
            # add padding required by convolution
            nearest_words.extend([[word_id + x*d for x in range(self.proto_size)] for word_id in nearest_conv[j:j+num_filters[i]]])
            dil2pad = ceil((self.proto_size-1) * (d-1) / 2)
            text_nearest.extend(F.pad(text_tknzd[nearest_sent[j:j+num_filters[i]]],pad=[dil2pad,dil2pad]))
            j += num_filters[i]

        for i, (s_index, w_indeces) in enumerate(zip(nearest_sent, nearest_words)):
            token2text = model.tokenizer.decode(text_nearest[i][w_indeces].tolist())
            proto_texts.append([s_index, token2text, text_train[s_index]])

        return proto_texts


class ProtoPNetDist(ProtoPNet):
    def __init__(self, args):
        super(ProtoPNetDist, self).__init__(args)

    def compute_distance(self, embedding):
        # this approach is not possible, to compute all available combinations it requires lots of memory and computational power
        combs_batch = []
        for batch in range(embedding.size(0)):
            combs_proto = []
            for proto in range(self.num_prototypes):
                combs_sentence = []
                for feature in range(self.enc_size):
                    combs_sentence.append(torch.combinations(embedding[batch,:,feature], r=self.proto_size))
                # proto_size x num_combinations x enc_size, where num_combinations is (word per sentence over words per prototype)
                stacked = torch.stack(combs_sentence).view(self.proto_size,-1,self.enc_size)
                dist = torch.cdist(stacked, self.protolayer[:,proto,:].unsqueeze(0)) # shape proto_size x num_combinations
                dist = torch.sum(dist, dim=1).squeeze() # sum along proto_size to get one dist value for the prototype
                dist = torch.min(dist)[0]
                combs_proto.append(dist)
            stacked_proto = torch.stack(combs_proto)
            combs_batch.append(stacked_proto)
        distances = torch.stack(combs_batch).view(-1,embedding.size(0),self.proto_size,self.enc_size)
        return distances

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
        text_tknzd = model.tokenizer(text_train, return_tensors="pt", truncation=True, padding=True).input_ids
        for (s_index, w_index) in zip(nearest_sentence, nearest_word):
            token2text = model.tokenizer.decode(text_tknzd[s_index][w_index].tolist())
            proto_texts.append([s_index, token2text, text_train[s_index]])

        return proto_texts


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