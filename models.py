import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel

class ProtoNet(nn.Module):
    def __init__(self, args):
        super(ProtoNet, self).__init__()

        self.sentBert = SentenceTransformer('bert-large-nli-mean-tokens', device=args.gpu[0])
        enc_size = self.sentBert.get_sentence_embedding_dimension()
        for param in self.sentBert.parameters():
            param.requires_grad = False
        self.protolayer = nn.Parameter(nn.init.uniform_(torch.empty((args.num_prototypes, enc_size))),
                                                             requires_grad=True)
        #self.protolayer = self.protolayer.repeat(hyperparams['num_prototypes'], 1)
        #self.pdist = nn.PairwiseDistance(p=2)
        self.fc = nn.Linear(args.num_prototypes, args.num_classes, bias=False)

    def forward(self, embedding):
        prototype_distances = torch.cdist(embedding, self.protolayer, p=2)# get prototype distances
        # feature_vector_distances = torch.cdist(self.protolayer, embedding, p=2) # get feature vector distances
        feature_vector_distances =  prototype_distances.T # get feature vector distances
        class_out = self.fc(prototype_distances)
        return prototype_distances, feature_vector_distances, class_out

    def get_protos(self):
        return self.protolayer

    def get_proto_weights(self):
        return self.fc.weight.T.cpu().detach().numpy()

    def compute_embedding(self, x, gpu):
        embedding = self.sentBert.encode(x, convert_to_tensor=True, device=gpu)
        embedding = embedding.to(f'cuda:{gpu}')
        return embedding

    @staticmethod
    def nearest_neighbors(prototype_distances):
        nearest_ids = torch.argmin(prototype_distances, dim=0)
        return nearest_ids.cpu().numpy()


class ProtoPNet(nn.Module):
    def __init__(self, args):
        super(ProtoPNet, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained('bert-large-cased', device=args.gpu[0])
        self.Bert = BertModel.from_pretrained('bert-large-cased')
        self.enc_size = 1024 # needs to be adjusted
        self.proto_size = args.proto_size
        for param in self.Bert.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(args.num_prototypes, args.num_classes, bias=False)

    def get_proto_weights(self):
        return self.fc.weight.T.cpu().detach().numpy()

    def compute_embedding(self, x, gpu):
        inputs = self.tokenizer(x, return_tensors="pt", truncation=True, padding=True)
        inputs['input_ids'] = inputs['input_ids'].to(f'cuda:{gpu}')
        inputs['attention_mask'] = inputs['attention_mask'].to(f'cuda:{gpu}')
        inputs['token_type_ids'] = inputs['token_type_ids'].to(f'cuda:{gpu}')
        outputs = self.Bert(**inputs)
        word_embedding = outputs[0]
        # cls_embedding = outputs[1]
        return word_embedding


class ProtoPNetConv(ProtoPNet):
    def __init__(self, args):
        super(ProtoPNetConv, self).__init__(args)

        self.protolayer = nn.Parameter(nn.init.uniform_(torch.empty((args.num_prototypes, self.enc_size, args.proto_size))),
                                       requires_grad=True)
        self.ones = nn.Parameter(torch.ones(args.num_prototypes, self.enc_size, args.proto_size), requires_grad=False)

    def forward(self, embedding):
        prototype_distances = self.l2_convolution(embedding)
        feature_vector_distances =  prototype_distances.T # get feature vector distances

        min_distances = -F.max_pool1d(-prototype_distances, kernel_size=(prototype_distances.size()[2]))
        min_distances.squeeze_()
        class_out = self.fc(min_distances)
        return prototype_distances, feature_vector_distances, class_out

    def l2_convolution(self, x):
        # l2-convolution filters on input x
        x2 = x ** 2
        x2 = x2.permute(0,2,1)
        x2_patch_sum = F.conv1d(input=x2, weight=self.ones)

        p2 = self.protolayer ** 2
        p2 = torch.sum(p2, dim=(1, 2))
        p2_reshape = p2.view(1, -1, 1)

        x = x.permute(0,2,1)
        xp = F.conv1d(input=x, weight=self.protolayer)
        distances = x2_patch_sum - 2 * xp + p2_reshape
        return distances

    def get_protos(self):
        return self.protolayer

    def nearest_neighbors(self,prototype_distances):
        min_distances_sent = -F.max_pool1d(-prototype_distances, kernel_size=prototype_distances.size(-1))
        nearest_sent = torch.argmin(min_distances_sent.squeeze(), dim=0).squeeze()

        min_distances_word = -F.max_pool1d(-prototype_distances.permute(1,2,0), kernel_size=prototype_distances.size(0))
        nearest_word = torch.argmin(min_distances_word.squeeze(), dim=1).squeeze()
        # only finds the beginning word id since we did a convolution. so we have to add the proto_size subsequent words
        nearest_words = [[word_id+x for x in range(self.proto_size)] for word_id in nearest_word.cpu().numpy()]
        return nearest_sent.cpu().numpy(), nearest_words


class ProtoPNetDist(ProtoPNet):
    def __init__(self, args):
        super(ProtoPNetDist, self).__init__(args)

        self.protolayer = nn.Parameter(nn.init.uniform_(torch.empty((1, args.num_prototypes, args.proto_size, self.enc_size))),
                                       requires_grad=True)

    def forward(self, embedding):
        # adjust embedding shape
        # embedding.unsqueeze_(1)
        prototype_distances = torch.cdist(embedding.unsqueeze(1), self.protolayer, p=2)  # get prototype distances
        feature_vector_distances = prototype_distances.T
        min_distances = -F.max_pool2d(-prototype_distances,
                                      kernel_size=(prototype_distances.size()[2],
                                                   prototype_distances.size()[3]))
        min_distances.squeeze_()
        class_out = self.fc(min_distances)
        return prototype_distances, feature_vector_distances, class_out

    def get_protos(self):
        return self.protolayer

    @staticmethod
    def nearest_neighbors(prototype_distances):
        min_distances_sent = -F.max_pool2d(-prototype_distances,
                                      kernel_size=(prototype_distances.size()[2],
                                                   prototype_distances.size()[3]))
        nearest_sentence = torch.argmin(min_distances_sent.squeeze(), dim=0).squeeze()

        min_distances_word = -F.max_pool2d(-prototype_distances.permute(1,2,0,3),
                                      kernel_size=(prototype_distances.size()[0],1))
        nearest_word = torch.argmin(min_distances_word, dim=1).squeeze()
        return nearest_sentence.cpu().numpy(), nearest_word.cpu().numpy()


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
        embedding = embedding.to(f'cuda:{gpu}')
        return embedding