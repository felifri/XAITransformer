import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel

class ProtoNet(nn.Module):
    def __init__(self, args):
        super(ProtoNet, self).__init__()

        self.sentBert = SentenceTransformer('bert-large-nli-mean-tokens', device=args.gpu)
        enc_size = self.sentBert.get_sentence_embedding_dimension()
        for param in self.sentBert.parameters():
            param.requires_grad = False
        # tensor of prototype feature vectors
        self.protolayer = nn.Parameter(nn.init.uniform_(torch.empty((args.num_prototypes, enc_size))),
                                                             requires_grad=True)
        #self.protolayer = self.protolayer.repeat(hyperparams['num_prototypes'], 1)
        #self.pdist = nn.PairwiseDistance(p=2)

        self.fc = nn.Sequential(
            nn.Linear(args.num_prototypes, args.num_classes, bias=False),
        )

    def forward(self, embedding):
        prototype_distances = torch.cdist(embedding, self.protolayer, p=2)# get prototype distances
        feature_vector_distances = torch.cdist(self.protolayer, embedding, p=2) # get feature vector distances
        class_out = self.fc(prototype_distances)
        return prototype_distances, feature_vector_distances, class_out

    def get_protos(self):
        return self.protolayer

    def get_proto_weights(self):
        return list(self.fc.children())[-1].weight.T.cpu().detach().numpy()

    def compute_embedding(self, x, gpu):
        embedding = self.sentBert.encode(x, convert_to_tensor=True, device=gpu)
        embedding = embedding.cuda(gpu)
        return embedding

    @staticmethod
    def nearest_neighbors(prototype_distances):
        nearest_ids = torch.argmin(prototype_distances, dim=0)
        return nearest_ids.cpu().numpy()


class ProtoPNet(nn.Module):
    def __init__(self, args):
        super(ProtoPNet, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained('bert-large-cased', device=args.gpu)
        self.Bert = BertModel.from_pretrained('bert-large-cased')
        enc_size = 1024 # needs to be adjusted
        for param in self.Bert.parameters():
            param.requires_grad = False
        # tensor of prototype feature vectors
        self.protolayer = nn.Parameter(nn.init.uniform_(torch.empty((1, args.num_prototypes, args.proto_size, enc_size))),
                                       requires_grad=True)

        self.fc = nn.Sequential(
            nn.Linear(args.num_prototypes, args.num_classes, bias=False),
        )

    def forward(self, embedding):
        # adjust embedding shape
        # embedding.unsqueeze_(1)
        prototype_distances = torch.cdist(embedding.unsqueeze(1), self.protolayer, p=2)  # get prototype distances
        feature_vector_distances = torch.cdist(self.protolayer, embedding.unsqueeze(1), p=2)  # get feature vector distances

        min_distances = -torch.nn.functional.max_pool2d(-prototype_distances,
                                      kernel_size=(prototype_distances.size()[2],
                                                   prototype_distances.size()[3]))
        min_distances.squeeze_()
        class_out = self.fc(min_distances)
        return prototype_distances, feature_vector_distances, class_out

    def get_protos(self):
        return self.protolayer

    def get_proto_weights(self):
        return list(self.fc.children())[-1].weight.T.cpu().detach().numpy()

    def compute_embedding(self, x, gpu):
        inputs = self.tokenizer(x, return_tensors="pt", truncation=True, padding=True)
        inputs['input_ids'] = inputs['input_ids'].cuda(gpu)
        inputs['attention_mask'] = inputs['attention_mask'].cuda(gpu)
        inputs['token_type_ids'] = inputs['token_type_ids'].cuda(gpu)
        outputs = self.Bert(**inputs)
        word_embedding = outputs[0]
        # cls_embedding = outputs[1]
        return word_embedding

    @staticmethod
    def nearest_neighbors(prototype_distances):
        # min_distances_sent = -torch.nn.functional.max_pool2d(-prototype_distances,
        #                               kernel_size=(prototype_distances.size()[2],1))
        min_distances_sent = -torch.nn.functional.max_pool2d(-prototype_distances,
                                      kernel_size=(prototype_distances.size()[2],
                                                   prototype_distances.size()[3]))
        nearest_sentence = torch.argmin(min_distances_sent.squeeze(), dim=0).squeeze()
        nearest_words = -torch.nn.functional.max_pool2d(-prototype_distances,
                                      kernel_size=(prototype_distances.size()[2],1))
        nearest_word = torch.argmin(nearest_words, dim=0)
        return nearest_sentence.cpu().numpy(), nearest_word.cpu().numpy()


class ProtoPNet_Cdist(nn.Module):
    def __init__(self, args):
        super(ProtoPNet_Cdist, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained('bert-large-cased', device=args.gpu)
        self.Bert = BertModel.from_pretrained('bert-large-cased')
        enc_size = 1024 # needs to be adjusted
        for param in self.Bert.parameters():
            param.requires_grad = False
        # tensor of prototype feature vectors
        self.protolayer = nn.Parameter(nn.init.uniform_(torch.empty((1, args.num_prototypes, args.proto_size, enc_size))),
                                       requires_grad=True)

        self.fc = nn.Sequential(
            nn.Linear(args.num_prototypes, args.num_classes, bias=False),
        )

    def forward(self, embedding):
        # adjust embedding shape
        # embedding.unsqueeze_(1)
        prototype_distances = torch.cdist(embedding.unsqueeze(1), self.protolayer, p=2)  # get prototype distances
        feature_vector_distances = torch.cdist(self.protolayer, embedding.unsqueeze(1), p=2)  # get feature vector distances

        min_distances = -torch.nn.functional.max_pool2d(-prototype_distances,
                                      kernel_size=(prototype_distances.size()[2],
                                                   prototype_distances.size()[3]))
        min_distances.squeeze_()
        class_out = self.fc(min_distances)
        return prototype_distances, feature_vector_distances, class_out

    def get_protos(self):
        return self.protolayer

    def get_proto_weights(self):
        return list(self.fc.children())[-1].weight.T.cpu().detach().numpy()

    def compute_embedding(self, x, gpu):
        inputs = self.tokenizer(x, return_tensors="pt", truncation=True, padding=True)
        inputs['input_ids'] = inputs['input_ids'].cuda(gpu)
        inputs['attention_mask'] = inputs['attention_mask'].cuda(gpu)
        inputs['token_type_ids'] = inputs['token_type_ids'].cuda(gpu)
        outputs = self.Bert(**inputs)
        word_embedding = outputs[0]
        # cls_embedding = outputs[1]
        return word_embedding

    @staticmethod
    def nearest_neighbors(prototype_distances):
        # min_distances_sent = -torch.nn.functional.max_pool2d(-prototype_distances,
        #                               kernel_size=(prototype_distances.size()[2],1))
        min_distances_sent = -torch.nn.functional.max_pool2d(-prototype_distances,
                                      kernel_size=(prototype_distances.size()[2],
                                                   prototype_distances.size()[3]))
        nearest_sentence = torch.argmin(min_distances_sent.squeeze(), dim=0).squeeze()
        nearest_words = -torch.nn.functional.max_pool2d(-prototype_distances,
                                      kernel_size=(prototype_distances.size()[2],1))
        nearest_word = torch.argmin(nearest_words, dim=0)
        return nearest_sentence.cpu().numpy(), nearest_word.cpu().numpy()


class BaseNet(nn.Module):
    def __init__(self, args):
        super(BaseNet, self).__init__()

        self.sentBert = SentenceTransformer('bert-large-nli-mean-tokens', device=args.gpu)
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
        embedding = embedding.cuda(gpu)
        return embedding