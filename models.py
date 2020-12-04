import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer



class ProtopNetNLP(nn.Module):
    def __init__(self, args):
        super(ProtopNetNLP, self).__init__()

        self.sentBert = SentenceTransformer('bert-large-nli-mean-tokens')
        enc_size = self.sentBert.get_sentence_embedding_dimension()
        for param in self.sentBert.parameters():
            param.requires_grad = False
        # tensor of prototype feature vectors
        self.protolayer = nn.Parameter(torch.nn.init.uniform_(torch.empty((args.num_prototypes, enc_size))),
                                                             requires_grad=True)
        #self.protolayer = self.protolayer.repeat(hyperparams['num_prototypes'], 1)
        #self.protolayer.to(device)
        #self.pdist = nn.PairwiseDistance(p=2)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(args.num_prototypes, args.num_classes),
            # torch.nn.Softmax(dim=1)
        )

    def forward(self, embedding):
        prototype_distances = torch.cdist(embedding, self.protolayer, p=2)# get prototype distances
        feature_vector_distances = torch.cdist(self.protolayer, embedding, p=2) # get feature vector distances

        #assert prototype_distances.shape == feature_vector_distances.T.shape

        class_out = self.fc(prototype_distances)

        #prototypes = self.ae.decoder(self.protolayers.unsqueeze(dim=1))

        return (prototype_distances, feature_vector_distances, class_out)

    def get_protos(self):
        return self.protolayer

    def get_proto_weights(self):
        return list(self.fc.children())[-1].weight.T.cpu().detach().numpy()

    def compute_embedding(self, x, gpu):
        embedding = self.sentBert.encode(x, convert_to_tensor=True)
        embedding = embedding.cuda(gpu)
        return embedding
