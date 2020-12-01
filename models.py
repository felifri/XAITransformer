import torch
import torch.nn as nn
# import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import math
import numpy as np
from sklearn.cluster import KMeans


class ProtopNetNLP(nn.Module):
    def __init__(self, args):
        super(ProtopNetNLP, self).__init__(args.gpu)

        self.device = args.device #'cuda'
        self.sentBert = SentenceTransformer('distilbert-base-nli-mean-tokens')
        # tensor of prototype feature vectors
        self.protolayer = nn.Parameter(torch.nn.init.uniform_(torch.empty((args.num_prototypes, args.enc_size),
                                                             requires_grad=True)))
                                                             #device=self.device)))
        #self.protolayer = self.protolayer.repeat(hyperparams['num_prototypes'], 1)
        #self.protolayer.to(device)
        #self.pdist = nn.PairwiseDistance(p=2)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(args.num_prototypes, args.num_classes),
            # torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        embedding = self.sentBert.encode(x, convert_to_tensor=True)

        prototype_distances = torch.cdist(embedding, self.protolayer, p=2)# get prototype distances
        feature_vector_distances = torch.cdist(self.protolayer, embedding, p=2) # get feature vector distances

        #assert prototype_distances.shape == feature_vector_distances.T.shape

        class_out = self.fc(prototype_distances)

        #prototypes = self.ae.decoder(self.protolayers.unsqueeze(dim=1))

        return (prototype_distances, feature_vector_distances, class_out, embedding)

    def init_protos(self, args, text, labels):
        # init via kmeans
        # self.protolayer = nn.Parameter(torch.nn.init.uniform_(torch.empty((hyperparams['num_prototypes'],
        #                                                           hyperparams['enc_size']),
        #                                                           requires_grad=True,
        #                                                           device=device)))
        self.eval()
        class2vecs = dict()
        class_id_list = [i for i in range(0, self.class_num)]

        for i in class_id_list:
            class2vecs[i] = []

        # for each batch, get vecs and ids, add vecs to class2vecs based on ids
        batch_size = args.batch_size
        batch_num = math.floor(len(text) / batch_size)
        if len(text) > 0 and len(text) < batch_size:
            batch_num = 1

        for n in range(batch_num):
            i = n * batch_size
            if n < batch_num - 1:
                j = (n + 1) * batch_size
            else:
                j = len(text)

            batch = text[i:j]
            targets = self.tag_seq_indexer.items2idx(labels[i:j])
            _, _, _, latents = self.forward(batch)  # latents: batch_size x proto_dim
            for k in range(len(batch)):
                class_id = targets[k]

                latent_vec = latents[k, :].detach().cpu().numpy()
                class2vecs[class_id].append(latent_vec)

        # there are a variety of ways to move the data from kmeans.cluster_centers_ to self.prototypes, of course
        # but copying with splices directly to self.prototypes_[idx,:,:] was silently failing, so we preallocate and .copy_ all at once
        new_prototypes = torch.empty(self.proto_dim, self.embedding_dim)
        for i in class_id_list:
            class_data = np.array(class2vecs[i])
            kmeans = KMeans(n_clusters=self.num_prototypes_per_class)
            kmeans.fit(class_data)

            centers = torch.Tensor(kmeans.cluster_centers_).view(self.num_prototypes_per_class, self.proto_dim)
            new_prototypes = torch.cat((new_prototypes, centers), 0)

        self.prototypes.data.copy_(new_prototypes)