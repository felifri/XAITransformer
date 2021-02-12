import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model
import transformers
import logging
from transformers import BertForSequenceClassification
# from utils import dist2similarity


class ProtoNet(nn.Module):
    def __init__(self, args):
        super(ProtoNet, self).__init__()
        self.language_model = args.language_model
        self.num_prototypes = args.num_prototypes
        self.LM = SentenceTransformer('bert-large-nli-mean-tokens', device=args.gpu[0])
        self.enc_size = self.LM.get_sentence_embedding_dimension()
        for param in self.LM.parameters():
            param.requires_grad = False
        self.protolayer = nn.Parameter(nn.init.uniform_(torch.empty((args.num_prototypes, self.enc_size, 1))),
                                                             requires_grad=True)
        self.fc = nn.Linear(args.num_prototypes, args.num_classes, bias=False)

    def forward(self, embedding):
        prototype_distances = torch.cdist(embedding, self.protolayer.squeeze(), p=2)
        # similarity = dist2similarity(prototype_distances)
        # class_out = self.fc(similarity)
        class_out = self.fc(prototype_distances)
        return prototype_distances, prototype_distances, class_out

    def get_protos(self):
        return self.protolayer.squeeze()

    def get_proto_weights(self):
        return self.fc.weight.T.cpu().detach().numpy()

    def compute_embedding(self, x, args):
        embedding = self.LM.encode(x, convert_to_tensor=True, device=args.gpu[0]).cpu()
        if len(embedding.size()) == 1:
            embedding.unsqueeze_(0)
        return embedding

    @staticmethod
    def nearest_neighbors(distances, text_train, labels_train):
        distances = torch.cat(distances)
        _, nearest_ids = torch.topk(distances, 1, dim=0, largest=False)
        proto_id = [f"P{proto+1} | sentence {index} | label {labels_train[index]} | text: " for proto, sent
                    in enumerate(nearest_ids.cpu().numpy().T) for index in sent]
        proto_texts = [f"{text_train[index]}" for sent in nearest_ids.cpu().numpy().T for index in sent]
        return proto_id, proto_texts


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
        elif 'GPT2':
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.LM = GPT2Model.from_pretrained('gpt2-xl')
        self.enc_size = self.LM.config.hidden_size
        for param in self.LM.parameters():
            param.requires_grad = False
        self.proto_size = args.proto_size
        self.num_prototypes = args.num_prototypes
        self.fc = nn.Linear(args.num_prototypes, args.num_classes, bias=False)
        self.emb_trafo = nn.Sequential(
                nn.Linear(in_features=self.enc_size, out_features=self.enc_size),
                nn.ReLU()
                )
        self.protolayer = nn.Parameter(nn.init.uniform_(torch.empty((args.num_prototypes, self.enc_size, self.proto_size))),
                                       requires_grad=True)

    def get_proto_weights(self):
        return self.fc.weight.T.cpu().detach().numpy()

    def get_protos(self):
        return self.protolayer

    def compute_embedding(self, x, args):
        bs = 10 # divide data by a batch size if too big for memory to process embedding at once
        word_embedding = []
        inputs = self.tokenizer(x, return_tensors="pt", padding=True, add_special_tokens=False)
        for i in range(0,len(x),bs):
            inputs_ = inputs.copy()
            inputs_['input_ids'] = inputs_['input_ids'][i:i+bs].to(f'cuda:{args.gpu[0]}')
            inputs_['attention_mask'] = inputs_['attention_mask'][i:i+bs].to(f'cuda:{args.gpu[0]}')
            outputs = self.LM(inputs_['input_ids'], attention_mask=inputs_['attention_mask'])
            if args.avoid_spec_token:
                # set embedding values of PAD token to a high number to make it "unlikely regarded" in distance computation
                inputs_['attention_mask'][inputs_['attention_mask']==0] = 1e3
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
        self.num_filters = [self.num_prototypes // len(self.dilated)] * len(self.dilated)
        self.num_filters[0] += self.num_prototypes % len(self.dilated)
        self.protolayer = nn.Parameter(nn.init.uniform_(torch.empty((args.num_prototypes, self.enc_size, self.proto_size))),
                                       requires_grad=True)

    def forward(self, embedding):
        # embedding = self.emb_trafo(embedding)
        distances = self.l2_convolution(embedding)
        prototype_distances = torch.cat([torch.min(dist, dim=2)[0] for dist in distances], dim=1)
        # similarity = dist2similarity(prototype_distances)
        # class_out = self.fc(similarity)
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
            dist = torch.sqrt(F.relu(x2_patch_sum - 2 * xp + p2_sum[j:j+n]))
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
        text_nearest, nearest_words, proto_texts, proto_ids = [], [], [], []
        text_tknzd = self.tokenizer(text_train, return_tensors="pt", padding=True, add_special_tokens=False).input_ids
        j = 0
        for d, n in zip(self.dilated, self.num_filters):
            # only finds the beginning word id since we did a convolution. so we have to add the subsequent words, also
            # add padding required by convolution
            nearest_words.extend([[word_id + x*d for x in range(self.proto_size)] for word_id in nearest_conv[j:j+n]])
            text_nearest.extend(text_tknzd[nearest_sent[j:j+n]])
            j += n

        for i, (s_index, w_indices) in enumerate(zip(nearest_sent, nearest_words)):
            token2text = self.tokenizer.decode(text_nearest[i][w_indices].tolist())
            proto_ids.append(f"P{i+1} | sentence {s_index} | label {labels_train[s_index]} | text: {text_train[s_index]}| proto: ")
            proto_texts.append(f"{token2text}")

        return proto_ids, proto_texts


class BasePartsNet(ProtoPNet):
    def __init__(self, args):
        super(BasePartsNet, self).__init__(args)

        self.fc1 = nn.Sequential(
            nn.Linear(self.enc_size, 1),
            # nn.Dropout(),
            # nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(args.seq_length, args.num_classes, bias=False),
            # nn.Dropout(),
            # nn.ReLU())
        )

    def forward(self, embedding):
        return self.fc2(self.fc1(embedding).squeeze())


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
