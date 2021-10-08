# Interactively Generating Explanations for Transformer Language Models
An XAI method to get insights into the decision-making process of transformer language models through prototypical explanations. Additionally a XIL method to interact with the trained (explainable) model.

## Setup and Run Model

To setup the model, we need to first:
* download https://github.com/peterbhase/InterpretableNLP-ACL2020/tree/master/text/data/rt-polarity to `data/rt-polarity/`
* download https://www.yelp.com/dataset to `data/restaurant/`
* download https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data to `data/jigsaw/`
* set up a (python) virtual environment meeting [requirements.txt](requirements.txt)

The core file is [run_proto_nlp.py](run_proto_nlp.py)

To run this model on sentence-/ word-level, e.g. do:

```
python run_proto_nlp.py --data_name rt-polarity --level sentence --language_model SentBert --compute_emb True
```

```
python run_proto_nlp.py --data_name rt-polarity --level word --language_model GPT2 --proto_size 4  --compute_emb True
```

You can find the results in `experiments/train_results/`. Further parameter choices are explained in the core file.

## Interaction

To interact with the model, you first need to train a model. After the training you can interact with the model in the following fashion:
```
python run_proto_nlp.py --data_name rt-polarity --level sentence --language_model SentBert --mode replace
```
The mode sets the interaction type you want to apply. In the core file you can refine the interaction method, i.e. set e.g. the prototype you want to remove or replace. The last trained model is automatically selected for the interaction. If you trained several models and want to interact with a certain one, the path has to be set manually.
