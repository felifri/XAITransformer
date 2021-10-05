# Interactively Generating Explanations for Transformer Language Models
An XAI/ XIL method to get insights into the decision-making process of transformer language models through prototypical explanations.

## Setup and Run Model

To setup the model, we need to first:
* download https://github.com/peterbhase/InterpretableNLP-ACL2020/tree/master/text/data/rt-polarity to `data/rt-polarity/`
* download https://www.yelp.com/dataset to `data/restaurant/`
* download https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data to `data/jigsaw/`
* set up a virtual env meeting requirements.txt

The core file is [run_proto_nlp.py](run_proto_nlp.py)

To run this model on sentence-/ word-level, e.g. do:
```
python run_proto_nlp.py --data_name rt-polarity --level sentence --language_model SentBert
python run_proto_nlp.py --data_name rt-polarity --level word --language_model GPT2 --proto_size 4 
```

You can find the results in `experiments/train_results/`
