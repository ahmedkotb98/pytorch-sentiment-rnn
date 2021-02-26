# Aspect-based Sentiment Analysis with RNN / GRUs / LSTMs on SemEval 2014.

Currently we implemented a baseline LSTM/RNN/GRU model with a linear layer on the last output along with a target-dependent, TD-LSTM (Tang et al 2015) model for Aspect based sentiment analysis (ABSA).

The sequences are padded with zeros from the front so that the last vector is not zero. We pad these in the prepare script using keras pad sequences. Nothing is masked so far and we pad to the max length.

There are two modes of prediction, namely term and aspect. Aspect refers to aspect categories while term refers to, well, terms which are sequences that can be found in the text itself. There are two datasets, Laptop and Restaurants. There are both term and aspect settings for Laptop but only aspect setting for restaurants.

Data Preperation


1.Download glove embeddings from paper.

Jeffrey Pennington, Richard Socher, and Christopher D
Manning. 2014. Glove: Global vectors for word representation.
Proceedings of the Empiricial Methods
in Natural Language Processing (EMNLP 2014),
12:1532–1543.

2.Make directory
 ```
mkdir ../glove_embeddings
```
3.place glove.840B.300d.txt into ../glove_embeddings

python prepare.py         # Will take awhile.

This should build into ./store and ./embeddings/. The former is the environment object that train.py reads while the file written into embeddings is just a smaller concised version of glove so that I can rerun prepare.py fast.

For training and evaluation, run the following script. Evaluates accuracy every epoch. (Note, it takes awhile for the model to stop predicting all the same class)

Notes

1.Basic LSTM/RNN/GRU works! Testing on SemEval (Term Category + Restaurants) give about 73-75% accuracy around epoch 20. This is the same result I previously got using TensorFlow. The algorithm constantly predicts the same class (2) for the first 10+ iterations though.

2.Handling Gradiet Clipping is done as follows:
```
if(self.args.clip_norm>0):
    coeff = clip_gradient(self.mdl, self.args.clip_norm)
    for p in self.mdl.parameters():
        p.grad.mul_(coeff)
self.optimizer.step()
```

Not sure if this is correct or not.

3.It seems like RNNs in pyTorch are batch-minor, i.e, seq length is dim 0 and batch is dim 1. Let's wait for more variable length support.

4.Pretrained embeddings are supported. (I loaded GloVe)

5.I wonder how to make the embedding layer non-trainable?

