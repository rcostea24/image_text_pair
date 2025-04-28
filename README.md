# Image-Text Pair Classification

## 1 Introduction

In this project, we had to classify using neural networks if image-text pairs
are matching or not. The approach is to learn embeddings for both images and
captions and then classify if they match or not using a fully connected neural
network.

## 2 Data preparation

## 2.1 Text data

Most of the processing is done on text data. Each caption is tokenized,
and a vocabulary is built, which is a simple word to index dictionary. The
tokenization process includes multiple spaces reduction, punctuation cleaning,
and stemming. The tokens are obtained using nltk library. After the vocabulary
is built, every caption is encoded and padded to the maximum length which is
32 words. The maximum length was determined by inspecting the data, i.e.
finding the maximum number of words of a caption. An encoded caption will
be a vector of 32 integers where the first N elements will be the index of the word
that was encoded, N meaning the length of the caption. The other 32-N elements
will be padding. The vocabulary contains the encoding of ”< mask >” (used
for the MLM pretraining experiment), ”< pad >”, ”< unk >” tokens.
The only pre-processing done to the images is a resize.
Everything is done in a Dataset class from torch.

## 3 Modeling

For this task the approach is to learn a representation of images and one for
captions and classify if they are a match using a fully connected network.

## 3.1 Image model

For the image representation we built two convolutional models, a CNN
encoder inspired by ResNet-18 and a custom Autoencoder that was used for unsupervised pretraining experiment.
The ResNet-18 inspired network has the architecture from Figure 1 and can
be seen in detail in the extended material (Figure 7). This network is used in
the experiments with random weights or with pretraind weights on a pretext
task of predicting the rotation of an image.
The Autoencoder is a simple Encoder-Decoder type of CNN. The use of this
network is to learn the representation of the images without the text network
training to interfere with it. Therefore, the network’s encoder is used with
pretrained weight on the reconstruction task. It’s architecture can be visualized
in the extended material

3.2 Text model

Because text data is sequential we used Recurrent Neural Networks. Because
the standard RNNs suffer from vanishing gradients for long sequences we employ
Long Short Term Memory architecture. It can also be seen in the extended
material Figure??. The number of layers is a hyperparameter that was tuned,
but from the experiments the best model has one LSTM layer.
The LSTM architecture was used in different experiments both with random
weight and pretrained weight. The pretraining was done on Mask Language
Modeling task. The other language model was a simple fully connected network used with
random weight:

4 Training and evaluating

4.1 Pretraining of image and text models

In the previous section we mentioned that each model, except FC for text,
was also used with pretrained weights on the given data.
The CNN was pretrained to predict the rotation of an image. This task is
a classification with 5 classes meaning a degree of rotation (-60, -30, 0, 30, 60).
The optimizer used is Adam and the loss function is CrossEntropyLoss
The LSTM was trained to predict the masked word in a caption. This is
done by using a special token named< mask >. A random word is replaced
with this token and then the model is trained to predict the missing token. It
is treated as a classification problem. The optimizer used is Adam and the loss
function is CrossEntropyLoss
The Autoencoder is trained in an unsupervised fashion. The model is trying
to reconstruct the input image, thus learning the representation of the it. In
the training for the image-text pair matching only the encoder is used. The
optimizer used is Adam and the loss function is MAE loss. The results can be
visualized in Figure 2


4.2 Final models

In the previous section we discussed the individual components of a final
model that will be trained. Now we defined how this components are used
for the classification task. Each final model has two components, a backbone
and a classifier, which is a FC network. The backbone is also composed of
two components, a visual model and a langauge model. Therefore the final
models are: (CN N+LST M)−> F C,(CN N+F C)−> F C,(AE−Encoder+
LST M)−> F C.
The problem was approached as a classification between two separate classes
(2 neurons as output, where each neuron represents the probability of that a
sample being in class 0 or class 1) and a binary classification based on a threshold
(1 neuron as output, where the output represents the probability that a sample
is matching or not).

4.3 Experiments

In this competition we have done a number of 46 experiments. All of
them have either CrossEntropyLoss, either BinaryCrossEntropy as loss func-
tion. Also, all of them all trained with 64 batch size on P100 GPU from kaggle
during 50 epochs. In general, each experiment have a validation accuracy around
60%−65% on validation data and the best experiments are up to 70% accuracy
on validation. The experiments are both with random and pretrained weights
with the models listed above. On the test data the models performed worse,
therefore we have a massive drop in accuracy, from 70% to around 52%−54%.


