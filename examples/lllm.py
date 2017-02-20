import numpy as np

from collections import defaultdict

wids = defaultdict(lambda: len(wids))

# Writing a function to read in the training and test corpora, and converting the words into numerical IDs.
def read_corpus(file):
  # for each line in the file, split the words and turn them into IDs like this:
  wid = wids[word]

# Writing the feature function, which takes in a string and returns which features are active (for example, as a baseline these can be features with the identity of the previous two words).
def feature_function(ctxt):
  features = []
  features.append(calculate_feature_1(ctxt))
  features.append(calculate_feature_2(ctxt))
  ...
  return features

# Input is a list of sparse features, output is a numpy array of dense features
# Actually, we don't need this when we're doing the sum of sparse feature vectors, but I'll leave this here for reference.
def sparse_features_to_dense_features(features):
  ret = np.zeros(number_of_features)
  for f in features:
    ret[f] += 1
  return ret

# How to create a sparse list of feature vectors, which is what we'll actually use.
list_of_feature_vectors = [np.zeros(vocab_size)] * [num_features]

# Writing code to calculate the loss function.
def loss_function(x, next_word):
  # Implement equations (25) to (28)

# Writing code to calculate gradients and perform stochastic gradient descent updates.
# Writing (or re-using from the previous exercise) code to evaluate the language models.

def run_training():
  train_loss = 0
  for i, training_example in enumerate(training_data):
    calculate_update(training_example)
    train_loss += calculate_loss(training_example)
    if i % 10000 == 0:
      dev_loss = 0
      for dev_example in development_data:
        dev_loss += calculate_loss(dev_example)
      print("Training_loss=%f, dev loss=%f" % (train_loss, dev_loss))
