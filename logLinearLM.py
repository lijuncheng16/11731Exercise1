import cPickle, pickle
import numpy as np
import os
from collections import defaultdict

class loglinearlm:
     
    def __init__(self):
        self.feats_and_values ={}
        self.wids = defaultdict(lambda: len(self.wids))
         
    def read_corpus(self, file):
        # for each line in the file, split the words and turn them into IDs like this:
        print file
        #self.accumulate_trigramfeatures(file)
        #self.accumulatebigramfeatures(file)
        #self.accumulateskipgramfeatures(file)
        f = open(file)
        self.data_array_train = []
        for line in f:
            line = '<s> ' + line.split('\n')[0] + ' </s>'
            self.data_array_train.append(line)
            words = line.split()
            for word in words:
                wid = self.wids[word]
        f.close()
        self.get_feature_vectors(file)  
       
     
    def accumulate_trigramfeatures(self, file):
        self.trigramfeaturedict = {}
        g = open(file)
        for line in g:
            line = '<s> ' + line.split('\n')[0] + ' </s>'
            line = line.split()
            contexts = zip(line[0:len(line)-2], line[1:len(line)-1], line[2:])
            for prev_2, prev_1, current in contexts:
            #print prev_2, prev_1, current
                context = prev_2 + ' ' + prev_1
                self.trigramfeaturedict[context] = current
        g.close()    

    def print_words(self):
        for wid in self.wids:
            print wid, self.wids[wid]
     
    def get_vocab_size(self):
        return len(self.wids)
     
    def calculate_feature_F1(self, file):
        # This is a trigram context feature
        features = []
        feature_vector_prime = np.zeros(self.get_vocab_size())
        g = open(file)
        for line in g:
            line = '<s> ' + line.split('\n')[0] + ' </s>'
            #print line
            line = line.split()
            contexts = zip(line[0:len(line)-2], line[1:len(line)-1], line[2:])
            for prev_2, prev_1, current in contexts:
                feature_vector = feature_vector_prime
                #print prev_2, prev_1, current, self.get_vocab_size()
                #print prev_2, self.wids[prev_2], feature_vector
                prev_2_id = self.wids[prev_2]
                feature_vector[prev_2_id] = 1.0
                prev_1_id = self.wids[prev_1]
                feature_vector[prev_1_id] = 1.0
                features.append(feature_vector)
                #print feature_vector
        #print features[0]   
        g.close()
        return features
    
    # Input is a list of sparse features, output is a numpy array of dense features
    # Actually, we don't need this when we're doing the sum of sparse feature vectors, but I'll leave this here for reference.    
    def sparse_features_to_dense_features(self, features):
        ret = np.zeros(len(features))
        print "sparse feature to dense:", ret
        for f in features:
            print f
            ret[f] += 1
        return ret

    def get_feature_vectors(self, file):
        features = []
        features.append(self.sparse_features_to_dense_features(self.calculate_feature_F1(file)))
        #features.append(calculate_feature_F2())
        #features.append(calculate_feature_F3())
        #features.append(calculate_feature_F4())
        return features 
    
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

    
if __name__ == "__main__":
    llm = loglinearlm()
    llm.read_corpus('./en-de/train.en-de.low.en')
    #lm_01.print_dicts()
    #lm_01.save_dicts()
    #t = lm_01.eqn8("hello")
    #print "hello ", lm_01.get_sentence_perplexity("hello", 0)
    #print '\n'
    #print "hello this is cool \n", lm_01.get_sentence_perplexity("hello this is cool",0)
    #print lm_01.get_vocab_size()
    #print llm.get_feature_vectors('./en-de/train.en-de.low.en')
    #print np.power(2, (-1.0 * np.log2(t)))
    #lm_01.get_file_perplexity('../data/en-de/test.en-de.low.en')
    #lm_01.get_file_perplexity('../data/en-de/test.en-de.low.en')
         