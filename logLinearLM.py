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
       #self.get_feature_vectors(file)  
       
     
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
	           
        
     def sparse_features_to_dense_features(self, features):
        ret = np.zeros(len(features))
        print ret
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
         