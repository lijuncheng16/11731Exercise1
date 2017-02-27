import numpy as np
import sys, math
import random
import math
import dynet as dy
from collections import defaultdict

class nnlm:
         
    def __init__(self):         
        self.feats_and_values ={}
        self.wids = defaultdict(lambda: len(self.wids))
        self.unigrams = {}
        self.model = dy.Model()
        self.EMB_SIZE = 128
        self.HID_SIZE  = 128
        self.N = 3
        M = self.model.add_lookup_parameters((len(self.wids), self.EMB_SIZE))
        W_mh = self.model.add_parameters((self.HID_SIZE, self.EMB_SIZE * (self.N-1)))
        b_hh = self.model.add_parameters((self.HID_SIZE))
        W_hs = self.model.add_parameters((len(self.wids), self.HID_SIZE))
        b_s = self.model.add_parameters((len(self.wids)))

    def read_corpus(self, file):
        #print file
        f = open(file)
        self.data_array_train = []
        for line in f:
            line = '<s> ' + line.split('\n')[0] + ' </s>'          
            words = line.split()
            for word in words:
                if word in self.unigrams:
                    self.unigrams[word] = self.unigrams[word] + 1
                else:
                    self.unigrams[word] = 1 
        f.close()         
        self.assign_ids()
        self.create_data(file)
        return self.trigramfeaturedict, self.wids
       
       
    def assign_ids(self):
        self.wids["<unk>"] = 0
        self.wids["<s>"] = 1
        self.wids["</s>"] = 2
        for w in self.unigrams:
            if self.unigrams[w] > 3:
                self.wids[w] = len(self.wids)
            else:
                self.wids[w] = 0
        return   
       
    def create_data(self, file):
        self.accumulate_trigramfeatures(file)
         
    def build_nnlm_graph(self, dictionary):
        dy.renew_cg()
        M = self.model.add_lookup_parameters((len(self.wids), self.EMB_SIZE))
        W_mh = self.model.add_parameters((self.HID_SIZE, self.EMB_SIZE * (self.N-1)))
        b_hh = self.model.add_parameters((self.HID_SIZE))
        W_hs = self.model.add_parameters((len(self.wids), self.HID_SIZE))
        b_s = self.model.add_parameters((len(self.wids)))

        w_xh = dy.parameter(W_mh)
        b_h = dy.parameter(b_hh)
        W_hy = dy.parameter(W_hs)
        b_y = dy.parameter(b_s)
        errs = []
        for context, next_word in dictionary:
            #print context, next_word
            k =  M[self.wids[context.split()[0]]]
            kk =  M[self.wids[context.split()[1]]]
            #print k , kk
            #print k.value()
            x = k.value() + kk.value()
            #print x
            h_val = dy.tanh(w_xh * dy.inputVector(x) + b_h)
            y_val = W_hy * h_val + b_y
            err = dy.pickneglogsoftmax(y_val,self.wids[next_word])
            errs.append(err)
        gen_err = dy.esum(errs)    
        return gen_err
    
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
        return
     
    def get_file_perplexity(self, file):
        f = open(file)
        print_flag = 0
        self.num_sentences = 0
        self.num_oovs = 0
        self.num_words = 0
        self.logprob = 0
        arr = []
        for line in f:
            line = line.split('\n')[0].lower()
            #line = '<s> ' + line.split('\n')[0] + ' </s>'
            ppl_sentence = self.get_sentence_perplexity(line,0)
            if print_flag ==1:
                print line, ppl_sentence, '\n'
            arr.append(ppl_sentence)
        #print np.mean(arr) 
        log_arr = np.log(arr)
        print log_arr
        ml_corpus = -1.0 * np.sum(log_arr) * 1.0/len(arr)
        print np.exp(ml_corpus)
        
        print 'Sentences: ', self.num_sentences
        print 'Words: ', self.num_words
        print 'OOVs: ' , self.num_oovs
        print 'Log probability: ', self.logprob
        self.perplexity = np.exp( -1.0 * self.logprob / ( self.num_words + self.num_sentences - self.num_oovs) * 2.71)  
        # SRILM constant
        print "Perplexity over corpus is: ", self.perplexity     
        
        
    def get_sentence_perplexity(self, string, print_flag):
        #print len(string)
        num_tokens = 0
        num_oovs = 0
        if len(string.split()) < 2:
            print "The Sentence you gave me is very short"
            return -1.0 * np.log(self.eqn8(string,0) / 2)
        else:
            mle_sentence = np.log(1.0)
            line =  string.split('\n')[0] + ' </s>'
            length_line = len(string.split())
            line = line.split()
            b = 0
            while b < len(line) - 1:
                if print_flag ==1:
                    print "The value of  b is ", b
                kv = line[b] + ' ' + line[b+1]
                if print_flag ==1:
                    print "I am looking for ", kv
                if line[b+1] == '</s>':
                    kv = line[b]
                if print_flag ==1:
                    print "I am looking for ", kv
                if kv in self.bigrams and self.bigrams[kv] > 0:  
                    if print_flag ==1:
                        print "Found ", kv , " in bigrams"
                    mle = self.eqn8(kv,0)
                    length_gram = 2
                else:
                    if print_flag ==1:
                        print  "I did not find ", kv, " in bigrams"  
                    kv = line[b]
                    if print_flag ==1:
                        print "Now, I am searching for ", kv
                    if kv in self.unigrams and self.unigrams[kv] > 0:
                        if print_flag ==1:
                            print "Found ",kv , " in unigrams"
                        mle = self.eqn8(kv,0)
                        length_gram = 1
                    else:
                        if print_flag ==1:
                            print  "I did not find ", kv, " in unigrams or it was a singleton. I think its an UNK"              
                            kv = line[b]
                            mle = self.alpha_unk * np.exp(1e-7)
                            length_gram = 1
                            num_oovs = num_oovs + 1
                b = b + length_gram
                num_tokens = num_tokens + 1 
                mle_sentence = mle_sentence + np.log(mle)
                self.num_oovs += num_oovs
            self.num_sentences += 1
            self.num_words += length_line
            self.logprob += mle_sentence
            print_flag = 0
            mle_sentence_old = mle_sentence 
            mle_sentence = mle_sentence * (- 1.0 / (length_line + 1 +1 - num_oovs )  )
            ppl_sentence = np.exp(mle_sentence * 2.3)
            if print_flag ==1:
                print "MLE of sentence is ", mle_sentence_old, " and PPL of sentence is ", ppl_sentence, " number of words: ", length_line, " number of OOVs: "  , num_oovs
                g = open('t','w')
                g.write(string + '\n')
                g.close()
                cmd = 'ngram -lm ../data/en-de/01_srilm_bigram.model -ppl t'
                os.system(cmd)
                print '\n\n'
            print_flag = 0                
            
            return ppl_sentence

if __name__ == "__main__":
    lm = nnlm()
    train_dict, wids = lm.read_corpus('./en-de/valid.en-de.de')
    #train_dict,wids = lm.read_corpus('txt.done.data')
    #print wids
    data = train_dict.items()
    #print data
    # Define the hyperparameters
    #N = 3
    #EVAL_EVERY = 10
    #EMB_SIZE = 128
    #HID_SIZE = 256

    model = dy.Model()
    trainer = dy.SimpleSGDTrainer(model)

    best_score = None
    token_count = sent_count = total_loss = cum_perplexity = 0.0
    sample_num = 0
    import time
    _start = time.time()
    print_flag = 0
    print 'Start Time: ',time.time()
  
    # Training
    for epoch in range(10000):
        random.shuffle(data)
        if print_flag == 1:
            print epoch, sample_num, " " , trainer.status(), 
            print "Average Loss ", total_loss / token_count,
            print "Expectation: ", math.exp(total_loss / token_count),
            print "Time: ", ( time.time() - _start)
        _start = time.time()
        losses = lm.build_nnlm_graph(data)
        #print _start
        gen_losses = losses.vec_value()
        #print gen_losses
        loss = dy.sum_batches(losses)
        total_loss += loss.value()
        cum_perplexity += math.exp(gen_losses[0]/ len(data))
        token_count += len(data)
        #sent_count += len(sents)
  
        loss.backward()
        trainer.update()
        sample_num += len(data)
        trainer.update_epoch(1)
        print_flag = 1
        #print "Epoch 0"
  

'''
  epoch_loss = 0
  random.shuffle(data)
  c = 1
  for x , ystar in data:
    c = c + 1
    dy.renew_cg()
    z = M[wids[x.split()[0]]].value() + M[wids[x.split()[1]]].value()
    y = calc_function(dy.inputVector(z))
    err = dy.pickneglogsoftmax(y, wids[ystar])
    epoch_loss = epoch_loss + err.value()
    err.backward()
    trainer.update()
    if c % 5000 == 1:
      print "    ", x, err.value(), ystar
  print("Epoch %d: loss=%f" % (epoch, epoch_loss))  
    
'''

