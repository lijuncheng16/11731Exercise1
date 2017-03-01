import nltk
import dynet as dy
import time
import math
start = time.time()
import random
from dynet import *
from util import CorpusReader

class Attention:

    def __init__(self, src_vocab_size, tgt_vocab_size, model, state_dim, embed_size, src_lookup, tgt_lookup, minibatch_size, builder=dy.LSTMBuilder):
        self.model = model
        #self.trainer = dy.SimpleSGDTrainer(self.model)
        self.layers = 1
        self.embed_size = 128
        self.hidden_size = 128
        self.state_size = 128
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.attention_size = 32
        self.minibatch_size = minibatch_size

        self.enc_fwd_lstm = dy.LSTMBuilder(self.layers, self.embed_size, self.state_size, model)
        self.enc_bwd_lstm = dy.LSTMBuilder(self.layers, self.embed_size,self.state_size, model)
        self.dec_lstm = dy.LSTMBuilder(self.layers, self.state_size*2 + self.embed_size,  self.state_size, model)

        self.input_lookup = src_lookup
        self.output_lookup = tgt_lookup
        self.attention_w1 = model.add_parameters( (self.attention_size, self.state_size*2))
        self.attention_w2 = model.add_parameters( (self.attention_size , self.state_size * self.layers* 2))
        self.attention_v = model.add_parameters( (1, self.attention_size))
        self.decoder_w = model.add_parameters( (self.src_vocab_size , self.state_size ))
        self.decoder_b = model.add_parameters( ( self.src_vocab_size ))
        #self.output_lookup = lookup
        self.duration_weight = model.add_parameters(( 1, self.state_size ))
        self.duration_bias = model.add_parameters( ( 1 ))
       

    def run_lstm(self, init_state, input_vecs):
        s = init_state
        out_vectors = []
        for vector in input_vecs:
            x_t = lookup(self.input_lookup, int(vector))
            s = s.add_input(x_t)
            out_vector = s.output()
            out_vectors.append(out_vector)
        return out_vectors 

     ## I am just gonna loop over them
    def run_lstm_batch(self, init_state, input_vecs_batch):
        out_vectors_array = []
        for input_vecs in input_vecs_batch:
            s = init_state
            out_vectors = []
            for vector in input_vecs:
                x_t = lookup(self.input_lookup, int(vector))
                s = s.add_input(x_t)
                out_vector = s.output()
                out_vectors.append(out_vector)
            out_vectors_array.append(out_vectors) 

    def embed_sentence(self, sentence):
        sentence = [EOS] + list(sentence) + [EOS]
        sentence = [char2int[c] for c in sentence]
        global input_lookup
        return [input_lookup[char] for char in sentence]

    def attend_batch(self, input_mat_array, state_array, w1dt_array):
        context_array = []
        for input_mat, state, w1dt in zip(input_mat_array, state_array, w1dt_array):
            context_array.append(attend(input_mat, state, w1dt))
        return context_array  


    def attend(self, input_mat, state, w1dt):
        #global self.attention_w2
        #global self.attention_v
        w2 = dy.parameter(self.attention_w2)
        v = dy.parameter(self.attention_v)
        w2dt = w2*dy.concatenate(list(state.s()))
        unnormalized = dy.transpose(v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
        att_weights = dy.softmax(unnormalized)
        context = input_mat * att_weights
        return context

    def test_duration(self, state, idx):
        dw = dy.parameter(self.duration_weight)
        db = dy.parameter(self.duration_bias)
        dur = dw * state.output() + db
        return dy.squared_norm(dur - idx)

    def decode_batch(self, vectors_array, output_array, end_token):
        loss_array = []
        for vector, output in zip(vectors_array, output_array):
            l,t = self.decode(vector, output, end_token)
            loss_array.append(l)
        return dy.esum(loss_array) , t  

    def decode(self,  vectors, output, end_token):
        #output = [EOS] + list(output) + [EOS]
        #output = [char2int[c] for c in output]

        w = dy.parameter(self.decoder_w)
        b = dy.parameter(self.decoder_b)
        w1 = dy.parameter(self.attention_w1)
        input_mat = dy.concatenate_cols(vectors)
        w1dt = None

        last_output_embeddings = self.output_lookup[2]
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.state_size *2), last_output_embeddings]))
        loss = []
        dur_loss = []
        c = 1
        for word in output:
            c += 1
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt = w1dt or w1 * input_mat
            vector = dy.concatenate([self.attend(input_mat, s, w1dt), last_output_embeddings])
            s = s.add_input(vector)
            k = s
            #print "Going"
            dloss = self.test_duration(k, c)
            #print "Back"
            dur_loss.append(dloss)
            out_vector = w * s.output() + b
            probs = dy.softmax(out_vector)
            last_output_embeddings = self.output_lookup[word]
            loss.append(-dy.log(dy.pick(probs, word)))
        loss = dy.esum(loss)
        return loss, c

    def generate(self, sentence):
        #embedded = embed_sentence(in_seq)
        encoded = self.encode_sentence(sentence)

        w = dy.parameter(self.decoder_w)
        b = dy.parameter(self.decoder_b)
        w1 = dy.parameter(self.attention_w1)
        input_mat = dy.concatenate_cols(encoded)
        w1dt = None

        last_output_embeddings = self.output_lookup[2]
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.state_size * 2), last_output_embeddings]))

        out = ''
        res = []
        count_EOS = 0
        for i in range(len(sentence)):
            if count_EOS == 2: break
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt = w1dt or w1 * input_mat
            vector = dy.concatenate([self.attend(input_mat, s, w1dt), last_output_embeddings])
            s = s.add_input(vector)
            #k = s
            #dloss = self.test_duration(k, i, b)
            out_vector = w * s.output() + b
            probs = dy.softmax(out_vector).vec_value()
            next_word = probs.index(max(probs))
            last_output_embeddings = self.output_lookup[next_word]
            if next_word == 2:
                count_EOS += 1
                continue
            res.append(next_word)
            #out += int2char[next_word]
        return res

    def get_loss(self, sentence):
        dy.renew_cg()
        #embedded = self.embed_sentence(sentence)
        encoded = self.encode_sentence(sentence)
        end_token = '</s>'
        return self.decode(encoded, sentence, end_token)

    def get_loss_batch(self, src_sentence_array, tgt_sentence_array):
        dy.renew_cg()
        #embedded = self.embed_sentence(sentence)
        encoded_array = self.encode_sentence_batch(src_sentence_array)
        end_token = '</s>'
        return self.decode_batch(encoded_array, tgt_sentence_array, end_token)

    def encode_sentence(self, sentence):
        sentence_rev = list(reversed(sentence))
        fwd_vectors = self.run_lstm(self.enc_fwd_lstm.initial_state(), sentence)
        bwd_vectors = self.run_lstm(self.enc_bwd_lstm.initial_state(), sentence_rev)
        bwd_vectors = list(reversed(bwd_vectors))
        vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]
        return vectors

    def encode_sentence_batch(self, sentence_array):
        vectors_array = []
        for sentence in sentence_array:
            vectors_array.append(self.encode_sentence(sentence))
        return vectors_array

    def encode_sentence_batch_advanced(self, sentence_array):
        sentence_rev_array = []
        for sent in sentence_array:
            sentence_rev_arrayy.append(list(reversed(sent)))
        fwd_vectors = self.run_lstm_batch(self.enc_fwd_lstm.initial_state(), sentence_arry)
        bwd_vectors = self.run_lstm_batch(self.enc_bwd_lstm.initial_state(), sentence_rev_array)
        bwd_vectors_array = []
        for v in bwd_vectors:
            bwd_vectors_array.append(list(reversed(v)))
        fwd_vectors_array = fwd_vectors
        vectors_batch = []
        for fwd_vector, bwd_vector in zip(fwd_vectors_array, bwd_vectors_array):
            vector = [dy.concatenate(list(p)) for p in zip(fwd_vector, bwd_vector)]
            vectors_batch.append(vector)   
        return vectors_batch
    
    
if __name__ == "__main__":
    src_filename = './en-de/train.en-de.low.de'
    tgt_filename = './en-de/train.en-de.low.en'
    #filename = '../../../../dynet-base/dynet/examples/python/written.txt'
    #filename = 'txt.done.data'

    src_filename_test = './en-de/test.en-de.low.de'
    tgt_filename_test = './en-de/test.en-de.low.en'
    reader = CorpusReader()
    print "we are here debugging..."
    src_wids = reader.read_corpus_word(src_filename, 0)
    tgt_wids = reader.read_corpus_word(tgt_filename, 0)
    src_i2w = {i:w for w,i in src_wids.iteritems()}
    tgt_i2w = {i:w for w,i in tgt_wids.iteritems()}

    model = Model()     
    trainer = SimpleSGDTrainer(model)
    num_layers = 1
    input_dim = 128
    embedding_dim = 128
    src_vocab_size = len(src_wids)
    tgt_vocab_size = len(tgt_wids)
    minibatch_size = 16

    src_lookup = model.add_lookup_parameters((len(src_wids), embedding_dim))
    tgt_lookup = model.add_lookup_parameters((len(tgt_wids), embedding_dim))
    builder = LSTMBuilder
    minibatch_size = 32
    aed_b =  Attention(len(src_wids), len(tgt_wids),  model, input_dim, embedding_dim, src_lookup, tgt_lookup, minibatch_size, builder)

    def get_indexed(arr, src_flag):
        ret_arr = []
        for a in arr:
            #print a, wids[a], M[wids[a]].value()
            if src_flag == 1:
                ret_arr.append(src_wids[a])
            else:
                ret_arr.append(tgt_wids[a])
        return ret_arr  

    def get_indexed_batch(sentence_array):
        ret_ssent_arr = []
        ret_tsent_arr  = []
        words_mb = 0
        for ssent,tsent in sentence_array:
            #print sent
            ar_s = get_indexed(ssent.split(),1)
            ret_ssent_arr.append(ar_s)
            ar = get_indexed(tsent.split(),0)
            ret_tsent_arr.append(ar)
            words_mb += len(ar_s)
        return ret_ssent_arr, ret_tsent_arr, words_mb  



    # Accumulate training data
    # I am using this simple version as I dont need to do tokenization for this assignment. Infact, tokenization might be bad in this case.
    src_sentences  = []
    f = open(src_filename)
    for  line in f:
        line = line.strip()
        src_sentences.append( '<s>' + ' ' + line + ' ' + '</s>')

    tgt_sentences  = []
    f = open(tgt_filename)
    for  line in f:
        line = line.strip()
        tgt_sentences.append( '<s>' + ' ' + line + ' ' + '</s>')

    # Batch the training data ##############################################
    # Sort
    sentences = zip(src_sentences, tgt_sentences)
    sentences.sort(key=lambda x: -len(x))
    train_order = [x*minibatch_size for x in range(int((len(sentences)-1)/minibatch_size + 1))]
    test_order = train_order[-1]
    train_order = train_order[:-1]


    print ("startup time: %r" % (time.time() - start))
    # Perform training

    i = words = sents = loss = cumloss = dloss = 0
    for epoch in range(100):
        random.shuffle(train_order) 
        loss = 0
        c = 1
        for sentence_id in train_order:
            #print "Processing ", sentence
            #sentence = train_order[sentence_id]
            #sentence = sentence.split() 
            #if len(sentence) > 2:  
            #print "This is a valid sentence"
            if 3 > 2:  
                #print "This is a valid sentence"
                c = c+1
                print c, " out of ", len(train_order)
                if c%250 == 1:
                    ##print "I will print trainer status now"
                    trainer.status()
                    print "Loss: ", loss / words
                    print "Perplexity: ", math.exp(loss / words)
                    print ("time: %r" % (time.time() - start))
                    for jj in range(minibatch_size):
                        sentence_id = jj + test_order[0]
                        isents, idurs, words_minibatch_indexing = get_indexed_batch(sentences[sentence_id:sentence_id+minibatch_size])
                        src,tgt = sentences[sentence_id]
                        resynth = aed_b.generate(src)
                        tgt_resynth = " ".join([tgt_i2w[c] for c in resynth]).strip()
                        BLEUscore = nltk.translate.bleu_score.sentence_bleu([src], tgt_resynth)
                        print "BLEU: ", BLEUscore
                    #isent = get_indexed(src_sentence, 1)
                    #itype = get_indexed(tgt_sentence,0)
                    #resynth= red.generate(isent)

                    #resythn = red.sample(nchars= len(sentence),stop=wids["</s>"])
                    #print(" ".join([tgt_i2w[c] for c in resynth]).strip())
                    #print '\n'
                    #durs = durs[0:5]
                    #hypothesis = resynth
                    #reference = itype
                    #BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
                    #print "BLEU: ", BLEUscore

                    #BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
                    #print "BLEU: ", BLEUscore

                #     #print dloss / words
                #     loss = 0
                #     words = 0
                #     dloss = 0
                #     for _ in range(1):
                # 	     print ' '.join(k for k in sentence)
                #         samp = red.sample(nchars= len(sentence),stop=wids["</s>"])
                #         res = red.generate(get_indexed(sentence))
                #         print(" ".join([i2w[c] for c in res]).strip())

                #words += len(sentence) - 1
                isents, idurs, words_minibatch_indexing = get_indexed_batch(sentences[sentence_id:sentence_id+minibatch_size])

                #print isent
                #print "I will try to calculate error now"
                error, words_minibatch_loss = aed_b.get_loss_batch(isents,idurs)
                ####### I need to fix this sometime
                #print words_minibatch_indexing , words_minibatch_loss
                #assert words_minibatch_indexing == words_minibatch_loss
                words += words_minibatch_indexing
                #print "Obtained loss ", error.value()
                loss += error.value()
                #print "Added error"
                #print error.value()
                error.backward()
                trainer.update(1.0)
            print '\n'   
            print("ITER",epoch,loss)
            print '\n'
            trainer.status()
            trainer.update_epoch(1)