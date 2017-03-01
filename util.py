from collections import defaultdict
from itertools import count
class Vocab:
    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(count(0).next)
        self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in w2i.iteritems()}
    @classmethod
    def from_corpus(cls, corpus):
        w2i = defaultdict(count(0).next)
        for sent in corpus:
            [w2i[word] for word in sent]
        return Vocab(w2i)

    def size(self): return len(self.w2i.keys())

class CorpusReader:
    def __init__(self, fname='txt.done.data'):
        self.fname = fname
        self.wids = defaultdict(lambda:  len(self.wids))
        self.wids["<unk>"] = 0
        self.wids["<s>"] = 1
        self.wids["</s>"] = 2
        self.wids["<S>"] = 3
    #def __iter__(self):
        #for line in file(self.fname):
            #line = line.strip().split()
            #line = [' ' if x == '' else x for x in line]
            #yield line
    def read_corpus_word(self, fname, tokenizer_flag):
        self.fname = fname
        print self.fname
        if tokenizer_flag == 1:
            tokenizer = RegexpTokenizer(r'\w+')
        f = open(self.fname)
        for line in f:
            line = line.split('\n')[0]          
            words = line.split()
            for word in words:
                if tokenizer_flag == 1:
                    word = tokenizer.tokenize(word)[0]
                else:
                    pass
                self.wids[word]
        f.close()         
        return self.wids

class CharsCorpusReader:
    def __init__(self, fname, begin=None):
        self.fname = fname
        self.begin = begin
    def __iter__(self):
        begin = self.begin
        for line in file(self.fname):
            line = list(line)
            if begin:
                line = [begin] + line
            yield line
