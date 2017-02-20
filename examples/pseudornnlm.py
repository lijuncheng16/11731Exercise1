'''
1) How to use RNN function in dynet
2) How to train minibatch using lookup_batch, sum_batches, pickneglogsoftmax_batch

'''
import dynet as dy
import numpy as np
import os,sys,time,random,math,argparse
from collections import defaultdict

# take (or set) hyper parameters
def main():
  parser = argparse.ArgumentParser()

  # Hyper-parameters
  parser.add_argument('-batch_size', default=32, type=int, help='Batch size')
  # rnn_size, emb_size, ...

  # dynet options
  parser.add_argument('--dynet-mem')
  parser.add_argument('--dynet-gpu',action='store_true',default=False)

  if args.dynet_gpu:  # the python gpu switch.
    print('using GPU')
    import _gdynet as dy

  # inputs parametesr
  parser.add_argument('training')
  parser.add_argument('validation')
  args = parser.parse_args()


  # read vocab, read_file
  vocab = XXX(args.training)
  trainig = XXX(args.training, vocab)
  validation = XXX(args.validation, vocab)

  rnnlm = RNNLM(
    ...
  )

class RNNLM():
  def __init__(self, training, validation, vocab, ...)

    # define dynet model
    model = dy.Model()
    trainer = dy.SimpleSGDTrainer(model)
    builder = dy.SimpleRNNBuilder(layer_depth, emb_size, hidden_size, model)
    lookup = model.add_lookup_parameters((self.vocab_size, self.emb_size))
    W_y = model.add_parameters((vocab_size, hidden_size))
    b_y = model.add_parameters((vocab_size))

    # sort train/valid sentences in descending order and prepare minibatch
    train_set.sort(key=lambda x: -len(x))
    # train_order = [0, 32, 64, ...]
    train_order = [x*batch_size for x in range((len(train_set)-1) / batch_size)]

    for eidx, epoch in enumerate(range(num_epochs)):
      train_loss, train_words = 0, 0
      random.shuffle(train_order)
      for iter, tidx in enumerate(train_order):
        batch = train_set[tidx:tidx+batch_size]
        # batch  = [ [a1,a2,a3,a4,a5], [b1,b2,b3,b4,b5], [c1,c2,c3,c4] ..]
        loss,words = _step_batch(batch)
        loss.backward()
        trainer.update()

        # check training loss/perplexity
        # check validation  loss/perplexity
        # check sampling
      trainer.update_epoch(1.0)




  def _step_batch(batch):
    dy.renew_cg()

    # batch  = [ [a1,a2,a3,a4,a5], [b1,b2,b3,b4,b5], [c1,c2,c3,c4] ..]
    # transpose the batch into 
    #   wids: [[a1,b1,c1,..], [a2,b2,c2,..], .. [a5,b5,START]]
    #   masks: [1,1,1,..], [1,1,1,..], ...[1,1,0,..]]
    wids = []
    masks = []
    total_words = XXX

    # initialize RNN: [<s>,a1,b1,c1,..]
    init_state = builder.initialize_state()


    # start the RNN by inputting <s>
    start_state = dy.lookup_batch(lookup, START * len(batch))
    state =  init_state.add_input(start_state)


    losses = []
    for ys, mask in zip(wids, masks):
      # batch and mask at a time step: [a1,b1,c1,..], [,1,1,1,...]

      output = state.output()
      ystar = XXXX[W_y, b_y, output]
      loss = XXXX[ystar, ys] #pickneglogsoftmax_batch

      # masking
      loss = XXX(loss, mask)

      # penalize UNK loss
      loss = XXX(loss, ys) 1e-7

      losses.append(loss)

      # update state of RNN
      state = state.add_input(dy.lookup_batch(lookup, ys))

    return dy.sum_batches(dy.esum(losses)), total_words

########################################################################
# Reference:
# http://dynet.readthedocs.io/en/latest/tutorials_notebooks/RNNs.html
# http://dynet.readthedocs.io/en/latest/minibatch.html
# Reference perplexity:
# after one epoch: 210.43 / 240.93
# after ten epochs: 110.09 / 138.07
########################################################################

if __name__ == '__main__': main()

