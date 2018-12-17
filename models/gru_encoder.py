from __future__ import division
import torch as tc
import torch.nn as nn
import wargs
#from gru import LGRU, TGRU
from tools.utils import *

'''
    Bi-directional Transition Gated Recurrent Unit network encoder
    input args:
        src_emb:        class WordEmbedding
        enc_hid_size:   the size of TGRU hidden state
        n_layers:       layer nubmer of encoder
'''
class StackedGRUEncoder(nn.Module):

    def __init__(self,
                 src_emb,
                 enc_hid_size=512,
                 dropout_prob=0.3,
                 n_layers=2,
                 bidirectional=True,
                 prefix='GRU_Encoder', **kwargs):

        super(StackedGRUEncoder, self).__init__()

        self.src_word_emb = src_emb
        n_embed = src_emb.n_embed
        self.enc_hid_size = enc_hid_size
        f = lambda name: str_cat(prefix, name)  # return 'Encoder_' + parameters name

        self.bigru = nn.GRU(input_size=n_embed, hidden_size=self.enc_hid_size,
                            num_layers=n_layers, bias=True, batch_first=True,
                            dropout=dropout_prob, bidirectional=bidirectional)
        self.bidirectional = bidirectional
        self.n_layers = n_layers

    def forward(self, xs, xs_mask=None):

        if xs.dim() == 3: xs_e = xs
        else: x_w_e, xs_e = self.src_word_emb(xs)

        self.bigru.flatten_parameters()

        #if self.bidirectional is False:
        #    h0 = tc.zeros(batch_size, self.enc_hid_size, requires_grad=False)
        #else:
        #    h0 = tc.zeros(2, batch_size, self.enc_hid_size, requires_grad=False)
        #print xs_e.size(), h0.size()
        #output, hn = self.bigru(xs_e, h0)
        output, hn = self.bigru(xs_e)

        return output * xs_mask[:, :, None]

