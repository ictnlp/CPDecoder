from __future__ import division, print_function

import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from tools.utils import MAX_SEQ_SIZE, wlog, PAD
from .nn_utils import PositionwiseFeedForward
from .attention import MultiHeadAttention

'''
Get an attention mask to avoid using the subsequent info.
Args: d_model: int
Returns: (LongTensor): future_mask [1, d_model, d_model]
'''
def get_attn_future_mask(size):

    attn_shape = (size, size)
    future_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    future_mask = tc.from_numpy(future_mask)

    return future_mask

'''
Compose with three layers
    Args:
        d_model(int): the dimension of keys/values/queries in
                      MultiHeadAttention, also the input size of
                      the first-layer of the PositionwiseFeedForward.
        n_head(int): the number of head for MultiHeadAttention.
        hidden_size(int): the second-layer of the PositionwiseFeedForward.
        droput(float): dropout probability(0-1.0).
'''
class SelfAttDecoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 n_head=8,
                 d_ff_filter=2048,
                 att_dropout=0.3,
                 residual_dropout=0.,
                 relu_dropout=0.,
                 self_attn_type='scaled-dot',
                 decoder_normalize_before=False):

        super(SelfAttDecoderLayer, self).__init__()

        self.decoder_normalize_before = decoder_normalize_before
        self.layer_norm_0 = nn.LayerNorm(d_model, elementwise_affine=True)

        self.self_attn_type = self_attn_type
        if self_attn_type == 'scaled-dot':
            self.self_attn = MultiHeadAttention(d_model, n_head, dropout_prob=att_dropout)
        elif self_attn_type == 'average':
            self.self_attn = AverageAttention(d_model, dropout_prob=att_dropout)

        self.residual_dropout_prob = residual_dropout

        self.layer_norm_1 = nn.LayerNorm(d_model, elementwise_affine=True)
        self.trg_src_attn = MultiHeadAttention(d_model, n_head, dropout_prob=att_dropout)

        self.layer_norm_2 = nn.LayerNorm(d_model, elementwise_affine=True)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_ff_filter, d_model, dropout_prob=relu_dropout)

    def forward(self, x, enc_output, trg_self_attn_mask=None, trg_src_attn_mask=None):
        '''
        Args:
            x (FloatTensor):                [batch_size, trg_len, d_model]
            enc_output (FloatTensor):       [batch_size, src_len, d_model]
            trg_self_attn_mask (LongTensor):[batch_size, trg_len, trg_len]
            trg_src_attn_mask  (LongTensor):[batch_size, trg_len, src_len]
        Returns: (FloatTensor, FloatTensor, FloatTensor, FloatTensor):
            dec_output:         [batch_size, trg_len, d_model]
            trg_self_attns:     [batch_size, n_head, trg_len, trg_len]
            trg_src_attns:      [batch_size, n_head, trg_len, src_len]
            one_dec_enc_attn:   [batch_size, trg_len, src_len]
        '''
        # target self-attention
        residual = x
        if self.decoder_normalize_before is True:
            x = self.layer_norm_0(x)     # before 'n' for preprocess

        # trg_self_attn_mask: (batch_size, trg_len, trg_len)
        if self.self_attn_type == 'scaled-dot':
            x, trg_self_attns = self.self_attn(x, x, x, attn_mask=trg_self_attn_mask)
            # query:                [batch_size, trg_len, d_model]
            # trg_self_attns:       [batch_size, n_head, trg_len, trg_len]
            # one_dec_self_attn:    [batch_size, trg_len, trg_len]
        elif self.self_attn_type == 'average':
            query, attn = self.self_attn(input_norm, mask=trg_self_attn_mask,
                                         layer_cache=layer_cache, step=step)

        x = F.dropout(x, p=self.residual_dropout_prob, training=self.training)
        x = residual + x    # 'da' for postprocess
        if self.decoder_normalize_before is False:
            x = self.layer_norm_0(x)

        # encoder-decoder attention
        residual = x
        if self.decoder_normalize_before is True:
            x = self.layer_norm_1(x)   # before 'n' for preprocess

        # trg_src_attn_mask: (batch_size, trg_len, src_len)
        x, trg_src_attns = self.trg_src_attn(enc_output, enc_output, x, attn_mask=trg_src_attn_mask)
        # x:                    [batch_size, trg_len, d_model]
        # trg_src_attns:        [batch_size, trg_len, src_len]

        x = F.dropout(x, p=self.residual_dropout_prob, training=self.training)
        x = residual + x    # before 'da' for postprocess
        if self.decoder_normalize_before is False:
            x = self.layer_norm_1(x)

        # feed forward
        residual = x
        if self.decoder_normalize_before is True:
            x = self.layer_norm_2(x)   # 'n' for preprocess

        x = self.pos_ffn(x)

        x = F.dropout(x, p=self.residual_dropout_prob, training=self.training)
        x = residual + x    # 'da' for postprocess
        if self.decoder_normalize_before is False:
            x = self.layer_norm_2(x)

        return x, trg_self_attns, trg_src_attns

''' A decoder model with self attention mechanism '''
class SelfAttDecoder(nn.Module):

    def __init__(self, trg_emb,
                 n_layers=6,
                 d_model=512,
                 n_head=8,
                 d_ff_filter=1024,
                 att_dropout=0.3,
                 residual_dropout=0.,
                 relu_dropout=0.,
                 self_attn_type='scaled-dot',
                 proj_share_weight=False,
                 decoder_normalize_before=False):

        wlog('Transformer decoder ========================= ')
        wlog('\ttrg_word_emb:       {}'.format(trg_emb.we.weight.size()))
        wlog('\tn_layers:           {}'.format(n_layers))
        wlog('\tn_head:             {}'.format(n_head))
        wlog('\td_model:            {}'.format(d_model))
        wlog('\td_ffn_filter:       {}'.format(d_ff_filter))
        wlog('\tatt_dropout:        {}'.format(att_dropout))
        wlog('\tresidual_dropout:   {}'.format(residual_dropout))
        wlog('\trelu_dropout:       {}'.format(relu_dropout))
        wlog('\tproj_share_weight:  {}'.format(proj_share_weight))

        super(SelfAttDecoder, self).__init__()

        self.layer_stack = nn.ModuleList([
            SelfAttDecoderLayer(d_model,
                                n_head,
                                d_ff_filter,
                                att_dropout=att_dropout,
                                residual_dropout=residual_dropout,
                                relu_dropout=relu_dropout,
                                self_attn_type=self_attn_type,
                                decoder_normalize_before=decoder_normalize_before)
            for _ in range(n_layers)])

        self.trg_word_emb = trg_emb
        if decoder_normalize_before is True:
            self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=True)
        self.decoder_normalize_before = decoder_normalize_before

    def forward(self, trg_seq, src_seq, enc_output, trg_mask=None, src_mask=None):

        src_B, src_L = src_seq.size()
        trg_B, trg_L = trg_seq.size()
        assert src_B == trg_B

        '''
        Get an attention mask to avoid using the subsequent info.
        array([[[0, 1, 1],
                [0, 0, 1],
                [0, 0, 0]]], dtype=uint8)
        '''
        trg_src_attn_mask = None if src_mask is None else src_mask.unsqueeze(1).expand(src_B, trg_L, src_L)
        trg_self_attn_mask = None if trg_mask is None else trg_mask.unsqueeze(1).expand(trg_B, trg_L, trg_L)

        with tc.no_grad():
            if trg_self_attn_mask is not None:
                future_mask = tc.tril(tc.ones(trg_L, trg_L), diagonal=0, out=None).cuda()
                trg_self_attn_mask = tc.gt(trg_self_attn_mask + future_mask[None, :, :], 1)

        _, dec_output = self.trg_word_emb(trg_seq)

        #nlayer_outputs, nlayer_self_attns, nlayer_attns = [], [], []
        for dec_layer in self.layer_stack:
            dec_output, trg_self_attns, trg_src_attns = dec_layer(
                dec_output, enc_output,
                trg_self_attn_mask=trg_self_attn_mask,
                trg_src_attn_mask=trg_src_attn_mask)
            #nlayer_outputs += [dec_output]
            #nlayer_self_attns += [trg_self_attns]
            #nlayer_attns += [trg_src_attns]

        if self.decoder_normalize_before is True:
            dec_output = self.layer_norm(dec_output)    # layer norm for the last layer output

        #return (dec_output, nlayer_self_attns, nlayer_attns)
        return dec_output, trg_self_attns, trg_src_attns


