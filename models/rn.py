import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import wargs

class ConvLayer(nn.Module):

    def __init__(self, d_in_chn, d_out_chn, n_windows=3, dense=False):

        super(ConvLayer, self).__init__()
        self.conv = nn.Conv1d(d_in_chn, d_out_chn, kernel_size=n_windows, stride=1,
                              padding=(n_windows -1 )/2)
        self.leakyRelu = nn.LeakyReLU(0.1)
        self.batchNorm = nn.BatchNorm1d(d_out_chn)
        self.dense = dense

    def forward(self, x):
        # (B, n_in_feats, L) -> (B, n_out_feats, L) or (B, n_in_feats+n_out_feats, L)
        y = self.conv(x)
        y = self.leakyRelu(y)
        y = self.batchNorm(y)
        return tc.cat([x, y], dim=1) if self.dense is True else y

class RelationLayer_new(nn.Module):

    def __init__(self, d_in, d_out, n_windows, d_chn, d_mlp=128):

        super(RelationLayer, self).__init__()

        self.C_in = input_size

        self.fws = n_windows
        self.ffs = d_chn
        self.N = len(self.fws)
        self.d_mlp = d_mlp

        self.convLayer = ConvLayer(input_size, d_chn, n_windows)
        cnn_feats_size = sum([k for k in self.ffs])

        self.g_mlp = nn.Sequential(
            nn.Linear(2 * (cnn_feats_size+2) + wargs.enc_hid_size, d_mlp),
            nn.LeakyReLU(0.1),
            nn.Linear(d_mlp, d_mlp),
            nn.LeakyReLU(0.1),
            nn.Linear(d_mlp, d_mlp),
            nn.LeakyReLU(0.1),
            nn.Linear(d_mlp, d_mlp),
            nn.LeakyReLU(0.1)
        )

        self.f_mlp = nn.Sequential(
            nn.Linear(d_mlp, d_mlp),
            nn.LeakyReLU(0.1),
            nn.Linear(d_mlp, wargs.dec_hid_size),
            nn.LeakyReLU(0.1)
        )

    # prepare coord tensor
    def cvt_coord(self, idx, L):
        return [( idx / np.sqrt(L) - 2 ) / 2., ( idx % np.sqrt(L) - 2 ) / 2.]

    def forward(self, x, h, xs_mask=None):

        L, B, E = x.size()
        x = x.permute(1, 2, 0)    # (B, E, L)

        ''' CNN Layer '''
        x = self.convLayer(x)   # (B, sum_feats_size, L')
        L = x.size(-1)

        # (B, sum_feats_size, L) -> (B, L, sum_feats_size)
        x = x.permute(0, 2, 1)
        # (B, L, sum_feats_size)

        self.coord_tensor = tc.FloatTensor(B, L, 2)
        if wargs.gpu_id is not None: self.coord_tensor = self.coord_tensor.cuda(wargs.gpu_id[0])
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((B, L, 2))
        for _i in range(L): np_coord_tensor[:, _i, :] = np.array( self.cvt_coord(_i, L) )
        self.coord_tensor.data.copy_(tc.from_numpy(np_coord_tensor))

        # add coordinates
        x = tc.cat([x, self.coord_tensor], dim=2)
        # (B, L, sum_feats_size+2)

        # add question everywhere
        h = tc.unsqueeze(h, 1)      # (B, E) -> (B, 1, E)
        h = h.repeat(1, L, 1)     # (B, 1, E) -> (B, L, E)
        h = tc.unsqueeze(h, 2)      # (B, L, E) -> (B, L, 1, E)

        # cast all pairs against each other
        x_i = tc.unsqueeze(x, 1)        # (B, 1, L, sum_feats_size+2)
        x_i = x_i.repeat(1, L, 1, 1)    # (B, L, L, sum_feats_size+2)

        x_j = tc.unsqueeze(x, 2)        # (B, L, 1, sum_feats_size+2)
        x_j = tc.cat([x_j, h], 3)       # (B, L, 1, sum_feats_size+2)
        x_j = x_j.repeat(1, 1, L, 1)    # (B, L, L, sum_feats_size+2+E)

        # concatenate all together
        x = tc.cat([x_i, x_j], 3)       # (B, L, L, 2*(sum_feats_size+2)+E)

        ''' Graph Propagation Layer '''
        #if xs_mask is not None: xs_h = xs_h * xs_mask[:, :, None]
        #x = x[None, :, :, :].expand(L, L, B, self.cnn_feats_size)
        #x = tc.cat([x, x.transpose(0, 1)], dim=-1)

        x = self.g_mlp(x)
        #if xs_mask is not None: xs_h = xs_h * xs_mask[:, :, None]

        x = x.view(B, L*L, self.d_mlp)    # (B, L*L, d_mlp)
        x = x.sum(1).squeeze()

        ''' MLP Layer '''
        return self.f_mlp(x)


class RelationLayer2(nn.Module):

    def __init__(self, d_in, d_out, n_windows, d_chns, d_mlp=128, n_conv_layers=4, dense=False):

        super(RelationLayer, self).__init__()

        self.fws = n_windows
        self.d_chns = d_chns
        self.N = len(self.fws)

        self.cnnlayer = nn.ModuleList(
            [
                nn.Sequential(*[
                    ConvLayer(d_in_chn=( d_in if i == 0 else ( (d_in + i * self.d_chns[k]) \
                                        if dense is True else self.d_chns[k] ) ),
                              d_out_chn=self.d_chns[k], n_windows=self.fws[k],
                              dense=False if i == n_conv_layers - 1 else dense) \
                    for i in range(n_conv_layers)
                ])
                for k in range(self.N)
            ]
        )

        self.d_chn = sum([k for k in self.d_chns])
        #self.mlp = nn.Sequential(
        #    *[nn.Linear(2 * self.d_chn if i == 0 else d_mlp, d_mlp), nn.LeakyReLU(0.1) \
        #      for i in range(n_mlp_layers)]
        #)

        self.mlp = nn.Sequential(
            nn.Linear(2 * self.d_chn, d_mlp),
            nn.LeakyReLU(0.1),
            nn.Linear(d_mlp, d_mlp),
            nn.LeakyReLU(0.1),
            nn.Linear(d_mlp, d_mlp),
            nn.LeakyReLU(0.1),
            nn.Linear(d_mlp, d_mlp),
            nn.LeakyReLU(0.1)
        )

        self.mlp_layer = nn.Sequential(
            nn.Linear(d_mlp, d_mlp),
            nn.LeakyReLU(0.1),
            nn.Linear(d_mlp, d_out),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x, xs_mask=None):

        L, B, E = x.size()
        if xs_mask is not None: x = x * xs_mask[:, :, None]
        x = x.permute(1, 2, 0)    # -> (B, E, L)
        # CNN Layer
        # (B, E, L) -> [(B, d_out_chn0, L), (B, d_out_chn1, L), ..., ]
        x = [self.cnnlayer[i](x) for i in range(self.N)]
        #x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = tc.cat(x, dim=1)
        # (B, d_out_chn_sum, L) -> (L, B, d_out_chn_sum)
        x = x.permute(2, 0, 1)

        # Graph Propagation Layer (in: (L, B, d_out_chn_sum))
        if xs_mask is not None: x = x * xs_mask[:, :, None]
        x1 = x[:, None, :, :].repeat(1, L, 1, 1)
        x2 = x[None, :, :, :].repeat(L, 1, 1, 1)
        x = tc.cat([x1, x2], dim=-1)    # (L, _L, B, d_out_chn_sum)

        x = self.mlp(x)
        if xs_mask is not None: x = x * xs_mask[:, None, :, None]
        x = x.mean(dim=1)

        # MLP Layer
        x = self.mlp_layer(x)
        if xs_mask is not None: x = x * xs_mask[:, :, None]

        return x

class RelationLayer(nn.Module):

    def __init__(self, d_in, d_out, fltr_windows, d_fltr_feats, d_mlp=128):

        super(RelationLayer, self).__init__()

        self.n_windows = len(fltr_windows)
        assert len(d_fltr_feats) == self.n_windows, 'Require same number of windows and features'
        self.cnnlayer = nn.ModuleList(
            [
                nn.Conv1d(in_channels=1,
                          out_channels=d_fltr_feats[k],
                          kernel_size=d_in * fltr_windows[k],
                          stride=d_in,
                          padding=( (fltr_windows[k] - 1) / 2 ) * d_in
                         ) for k in range(self.n_windows)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(d_fltr_feats[k]) for k in range(self.n_windows)])
        self.leakyRelu = nn.LeakyReLU(0.1)

        self.d_sum_fltr_feats = sum(d_fltr_feats)

        self.mlp = nn.Sequential(
            nn.Linear(2 * self.d_sum_fltr_feats, d_mlp),
            nn.LeakyReLU(0.1),
            nn.Linear(d_mlp, d_mlp),
            nn.LeakyReLU(0.1),
            nn.Linear(d_mlp, d_mlp),
            nn.LeakyReLU(0.1),
            nn.Linear(d_mlp, d_mlp),
            nn.LeakyReLU(0.1)
        )

        self.mlp_layer = nn.Sequential(
            nn.Linear(d_mlp, d_mlp),
            nn.LeakyReLU(0.1),
            nn.Linear(d_mlp, d_out),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x, xs_mask=None):

        L, B, E = x.size()
        if xs_mask is not None: x = x * xs_mask[:, :, None]
        x = x.permute(1, 0, 2)    # (B, L, d_in)

        ''' CNN Layer '''
        # (B, L, d_in) -> (B, d_in * L) -> (B, 1, d_in * L)
        x = x.contiguous().view(B, -1)[:, None, :]
        # (B, 1, d_in * L) -> [ (B, d_fltr_feats[k], L) ]
        x = [self.leakyRelu(self.bns[k](self.cnnlayer[k](x))) for k in range(self.n_windows)]
        #x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        # [ (B, d_fltr_feats[k], L) ] -> (B, d_sum_fltr_feats, L)
        x = tc.cat(x, dim=1)
        # (B, d_sum_fltr_feats, L) -> (L, B, d_sum_fltr_feats)
        x = x.permute(2, 0, 1)

        ''''' Graph Propagation Layer '''''
        # (in: (L, B, d_sum_fltr_feats))
        if xs_mask is not None: x = x * xs_mask[:, :, None]
        x1 = x[:, None, :, :].repeat(1, L, 1, 1)
        x2 = x[None, :, :, :].repeat(L, 1, 1, 1)
        x = tc.cat([x1, x2], dim=-1)    # (L, _L, B, 2 * d_sum_fltr_feats)
        x = self.mlp(x)
        if xs_mask is not None: x = x * xs_mask[:, None, :, None]
        # (L, _L, B, 2 * d_sum_fltr_feats) -> (L, B, 2 * d_sum_fltr_feats)
        x = x.mean(dim=1)

        ''' MLP Layer '''
        x = self.mlp_layer(x)
        if xs_mask is not None: x = x * xs_mask[:, :, None]

        return x



