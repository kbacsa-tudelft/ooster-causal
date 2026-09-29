# Vendored (near-verbatim) from https://github.com/jarrycyx/UNN/blob/main/CUTS_Plus/model/cuts_plus_net.py
# (MIT license, see ../LICENSE). One deliberate deviation from upstream, kept as the only edit so this
# stays otherwise faithful to the source:
#   1. Performance-only (same math, same results, no behavior change): LocalConv1D's forward used to
#      loop over n_nodes in Python, calling one separate nn.Conv1d per node - each node's independent
#      (kernel_size=1) affine map stayed mathematically independent (no weight sharing across nodes,
#      identical capacity), but was computed as n_nodes separate tiny CUDA kernel launches. On real
#      workloads (n_nodes~490, called thousands of times/epoch via CUTS_Plus_Net's S1/data-prediction
#      and S2/graph-discovery training loops in vendor/cuts_plus.py), this Python-loop kernel-launch
#      overhead was the dominant cost of an observed ~30 min/epoch. Reimplemented using a single
#      grouped nn.Conv1d (groups=n_nodes) that computes the same n_nodes independent affine maps in
#      one fused kernel launch - see tests/test_local_conv1d_equivalence.py for a numerical
#      equivalence check (forward and backward) against the original per-node-loop algorithm.

import torch
from einops import rearrange
from torch import nn

class GRUCell(nn.Module):

    def __init__(self, d_in, num_units, n_nodes, concat_h=False, activation='tanh'):
        super(GRUCell, self).__init__()
        self.activation_fn = getattr(torch, activation)

        mpnn_channel = d_in*n_nodes+num_units if concat_h else d_in*n_nodes
        self.forget_gate = MPNN(c_in=mpnn_channel, c_out=num_units, concat_h=concat_h)
        self.update_gate = MPNN(c_in=mpnn_channel, c_out=num_units, concat_h=concat_h)
        self.c_gate = MPNN(c_in=mpnn_channel, c_out=num_units, concat_h=concat_h)

    def forward(self, x, h, adj):
        """
        :param x: (B, input_dim, num_nodes)
        :param h: (B, num_units, num_nodes)
        :param adj: (num_nodes, num_nodes)
        :return:
        """
        # we start with bias 1.0 to not reset and not update
        r = torch.sigmoid(self.forget_gate(x, h, adj))
        u = torch.sigmoid(self.update_gate(x, h, adj))
        c = self.c_gate(x, r * h, adj)  # batch_size, self._num_nodes * output_size
        c = self.activation_fn(c)
        return u * h + (1. - u) * c


class MPNN(nn.Module):
    def __init__(self, c_in, c_out, concat_h=True):
        super(MPNN, self).__init__()
        self.concat_h = concat_h
        self.mlp = nn.Conv1d(c_in, c_out, kernel_size=1)
        
    def forward(self, x, h, graph):
        b, c, n = x.shape
        
        x_repeat = x[:, :, :, None].expand(-1, -1, -1, n) # [b, c, n, n]
        # graph = rearrange(graph, 'b n m -> b m n')
        x_messages = torch.einsum('bcmn,bmn->bcmn', (x_repeat, graph))
        x_messages = rearrange(x_messages, 'b c m n -> b (c m) n')
        
        if self.concat_h:
            out = self.mlp(torch.cat([x_messages, h], dim=1))
        else:
            out = self.mlp(x_messages)
        return out


class LocalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_nodes):
        super(LocalConv1D, self).__init__()
        assert kernel_size == 1, (
            "grouped-conv LocalConv1D only supports kernel_size=1 (the only value ever "
            "instantiated in this codebase - larger kernels can't slide over the length-1 "
            "input this layer always receives)"
        )
        self.out_channel = out_channels
        self.n_nodes = n_nodes
        self.grouped_conv = nn.Conv1d(
            in_channels=in_channels * n_nodes,
            out_channels=out_channels * n_nodes,
            kernel_size=1,
            groups=n_nodes,
        )

    def forward(self, x): # x: [batch, features, nodes]
        b, h, n = x.shape
        x_node_major = x.permute(0, 2, 1).reshape(b, n * h, 1)  # node-major channel layout
        out = self.grouped_conv(x_node_major)                   # [b, n*out_channel, 1]
        out = out.reshape(b, n, self.out_channel).permute(0, 2, 1)
        return out


class CUTS_Plus_Net(nn.Module):
    def __init__(self, n_nodes,
                 in_ch=1,
                 hidden_ch=32,
                 n_layers=1,
                 shared_weights_decoder=False,
                 concat_h=False,):
        super().__init__()
        self.in_ch = in_ch
        self.hidden_ch = hidden_ch
        self.n_layers = n_layers
        
        self.conv_encoder1 = nn.Conv1d(in_channels=hidden_ch, out_channels=hidden_ch, kernel_size=1)
        self.conv_encoder2 = nn.Conv1d(in_channels=2*hidden_ch, out_channels=hidden_ch, kernel_size=1)
        if shared_weights_decoder:
            self.decoder = nn.Sequential(
                nn.Conv1d(in_channels=2*hidden_ch, out_channels=in_ch, kernel_size=1),
                # nn.LeakyReLU(),
                # nn.Conv1d(in_channels=hidden_ch, out_channels=hidden_ch, kernel_size=1),
                # nn.LeakyReLU(),
                # nn.Conv1d(in_channels=hidden_ch, out_channels=in_ch, kernel_size=1),
                # nn.LeakyReLU(),
            )
        else:
            self.decoder = nn.Sequential(
                LocalConv1D(in_channels=2*hidden_ch, out_channels=in_ch, kernel_size=1, n_nodes=n_nodes),
                # nn.LeakyReLU(),
                # LocalConv1D(in_channels=hidden_ch, out_channels=hidden_ch, kernel_size=1, n_nodes=n_nodes),
                # nn.LeakyReLU(),
                # LocalConv1D(in_channels=hidden_ch, out_channels=in_ch, kernel_size=1, n_nodes=n_nodes),
                # nn.LeakyReLU(),
            )
        # self.act = nn.PReLU()
        self.act = nn.LeakyReLU()
        
        self.cells = nn.ModuleList()
        for i in range(self.n_layers):
            self.cells.append(GRUCell(d_in=in_ch if i==0 else hidden_ch, 
                                      num_units=hidden_ch, 
                                      n_nodes=n_nodes,
                                      concat_h=concat_h))
                
        self.h0 = self.init_state(n_nodes)
    
    def init_state(self, n_nodes):
        h = []
        for layer in range(self.n_layers):
            h.append(nn.parameter.Parameter(torch.zeros([self.hidden_ch, n_nodes])))
        return nn.ParameterList(h)
    
    def update_state(self, x, h, graph):
        rnn_in = x
        for layer in range(self.n_layers):
            rnn_in = h[layer] = self.cells[layer](rnn_in, h[layer], graph)
        return h
    
    def forward(self, x, mask, fwd_graph):
        x = rearrange(x, 'b n s c -> b c n s')
        # fwd_graph = torch.ones_like(fwd_graph)
        # mask = torch.ones_like(x).byte()
        bs, in_ch, n_nodes, steps = x.shape
        
        h = [h_.expand(bs, -1, -1) for h_ in self.h0.to(x.device)]
        
        pred = []
        for step in range(steps):
            x_now = x[..., step] # [batches, in_ch, nodes]
            
            """Update state"""
            h = self.update_state(x_now, h, fwd_graph)
            h_now = h[-1]
            
            """Prediction"""
            x_repr = self.act(self.conv_encoder1(h_now)) # [batches, hidden_ch, nodes]
            x_repr = self.act(self.conv_encoder2(torch.cat([x_repr, h_now], dim=1))) # [batches, hidden_ch, nodes]
            x_repr = torch.cat([x_repr, h_now], dim=1) # [batches, 2*hidden_ch, nodes]
            x_hat2 = self.decoder(x_repr) # [batches, in_ch, nodes]
            pred.append(x_hat2)
            
        
        pred = torch.stack(pred, dim=-1)
        pred = rearrange(pred, 'b c n s -> b n s c')
        return pred[:, :, -1:]

