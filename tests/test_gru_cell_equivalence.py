import os
import sys

import torch
from einops import rearrange
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'cuts_plus_prototype', 'vendor'))

from model.cuts_plus_net import GRUCell, MPNN  # noqa: E402


class _OldMPNN(nn.Module):
    """Reference copy of MPNN as it was BEFORE deviation 2: recomputes x_messages internally on every
    call, instead of receiving it precomputed. Kept only here to verify the shared-computation version
    produces identical results."""

    def __init__(self, c_in, c_out, concat_h=True):
        super().__init__()
        self.concat_h = concat_h
        self.mlp = nn.Conv1d(c_in, c_out, kernel_size=1)

    def forward(self, x, h, graph):
        b, c, n = x.shape
        x_repeat = x[:, :, :, None].expand(-1, -1, -1, n)
        x_messages = torch.einsum('bcmn,bmn->bcmn', (x_repeat, graph))
        x_messages = rearrange(x_messages, 'b c m n -> b (c m) n')
        if self.concat_h:
            return self.mlp(torch.cat([x_messages, h], dim=1))
        return self.mlp(x_messages)


class _OldGRUCell(nn.Module):
    def __init__(self, d_in, num_units, n_nodes, concat_h=False, activation='tanh'):
        super().__init__()
        self.activation_fn = getattr(torch, activation)
        mpnn_channel = d_in * n_nodes + num_units if concat_h else d_in * n_nodes
        self.forget_gate = _OldMPNN(c_in=mpnn_channel, c_out=num_units, concat_h=concat_h)
        self.update_gate = _OldMPNN(c_in=mpnn_channel, c_out=num_units, concat_h=concat_h)
        self.c_gate = _OldMPNN(c_in=mpnn_channel, c_out=num_units, concat_h=concat_h)

    def forward(self, x, h, adj):
        r = torch.sigmoid(self.forget_gate(x, h, adj))
        u = torch.sigmoid(self.update_gate(x, h, adj))
        c = self.c_gate(x, r * h, adj)
        c = self.activation_fn(c)
        return u * h + (1. - u) * c


def _copy_weights(old, new):
    with torch.no_grad():
        for gate_name in ('forget_gate', 'update_gate', 'c_gate'):
            old_gate = getattr(old, gate_name).mlp
            new_gate = getattr(new, gate_name).mlp
            new_gate.weight.copy_(old_gate.weight)
            new_gate.bias.copy_(old_gate.bias)


def test_gru_cell_matches_reference_forward_and_backward():
    torch.manual_seed(0)
    b, d_in, num_units, n = 3, 2, 5, 13  # deliberately non-round sizes

    old = _OldGRUCell(d_in, num_units, n, concat_h=True).double()
    new = GRUCell(d_in, num_units, n, concat_h=True).double()
    _copy_weights(old, new)

    x = torch.randn(b, d_in, n, dtype=torch.float64, requires_grad=True)
    h = torch.randn(b, num_units, n, dtype=torch.float64, requires_grad=True)
    adj = torch.rand(b, n, n, dtype=torch.float64)
    x2, h2 = x.clone().detach().requires_grad_(True), h.clone().detach().requires_grad_(True)

    out_old = old(x, h, adj)
    out_new = new(x2, h2, adj)
    torch.testing.assert_close(out_old, out_new, rtol=1e-7, atol=1e-9)

    out_old.sum().backward()
    out_new.sum().backward()
    torch.testing.assert_close(x.grad, x2.grad, rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(h.grad, h2.grad, rtol=1e-7, atol=1e-9)
    for gate_name in ('forget_gate', 'update_gate', 'c_gate'):
        old_mlp = getattr(old, gate_name).mlp
        new_mlp = getattr(new, gate_name).mlp
        torch.testing.assert_close(old_mlp.weight.grad, new_mlp.weight.grad, rtol=1e-7, atol=1e-9)
        torch.testing.assert_close(old_mlp.bias.grad, new_mlp.bias.grad, rtol=1e-7, atol=1e-9)


def test_gru_cell_matches_reference_float32_concat_h_false():
    """Secondary check at the training dtype (float32) and the other concat_h setting."""
    torch.manual_seed(1)
    b, d_in, num_units, n = 4, 1, 32, 21

    old = _OldGRUCell(d_in, num_units, n, concat_h=False)
    new = GRUCell(d_in, num_units, n, concat_h=False)
    _copy_weights(old, new)

    x = torch.randn(b, d_in, n)
    h = torch.randn(b, num_units, n)
    adj = torch.rand(b, n, n)
    torch.testing.assert_close(old(x, h, adj), new(x, h, adj), rtol=1e-4, atol=1e-5)
