import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'cuts_plus_prototype', 'vendor'))

from model.cuts_plus_net import LocalConv1D  # noqa: E402


class _OldLocalConv1D(torch.nn.Module):
    """Reference copy of the pre-vectorization LocalConv1D (Python loop over per-node nn.Conv1d
    submodules), kept only here to verify the grouped-conv reimplementation produces equivalent
    results to the original per-node-loop algorithm."""

    def __init__(self, in_channels, out_channels, kernel_size, n_nodes):
        super().__init__()
        self.out_channel = out_channels
        self.conv_list = torch.nn.ModuleList([
            torch.nn.Conv1d(in_channels, out_channels, kernel_size) for _ in range(n_nodes)
        ])

    def forward(self, x):
        b, h, n = x.shape
        out = torch.zeros((b, self.out_channel, n), device=x.device, dtype=x.dtype)
        for i in range(n):
            out[..., i] = self.conv_list[i](x[..., i].unsqueeze(-1)).squeeze(-1)
        return out


def _copy_weights(old, new):
    with torch.no_grad():
        new.grouped_conv.weight.copy_(torch.cat([c.weight for c in old.conv_list], dim=0))
        new.grouped_conv.bias.copy_(torch.cat([c.bias for c in old.conv_list], dim=0))


def test_local_conv1d_matches_reference_forward_and_backward():
    torch.manual_seed(0)
    b, in_ch, out_ch, n = 4, 6, 3, 17  # deliberately non-power-of-2 sizes, small n for a fast test

    old = _OldLocalConv1D(in_ch, out_ch, kernel_size=1, n_nodes=n).double()
    new = LocalConv1D(in_ch, out_ch, kernel_size=1, n_nodes=n).double()
    _copy_weights(old, new)

    x = torch.randn(b, in_ch, n, dtype=torch.float64, requires_grad=True)
    x2 = x.clone().detach().requires_grad_(True)

    y_old = old(x)
    y_new = new(x2)
    torch.testing.assert_close(y_old, y_new, rtol=1e-7, atol=1e-9)

    y_old.sum().backward()
    y_new.sum().backward()
    torch.testing.assert_close(x.grad, x2.grad, rtol=1e-7, atol=1e-9)
    for i, c in enumerate(old.conv_list):
        torch.testing.assert_close(
            c.weight.grad, new.grouped_conv.weight.grad[i * out_ch:(i + 1) * out_ch], rtol=1e-7, atol=1e-9)
        torch.testing.assert_close(
            c.bias.grad, new.grouped_conv.bias.grad[i * out_ch:(i + 1) * out_ch], rtol=1e-7, atol=1e-9)


def test_local_conv1d_matches_reference_float32():
    """Secondary check at the actual training dtype (float32) with a looser tolerance, since fused-
    kernel vs. per-node accumulation can differ slightly in rounding."""
    torch.manual_seed(1)
    b, in_ch, out_ch, n = 8, 64, 1, 37  # in_ch=64 (2*hidden_ch=32), out_ch=1 mirrors real usage

    old = _OldLocalConv1D(in_ch, out_ch, kernel_size=1, n_nodes=n)
    new = LocalConv1D(in_ch, out_ch, kernel_size=1, n_nodes=n)
    _copy_weights(old, new)

    x = torch.randn(b, in_ch, n)
    torch.testing.assert_close(old(x), new(x), rtol=1e-4, atol=1e-5)
