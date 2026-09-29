import os
import sys
from copy import deepcopy

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'cuts_plus_prototype'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'cuts_plus_prototype', 'vendor'))

from cuts_plus_rca import CUTSPlusRCAConfig, build_opt, make_synthetic_series, set_seed  # noqa: E402
from cuts_plus import MultiCAD, generate_batch_index_groups, materialize_batch  # noqa: E402
from utils.logger import MyLogger  # noqa: E402


def _old_latent_data_pred(multicad, x, y, mask_x, mask_y):
    """Reference copy of latent_data_pred as it was BEFORE deviation 7: recomputes Graph internally
    on every call, instead of receiving it precomputed. Kept only here to verify the hoisted version
    produces identical training results."""

    def sample_bernoulli(sample_matrix, batch_size):
        sample_matrix = sample_matrix[None].expand(batch_size, -1, -1)
        return torch.bernoulli(sample_matrix).float()

    multicad.fitting_model.train()
    multicad.data_pred_optimizer.zero_grad()

    graph = torch.einsum("nm,ml->nl", multicad.G, torch.sigmoid(multicad.GT))
    graph_sampled = sample_bernoulli(graph, multicad.args.batch_size)

    y_pred = multicad.fitting_model(x, mask_x, graph_sampled)
    loss = multicad.data_pred_loss(y * mask_y, y_pred * mask_y) / torch.mean(mask_y)
    loss.backward()
    multicad.data_pred_optimizer.step()
    return y_pred, loss


def _build_multicad_and_batches(tmp_path):
    config = CUTSPlusRCAConfig(
        synthetic_num_vars=10, synthetic_series_len=600, synthetic_num_anomalies=3,
        total_epoch=2, n_groups=4, group_policy='multiply_2_every_1', batch_size=16, seed=0,
        save_dir=str(tmp_path / 'models'), log_dir=str(tmp_path / 'runs'),
    )
    set_seed(config.seed)
    x, _, _, _ = make_synthetic_series(config)
    data = torch.from_numpy(x[:, :, None]).float()
    mask = torch.ones_like(data)

    opt = build_opt(config, config.synthetic_num_vars)
    log = MyLogger(log_dir=str(tmp_path / 'log'), stdout=False, stderr=False, tensorboard=False)
    multicad = MultiCAD(opt, log, device='cpu')

    # Force past epoch 0's group-refinement skip so self.GT/self.G already exist (mirrors what a real
    # epoch > 0 looks like - see the `if epoch_i != 0` skip in train()'s group-refinement block).
    # n_groups == n_nodes (full resolution) keeps the shapes simple: G is (n_nodes, n_groups).
    n = config.synthetic_num_vars
    multicad.n_groups = n
    multicad.G = torch.eye(n)
    multicad.GT = torch.nn.Parameter(torch.randn((n, n)) * 0.1)
    multicad.set_graph_optimizer(0)

    batch_index_groups = generate_batch_index_groups(
        config.input_step, opt.data_pred.pred_step, data.shape[0], bs=config.batch_size)
    batches = []
    x_offsets = torch.arange(-config.input_step, 0, dtype=torch.long)
    y_offsets = torch.arange(0, opt.data_pred.pred_step, dtype=torch.long)
    for g in batch_index_groups[:5]:
        t_idx = torch.tensor(g, dtype=torch.long)
        batches.append(materialize_batch(data, mask, t_idx, x_offsets, y_offsets))
    return multicad, batches


def test_hoisted_s1_matches_reference_training(tmp_path):
    multicad_old, batches = _build_multicad_and_batches(tmp_path / 'old')
    multicad_new, _ = _build_multicad_and_batches(tmp_path / 'new')
    # Same starting weights for both (both built with the same seed/config, but re-verify directly).
    for k in multicad_old.fitting_model.state_dict():
        assert torch.equal(multicad_old.fitting_model.state_dict()[k],
                            multicad_new.fitting_model.state_dict()[k])

    for x, y, t, mask_x, mask_y in batches:
        torch.manual_seed(123)  # reset RNG immediately before the stochastic sampling step in both
        _old_latent_data_pred(multicad_old, x, y, mask_x, mask_y)

    s1_graph = torch.einsum("nm,ml->nl", multicad_new.G, torch.sigmoid(multicad_new.GT)).detach()
    for x, y, t, mask_x, mask_y in batches:
        torch.manual_seed(123)
        multicad_new.latent_data_pred(x, y, mask_x, mask_y, s1_graph)

    old_state = multicad_old.fitting_model.state_dict()
    new_state = multicad_new.fitting_model.state_dict()
    for k in old_state:
        torch.testing.assert_close(old_state[k], new_state[k], rtol=1e-6, atol=1e-8)
