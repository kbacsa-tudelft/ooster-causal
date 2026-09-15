import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'cuts_plus_prototype'))

from cuts_plus_rca import CUTSPlusRCAConfig, run_pipeline  # noqa: E402


def _build_config(tmp_path):
    return CUTSPlusRCAConfig(
        synthetic_num_vars=10,
        synthetic_series_len=1200,
        synthetic_num_anomalies=5,
        total_epoch=3,
        n_groups=4,
        group_policy='multiply_2_every_1',
        save_dir=str(tmp_path / 'saved_models'),
        log_dir=str(tmp_path / 'runs'),
        seed=0,
    )


def test_full_pipeline_on_synthetic_data(tmp_path):
    config = _build_config(tmp_path)

    multicad, graph, causal_metrics, rc_metrics = run_pipeline(config, log_dir_name='test')

    assert graph.shape == (config.synthetic_num_vars, config.synthetic_num_vars)
    assert np.isfinite(causal_metrics['auroc'])
    assert np.isfinite(causal_metrics['f1'])

    for key in ('ac@1', 'ac@10', 'avg@10', 'ac*@1', 'avg*@500'):
        assert np.isfinite(rc_metrics[key])

    checkpoint_path = os.path.join(str(tmp_path / 'saved_models'), 'cuts_plus_fitting_model.pt')
    assert os.path.exists(checkpoint_path)
    assert os.path.exists(os.path.join(str(tmp_path / 'saved_models'), 'cuts_plus_graph.npy'))
