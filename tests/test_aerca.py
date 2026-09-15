import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aerca import AERCA, AERCAConfig, make_dataloader, make_synthetic_dataset, split_series_dict


def _build_config(tmp_path):
    return AERCAConfig(
        hidden_layer_size=8,
        num_hidden_layers=1,
        window_size=3,
        epochs=2,
        patience=2,
        chunk_len=60,
        val_ratio=0.3,
        synthetic_num_vars=4,
        synthetic_num_series=3,
        synthetic_series_len=120,
        synthetic_num_anomalies=3,
        save_dir=str(tmp_path / 'saved_models'),
        log_dir=str(tmp_path / 'runs'),
        seed=0,
    )


def test_full_pipeline_on_synthetic_data(tmp_path):
    config = _build_config(tmp_path)

    series_dict, label_dict, causal_struct_value = make_synthetic_dataset(config)
    assert causal_struct_value.shape == (config.synthetic_num_vars, config.synthetic_num_vars)
    assert 'series_test' in series_dict

    test_dict = {'series_test': series_dict.pop('series_test')}
    train_dict, val_dict = split_series_dict(series_dict, val_ratio=config.val_ratio, seed=config.seed)

    train_loader = make_dataloader(train_dict, shuffle=True)
    val_loader = make_dataloader(val_dict, shuffle=False)
    test_loader = make_dataloader(test_dict, label_dict=label_dict, shuffle=False)

    num_vars = next(iter(series_dict.values())).shape[1]
    model = AERCA(num_vars=num_vars, device=torch.device('cpu'), config=config)

    model._training(train_loader, val_loader)

    checkpoint_path = os.path.join(model.save_dir, f'{model.model_name}.pt')
    assert os.path.exists(checkpoint_path)
    assert np.isfinite(model.recon_threshold_value)
    assert model.us_mean_encoder.shape == (num_vars,)
    assert model.us_std_encoder.shape == (num_vars,)

    model._testing_root_cause(test_loader)
    model._testing_causal_discover(test_loader, causal_struct_value)

    model.writer.close()
