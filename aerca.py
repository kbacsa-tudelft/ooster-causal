import argparse
import dataclasses
import os
import random
from dataclasses import dataclass, fields
from math import log
from pathlib import Path

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import minimize
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

@dataclass
class AERCAConfig:
    # Architecture
    hidden_layer_size: int = 32
    num_hidden_layers: int = 2
    window_size: int = 8
    stride: int = 1

    # Loss weights
    encoder_alpha: float = 0.5
    decoder_alpha: float = 0.5
    encoder_gamma: float = 0.5
    decoder_gamma: float = 0.5
    encoder_lambda: float = 0.5
    decoder_lambda: float = 0.5
    beta: float = 0.5

    # Optimization
    lr: float = 1e-4
    epochs: int = 100
    patience: int = 20
    chunk_len: int = 1000
    val_ratio: float = 0.2
    seed: int = 42

    # EVT / thresholds
    recon_threshold: float = 0.95
    causal_quantile: float = 0.80
    root_cause_threshold_encoder: float = 0.95
    root_cause_threshold_decoder: float = 0.95
    initial_z_score: float = 3.0
    risk: float = 1e-2
    initial_level: float = 0.98
    num_candidates: int = 100

    # Runtime / paths
    data_name: str = 'synthetic'
    data_dir: str = ''
    save_dir: str = 'saved_models'
    log_dir: str = 'runs'
    device: str = ''

    # Synthetic-data generation (only used when data_dir is empty)
    synthetic_num_vars: int = 6
    synthetic_num_series: int = 5
    synthetic_series_len: int = 600
    synthetic_num_anomalies: int = 5
    synthetic_edge_prob: float = 0.3


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train AERCA on synthetic or real data.')
    defaults = AERCAConfig()
    for f in fields(defaults):
        parser.add_argument(f'--{f.name.replace("_", "-")}', type=f.type, default=getattr(defaults, f.name))
    return parser


# --------------------------------------------------------------------------------------
# Math / statistics helpers
# --------------------------------------------------------------------------------------

def compute_kl_divergence(us, device: torch.device):
    """
    Compute the KL divergence between the empirical distribution of the input samples
    and an isotropic standard Gaussian distribution using PyTorch.

    Parameters:
    us (Tensor): A 2D tensor with rows as samples and columns as features.

    Returns:
    Tensor: The KL divergence between the empirical distribution of the samples
            and the standard Gaussian distribution.
    """
    mean_p = torch.mean(us, dim=0)
    cov_p = torch.cov(us.t())

    d = mean_p.shape[0]

    eigenvalues = torch.linalg.eigvalsh(cov_p)
    condition_number = eigenvalues.max() / eigenvalues.clamp(min=1e-9).min()
    regularization_term = condition_number * 1e-6
    cov_p += torch.eye(d, device=device) * regularization_term

    trace_term = torch.trace(cov_p)
    means_term = torch.dot(mean_p, mean_p)

    try:
        L = torch.linalg.cholesky(cov_p)
        log_det_cov_p = 2 * torch.log(torch.diagonal(L)).sum()
    except RuntimeError:
        log_det_cov_p = torch.logdet(cov_p)

    kl_div = means_term + trace_term - d + log_det_cov_p
    if torch.isnan(kl_div).any():
        raise ValueError(
            f'KL divergence is NaN (mean_p={mean_p}, cov_p={cov_p}, trace_term={trace_term}, '
            f'means_term={means_term}, log_det_cov_p={log_det_cov_p})'
        )

    return kl_div


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sliding_window_view_torch(x, window_size: int):
    """
    A function to create a 2D sliding window view of a 2D PyTorch tensor.

    Args:
    x (torch.Tensor): The input 2D tensor.
    window_size (int): Window size.

    Returns:
    torch.Tensor: A tensor with the sliding windows.
    """
    output_shape = (x.size(0) - window_size + 1, window_size, x.size(1))
    strides = (x.stride(0), x.stride(0), x.stride(1))
    return x.as_strided(size=output_shape, stride=strides)


def eval_causal_structure_binary(a_true: np.ndarray, a_pred: np.ndarray, diagonal=False):
    if not diagonal:
        a_true_offdiag = a_true[np.logical_not(np.eye(a_true.shape[0]))].flatten()
        a_pred_offdiag = a_pred[np.logical_not(np.eye(a_true.shape[0]))].flatten()
        precision = precision_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
        recall = recall_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
        accuracy = accuracy_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
        bal_accuracy = balanced_accuracy_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
        hamming_dist = np.sum(np.abs(a_true_offdiag - a_pred_offdiag)) / len(a_true_offdiag)
    else:
        precision = precision_score(y_true=a_true.flatten(), y_pred=a_pred.flatten())
        recall = recall_score(y_true=a_true.flatten(), y_pred=a_pred.flatten())
        accuracy = accuracy_score(y_true=a_true.flatten(), y_pred=a_pred.flatten())
        bal_accuracy = balanced_accuracy_score(y_true=a_true.flatten(), y_pred=a_pred.flatten())
        hamming_dist = np.sum(np.abs(a_true.flatten() - a_pred.flatten())) / len(a_true.flatten())
    return accuracy, bal_accuracy, precision, recall, hamming_dist


def eval_causal_structure(a_true: np.ndarray, a_pred: np.ndarray, diagonal=False):
    if not diagonal:
        a_true_offdiag = a_true[np.logical_not(np.eye(a_true.shape[0]))]
        a_pred_offdiag = a_pred[np.logical_not(np.eye(a_true.shape[0]))]
        auroc = roc_auc_score(y_true=a_true_offdiag.flatten(), y_score=a_pred_offdiag.flatten())
        auprc = average_precision_score(y_true=a_true_offdiag.flatten(), y_score=a_pred_offdiag.flatten())
    else:
        auroc = roc_auc_score(y_true=a_true.flatten(), y_score=a_pred.flatten())
        auprc = average_precision_score(y_true=a_true.flatten(), y_score=a_pred.flatten())
    return auroc, auprc


def grimshaw(peaks: np.array, threshold: float, num_candidates: int = 10, epsilon: float = 1e-8):
    """ The Grimshaw's Trick Method

    The trick of the Grimshaw's procedure is to reduce the two variables
    optimization problem to a single variable equation.

    Args:
        peaks: peak nodes from original dataset.
        threshold: init threshold
        num_candidates: the maximum number of nodes we choose as candidates
        epsilon: numerical parameter to perform

    Returns:
        gamma: estimate
        sigma: estimate
    """
    peak_min = peaks.min()
    peak_max = peaks.max()
    mean = peaks.mean()

    if abs(-1 / peak_max) < 2 * epsilon:
        epsilon = abs(-1 / peak_max) / num_candidates

    a = -1 / peak_max + epsilon
    b = 2 * (mean - peak_min) / (mean * peak_min)
    c = 2 * (mean - peak_min) / (peak_min ** 2)

    candidate_gamma = solve(function=lambda t: function(peaks, t),
                             dev_function=lambda t: dev_function(peaks, t),
                             bounds=(a + epsilon, -epsilon),
                             num_candidates=num_candidates
                             )
    candidate_sigma = solve(function=lambda t: function(peaks, t),
                             dev_function=lambda t: dev_function(peaks, t),
                             bounds=(b, c),
                             num_candidates=num_candidates
                             )
    candidates = np.concatenate([candidate_gamma, candidate_sigma])

    gamma_best = 0
    sigma_best = mean
    log_likelihood_best = cal_log_likelihood(peaks, gamma_best, sigma_best)

    for candidate in candidates:
        if candidate == 0 or np.isnan(candidate):
            continue
        gamma = np.log(1 + candidate * peaks).mean()
        sigma = gamma / candidate
        log_likelihood = cal_log_likelihood(peaks, gamma, sigma)
        if log_likelihood > log_likelihood_best:
            gamma_best = gamma
            sigma_best = sigma
            log_likelihood_best = log_likelihood

    return gamma_best, sigma_best


def function(x, threshold):
    s = 1 + threshold * x
    u = 1 + np.log(s).mean()
    v = np.mean(1 / s)
    return u * v - 1


def dev_function(x, threshold):
    s = 1 + threshold * x
    u = 1 + np.log(s).mean()
    v = np.mean(1 / s)
    dev_u = (1 / threshold) * (1 - v)
    dev_v = (1 / threshold) * (-v + np.mean(1 / s ** 2))
    return u * dev_v + v * dev_u


def obj_function(x, function, dev_function):
    m = 0
    n = np.zeros(x.shape)
    for index, item in enumerate(x):
        y = function(item)
        m = m + y ** 2
        n[index] = 2 * y * dev_function(item)
    return m, n


def solve(function, dev_function, bounds, num_candidates):
    step = (bounds[1] - bounds[0]) / (num_candidates + 1)
    x0 = np.arange(bounds[0] + step, bounds[1], step)
    optimization = minimize(lambda x: obj_function(x, function, dev_function),
                             x0,
                             method='L-BFGS-B',
                             jac=True,
                             bounds=[bounds] * len(x0)
                             )
    x = np.round(optimization.x, decimals=5)
    return np.unique(x)


def cal_log_likelihood(peaks, gamma, sigma):
    if gamma != 0:
        tau = gamma / sigma
        log_likelihood = -peaks.size * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * peaks)).sum()
    else:
        log_likelihood = peaks.size * (1 + log(peaks.mean()))
    return log_likelihood


def pot(data: np.array, risk: float = 1e-2, init_level: float = 0.98, num_candidates: int = 10,
        epsilon: float = 1e-8) -> float:
    """ Peak-over-Threshold Algorithm

    References:
    Siffer, Alban, et al. "Anomaly detection in streams with extreme value theory."
    Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge
    Discovery and Data Mining. 2017.

    Args:
        data: data to process
        risk: detection level
        init_level: probability associated with the initial threshold
        num_candidates: the maximum number of nodes we choose as candidates
        epsilon: numerical parameter to perform

    Returns:
        z: threshold searching by pot
        t: init threshold
    """
    t = np.sort(data)[int(init_level * data.size)]
    peaks = data[data > t] - t

    gamma, sigma = grimshaw(peaks=peaks,
                             threshold=t,
                             num_candidates=num_candidates,
                             epsilon=epsilon
                             )

    r = data.size * risk / peaks.size
    if gamma != 0:
        z = t + (sigma / gamma) * (pow(r, -gamma) - 1)
    else:
        z = t - sigma * log(r)

    return z, t


def topk(z_scores, label, threshold, k_range=500):
    """ Top-k method

    Args:
        z_scores: anomaly scores
        label: ground truth

    Returns:
        k: the number of top-k nodes
    """
    z_scores = np.array(z_scores)
    us_above_threshold = np.where(z_scores > threshold, z_scores, 0.0)
    label = np.array(label)
    us_above_threshold = us_above_threshold.flatten()
    label = label.flatten()
    ranking = np.argsort(us_above_threshold)
    label_ind = np.where(label == 1)[0]
    k_lst = []
    for k in range(1, k_range + 1):
        count = [1 if i in label_ind else 0 for i in ranking[-k:]]
        k_lst.append(sum(count) / min(k, len(label_ind)))
    return np.array(k_lst)


def topk_at_step(scores, labels, k_range=10):
    k_lst = []
    for i in range(len(labels)):
        if sum(labels[i]) > 0:
            ranking = np.argsort(scores[i])
            label_ind = np.where(labels[i] == 1)[0]
            for k in range(1, k_range + 1):
                count = [1 if i in label_ind else 0 for i in ranking[-k:]]
                k_lst.append(sum(count) / min(k, len(label_ind)))
    return np.array(k_lst).reshape(-1, k_range).mean(axis=0)


# --------------------------------------------------------------------------------------
# Model
# https://github.com/hanxiao0607/AERCA/blob/main/models/senn.py
# https://github.com/hanxiao0607/AERCA/blob/main/models/aerca.py
# --------------------------------------------------------------------------------------

class SENNGC(nn.Module):
    def __init__(self, num_vars: int, order: int, hidden_layer_size: int, num_hidden_layers: int, device: torch.device):
        """
        Generalised VAR (GVAR) model based on self-explaining neural networks.
        @param num_vars: number of variables (p).
        @param order:  model order (maximum lag, K).
        @param hidden_layer_size: number of units in the hidden layer.
        @param num_hidden_layers: number of hidden layers.
        @param device: Torch device.
        """
        super(SENNGC, self).__init__()

        # Networks for amortising generalised coefficient matrices.
        self.coeff_nets = nn.ModuleList()

        for k in range(order):
            layers = [nn.Linear(num_vars, hidden_layer_size), nn.ReLU()]
            for _ in range(num_hidden_layers - 1):
                layers += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()]
            layers += [nn.Linear(hidden_layer_size, num_vars ** 2), nn.Tanh()]
            self.coeff_nets.append(nn.Sequential(*layers))

        self.num_vars = num_vars
        self.order = order
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layer_size = num_hidden_layers
        self.device = device

    def forward(self, inputs: torch.Tensor):
        assert inputs[0, :, :].shape == torch.Size([self.order, self.num_vars]), \
            f'inputs should be of shape BS x K x p, got {inputs.shape}'

        coeffs = None
        preds = torch.zeros((inputs.shape[0], self.num_vars)).to(self.device)
        for k in range(self.order):
            coeff_net_k = self.coeff_nets[k]
            coeffs_k = coeff_net_k(inputs[:, k, :])
            coeffs_k = torch.reshape(coeffs_k, (inputs.shape[0], self.num_vars, self.num_vars))
            if coeffs is None:
                coeffs = torch.unsqueeze(coeffs_k, 1)
            else:
                coeffs = torch.cat((coeffs, torch.unsqueeze(coeffs_k, 1)), 1)
            preds = preds + torch.matmul(coeffs_k, inputs[:, k, :].unsqueeze(dim=2)).squeeze(-1)
        return preds, coeffs


class AERCA(nn.Module):
    def __init__(self, num_vars: int, device: torch.device, config: AERCAConfig):
        super(AERCA, self).__init__()
        self.config = config
        self.encoder = SENNGC(num_vars, config.window_size, config.hidden_layer_size, config.num_hidden_layers, device)
        self.decoder = SENNGC(num_vars, config.window_size, config.hidden_layer_size, config.num_hidden_layers, device)
        self.decoder_prev = SENNGC(num_vars, config.window_size, config.hidden_layer_size, config.num_hidden_layers, device)
        self.device = device
        self.num_vars = num_vars
        self.hidden_layer_size = config.hidden_layer_size
        self.num_hidden_layers = config.num_hidden_layers
        self.window_size = config.window_size
        self.stride = config.stride
        self.encoder_alpha = config.encoder_alpha
        self.decoder_alpha = config.decoder_alpha
        self.encoder_gamma = config.encoder_gamma
        self.decoder_gamma = config.decoder_gamma
        self.encoder_lambda = config.encoder_lambda
        self.decoder_lambda = config.decoder_lambda
        self.beta = config.beta
        self.lr = config.lr
        self.epochs = config.epochs
        self.patience = config.patience
        self.chunk_len = config.chunk_len
        self.recon_threshold = config.recon_threshold
        self.root_cause_threshold_encoder = config.root_cause_threshold_encoder
        self.root_cause_threshold_decoder = config.root_cause_threshold_decoder
        self.initial_z_score = config.initial_z_score
        self.mse_loss = nn.MSELoss()
        self.mse_loss_wo_reduction = nn.MSELoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.decoder_prev.to(self.device)
        self.model_name = ('AERCA_' + config.data_name + '_ws_' + str(config.window_size) + '_stride_' + str(config.stride) +
                            '_encoder_alpha_' + str(config.encoder_alpha) + '_decoder_alpha_' + str(config.decoder_alpha) +
                            '_encoder_gamma_' + str(config.encoder_gamma) + '_decoder_gamma_' + str(config.decoder_gamma) +
                            '_encoder_lambda_' + str(config.encoder_lambda) + '_decoder_lambda_' + str(config.decoder_lambda) +
                            '_beta_' + str(config.beta) + '_lr_' + str(config.lr) + '_epochs_' + str(config.epochs) +
                            '_hidden_layer_size_' + str(config.hidden_layer_size) + '_num_hidden_layers_' +
                            str(config.num_hidden_layers))
        self.causal_quantile = config.causal_quantile
        self.risk = config.risk
        self.initial_level = config.initial_level
        self.num_candidates = config.num_candidates

        self.save_dir = os.path.join(os.getcwd(), config.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=os.path.join(os.getcwd(), config.log_dir, self.model_name))
        self.global_step = 0

    def _log_and_print(self, msg, *args):
        """Helper method to print and record final testing results."""
        final_msg = msg.format(*args) if args else msg
        print(final_msg)

    def _sparsity_loss(self, coeffs, alpha):
        norm2 = torch.mean(torch.norm(coeffs, dim=1, p=2))
        norm1 = torch.mean(torch.norm(coeffs, dim=1, p=1))
        return (1 - alpha) * norm2 + alpha * norm1

    def _smoothness_loss(self, coeffs):
        return torch.norm(coeffs[:, 1:, :, :] - coeffs[:, :-1, :, :], dim=1).mean()

    def encoding(self, xs):
        windows = sliding_window_view(xs, (self.window_size + 1, self.num_vars))[:, 0, :, :]
        winds = windows[:, :-1, :]
        nexts = windows[:, -1, :]
        winds = torch.tensor(winds).float().to(self.device)
        nexts = torch.tensor(nexts).float().to(self.device)
        preds, coeffs = self.encoder(winds)
        us = preds - nexts
        return us, coeffs, nexts[self.window_size:], winds[:-self.window_size]

    def decoding(self, us, winds, add_u=True):
        u_windows = sliding_window_view_torch(us, self.window_size + 1)
        u_winds = u_windows[:, :-1, :]
        u_next = u_windows[:, -1, :]

        preds, coeffs = self.decoder(u_winds)
        prev_preds, prev_coeffs = self.decoder_prev(winds)

        if add_u:
            nexts_hat = preds + u_next + prev_preds
        else:
            nexts_hat = preds + prev_preds
        return nexts_hat, coeffs, prev_coeffs

    def forward(self, x, add_u=True):
        us, encoder_coeffs, nexts, winds = self.encoding(x)
        kl_div = compute_kl_divergence(us, self.device)
        nexts_hat, decoder_coeffs, prev_coeffs = self.decoding(us, winds, add_u=add_u)
        return nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us

    def _compute_losses(self, nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div):
        loss_recon = self.mse_loss(nexts_hat, nexts)
        loss_encoder_coeffs = self._sparsity_loss(encoder_coeffs, self.encoder_alpha)
        loss_decoder_coeffs = self._sparsity_loss(decoder_coeffs, self.decoder_alpha)
        loss_prev_coeffs = self._sparsity_loss(prev_coeffs, self.decoder_alpha)
        loss_encoder_smooth = self._smoothness_loss(encoder_coeffs)
        loss_decoder_smooth = self._smoothness_loss(decoder_coeffs)
        loss_prev_smooth = self._smoothness_loss(prev_coeffs)
        loss_kl = kl_div

        loss_total = (loss_recon +
                      self.encoder_lambda * loss_encoder_coeffs +
                      self.decoder_lambda * (loss_decoder_coeffs + loss_prev_coeffs) +
                      self.encoder_gamma * loss_encoder_smooth +
                      self.decoder_gamma * (loss_decoder_smooth + loss_prev_smooth) +
                      self.beta * loss_kl)

        return {
            'loss_recon': loss_recon,
            'loss_encoder_coeffs': loss_encoder_coeffs,
            'loss_decoder_coeffs': loss_decoder_coeffs,
            'loss_prev_coeffs': loss_prev_coeffs,
            'loss_encoder_smooth': loss_encoder_smooth,
            'loss_decoder_smooth': loss_decoder_smooth,
            'loss_prev_smooth': loss_prev_smooth,
            'loss_kl': loss_kl,
            'loss_total': loss_total,
        }

    def _log_losses(self, losses: dict, split: str, step: int):
        for name, value in losses.items():
            self.writer.add_scalar(f'{split}/{name}', value.item(), step)

    def _training_step_long(self, x, chunk_len=None, add_u=True, train=True):
        """
        Splits a long raw sequence into overlapping chunks (overlap = window_size,
        so every chunk still has enough context to form valid windows), and calls
        the existing, unmodified _training_step on each piece.
        """
        chunk_len = chunk_len or self.chunk_len
        n = len(x)
        step = chunk_len - self.window_size  # overlap ensures no windows are lost at boundaries
        starts = list(range(0, n, step))
        n_chunks = len(starts)

        if train:
            self.optimizer.zero_grad()
        total_loss = 0.0
        for start in starts:
            end = min(start + chunk_len, n)
            x_chunk = x[start:end]
            if len(x_chunk) < 2 * self.window_size + 1:
                continue  # too short for the encode/decode pipeline to produce even one row, skip

            if train:
                loss = self._training_step(x_chunk, add_u=add_u)
                (loss / n_chunks).backward()
            else:
                with torch.no_grad():
                    loss = self._training_step(x_chunk, add_u=add_u)
            total_loss += loss.item()
        if train:
            self.optimizer.step()
        return total_loss / n_chunks

    def _training_step(self, x, add_u=True):
        nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us = self.forward(x, add_u=add_u)
        losses = self._compute_losses(nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div)
        self._log_losses(losses, 'train', self.global_step)
        self.global_step += 1
        return losses['loss_total']

    def _training(self, train_loader, val_loader):
        best_val_loss = np.inf
        count = 0
        for epoch in tqdm(range(self.epochs), desc='Epoch'):
            count += 1
            epoch_loss = 0
            self.train()
            for x, _, _ in train_loader:
                loss = self._training_step_long(x)
                epoch_loss += loss

            epoch_val_loss = 0
            self.eval()
            with torch.no_grad():
                for x, _, _ in val_loader:
                    loss = self._training_step_long(x, train=False)
                    epoch_val_loss += loss

            self.writer.add_scalar('epoch/train_loss', epoch_loss, epoch)
            self.writer.add_scalar('epoch/val_loss', epoch_val_loss, epoch)

            if epoch_val_loss < best_val_loss:
                count = 0
                best_val_loss = epoch_val_loss
                torch.save(self.state_dict(), os.path.join(self.save_dir, f'{self.model_name}.pt'))
            if count >= self.patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        self.load_state_dict(
            torch.load(os.path.join(self.save_dir, f'{self.model_name}.pt'), map_location=self.device)
        )
        self._get_recon_threshold(val_loader)
        self._get_root_cause_threshold_encoder(val_loader)
        self._get_root_cause_threshold_decoder(val_loader)

    def _testing_step(self, x, label=None, add_u=True):
        nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us = self.forward(x, add_u=add_u)

        if label is not None:
            preprocessed_label = sliding_window_view(label, (self.window_size + 1, self.num_vars))[self.window_size:, 0, :-1, :]
        else:
            preprocessed_label = None

        losses = self._compute_losses(nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div)
        self._log_losses(losses, 'test', self.global_step)

        return losses['loss_total'], nexts_hat, nexts, encoder_coeffs, decoder_coeffs, kl_div, preprocessed_label, us

    def _get_recon_threshold(self, loader):
        self.eval()
        losses_list = []
        with torch.no_grad():
            for x, _, _ in loader:
                _, nexts_hat, nexts, _, _, _, _, _ = self._testing_step(x, add_u=False)
                loss_arr = self.mse_loss_wo_reduction(nexts_hat, nexts).cpu().numpy().ravel()
                losses_list.append(loss_arr)
        recon_losses = np.concatenate(losses_list)
        self.recon_threshold_value = np.quantile(recon_losses, self.recon_threshold)
        self.recon_mean = np.mean(recon_losses)
        self.recon_std = np.std(recon_losses)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_recon_threshold.npy'), self.recon_threshold_value)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_recon_mean.npy'), self.recon_mean)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_recon_std.npy'), self.recon_std)

    def _get_root_cause_threshold_encoder(self, loader):
        self.eval()
        us_list = []
        with torch.no_grad():
            for x, _, _ in loader:
                us = self._testing_step(x)[-1]
                us_list.append(us.cpu().numpy())
        us_all = np.concatenate(us_list, axis=0).reshape(-1, self.num_vars)
        self.lower_encoder = np.quantile(us_all, (1 - self.root_cause_threshold_encoder) / 2, axis=0)
        self.upper_encoder = np.quantile(us_all, 1 - (1 - self.root_cause_threshold_encoder) / 2, axis=0)
        self.us_mean_encoder = np.median(us_all, axis=0)
        self.us_std_encoder = np.std(us_all, axis=0)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_lower_encoder.npy'), self.lower_encoder)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_upper_encoder.npy'), self.upper_encoder)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_us_mean_encoder.npy'), self.us_mean_encoder)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_us_std_encoder.npy'), self.us_std_encoder)

    def _get_root_cause_threshold_decoder(self, loader):
        self.eval()
        diff_list = []
        with torch.no_grad():
            for x, _, _ in loader:
                _, nexts_hat, nexts, _, _, _, _, _ = self._testing_step(x, add_u=False)
                diff = (nexts - nexts_hat).cpu().numpy().ravel()
                diff_list.append(diff)
        us_all = np.concatenate(diff_list, axis=0).reshape(-1, self.num_vars)
        self.lower_decoder = np.quantile(us_all, (1 - self.root_cause_threshold_decoder) / 2, axis=0)
        self.upper_decoder = np.quantile(us_all, 1 - (1 - self.root_cause_threshold_decoder) / 2, axis=0)
        self.us_mean_decoder = np.mean(us_all, axis=0)
        self.us_std_decoder = np.std(us_all, axis=0)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_lower_decoder.npy'), self.lower_decoder)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_upper_decoder.npy'), self.upper_decoder)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_us_mean_decoder.npy'), self.us_mean_decoder)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_us_std_decoder.npy'), self.us_std_decoder)

    def _testing_root_cause(self, test_loader):
        self.load_state_dict(
            torch.load(os.path.join(self.save_dir, f'{self.model_name}.pt'), map_location=self.device)
        )
        self.eval()
        self.us_mean_encoder = np.load(os.path.join(self.save_dir, f'{self.model_name}_us_mean_encoder.npy'))
        self.us_std_encoder = np.load(os.path.join(self.save_dir, f'{self.model_name}_us_std_encoder.npy'))

        us_list, us_sample_list, labels_list = [], [], []
        with torch.no_grad():
            for x, label, _ in test_loader:
                us = self._testing_step(x, label, add_u=False)[-1]
                us_sample_list.append(us[self.window_size:].cpu().numpy())
                us_list.append(us.cpu().numpy())
                labels_list.append(label)

        us_all = np.concatenate(us_list, axis=0).reshape(-1, self.num_vars)
        self._log_and_print('=' * 50)
        us_all_z_score = (-(us_all - self.us_mean_encoder) / self.us_std_encoder)
        us_all_z_score_pot = np.array([
            pot(us_all_z_score[:, i], self.risk, self.initial_level, self.num_candidates)[0]
            for i in range(self.num_vars)
        ])

        k_all, k_at_step_all = [], []
        for i in range(len(us_sample_list)):
            us_sample = us_sample_list[i]
            z_scores = (-(us_sample - self.us_mean_encoder) / self.us_std_encoder)
            k_lst = topk(z_scores, labels_list[i][self.window_size * 2:], us_all_z_score_pot)
            k_at_step = topk_at_step(z_scores, labels_list[i][self.window_size * 2:])
            k_all.append(k_lst)
            k_at_step_all.append(k_at_step)
        k_all = np.array(k_all).mean(axis=0)
        k_at_step_all = np.array(k_at_step_all).mean(axis=0)

        ac_at = [k_at_step_all[0], k_at_step_all[2], k_at_step_all[4], k_at_step_all[9]]
        self._log_and_print('Root cause analysis AC@1: {:.5f}', ac_at[0])
        self._log_and_print('Root cause analysis AC@3: {:.5f}', ac_at[1])
        self._log_and_print('Root cause analysis AC@5: {:.5f}', ac_at[2])
        self._log_and_print('Root cause analysis AC@10: {:.5f}', ac_at[3])
        self._log_and_print('Root cause analysis Avg@10: {:.5f}', np.mean(k_at_step_all))

        ac_star_at = [k_all[0], k_all[9], k_all[99], k_all[499]]
        self._log_and_print('Root cause analysis AC*@1: {:.5f}', ac_star_at[0])
        self._log_and_print('Root cause analysis AC*@10: {:.5f}', ac_star_at[1])
        self._log_and_print('Root cause analysis AC*@100: {:.5f}', ac_star_at[2])
        self._log_and_print('Root cause analysis AC*@500: {:.5f}', ac_star_at[3])
        self._log_and_print('Root cause analysis Avg*@500: {:.5f}', np.mean(k_all))

        self.writer.add_scalar('test/root_cause_AC@1', ac_at[0])
        self.writer.add_scalar('test/root_cause_AC@3', ac_at[1])
        self.writer.add_scalar('test/root_cause_AC@5', ac_at[2])
        self.writer.add_scalar('test/root_cause_AC@10', ac_at[3])
        self.writer.add_scalar('test/root_cause_Avg@10', np.mean(k_at_step_all))

        return {
            'ac@1': ac_at[0], 'ac@3': ac_at[1], 'ac@5': ac_at[2], 'ac@10': ac_at[3],
            'avg@10': np.mean(k_at_step_all),
            'ac*@1': ac_star_at[0], 'ac*@10': ac_star_at[1], 'ac*@100': ac_star_at[2],
            'ac*@500': ac_star_at[3], 'avg*@500': np.mean(k_all),
        }

    def _testing_causal_discover(self, test_loader, causal_struct_value):
        self.load_state_dict(
            torch.load(os.path.join(self.save_dir, f'{self.model_name}.pt'), map_location=self.device)
        )
        self.eval()
        encoder_causal_list = []
        with torch.no_grad():
            for x, _, _ in test_loader:
                _, _, _, encoder_coeffs, _, _, _, _ = self._testing_step(x)
                encoder_estimate = torch.max(torch.median(torch.abs(encoder_coeffs), dim=0)[0],
                                              dim=0).values.cpu().numpy()
                encoder_causal_list.append(encoder_estimate)
        encoder_causal_struct_estimate_lst = np.stack(encoder_causal_list, axis=0)

        encoder_auroc = []
        encoder_auprc = []
        encoder_hamming = []
        encoder_f1 = []
        for i in range(len(encoder_causal_struct_estimate_lst)):
            encoder_auroc_temp, encoder_auprc_temp = eval_causal_structure(
                a_true=causal_struct_value, a_pred=encoder_causal_struct_estimate_lst[i])

            encoder_auroc.append(encoder_auroc_temp)
            encoder_auprc.append(encoder_auprc_temp)
            encoder_q = np.quantile(encoder_causal_struct_estimate_lst[i], q=self.causal_quantile)
            encoder_a_hat_binary = (encoder_causal_struct_estimate_lst[i] >= encoder_q).astype(float)
            _, _, _, _, ham_e = eval_causal_structure_binary(a_true=causal_struct_value,
                                                              a_pred=encoder_a_hat_binary)
            encoder_hamming.append(ham_e)
            encoder_f1.append(f1_score(causal_struct_value.flatten(), encoder_a_hat_binary.flatten()))
        self._log_and_print('Causal discovery F1: {:.5f} std: {:.5f}',
                             np.mean(encoder_f1), np.std(encoder_f1))
        self._log_and_print('Causal discovery AUROC: {:.5f} std: {:.5f}',
                             np.mean(encoder_auroc), np.std(encoder_auroc))
        self._log_and_print('Causal discovery AUPRC: {:.5f} std: {:.5f}',
                             np.mean(encoder_auprc), np.std(encoder_auprc))
        self._log_and_print('Causal discovery Hamming Distance: {:.5f} std: {:.5f}',
                             np.mean(encoder_hamming), np.std(encoder_hamming))

        self.writer.add_scalar('test/causal_f1', np.mean(encoder_f1))
        self.writer.add_scalar('test/causal_auroc', np.mean(encoder_auroc))
        self.writer.add_scalar('test/causal_auprc', np.mean(encoder_auprc))
        self.writer.add_scalar('test/causal_hamming', np.mean(encoder_hamming))


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    def __init__(self, series_dict, label_dict=None):
        self.sample_ids = list(series_dict.keys())
        self.series_dict = series_dict
        self.label_dict = label_dict or {}

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        x = self.series_dict[sample_id].values.astype(np.float32)

        label = self.label_dict.get(sample_id)
        if label is None:
            label = np.zeros_like(x)
        elif hasattr(label, 'values'):
            label = label.values.astype(np.float32)
        else:
            label = np.asarray(label, dtype=np.float32)

        return x, label, sample_id


def identity_collate(batch):
    # batch_size=1 -> just unwrap; sequences differ in length so we never stack them
    return batch[0]


def make_dataloader(series_dict, label_dict=None, shuffle=False):
    dataset = TimeSeriesDataset(series_dict, label_dict=label_dict)
    return DataLoader(dataset, batch_size=1, shuffle=shuffle, collate_fn=identity_collate)


def split_series_dict(series_dict, val_ratio=0.2, seed=42):
    """Sample-level split. Falls back to a temporal split if there's only one series."""
    sample_ids = list(series_dict.keys())

    if len(sample_ids) == 1:
        sample_id = sample_ids[0]
        df = series_dict[sample_id]
        split_idx = int((1 - val_ratio) * len(df))
        return {sample_id: df.iloc[:split_idx]}, {sample_id: df.iloc[split_idx:]}

    rng = random.Random(seed)
    shuffled = sample_ids[:]
    rng.shuffle(shuffled)
    split_idx = int((1 - val_ratio) * len(shuffled))
    train_ids, val_ids = shuffled[:split_idx], shuffled[split_idx:]

    return ({k: series_dict[k] for k in train_ids},
            {k: series_dict[k] for k in val_ids})


def load_series_dict(input_dir):
    """Restore series_dict from a directory of parquet files.
    Also returns means/stds (as pd.Series) if a stats file is present,
    otherwise returns None for both."""
    input_dir = Path(input_dir)

    series_dict = {
        f.stem: pd.read_parquet(f)
        for f in sorted(input_dir.glob("*.parquet"))
    }

    means, stds = None, None
    stats_path = input_dir / 'normalization_stats.json'
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        means = pd.Series(stats['mean'])
        stds = pd.Series(stats['std'])

    return series_dict, means, stds


def make_synthetic_dataset(config: AERCAConfig, rng: np.random.Generator = None):
    """
    Generate a synthetic multivariate time series dataset driven by a sparse random
    VAR(window_size) process, for exercising training / root-cause detection / causal
    discovery without real data.

    Returns:
        series_dict: sample_id -> pd.DataFrame (T x num_vars), including one held-out
            "series_test" sample with injected anomalies.
        label_dict: sample_id -> np.ndarray (T x num_vars) binary anomaly labels
            (all zero except for "series_test").
        causal_struct_value: (num_vars x num_vars) ground-truth binary adjacency matrix,
            causal_struct_value[i, j] == 1 means variable j Granger-causes variable i.
    """
    rng = rng or np.random.default_rng(config.seed)
    p = config.synthetic_num_vars
    order = config.window_size
    noise_std = 0.1

    causal_struct_value = (rng.random((p, p)) < config.synthetic_edge_prob).astype(float)
    np.fill_diagonal(causal_struct_value, 1.0)

    # One random coefficient matrix per lag, masked by the causal structure and scaled
    # down so the process stays stable.
    coeffs = [causal_struct_value * rng.uniform(-0.3, 0.3, size=(p, p)) / order for _ in range(order)]

    def simulate(length, shocks=None):
        x = np.zeros((length, p))
        x[:order] = rng.normal(scale=noise_std, size=(order, p))
        for t in range(order, length):
            value = sum(coeffs[k] @ x[t - k - 1] for k in range(order))
            value += rng.normal(scale=noise_std, size=p)
            if shocks and t in shocks:
                value[shocks[t]] += 5.0
            x[t] = value
        return x

    series_dict, label_dict = {}, {}
    for i in range(config.synthetic_num_series):
        data = simulate(config.synthetic_series_len)
        series_dict[f'series_{i}'] = pd.DataFrame(data, columns=[f'var_{j}' for j in range(p)])
        label_dict[f'series_{i}'] = np.zeros_like(data)

    shock_times = rng.choice(
        range(order * 2, config.synthetic_series_len), size=config.synthetic_num_anomalies, replace=False
    )
    shock_vars = rng.integers(0, p, size=config.synthetic_num_anomalies)
    shocks = dict(zip(shock_times.tolist(), shock_vars.tolist()))

    test_data = simulate(config.synthetic_series_len, shocks=shocks)
    test_labels = np.zeros_like(test_data)
    for t, var in shocks.items():
        # The shocked variable and anything it directly causes are labelled anomalous.
        affected = np.where(causal_struct_value[:, var] > 0)[0]
        test_labels[t:t + order, affected] = 1.0

    series_dict['series_test'] = pd.DataFrame(test_data, columns=[f'var_{j}' for j in range(p)])
    label_dict['series_test'] = test_labels

    return series_dict, label_dict, causal_struct_value


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def main():
    config = AERCAConfig(**vars(build_arg_parser().parse_args()))
    set_seed(config.seed)
    device = torch.device(config.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    if config.data_dir:
        series_dict, _, _ = load_series_dict(config.data_dir)
        label_dict, causal_struct_value = None, None
    else:
        series_dict, label_dict, causal_struct_value = make_synthetic_dataset(config)

    test_dict = {'series_test': series_dict.pop('series_test')} if 'series_test' in series_dict else {}
    train_dict, val_dict = split_series_dict(series_dict, val_ratio=config.val_ratio, seed=config.seed)

    train_loader = make_dataloader(train_dict, shuffle=True)
    val_loader = make_dataloader(val_dict, shuffle=False)

    num_vars = next(iter(series_dict.values())).shape[1]
    print(f'Training AERCA on {len(series_dict)} series ({num_vars} variables) using device={device}')

    model = AERCA(num_vars=num_vars, device=device, config=config)
    model._training(train_loader, val_loader)

    if test_dict:
        test_loader = make_dataloader(test_dict, label_dict=label_dict, shuffle=False)
        model._testing_root_cause(test_loader)
        model._testing_causal_discover(test_loader, causal_struct_value)

    model.writer.close()


if __name__ == '__main__':
    main()
