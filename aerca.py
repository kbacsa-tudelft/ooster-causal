import os
import logging
import numpy as np
import pandas as pd
import torch.nn as nn
import torch

from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path


from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import f1_score

from math import log
from scipy.optimize import minimize
import random
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score, \
    precision_score, recall_score



def compute_kl_divergence(us, device: torch.device):
    """
    Compute the KL divergence between the empirical distribution of the input samples
    and an isotropic standard Gaussian distribution using PyTorch.

    Parameters:
    samples (Tensor): A 2D tensor with rows as samples and columns as features.

    Returns:
    Tensor: The KL divergence between the empirical distribution of the samples
            and the standard Gaussian distribution.
    """

    # Calculate the empirical mean and covariance matrix of the samples
    mean_p = torch.mean(us, dim=0)
    cov_p = torch.cov(us.t())

    # Dimensionality of the distribution
    d = mean_p.shape[0]

    eigenvalues = torch.linalg.eigvalsh(cov_p)
    condition_number = eigenvalues.max() / eigenvalues.clamp(min=1e-9).min()
    regularization_term = condition_number * 1e-6
    cov_p += torch.eye(d, device=device) * regularization_term
    # Ensure the covariance matrix is full rank
    # cov_p += 1e-9 * torch.eye(d).to(device)

    # Compute the trace term
    trace_term = torch.trace(cov_p)

    # Compute the product of means term (since mean_q is zero, this is just mean_p squared)
    means_term = torch.dot(mean_p, mean_p)

    # # Compute the determinant term
    # log_det_cov_p = torch.logdet(cov_p)
    try:
        L = torch.linalg.cholesky(cov_p)
        log_det_cov_p = 2 * torch.log(torch.diagonal(L)).sum()
    except RuntimeError:
        # Handle the case where Cholesky decomposition fails
        log_det_cov_p = torch.logdet(cov_p)

    # Compute the KL divergence using the formula
    kl_div = means_term + trace_term - d + log_det_cov_p
    if torch.isnan(kl_div).any():
        print('nan')
        print(f'mean_p: {mean_p}')
        print(f'cov_p: {cov_p}')
        print(f'trace_term: {trace_term}')
        print(f'means_term: {means_term}')
        print(f'log_det_cov_p: {log_det_cov_p}')
        print(f'kl_div: {kl_div}')
        raise ValueError('KL divergence is NaN')


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
    # Calculate output shape
    output_shape = (x.size(0) - window_size + 1, window_size, x.size(1))
    # Calculate strides
    strides = (x.stride(0), x.stride(0), x.stride(1))
    # Create a view
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
        #if np.max(a_true_offdiag) == np.min(a_true_offdiag):
        #    auroc = None
        #    auprc = None
        #else:
        auroc = roc_auc_score(y_true=a_true_offdiag.flatten(), y_score=a_pred_offdiag.flatten())
        auprc = average_precision_score(y_true=a_true_offdiag.flatten(), y_score=a_pred_offdiag.flatten())
    else:
        auroc = roc_auc_score(y_true=a_true.flatten(), y_score=a_pred.flatten())
        auprc = average_precision_score(y_true=a_true.flatten(), y_score=a_pred.flatten())
    return auroc, auprc


def construct_training_dataset(data, order):
    # Pack the data, if it is not in a list already
    if not isinstance(data, list):
        data = [data]

    data_out = None
    response = None
    time_idx = None
    # Iterate through time series replicates
    offset = 0
    for r in range(len(data)):
        data_r = data[r]
        # data: T x p
        T_r = data_r.shape[0]
        p_r = data_r.shape[1]
        inds_r = np.arange(order, T_r)
        data_out_r = np.zeros((T_r - order, order, p_r))
        response_r = np.zeros((T_r - order, p_r))
        time_idx_r = np.zeros((T_r - order, ))
        for i in range(T_r - order):
            j = inds_r[i]
            data_out_r[i, :, :] = data_r[(j - order):j, :]
            response_r[i] = data_r[j, :]
            time_idx_r[i] = j
        time_idx_r = time_idx_r + offset + 200 * (r >= 1)
        time_idx_r = time_idx_r.astype(int)
        if data_out is None:
            data_out = data_out_r
            response = response_r
            time_idx = time_idx_r
        else:
            data_out = np.concatenate((data_out, data_out_r), axis=0)
            response = np.concatenate((response, response_r), axis=0)
            time_idx = np.concatenate((time_idx, time_idx_r))
        offset = np.max(time_idx_r)
    return data_out, response, time_idx

def grimshaw(peaks:np.array, threshold:float, num_candidates:int=10, epsilon:float=1e-8):
    ''' The Grimshaw's Trick Method

    The trick of thr Grimshaw's procedure is to reduce the two variables
    optimization problem to a signle variable equation.

    Args:
        peaks: peak nodes from original dataset.
        threshold: init threshold
        num_candidates: the maximum number of nodes we choose as candidates
        epsilon: numerical parameter to perform

    Returns:
        gamma: estimate
        sigma: estimate
    '''
    min = peaks.min()
    max = peaks.max()
    mean = peaks.mean()

    if abs(-1 / max) < 2 * epsilon:
        epsilon = abs(-1 / max) / num_candidates

    a = -1 / max + epsilon
    b = 2 * (mean - min) / (mean * min)
    c = 2 * (mean - min) / (min ** 2)

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
                            bounds=[bounds]*len(x0)
                            )
    x = np.round(optimization.x, decimals=5)
    return np.unique(x)


def cal_log_likelihood(peaks, gamma, sigma):
    if gamma != 0:
        tau = gamma/sigma
        log_likelihood = -peaks.size * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * peaks)).sum()
    else:
        log_likelihood = peaks.size * (1 + log(peaks.mean()))
    return log_likelihood



def pot(data: np.array, risk: float = 1e-2, init_level: float = 0.98, num_candidates: int = 10,
        epsilon: float = 1e-8) -> float:
    ''' Peak-over-Threshold Alogrithm

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
    '''
    # Set init threshold0
    t = np.sort(data)[int(init_level * data.size)]
    peaks = data[data > t] - t

    # Grimshaw
    gamma, sigma = grimshaw(peaks=peaks,
                            threshold=t,
                            num_candidates=num_candidates,
                            epsilon=epsilon
                            )

    # Calculate Threshold
    r = data.size * risk / peaks.size
    if gamma != 0:
        z = t + (sigma / gamma) * (pow(r, -gamma) - 1)
    else:
        z = t - sigma * log(r)

    return z, t

def topk(z_scores, label, threshold, k_range=500):
    ''' Top-k method

    Args:
        us: anomaly scores
        label: ground truth

    Returns:
        k: the number of top-k nodes
    '''
    z_scores = np.array(z_scores)
    us_above_threshold = np.where(z_scores > threshold, z_scores, 0.0)
    label = np.array(label)
    us_above_threshold = us_above_threshold.flatten()
    label = label.flatten()
    ranking = np.argsort(us_above_threshold)
    label_ind = np.where(label == 1)[0]
    k_lst = []
    for k in range(1, k_range+1):
        count = [1 if i in label_ind else 0 for i in ranking[-k:]]
        k_lst.append(sum(count)/min(k, len(label_ind)))
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


# https://github.com/hanxiao0607/AERCA/blob/main/models/senn.py

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

        # Instantiate coefficient networks
        for k in range(order):
            modules = [nn.Sequential(nn.Linear(num_vars, hidden_layer_size), nn.ReLU())]
            if num_hidden_layers > 1:
                for j in range(num_hidden_layers - 1):
                    modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()))
            modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, num_vars**2), nn.Tanh()))
            self.coeff_nets.append(nn.Sequential(*modules))

        # Some bookkeeping
        self.num_vars = num_vars
        self.order = order
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layer_size = num_hidden_layers
        self.device = device


    # Initialisation
    def init_weights(self):
        for m in self.modules():
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.fill_(0.1)

    # Forward propagation,
    # returns predictions and generalised coefficients corresponding to each prediction
    def forward(self, inputs: torch.Tensor):
        if inputs[0, :, :].shape != torch.Size([self.order, self.num_vars]):
            print("WARNING: inputs should be of shape BS x K x p")

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
            # coeffs[:, k, :, :] = coeffs_k
            preds = preds + torch.matmul(coeffs_k, inputs[:, k, :].unsqueeze(dim=2)).squeeze()
        return preds, coeffs


# https://github.com/hanxiao0607/AERCA/blob/main/models/aerca.py

class AERCA(nn.Module):
    def __init__(self, num_vars: int, hidden_layer_size: int, num_hidden_layers: int, device: torch.device,
                 window_size: int, stride: int = 1, encoder_alpha: float = 0.5, decoder_alpha: float = 0.5,
                 encoder_gamma: float = 0.5, decoder_gamma: float = 0.5,
                 encoder_lambda: float = 0.5, decoder_lambda: float = 0.5,
                 beta: float = 0.5, lr: float = 0.0001, epochs: int = 100,
                 recon_threshold: float = 0.95, data_name: str = 'ld',
                 causal_quantile: float = 0.80, root_cause_threshold_encoder: float = 0.95,
                 root_cause_threshold_decoder: float = 0.95, initial_z_score: float = 3.0,
                 risk: float = 1e-2, initial_level: float = 0.98, num_candidates: int = 100):
        super(AERCA, self).__init__()
        self.encoder = SENNGC(num_vars, window_size, hidden_layer_size, num_hidden_layers, device)
        self.decoder = SENNGC(num_vars, window_size, hidden_layer_size, num_hidden_layers, device)
        self.decoder_prev = SENNGC(num_vars, window_size, hidden_layer_size, num_hidden_layers, device)
        self.device = device
        self.num_vars = num_vars
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.window_size = window_size
        self.stride = stride
        self.encoder_alpha = encoder_alpha
        self.decoder_alpha = decoder_alpha
        self.encoder_gamma = encoder_gamma
        self.decoder_gamma = decoder_gamma
        self.encoder_lambda = encoder_lambda
        self.decoder_lambda = decoder_lambda
        self.beta = beta
        self.lr = lr
        self.epochs = epochs
        self.recon_threshold = recon_threshold
        self.root_cause_threshold_encoder = root_cause_threshold_encoder
        self.root_cause_threshold_decoder = root_cause_threshold_decoder
        self.initial_z_score = initial_z_score
        self.mse_loss = nn.MSELoss()
        self.mse_loss_wo_reduction = nn.MSELoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.decoder_prev.to(self.device)
        self.model_name = 'AERCA_' + data_name + '_ws_' + str(window_size) + '_stride_' + str(stride) + \
                          '_encoder_alpha_' + str(encoder_alpha) + '_decoder_alpha_' + str(decoder_alpha) + \
                          '_encoder_gamma_' + str(encoder_gamma) + '_decoder_gamma_' + str(decoder_gamma) + \
                          '_encoder_lambda_' + str(encoder_lambda) + '_decoder_lambda_' + str(decoder_lambda) + \
                          '_beta_' + str(beta) + '_lr_' + str(lr) + '_epochs_' + str(epochs) + \
                          '_hidden_layer_size_' + str(hidden_layer_size) + '_num_hidden_layers_' + \
                          str(num_hidden_layers)
        self.causal_quantile = causal_quantile
        self.risk = risk
        self.initial_level = initial_level
        self.num_candidates = num_candidates

        # Create an absolute path for saving models and thresholds
        self.save_dir = os.path.join(os.getcwd(), 'saved_models')
        os.makedirs(self.save_dir, exist_ok=True)

    def _log_and_print(self, msg, *args):
        """Helper method to log and print testing results."""
        final_msg = msg.format(*args) if args else msg
        logging.info(final_msg)
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

    def _training_step_long(self, x, chunk_len=1000, add_u=True, train=True):
            """
            Splits a long raw sequence into overlapping chunks (overlap = window_size,
            so every chunk still has enough context to form valid windows), and calls
            the existing, unmodified _training_step on each piece.
            """
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
                if len(x_chunk) <= self.window_size + 1:
                    continue  # too short to form even one window, skip

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
        loss_recon = self.mse_loss(nexts_hat, nexts)
        logging.info('Reconstruction loss: %s', loss_recon.item())

        loss_encoder_coeffs = self._sparsity_loss(encoder_coeffs, self.encoder_alpha)
        logging.info('Encoder coeffs loss: %s', loss_encoder_coeffs.item())

        loss_decoder_coeffs = self._sparsity_loss(decoder_coeffs, self.decoder_alpha)
        logging.info('Decoder coeffs loss: %s', loss_decoder_coeffs.item())

        loss_prev_coeffs = self._sparsity_loss(prev_coeffs, self.decoder_alpha)
        logging.info('Prev coeffs loss: %s', loss_prev_coeffs.item())

        loss_encoder_smooth = self._smoothness_loss(encoder_coeffs)
        logging.info('Encoder smooth loss: %s', loss_encoder_smooth.item())

        loss_decoder_smooth = self._smoothness_loss(decoder_coeffs)
        logging.info('Decoder smooth loss: %s', loss_decoder_smooth.item())

        loss_prev_smooth = self._smoothness_loss(prev_coeffs)
        logging.info('Prev smooth loss: %s', loss_prev_smooth.item())

        loss_kl = kl_div
        logging.info('KL loss: %s', loss_kl.item())

        loss = (loss_recon +
                self.encoder_lambda * loss_encoder_coeffs +
                self.decoder_lambda * (loss_decoder_coeffs + loss_prev_coeffs) +
                self.encoder_gamma * loss_encoder_smooth +
                self.decoder_gamma * (loss_decoder_smooth + loss_prev_smooth) +
                self.beta * loss_kl)
        logging.info('Total loss: %s', loss.item())

        return loss

    def _training(self, train_loader, val_loader):
        best_val_loss = np.inf
        count = 0
        for epoch in tqdm(range(self.epochs), desc='Epoch'):
            count += 1
            epoch_loss = 0
            self.train()
            for x, _ in train_loader:
                #self.optimizer.zero_grad()
                loss = self._training_step_long(x)
                epoch_loss += loss
                #loss.backward()
                #self.optimizer.step()
            logging.info('Epoch %s/%s', epoch + 1, self.epochs)
            logging.info('Epoch training loss: %s', epoch_loss)
            logging.info('-------------------')

            epoch_val_loss = 0
            self.eval()
            with torch.no_grad():
                for x, _ in val_loader:
                    loss = self._training_step_long(x, train=False)
                    epoch_val_loss += loss
            logging.info('Epoch val loss: %s', epoch_val_loss)
            logging.info('-------------------')

            if epoch_val_loss < best_val_loss:
                count = 0
                logging.info(f'Saving model at epoch {epoch + 1}')
                best_val_loss = epoch_val_loss
                torch.save(self.state_dict(), os.path.join(self.save_dir, f'{self.model_name}.pt'))
            if count >= 20:
                print('Early stopping')
                break

        self.load_state_dict(
            torch.load(os.path.join(self.save_dir, f'{self.model_name}.pt'), map_location=self.device)
        )
        logging.info('Training complete')
        self._get_recon_threshold(val_loader)
        self._get_root_cause_threshold_encoder(val_loader)
        self._get_root_cause_threshold_decoder(val_loader)

    def _testing_step(self, x, label=None, add_u=True):
        nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us = self.forward(x, add_u=add_u)

        if label is not None:
            preprocessed_label = sliding_window_view(label, (self.window_size + 1, self.num_vars))[self.window_size:, 0, :-1, :]
        else:
            preprocessed_label = None

        loss_recon = self.mse_loss(nexts_hat, nexts)
        logging.info('Reconstruction loss: %s', loss_recon.item())

        loss_encoder_coeffs = self._sparsity_loss(encoder_coeffs, self.encoder_alpha)
        logging.info('Encoder coeffs loss: %s', loss_encoder_coeffs.item())

        loss_decoder_coeffs = self._sparsity_loss(decoder_coeffs, self.decoder_alpha)
        logging.info('Decoder coeffs loss: %s', loss_decoder_coeffs.item())

        loss_prev_coeffs = self._sparsity_loss(prev_coeffs, self.decoder_alpha)
        logging.info('Prev coeffs loss: %s', loss_prev_coeffs.item())

        loss_encoder_smooth = self._smoothness_loss(encoder_coeffs)
        logging.info('Encoder smooth loss: %s', loss_encoder_smooth.item())

        loss_decoder_smooth = self._smoothness_loss(decoder_coeffs)
        logging.info('Decoder smooth loss: %s', loss_decoder_smooth.item())

        loss_prev_smooth = self._smoothness_loss(prev_coeffs)
        logging.info('Prev smooth loss: %s', loss_prev_smooth.item())

        loss_kl = kl_div
        logging.info('KL loss: %s', loss_kl.item())

        loss = (loss_recon +
                self.encoder_lambda * loss_encoder_coeffs +
                self.decoder_lambda * (loss_decoder_coeffs + loss_prev_coeffs) +
                self.encoder_gamma * loss_encoder_smooth +
                self.decoder_gamma * (loss_decoder_smooth + loss_prev_smooth) +
                self.beta * loss_kl)
        logging.info('Total loss: %s', loss.item())

        return loss, nexts_hat, nexts, encoder_coeffs, decoder_coeffs, kl_div, preprocessed_label, us

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


class TimeSeriesDataset(Dataset):
    def __init__(self, series_dict):
        self.sample_ids = list(series_dict.keys())
        self.series_dict = series_dict

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        x = self.series_dict[sample_id].values.astype(np.float32)

        return x, sample_id


def identity_collate(batch):
    # batch_size=1 -> just unwrap; sequences differ in length so we never stack them
    return batch[0]

def make_dataloader(series_dict, label_dict=None, shuffle=False):
    dataset = TimeSeriesDataset(series_dict)
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


# Set up logging: create a logs directory relative to the current working directory if it doesn't exist.
logging_dir = os.path.join(os.getcwd(), "logs")
if not os.path.exists(logging_dir):
    os.makedirs(logging_dir)
log_file_path = os.path.join(logging_dir)
logging.basicConfig(
        filename=os.path.join(logging_dir, "out.log"),
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')

data_dict, data_means, data_stds = load_series_dict('/home/kbacsa/data/oosterscheldkering/Normal_5')
train_dict, val_dict = split_series_dict(data_dict, val_ratio=0.2)
train_loader = make_dataloader(train_dict, shuffle=True)
val_loader = make_dataloader(val_dict, shuffle=False)
sample, _ = next(iter(train_loader))

print(sample.shape)

in_ch = sample.shape[1]
hidden_layer_size = 8
num_layers = 2
window_size = 8

model = AERCA(num_vars=in_ch, hidden_layer_size=hidden_layer_size, num_hidden_layers=num_layers, device=torch.device('cuda'), window_size=window_size)

print(model)

model._training(train_loader, val_loader)

