# Vendored (near-verbatim) from https://github.com/jarrycyx/UNN/blob/main/CUTS_Plus/cuts_plus.py (MIT license,
# see ../LICENSE). Seven deliberate deviations from upstream, kept as the only edits so this stays
# otherwise faithful to the source:
#   1. The trailing `if __name__ == "__main__":` block was removed: it referenced a yaml file not
#      present in the published CUTS_Plus subfolder and called main() with the wrong arity, since the
#      real CLI entry point lives in the parent UNN repo.
#   2. Performance-only (same math, same results, no behavior change): batch_generater's per-sample
#      Python loop was vectorized, and MultiCAD.train()'s per-batch `.item()` calls (each a blocking
#      CPU/GPU sync) were deduplicated and rate-limited to every LOG_EVERY_N_BATCHES batches instead
#      of every batch - see LOG_EVERY_N_BATCHES below. Both were identified as the cause of low GPU
#      utilization on real hardware.
#   3. MultiCAD.train() gained an optional `epoch_callback` parameter (default None, so any existing
#      caller is unaffected) invoked once per epoch with the current graph - lets a caller plot it
#      periodically without touching this training loop.
#   4. Bug fix: the end-of-epoch `Graph` computed for plot_matrix/calc_and_log_metrics/the returned
#      value used the raw `self.GT` parameter (an unconstrained logit, can be negative) instead of
#      `torch.sigmoid(self.GT)` - the version actually used everywhere else in this file (the S1/S2
#      training forward passes, which require a [0,1] probability for torch.bernoulli/gumbel_softmax
#      sampling to be well-defined). This made every plotted/saved/returned Graph a raw logit rather
#      than an edge probability - harmless to rank-based metrics (AUROC/AUPRC, quantile-based
#      binarization) since sigmoid is monotonic, but confusing to read directly (negative values look
#      like "inverse causation" when they just mean "logit < 0, i.e. probability < 0.5").
#   5. Performance-only (same math, same results, no behavior change): MultiCAD.train() used to build
#      every batch's actual tensors for the whole epoch via batch_generater() + list(...), because its
#      two loops (data prediction, then graph discovery) need to see the identical batch grouping. That
#      holds all of an epoch's batches in memory simultaneously, independent of --batch-size - OOM'd on
#      a wide dataset (1352 channels) even at batch_size=2, since the fixed cost scales with the total
#      number of training windows, not the batch size. generate_batch_index_groups() now plans the same
#      grouping cheaply (a shuffled list of window-start integers), and materialize_batch() builds one
#      batch's real tensors on demand inside each loop, so peak memory is one batch at a time - both
#      loops still iterate the exact same index groups, so results are unchanged.
#   6. MultiCAD.train() gained optional checkpoint_path/checkpoint_every parameters (default None/1,
#      so any existing caller is unaffected): every checkpoint_every epochs, the model, both
#      optimizers/schedulers, and the coarse-to-fine graph state (G, GT, n_groups, lambda_s,
#      gumbel_tau) are saved to checkpoint_path; if that file already exists when train() is called,
#      it's loaded and training resumes right after the epoch it was saved at. Added for real-dataset
#      runs long enough (hours) that losing all progress to an interruption is a real cost - a pure
#      addition, no effect on any run that doesn't pass checkpoint_path.
#   7. Performance-only (same math, same results, no behavior change): two more per-batch costs found
#      after fixing LocalConv1D's kernel-launch storm (see the vendored model file's own deviation).
#      First, latent_data_pred (S1) recomputed an O(n_nodes^2) Graph tensor (einsum + sigmoid) on
#      every one of ~5,765 batches/epoch, even though self.GT/self.G never change during S1 (GT only
#      updates via S2's graph_optimizer.step(), G only at group-refinement epoch boundaries) - now
#      computed once per epoch (detached; this path never backpropped into GT/G anyway, since
#      torch.bernoulli has no gradient w.r.t. its input) and passed into latent_data_pred instead of
#      recomputed inside it. Second, materialize_batch rebuilt x_offsets/y_offsets (constant for the
#      whole run) and re-converted each batch's index list to a tensor independently in both the S1
#      and S2 calls for the same batch (~11,530 conversions/epoch instead of ~5,765) - both are now
#      precomputed once per epoch/batch respectively and passed in. Also removed sample_multinorm, a
#      nested function in latent_data_pred that was redefined on every S1 call but never actually
#      called. Unlike deviation 6, this one does not change the model's state_dict or any checkpointed
#      state's shape - a checkpoint saved before this change still loads and resumes correctly.

import logging
import os, sys
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops

sys.path.append(opj(opd(__file__), ".."))

import tqdm
import numpy as np
from matplotlib import pyplot as plt
import argparse
from omegaconf import OmegaConf
from copy import deepcopy
import torch
from torch import dropout, nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve, roc_auc_score

from utils.gumbel_softmax import gumbel_softmax
from utils.misc import calc_and_log_metrics, log_time_series, plot_causal_matrix
from utils.opt_type import MultiCADopt
from utils.logger import MyLogger

from datetime import datetime
from model.cuts_plus_net import CUTS_Plus_Net

import os
from einops import rearrange

LOG_EVERY_N_BATCHES = 10  # perf patch: how often to sync losses to CPU for logging/progress display


def plot_matrix(name, mat, log, log_step, vmin=None, vmax=None):
    if len(mat.shape) == 3:
        mat = np.max(mat, axis=-1)
    n, m = mat.shape

    # Show Discovered Graph (Probability)
    sub_cg = plot_causal_matrix(
        mat,
        figsize=[1.5*n, 1*n],
        show_text=False,
        vmin=vmin, vmax=vmax)
    log.log_figures(sub_cg, name=name, iters=log_step)


def generate_indices(input_step, pred_step, t_length, block_size=None):
    if block_size is None:
        block_size = t_length

    offsets_in_block = np.arange(input_step, block_size-pred_step+1)
    assert t_length % block_size == 0, "t_length % block_size != 0"
    random_t_list = []
    for block_start in range(0, t_length, block_size):
        random_t_list += (offsets_in_block + block_start).tolist()

    np.random.shuffle(random_t_list)
    return random_t_list



def batch_generater(data, observ_mask, bs, n_nodes, input_step, pred_step, block_size=None):
    t, n, d = data.shape
    first_sample_t = input_step
    random_t_list = generate_indices(input_step, pred_step, t_length=t, block_size=block_size)

    # perf patch: the original built each batch with a Python for-loop assigning one sample's
    # window at a time (bs individual GPU slice-and-copy ops per batch). Vectorized into a single
    # gather per tensor - same result, far fewer GPU kernel launches per batch.
    x_offsets = torch.arange(-input_step, 0, device=data.device, dtype=torch.long)
    y_offsets = torch.arange(0, pred_step, device=data.device, dtype=torch.long)

    for batch_i in range(len(random_t_list) // bs):
        batch_t = [random_t_list.pop() for _ in range(bs)]
        t_idx = torch.tensor(batch_t, device=data.device, dtype=torch.long)  # (bs,)

        x_idx = t_idx.unsqueeze(1) + x_offsets.unsqueeze(0)  # (bs, input_step)
        y_idx = t_idx.unsqueeze(1) + y_offsets.unsqueeze(0)  # (bs, pred_step)

        x = data[x_idx].permute(0, 2, 1, 3)          # (bs, input_step, n, d) -> (bs, n, input_step, d)
        y = data[y_idx].permute(0, 2, 1, 3)           # (bs, pred_step, n, d) -> (bs, n, pred_step, d)
        mask_x = observ_mask[x_idx].permute(0, 2, 1, 3)
        mask_y = observ_mask[y_idx].permute(0, 2, 1, 3)

        yield x, y, t_idx, mask_x, mask_y


def generate_batch_index_groups(input_step, pred_step, t_length, bs, block_size=None):
    """Plans which window-start indices belong to each batch for one epoch - cheap (a shuffled list
    of plain Python ints), unlike materialize_batch's actual tensor gathering below. Returns a list
    of bs-sized index lists, one per batch - see deviation 5."""
    random_t_list = generate_indices(input_step, pred_step, t_length=t_length, block_size=block_size)
    n_batches = len(random_t_list) // bs
    return [random_t_list[i * bs:(i + 1) * bs] for i in range(n_batches)]


def materialize_batch(data, observ_mask, t_idx, x_offsets, y_offsets):
    """Gathers one batch's x/y/mask_x/mask_y tensors from a precomputed window-start-index tensor and
    offset tensors - the same tensor construction batch_generater does per batch, factored out so it
    can be called fresh for each batch instead of once for every batch in the epoch up front - see
    deviation 5. x_offsets/y_offsets depend only on input_step/pred_step (constant for the whole run),
    so the caller computes them once per epoch instead of passing input_step/pred_step here to be
    rebuilt on every call; t_idx is precomputed once per batch and shared between this batch's S1 and
    S2 calls, instead of being converted from a Python list independently in each - see deviation 7."""
    x_idx = t_idx.unsqueeze(1) + x_offsets.unsqueeze(0)
    y_idx = t_idx.unsqueeze(1) + y_offsets.unsqueeze(0)
    x = data[x_idx].permute(0, 2, 1, 3)
    y = data[y_idx].permute(0, 2, 1, 3)
    mask_x = observ_mask[x_idx].permute(0, 2, 1, 3)
    mask_y = observ_mask[y_idx].permute(0, 2, 1, 3)
    return x, y, t_idx, mask_x, mask_y





class MultiCAD(object):
    def __init__(self, args: MultiCADopt.MultiCADargs, log, device="cuda"):
        self.log: MyLogger = log
        self.args = args
        self.device = device

        # self.fitting_model = CUTS_Plus_LSTM(self.args.data_dim,
        #                                self.args.data_pred.mlp_hid,
        #                                self.args.data_dim * self.args.data_pred.pred_step,
        #                                self.args.data_pred.mlp_layers,
        #                                self.args.n_nodes).to(self.device)
        self.fitting_model = CUTS_Plus_Net(self.args.n_nodes, in_ch=self.args.data_dim,
                                           n_layers=self.args.data_pred.gru_layers,
                                           hidden_ch=self.args.data_pred.mlp_hid,
                                           shared_weights_decoder=self.args.data_pred.shared_weights_decoder,
                                           concat_h=self.args.data_pred.concat_h,
                                           ).to(self.device)

        self.data_pred_loss = nn.MSELoss()
        self.data_pred_optimizer = torch.optim.Adam(self.fitting_model.parameters(),
                                                    lr=self.args.data_pred.lr_data_start,
                                                    weight_decay=self.args.data_pred.weight_decay)


        if "every" in self.args.fill_policy:
            lr_schedule_length = int(self.args.fill_policy.split("_")[-1])
        else:
            lr_schedule_length = self.args.total_epoch

        gamma = (self.args.data_pred.lr_data_end / self.args.data_pred.lr_data_start) ** (1 / lr_schedule_length)
        self.data_pred_scheduler = torch.optim.lr_scheduler.StepLR(
            self.data_pred_optimizer, step_size=1, gamma=gamma)

        self.n_groups = self.args.n_groups
        print("n_groups: ", self.n_groups)
        if self.args.group_policy == "None":
            self.args.group_policy = None

        end_tau, start_tau = self.args.graph_discov.end_tau, self.args.graph_discov.start_tau
        self.gumbel_tau_gamma = (end_tau / start_tau) ** (1 / self.args.total_epoch)
        self.gumbel_tau = start_tau
        self.start_tau = start_tau

        end_lmd, start_lmd = self.args.graph_discov.lambda_s_end, self.args.graph_discov.lambda_s_start
        self.lambda_gamma = (end_lmd / start_lmd) ** (1 / self.args.total_epoch)
        self.lambda_s = start_lmd

    def set_graph_optimizer(self, epoch=None):
        if epoch == None:
            epoch = 0

        gamma = (self.args.graph_discov.lr_graph_end / self.args.graph_discov.lr_graph_start) ** (1 / self.args.total_epoch)
        self.graph_optimizer = torch.optim.Adam([self.GT], lr=self.args.graph_discov.lr_graph_start * gamma ** epoch)
        self.graph_scheduler = torch.optim.lr_scheduler.StepLR(self.graph_optimizer, step_size=1, gamma=gamma)


    def latent_data_pred(self, x, y, mask_x, mask_y, graph):
        # graph is precomputed once per epoch by the caller (train()'s S1 loop), not recomputed here
        # on every batch - see deviation 7. self.GT/self.G never change during S1 (only S2's
        # graph_optimizer.step() updates GT; G only changes at group-refinement epoch boundaries,
        # outside both batch loops), so graph's value would be identical on every call anyway; it's
        # passed in detached, which is safe since torch.bernoulli has no gradient w.r.t. its input, so
        # this path never needed to backprop into self.GT/self.G in the first place.

        def sample_bernoulli(sample_matrix, batch_size):
            sample_matrix = sample_matrix[None].expand(batch_size, -1, -1)
            return torch.bernoulli(sample_matrix).float()

        bs, n, t, d = x.shape
        self.fitting_model.train()
        self.data_pred_optimizer.zero_grad()

        graph_sampled = sample_bernoulli(graph, self.args.batch_size)

        y_pred = self.fitting_model(x, mask_x, graph_sampled)

        # print(y_pred.shape, y.shape, observ_mask.shape)
        loss = self.data_pred_loss(y * mask_y, y_pred * mask_y) / torch.mean(mask_y)
        loss.backward()
        self.data_pred_optimizer.step()
        return y_pred, loss

    def graph_discov(self, x, y, mask_x, mask_y):

        def gumbel_sigmoid_sample(graph, batch_size, tau=1):
            prob = graph[None, :, :, None].expand(batch_size, -1, -1, -1)
            logits = torch.concat([prob, (1-prob)], axis=-1)
            samples = gumbel_softmax(logits, tau=tau, hard=True)[:, :, :, 0]
            return samples

        gn, n = self.GT.shape
        # self.fitting_model.eval()
        self.graph_optimizer.zero_grad()
        GT_prob = self.GT
        G_prob = self.G

        Graph = torch.einsum("nm,ml->nl", G_prob, torch.sigmoid(GT_prob))
        graph_sampled = gumbel_sigmoid_sample(Graph, self.args.batch_size)

        loss_sparsity = torch.linalg.norm(Graph.flatten(), ord=1) / (n * n)

        y_pred = self.fitting_model(x, mask_x, graph_sampled)

        loss_data = self.data_pred_loss(y * mask_y, y_pred * mask_y) / torch.mean(mask_y)
        loss = loss_sparsity * self.lambda_s + loss_data
        loss.backward()
        self.graph_optimizer.step()

        return loss, loss_sparsity, loss_data



    def train(self, data, observ_mask, original_data, true_cm=None, epoch_callback=None,
              checkpoint_path=None, checkpoint_every=1):
        # perf patch note above covers deviations 1-2; this optional epoch_callback param is a third,
        # purely additive one: defaults to None (no-op, unchanged behavior for any existing caller),
        # invoked as epoch_callback(epoch_i, Graph) once per epoch with the raw (untransposed,
        # source->target) graph, so a caller can e.g. plot it periodically without modifying this loop.
        # checkpoint_path/checkpoint_every are a sixth, also purely additive deviation - both default
        # to a no-op for any existing caller. See deviation 6 in the header comment.

        original_data = torch.from_numpy(original_data).float().to(self.device)
        observ_mask = torch.from_numpy(observ_mask).float().to(self.device)
        data = torch.from_numpy(data).float().to(self.device)

        if self.args.supervision_policy == "masked":
            print("Using masked supervision for data prediction...")
        elif self.args.supervision_policy == "full":
            print("Using full supervision for data prediction......")
            observ_mask = torch.ones_like(observ_mask)
        elif "masked_before" in self.args.supervision_policy:
            print(f"Using masked supervision for data prediction ({self.args.supervision_policy:s})......")

        latent_pred_step = 0
        graph_discov_step = 0
        start_epoch = 0
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.fitting_model.load_state_dict(ckpt["fitting_model_state"])
            self.data_pred_optimizer.load_state_dict(ckpt["data_pred_optimizer_state"])
            self.data_pred_scheduler.load_state_dict(ckpt["data_pred_scheduler_state"])
            self.n_groups = ckpt["n_groups"]
            self.G = ckpt["G"].to(self.device)
            self.GT = nn.Parameter(ckpt["GT"].to(self.device))
            self.lambda_s = ckpt["lambda_s"]
            self.gumbel_tau = ckpt["gumbel_tau"]
            # set_graph_optimizer rebuilds graph_optimizer/graph_scheduler bound to the just-restored
            # self.GT object (a fresh nn.Parameter, so any prior optimizer's reference is stale); the
            # epoch argument only sets the initial lr, immediately overwritten by load_state_dict below.
            self.set_graph_optimizer(0)
            self.graph_optimizer.load_state_dict(ckpt["graph_optimizer_state"])
            self.graph_scheduler.load_state_dict(ckpt["graph_scheduler_state"])
            latent_pred_step = ckpt["latent_pred_step"]
            graph_discov_step = ckpt["graph_discov_step"]
            start_epoch = ckpt["epoch"] + 1
            print(f"Resumed from checkpoint {checkpoint_path} at epoch {start_epoch}")
        pbar = tqdm.tqdm(total=self.args.total_epoch, initial=start_epoch)
        data_interp = deepcopy(data)
        original_mask = deepcopy(observ_mask)
        auc = 0
        for epoch_i in range(start_epoch, self.args.total_epoch):

            if self.args.group_policy is not None:
                group_mul = int(self.args.group_policy.split("_")[1])
                group_every = int(self.args.group_policy.split("_")[3])
                if epoch_i % group_every == 0 and self.n_groups < self.args.n_nodes:
                    if epoch_i != 0:
                        self.n_groups *= group_mul
                    if self.n_groups > self.args.n_nodes:
                        self.n_groups = self.args.n_nodes

                    self.G = torch.zeros([self.args.n_nodes, self.n_groups]).to(self.device)

                    for i in range(0, self.n_groups):
                        for j in range(0, self.args.n_nodes // self.n_groups):
                            self.G[i*(self.args.n_nodes // self.n_groups) + j, i] = 1
                    for k in range(i*(self.args.n_nodes // self.n_groups) + j, self.args.n_nodes):
                        self.G[k, i] = 1

                    # inv_A = torch.linalg.inv(torch.mm(torch.t(self.fwd_graphA), self.fwd_graphA))
                    # fwd_graphB_init = torch.mm(inv_A, torch.mm(torch.t(self.fwd_graphA), self.fwd_graph))

                    if hasattr(self, "GT"):
                        GT_init = torch.sigmoid(self.GT).detach().cpu().repeat_interleave(group_mul, 0)[:self.n_groups, :]
                        GT_init = 1 - (1 - GT_init)**(1 / group_mul)
                    else:
                        GT_init = torch.ones((self.n_groups, self.args.n_nodes))*0.5

                    self.GT = nn.Parameter(GT_init.to(self.device))

                    self.set_graph_optimizer(epoch_i)
                elif epoch_i == 0 and self.n_groups == self.args.n_nodes:
                    self.G = torch.eye(self.args.n_nodes).to(self.device)
                    GT_init = torch.ones((self.n_groups, self.args.n_nodes))*0.5
                    self.GT = nn.Parameter(GT_init.to(self.device))
                    self.set_graph_optimizer(epoch_i)


            if "every" in self.args.fill_policy:
                update_every = int(self.args.fill_policy.split("_")[-1])
                if (epoch_i+1) % update_every == 0:
                    data = data_pred
                    print("Update data!")
                    # self.graph_optimizer.param_groups[0]['lr'] = self.args.graph_discov.lr_graph_start
                    self.data_pred_optimizer.param_groups[0]['lr'] = self.args.data_pred.lr_data_start
                    observ_mask = torch.ones_like(original_mask)
            elif "rate" in self.args.fill_policy:
                update_rate = float(self.args.fill_policy.split("_")[1])
                update_after = int(self.args.fill_policy.split("_")[3])
                if epoch_i+1 > update_after:
                    if epoch_i == update_after:
                        print("Data update started!")
                    data = data * (1 - update_rate) + data_pred * update_rate
            else:
                # no data update
                pass

            if "masked_before" in self.args.supervision_policy:
                masked_before = int(self.args.supervision_policy.split("_")[2])
                if epoch_i == masked_before:
                    print("Using full supervision for data prediction......")
                    observ_mask = torch.ones_like(original_mask)
                    self.gumbel_tau = self.start_tau

            # Data Prediction
            if hasattr(self.args, "data_pred"):
                if hasattr(self.args, "block_size"):
                    block_size = self.args.block_size
                else:
                    block_size = None
                ##
                # perf patch: previously built every batch's actual x/y/mask tensors up front via
                # batch_generater() + list(...), so the two loops below (S1, S2) could share the same
                # batch grouping - but that means every batch in the epoch is held in memory at once,
                # independent of --batch-size, which OOM's on a wide dataset (1352 channels) even at
                # batch_size=2. generate_batch_index_groups() plans the same grouping cheaply (just
                # window-start integers); materialize_batch() (called in each loop below) builds one
                # batch's actual tensors on demand - see deviation 5.
                batch_index_groups = generate_batch_index_groups(
                    self.args.input_step, self.args.data_pred.pred_step, data.shape[0],
                    bs=self.args.batch_size, block_size=block_size)
                # perf patch (deviation 7): x_offsets/y_offsets depend only on input_step/pred_step
                # (constant for the whole run) and each batch's t_idx tensor is shared between the S1
                # and S2 loops below - all previously rebuilt inside materialize_batch on every one of
                # the ~11,530 calls/epoch (once per batch per phase) despite never changing per-call.
                batch_index_tensors = [torch.tensor(g, device=data.device, dtype=torch.long)
                                       for g in batch_index_groups]
                x_offsets = torch.arange(-self.args.input_step, 0, device=data.device, dtype=torch.long)
                y_offsets = torch.arange(0, self.args.data_pred.pred_step, device=data.device, dtype=torch.long)
                # S1's Graph never changes across an epoch's S1 batches (self.GT/self.G are read-only
                # here - GT only updates via S2's graph_optimizer.step(), G only at group-refinement
                # epoch boundaries), so compute it once instead of on every one of ~5,765 S1 batches.
                # Detached: this path never backpropped into self.GT/self.G anyway (torch.bernoulli has
                # no gradient w.r.t. its input), so this changes nothing about what gets learned.
                s1_graph = torch.einsum("nm,ml->nl", self.G, torch.sigmoid(self.GT)).detach()

                # perf patch (deviation 7): data_pred_all is only ever read by log_time_series below,
                # itself gated by show_graph_every - building the full-dataset clone and scatter-
                # writing into it on every S1 batch was pure waste on the (typically vast majority of)
                # epochs that won't actually use it.
                need_data_pred_all = (epoch_i + 1) % self.args.show_graph_every == 0
                data_pred = deepcopy(data) # masked data points are predicted
                data_pred_all = deepcopy(data) if need_data_pred_all else None
                for batch_idx, t_idx in enumerate(batch_index_tensors):
                    x, y, t, mask_x, mask_y = materialize_batch(data, observ_mask, t_idx, x_offsets, y_offsets)
                    latent_pred_step += self.args.batch_size
                    y_pred, loss = self.latent_data_pred(x, y, mask_x, mask_y, s1_graph)
                    data_pred[t] = (y_pred*(1-mask_y) + y*mask_y).clone().detach()[:,:,0]
                    if need_data_pred_all:
                        data_pred_all[t] = y_pred.clone().detach()[:,:,0]
                    # perf patch: .item() is a blocking CPU/GPU sync - only pay for it every
                    # LOG_EVERY_N_BATCHES batches, and only once (previously called twice for the
                    # same value: once for logging, once for the progress bar).
                    if batch_idx % LOG_EVERY_N_BATCHES == 0:
                        loss_value = loss.item()
                        self.log.log_metrics({"latent_data_pred/pred_loss": loss_value}, latent_pred_step)
                        pbar.set_postfix_str(f"S1 loss={loss_value:.2f}, spr=IDLE, auc={auc:.4f}")

                current_data_pred_lr = self.data_pred_optimizer.param_groups[0]['lr']
                self.log.log_metrics({"graph_discov/lr": current_data_pred_lr}, latent_pred_step)
                self.data_pred_scheduler.step()
                mse_pred_to_original = self.data_pred_loss(original_data, data_pred)
                mse_interp_to_original = self.data_pred_loss(original_data, data_interp)

                self.log.log_metrics({"latent_data_pred/mse_pred_to_original": mse_pred_to_original,
                                      "latent_data_pred/mse_interp_to_original": mse_interp_to_original}, latent_pred_step)

            # Graph Discovery
            if hasattr(self.args, "graph_discov"):
                for batch_idx, t_idx in enumerate(batch_index_tensors):
                    x, y, t, mask_x, mask_y = materialize_batch(data, observ_mask, t_idx, x_offsets, y_offsets)
                    graph_discov_step += self.args.batch_size
                    if hasattr(self.args, "disable_graph") and self.args.disable_graph:
                        pass
                    else:
                        loss, loss_sparsity, loss_data = self.graph_discov(x, y, mask_x, mask_y)
                        # perf patch: see the matching comment in the S1 loop above.
                        if batch_idx % LOG_EVERY_N_BATCHES == 0:
                            loss_value, loss_sparsity_value, loss_data_value = (
                                loss.item(), loss_sparsity.item(), loss_data.item())
                            self.log.log_metrics({"graph_discov/sparsity_loss": loss_sparsity_value,
                                                "graph_discov/data_loss": loss_data_value,
                                                "graph_discov/total_loss": loss_value}, graph_discov_step)
                            pbar.set_postfix_str(
                                f"S2 loss={loss_data_value:.2f}, spr={loss_sparsity_value:.2f}, auc={auc:.4f}")

                self.graph_scheduler.step()
                # self.group_scheduler.step()
                current_graph_disconv_lr = self.graph_optimizer.param_groups[0]['lr']
                self.log.log_metrics({"graph_discov/lr": current_graph_disconv_lr}, graph_discov_step)
                self.log.log_metrics({"graph_discov/tau": self.gumbel_tau}, graph_discov_step)
                self.gumbel_tau *= self.gumbel_tau_gamma
                self.lambda_s *= self.lambda_gamma

            pbar.update(1)

            plot_roc = False

            G_prob = self.G.detach().cpu().numpy()
            GT_prob = torch.sigmoid(self.GT).detach().cpu().numpy()  # see deviation 4 in header comment
            Graph = np.einsum("nm,ml->nl", G_prob, GT_prob)

            if epoch_callback is not None:
                epoch_callback(epoch_i, Graph)

            if checkpoint_path is not None and (epoch_i + 1) % checkpoint_every == 0:
                # Written to a temp file then renamed (atomic on the same filesystem) so a checkpoint
                # is never left half-written if the process is killed mid-save.
                tmp_path = checkpoint_path + ".tmp"
                torch.save({
                    "epoch": epoch_i,
                    "fitting_model_state": self.fitting_model.state_dict(),
                    "data_pred_optimizer_state": self.data_pred_optimizer.state_dict(),
                    "data_pred_scheduler_state": self.data_pred_scheduler.state_dict(),
                    "graph_optimizer_state": self.graph_optimizer.state_dict(),
                    "graph_scheduler_state": self.graph_scheduler.state_dict(),
                    "n_groups": self.n_groups,
                    "G": self.G.detach().cpu(),
                    "GT": self.GT.detach().cpu(),
                    "lambda_s": self.lambda_s,
                    "gumbel_tau": self.gumbel_tau,
                    "latent_pred_step": latent_pred_step,
                    "graph_discov_step": graph_discov_step,
                }, tmp_path)
                os.replace(tmp_path, checkpoint_path)

            if (epoch_i+1) % self.args.show_graph_every == 0:
                avg_mask = np.mean(observ_mask.cpu().numpy(), axis=(0,2))
                if np.min(avg_mask) < 1:
                    time_series_idx = int(np.argwhere(avg_mask < 1)[0])
                else:
                    time_series_idx = 0
                log_time_series(original_data.cpu()[-100:,time_series_idx],
                                data_interp.cpu()[-100:,time_series_idx],
                                data_pred_all.cpu()[-100:,time_series_idx], log=self.log, log_step=latent_pred_step)
                # plot_causal_matrix_in_training(G_A0_GT, self.log, graph_discov_step, threshold=threshold)
                plot_matrix("G", G_prob, self.log, graph_discov_step, vmin=0, vmax=1)
                plot_matrix("GT", GT_prob, self.log, graph_discov_step, vmin=0, vmax=1)
                plot_matrix("Graph", Graph, self.log, graph_discov_step, vmin=0, vmax=1)
                np.save(os.path.join(self.log.log_dir, 'Graph.npy'), Graph)
                plot_roc = True

            # Show TPR FPR AUC ROC
            if true_cm is not None:
                Graph = rearrange(Graph, "n m -> m n")
                auc = calc_and_log_metrics(Graph, true_cm,
                                           self.log, graph_discov_step, plot_roc=plot_roc)

        return Graph


def prepross_data(data):
    T, N, D = data.shape
    new_data = np.zeros_like(data, dtype=float)
    for i in range(N):
        node = data[:,i,:]
        new_data[:,i,:] = (node - np.mean(node)) / np.std(node)

    return new_data



def main(data, mask, true_cm, opt, log, device="cuda"):
    if opt.n_nodes == "auto":
        opt.n_nodes = data.shape[1]

    data = data[:,:,None]
    mask = mask[:,:,None]
    data = prepross_data(data)

    multicad = MultiCAD(opt, log, device=device)
    Graph = multicad.train(data, mask, data, true_cm)
    return Graph
