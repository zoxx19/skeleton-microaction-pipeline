#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import csv
import glob
import inspect
import os
import pickle
import random
import shutil
import sys
import time
import traceback
from collections import OrderedDict
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
from timm.scheduler.cosine_lr import CosineLRScheduler
from tqdm import tqdm

from torchlight import DictAction


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
def init_seed(seed: int) -> None:
    """Fix random seeds for reproducibility (main process)."""
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_worker_init_fn(base_seed: int):
    """
    DataLoader worker_init_fn must accept worker_id.
    Seed numpy/random per-worker to avoid duplicated augmentation sampling.
    """
    def _init_fn(worker_id: int):
        seed = (base_seed + worker_id) % (2**32)
        np.random.seed(seed)
        random.seed(seed)
    return _init_fn


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def import_class(import_str: str):
    """Dynamically import a class from 'package.module.ClassName'."""
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError(
            f'Class {class_str} cannot be found ({traceback.format_exception(*sys.exc_info())})'
        )


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if not isinstance(v, str):
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Unsupported value encountered.')


# ------------------------------------------------------------
# Loss
# ------------------------------------------------------------
class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross-entropy for single-label classification."""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = float(smoothing)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth = -logprobs.mean(dim=-1)
        loss = confidence * nll + self.smoothing * smooth
        return loss.mean()


class AsymmetricLoss(nn.Module):
    """ASL: https://arxiv.org/abs/2009.14119. gamma_neg > gamma_pos focuses on hard negatives."""
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0,
                 clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: [B, C] logits,  y: [B, C] float binary labels
        xs_pos = torch.sigmoid(x)
        xs_neg = (1.0 - xs_pos + self.clip).clamp(max=1.0)
        loss = (y       * torch.log(xs_pos.clamp(min=self.eps)) +
                (1 - y) * torch.log(xs_neg.clamp(min=self.eps)))
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = xs_pos * y + (1.0 - xs_pos) * (1.0 - y)
            loss = loss * (1.0 - pt) ** (self.gamma_pos * y + self.gamma_neg * (1.0 - y))
        return -loss.mean()


# ------------------------------------------------------------
# Args
# ------------------------------------------------------------
def get_parser():
    parser = argparse.ArgumentParser(
        description='MMN Single-label Training / Fine-tuning (Unified)'
    )
    parser.add_argument('--work-dir', default=None,
                        help='Work folder for storing results (logs, checkpoints, etc.).')
    parser.add_argument('--model_saved_name', default='',
                        help='Base name used when saving model checkpoints.')

    # config / weights
    parser.add_argument('--weights', default=None,
                        help='Path to pre-trained weights.')
    parser.add_argument('--config', default='./config/train/MA52_J.yaml',
                        help='Path to YAML configuration file.')

    # phase / logging
    parser.add_argument('--phase', default='train', help='train | test')
    parser.add_argument('--save-score', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--save-interval', type=int, default=1)
    parser.add_argument('--save-epoch', type=int, default=30)
    parser.add_argument('--eval-interval', type=int, default=5)
    parser.add_argument('--print-log', type=str2bool, default=True)
    parser.add_argument('--show-topk', type=int, default=[1, 5], nargs='+')

    # data
    parser.add_argument('--feeder', default=None,
                        help='Dataloader class (e.g., feeders.feeder_mma33_single.Feeder).')
    parser.add_argument('--num-worker', type=int)
    parser.add_argument('--train-feeder-args', action=DictAction, default=dict())
    parser.add_argument('--test-feeder-args', action=DictAction, default=dict())

    # Optional: override data_root via CLI
    parser.add_argument('--train-data-root', type=str, default=None,
                        help='Override training data root (passed as data_root to Feeder).')
    parser.add_argument('--test-data-root', type=str, default=None,
                        help='Override test/val data root (passed as data_root to Feeder).')

    # model
    parser.add_argument('--model', default=None,
                        help='Model class (e.g., model.MMN.MMN_).')
    parser.add_argument('--model-args', action=DictAction, default=dict())
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+')

    # optimization
    parser.add_argument('--base-lr', type=float, default=1e-3)
    parser.add_argument('--min-lr', type=float, default=1e-5)
    parser.add_argument('--warmup-lr', type=float, default=1e-6)
    parser.add_argument('--warmup_prefix', type=str2bool, default=False)
    parser.add_argument('--warm_up_epoch', type=int, default=0)
    parser.add_argument('--grad-clip', type=str2bool, default=False)
    parser.add_argument('--grad-max', type=float, default=1.0)

    parser.add_argument('--device', type=int, default=0, nargs='+')
    parser.add_argument('--optimizer', default='AdamW', help='SGD | Adam | AdamW')
    parser.add_argument('--lr-scheduler', default='cosine')
    parser.add_argument('--nesterov', type=str2bool, default=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--test-batch-size', type=int, default=256)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--num-epoch', type=int, default=80)
    parser.add_argument('--weight-decay', type=float, default=0.0005)

    # loss
    parser.add_argument('--loss-type', type=str, default='CE', help='CE | LSCE | BCE | ASL')
    parser.add_argument('--asl-gamma-neg', type=float, default=4.0,
                        help='ASL gamma for negatives (default 4.0).')
    parser.add_argument('--asl-gamma-pos', type=float, default=0.0,
                        help='ASL gamma for positives (default 0.0).')

    # ✅ fine-tuning controls (unified)
    parser.add_argument('--head-only', type=str2bool, default=False,
                        help='Freeze backbone and train only classification head (after loading weights).')
    parser.add_argument('--overwrite', type=str2bool, default=True,
                        help='If True, delete work_dir/runs before training.')

    return parser


# ------------------------------------------------------------
# Processor
# ------------------------------------------------------------
class Processor:
    def __init__(self, arg):
        self.arg = arg
        self.global_step = 0
        self.best_acc = 0.0
        self.best_acc_epoch = 0

        # output device
        self.output_device = arg.device[0] if isinstance(arg.device, list) else arg.device

        # prepare work_dir
        if self.arg.work_dir is None:
            raise ValueError("--work-dir must be provided (or set in config).")
        os.makedirs(self.arg.work_dir, exist_ok=True)

        # save config snapshot
        self.save_arg()

        # writers / runs dir (train only)
        if arg.phase == 'train':
            arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
            if arg.overwrite and os.path.isdir(arg.model_saved_name):
                print('[INFO] Removing existing runs dir:', arg.model_saved_name)
                shutil.rmtree(arg.model_saved_name)
            os.makedirs(arg.model_saved_name, exist_ok=True)

            self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'))
            self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'))

        # build model / data
        self.load_model()
        self.load_data()

        # optimizer/scheduler (train only)
        if self.arg.phase == 'train':
            self.load_optimizer()
            self.load_scheduler(len(self.data_loader['train']))

        # move model to GPU
        self.model = self.model.cuda(self.output_device)
        if isinstance(self.arg.device, list) and len(self.arg.device) > 1:
            self.model = nn.DataParallel(
                self.model,
                device_ids=self.arg.device,
                output_device=self.output_device
            )

    def save_arg(self):
        arg_dict = vars(self.arg)
        os.makedirs(self.arg.work_dir, exist_ok=True)
        with open(os.path.join(self.arg.work_dir, 'config.yaml'), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def print_log(self, s: str, print_time: bool = True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            s = "[ " + localtime + ' ] ' + s
        print(s)
        if self.arg.print_log:
            with open(os.path.join(self.arg.work_dir, 'log.txt'), 'a') as f:
                print(s, file=f)

    # ----------------------------
    # Data
    # ----------------------------
    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader: Dict[str, torch.utils.data.DataLoader] = {}

        worker_init = make_worker_init_fn(self.arg.seed)

        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=worker_init,
            )

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=worker_init,
        )

    # ----------------------------
    # Model / loss / weights / head-only
    # ----------------------------
    def load_model(self):
        Model = import_class(self.arg.model)

        # copy model file into work_dir for reproducibility
        try:
            shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        except Exception:
            pass

        self.model = Model(**self.arg.model_args)

        # loss
        if self.arg.loss_type == 'CE':
            self.loss = nn.CrossEntropyLoss().cuda(self.output_device)
        elif self.arg.loss_type == 'LSCE':
            self.loss = LabelSmoothingCrossEntropy(smoothing=0.1).cuda(self.output_device)
        elif self.arg.loss_type in ('BCE', 'ASL'):
            self.loss = nn.BCEWithLogitsLoss().cuda(self.output_device)
        elif self.arg.loss_type == 'ASL':
            self.loss = AsymmetricLoss(
                gamma_neg=self.arg.asl_gamma_neg,
                gamma_pos=self.arg.asl_gamma_pos,
            ).cuda(self.output_device)
        else:
            raise ValueError(f"Unknown loss_type: {self.arg.loss_type}")

        # load weights (optional)
        if self.arg.weights:
            self.print_log(f'Load weights from {self.arg.weights}.')
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'rb') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights, weights_only=True)

            # remove "module." prefix if exists
            weights = OrderedDict([[k.split('module.')[-1], v.cuda(self.output_device)]
                                   for k, v in weights.items()])

            # ignore weights
            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        weights.pop(key, None)

            # load
            try:
                self.model.load_state_dict(weights, strict=False)
            except Exception:
                state = self.model.state_dict()
                state.update(weights)
                self.model.load_state_dict(state, strict=False)

        # ✅ head-only fine-tuning (AFTER loading weights)
        if getattr(self.arg, "head_only", False):
            self.print_log('[INFO] Head-only mode: freeze backbone, train head only.', print_time=False)
            for p in self.model.parameters():
                p.requires_grad = False

            if hasattr(self.model, 'head'):
                for p in self.model.head.parameters():
                    p.requires_grad = True
            else:
                # fallback: unfreeze any params whose name suggests classifier/head
                for name, p in self.model.named_parameters():
                    if any(k in name.lower() for k in ["head", "cls", "classifier", "fc"]):
                        p.requires_grad = True

            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            self.print_log(f"[INFO] Trainable params: {trainable}/{total} ({trainable/total*100:.2f}%)",
                           print_time=False)

    # ----------------------------
    # Optimizer / scheduler
    # ----------------------------
    def load_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if len(params) == 0:
            raise RuntimeError("No trainable parameters. Check head_only / model head naming.")

        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                params,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(params, lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(params, lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.arg.optimizer}")

        self.print_log(f'Using warm up, epoch: {self.arg.warm_up_epoch}')

    def load_scheduler(self, n_iter_per_epoch: int):
        if self.arg.lr_scheduler != 'cosine':
            raise ValueError(f"Unknown lr_scheduler: {self.arg.lr_scheduler}")

        num_steps = int(self.arg.num_epoch * n_iter_per_epoch)
        warmup_steps = int(self.arg.warm_up_epoch * n_iter_per_epoch)

        self.lr_scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=(num_steps - warmup_steps) if self.arg.warmup_prefix else num_steps,
            lr_min=self.arg.min_lr,
            warmup_lr_init=self.arg.warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=2,
            t_in_epochs=False,
            warmup_prefix=self.arg.warmup_prefix,
        )

    # ----------------------------
    # Train / Eval
    # ----------------------------
    def train_one_epoch(self, epoch: int, save_model: bool):
        self.model.train()
        self.print_log(f'Training epoch: {epoch + 1}')
        loader = self.data_loader['train']

        loss_value, acc_value = [], []
        process = tqdm(loader, ncols=100)

        for batch_idx, (data, index_t, label, index) in enumerate(process):
            self.lr_scheduler.step_update(self.global_step)
            self.global_step += 1

            data = data.float().cuda(self.output_device)
            index_t = index_t.float().cuda(self.output_device)
            if self.arg.loss_type in ('BCE', 'ASL'):
                label = label.float().cuda(self.output_device)
            else:
                label = label.long().cuda(self.output_device)

            output = self.model(data, index_t)
            loss = self.loss(output, label)

            self.optimizer.zero_grad()
            loss.backward()
            if self.arg.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arg.grad_max)
            self.optimizer.step()

            loss_value.append(loss.item())
            if self.arg.loss_type in ('BCE', 'ASL'):
                pred = (torch.sigmoid(output.data) > 0.5).float()
                acc = torch.mean((pred == label.data).float()).item()
            else:
                _, pred = torch.max(output.data, 1)
                acc = torch.mean((pred == label.data).float()).item()
            acc_value.append(acc)

            # tensorboard
            self.train_writer.add_scalar('loss', loss.item(), self.global_step)
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.global_step)

        self.print_log(f'\tMean training loss: {np.mean(loss_value):.4f}.')
        self.print_log(f'\tMean training acc: {np.mean(acc_value) * 100:.2f}%.')

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            save_path = f"{self.arg.model_saved_name}-{epoch + 1}-{int(self.global_step)}.pt"
            torch.save(weights, save_path)
            self.print_log(f"\tSaved checkpoint: {save_path}")

    def eval(self, epoch: int, save_score: bool, loader_name=('test',)):
        self.model.eval()
        self.print_log(f'Eval epoch: {epoch + 1}')

        for ln in loader_name:
            loss_value, score_frag, label_list, pred_list = [], [], [], []
            process = tqdm(self.data_loader[ln], ncols=100)

            for batch_idx, (data, index_t, label, index) in enumerate(process):
                label_list.append(label.numpy())
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    index_t = index_t.float().cuda(self.output_device)
                    if self.arg.loss_type in ('BCE', 'ASL'):
                        label_t = label.float().cuda(self.output_device)
                    else:
                        label_t = label.long().cuda(self.output_device)

                    output = self.model(data, index_t)
                    loss = self.loss(output, label_t)

                    loss_value.append(loss.item())
                    score_frag.append(output.data.cpu().numpy())
                    if self.arg.loss_type in ('BCE', 'ASL'):
                        pred = (torch.sigmoid(output.data) > 0.5).float()
                    else:
                        _, pred = torch.max(output.data, 1)
                    pred_list.append(pred.data.cpu().numpy())

            score = np.concatenate(score_frag) if score_frag else np.zeros((0, 1), dtype=np.float32)
            loss = float(np.mean(loss_value)) if loss_value else 0.0

            # dataset.top_k if available
            self.data_loader[ln].dataset.sample_name = np.arange(len(score))
            if self.arg.loss_type in ('BCE', 'ASL'):
                preds = (torch.sigmoid(torch.tensor(score)) > 0.5).float().numpy()
                labels = np.array([self.data_loader[ln].dataset.label[i] for i in range(len(score))])
                acc1 = float(np.mean((preds == labels)))
            else:
                acc1 = self.data_loader[ln].dataset.top_k(score, 1) if len(score) > 0 else 0.0

            if acc1 > self.best_acc:
                self.best_acc = acc1
                self.best_acc_epoch = epoch + 1

            self.print_log(f'\tMean {ln} loss: {loss:.6f}.')
            if self.arg.loss_type not in ('BCE', 'ASL'):
                for k in self.arg.show_topk:
                    topk = self.data_loader[ln].dataset.top_k(score, k) if len(score) > 0 else 0.0
                    self.print_log(f'\tTop{k}: {topk * 100:.2f}%')

            if self.arg.phase == 'train':
                self.val_writer.add_scalar(f'{ln}_loss', loss, self.global_step)
                self.val_writer.add_scalar(f'{ln}_acc1', acc1, self.global_step)

            if save_score:
                score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))
                with open(os.path.join(self.arg.work_dir, f'epoch{epoch+1}_{ln}_score.pkl'), 'wb') as f:
                    pickle.dump(score_dict, f)

            # confusion matrix + per-class acc
            if len(label_list) > 0 and len(pred_list) > 0:
                y_true = np.concatenate(label_list)
                y_pred = np.concatenate(pred_list)
                if self.arg.loss_type not in ('BCE', 'ASL'):
                    cm = confusion_matrix(y_true, y_pred)
                    self.print_log(str(cm))

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = int(self.arg.start_epoch * len(self.data_loader['train']))

            # initial eval
            self.eval(-1, save_score=True, loader_name=('test',))

            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = ((epoch + 1) % self.arg.save_interval == 0) and ((epoch + 1) >= self.arg.save_epoch)
                self.train_one_epoch(epoch, save_model=save_model)

                if (epoch + 1) % self.arg.eval_interval == 0:
                    self.eval(epoch, save_score=True, loader_name=('test',))

            # load best checkpoint (optional final eval)
            cand = glob.glob(os.path.join(self.arg.work_dir, f"runs-{self.best_acc_epoch}-*.pt"))
            if cand:
                best_path = sorted(cand)[0]
                self.print_log(f"[INFO] Load best checkpoint: {best_path}")
                weights = torch.load(best_path, weights_only=True)
                self.model.load_state_dict(weights, strict=False)

                self.arg.print_log = False
                self.eval(0, save_score=True, loader_name=('test',))
                self.arg.print_log = True

            self.print_log(f'Best acc@1: {self.best_acc}')
            self.print_log(f'Best epoch: {self.best_acc_epoch}')

        elif self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log(f'Model:   {self.arg.model}.')
            self.print_log(f'Weights: {self.arg.weights}.')
            self.eval(0, save_score=self.arg.save_score, loader_name=('test',))
            self.print_log('Done.\n')


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == '__main__':
    parser = get_parser()

    # first parse to read config path
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)

        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG:', k)
                assert (k in key)

        parser.set_defaults(**default_arg)

    # parse again with defaults loaded from config
    arg = parser.parse_args()

    # override data_root if provided
    if arg.train_data_root is not None:
        arg.train_feeder_args['data_root'] = arg.train_data_root
    if arg.test_data_root is not None:
        arg.test_feeder_args['data_root'] = arg.test_data_root

    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
