import logging
import math
import numpy as np
import random
import functools
import glog as log

import torch
from torch import nn, optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from pytorch_transformers.optimization import AdamW


class WarmupLinearScheduleNonZero(_LRScheduler):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to max_lr over `warmup_steps` training steps.
        Linearly decreases learning rate linearly to min_lr over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, min_lr=1e-5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.min_lr = min_lr
        super(WarmupLinearScheduleNonZero, self).__init__(optimizer, last_epoch=last_epoch)
    
    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            lr_factor =  float(step) / float(max(1, self.warmup_steps))
        else:
            lr_factor = max(0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))

        return [base_lr * lr_factor if (base_lr * lr_factor) > self.min_lr else self.min_lr for base_lr in self.base_lrs]


def init_optim(model, config):
    optimizer_grouped_parameters = []

    exclude_from_weight_decay=['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    for module_name, module in model.named_children():
        if config['model_type'] == 'vonly' or module_name == 'encoder':
            lr = config['learning_rate_bert']
        else:
            lr = config['learning_rate_task']
        for param_name, param in module.named_parameters():
            if param.requires_grad:
                if any(ex in param_name for ex in exclude_from_weight_decay):
                    optimizer_grouped_parameters += [
                        {"params": [param], 
                        "lr": lr,
                        "weight_decay": 0}
                    ]
                else:
                    optimizer_grouped_parameters += [
                        {"params": [param], 
                        "lr": lr,
                        "weight_decay": 0.01}
                    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config['learning_rate_task'])
    scheduler = WarmupLinearScheduleNonZero(optimizer, 
                                            warmup_steps=config['warmup_steps'], 
                                            t_total=config['train_steps'],
                                            min_lr=config['min_lr'])

    return optimizer, scheduler

def build_torch_optimizer(model, config):
    """Builds the PyTorch optimizer.

    We use the default parameters for Adam that are suggested by
    the original paper https://arxiv.org/pdf/1412.6980.pdf
    These values are also used by other established implementations,
    e.g. https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
    https://keras.io/optimizers/
    Recently there are slightly different values used in the paper
    "Attention is all you need"
    https://arxiv.org/pdf/1706.03762.pdf, particularly the value beta2=0.98
    was used there however, beta2=0.999 is still arguably the more
    established value, so we use that here as well

    Args:
      model: The model to optimize.
      config: The dictionary of options.

    Returns:
      A ``torch.optim.Optimizer`` instance.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    betas = [0.9, 0.999]
    exclude_from_weight_decay=['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    params = {'bert': [], 'task': []}
    for module_name, module in model.named_children():
        if module_name == 'encoder':
            param_type = 'bert'
        else:
            param_type = 'task'
        for param_name, param in module.named_parameters():
            if param.requires_grad:
                if any(ex in param_name for ex in exclude_from_weight_decay):
                    params[param_type] += [
                        {"params": [param], 
                        "weight_decay": 0}
                    ]
                else:
                    params[param_type] += [
                        {"params": [param], 
                        "weight_decay": 0.01}
                    ]
    if config['task_optimizer'] == 'adamw':
        log.info('Using AdamW as task optimizer')
        task_optimizer = AdamWeightDecay(params['task'],
                                         lr=config["learning_rate_task"],
                                         betas=betas,
                                         eps=1e-6)
    elif config['task_optimizer'] == 'adam':
        log.info('Using Adam as task optimizer')
        task_optimizer = optim.Adam(params['task'],
                                    lr=config["learning_rate_task"],
                                    betas=betas,
                                    eps=1e-6)
    if len(params['bert']) > 0:
        bert_optimizer = AdamWeightDecay(params['bert'],
                                         lr=config["learning_rate_bert"],
                                         betas=betas,
                                         eps=1e-6)
        optimizer = MultipleOptimizer([bert_optimizer, task_optimizer])
    else:
        optimizer = task_optimizer

    return optimizer


def make_learning_rate_decay_fn(decay_method, train_steps, **kwargs):
    """Returns the learning decay function from options."""
    if decay_method == "linear":
        return functools.partial(
            linear_decay,
            global_steps=train_steps,
            **kwargs)
    elif decay_method == "exp":
        return functools.partial(
            exp_decay,
            global_steps=train_steps,
            **kwargs)
    else:
        raise ValueError(f'{decay_method} not found')


def linear_decay(step, global_steps, warmup_steps=100, initial_learning_rate=1, end_learning_rate=0, **kargs):
    if step < warmup_steps:
        return initial_learning_rate * step / warmup_steps
    else:
        return (initial_learning_rate - end_learning_rate) * \
               (1 - (step - warmup_steps) / (global_steps - warmup_steps)) + \
               end_learning_rate

def exp_decay(step, global_steps, decay_exp=1, warmup_steps=100, initial_learning_rate=1, end_learning_rate=0, **kargs):
    if step < warmup_steps:
        return initial_learning_rate * step / warmup_steps
    else:
        return (initial_learning_rate - end_learning_rate) * \
               ((1 - (step - warmup_steps) / (global_steps - warmup_steps)) ** decay_exp) + \
               end_learning_rate


class MultipleOptimizer(object):
    """ Implement multiple optimizers needed for sparse adam """

    def __init__(self, op):
        """ ? """
        self.optimizers = op

    @property
    def param_groups(self):
        param_groups = []
        for optimizer in self.optimizers:
            param_groups.extend(optimizer.param_groups)
        return param_groups

    def zero_grad(self):
        """ ? """
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        """ ? """
        for op in self.optimizers:
            op.step()

    @property
    def state(self):
        """ ? """
        return {k: v for op in self.optimizers for k, v in op.state.items()}

    def state_dict(self):
        """ ? """
        return [op.state_dict() for op in self.optimizers]

    def load_state_dict(self, state_dicts):
        """ ? """
        assert len(state_dicts) == len(self.optimizers)
        for i in range(len(state_dicts)):
            self.optimizers[i].load_state_dict(state_dicts[i])


class OptimizerBase(object):
    """
    Controller class for optimization. Mostly a thin
    wrapper for `optim`, but also useful for implementing
    rate scheduling beyond what is currently available.
    Also implements necessary methods for training RNNs such
    as grad manipulations.
    """

    def __init__(self,
                 optimizer,
                 learning_rate,
                 learning_rate_decay_fn=None,
                 max_grad_norm=None):
        """Initializes the controller.

       Args:
         optimizer: A ``torch.optim.Optimizer`` instance.
         learning_rate: The initial learning rate.
         learning_rate_decay_fn: An optional callable taking the current step
           as argument and return a learning rate scaling factor.
         max_grad_norm: Clip gradients to this global norm.
        """
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._learning_rate_decay_fn = learning_rate_decay_fn
        self._max_grad_norm = max_grad_norm or 0
        self._training_step = 1
        self._decay_step = 1

    @classmethod
    def from_opt(cls, model, config, checkpoint=None):
        """Builds the optimizer from options.

        Args:
          cls: The ``Optimizer`` class to instantiate.
          model: The model to optimize.
          opt: The dict of user options.
          checkpoint: An optional checkpoint to load states from.

        Returns:
          An ``Optimizer`` instance.
        """
        optim_opt = config
        optim_state_dict = None

        if config["loads_ckpt"] and checkpoint is not None:
            optim = checkpoint['optim']
            ckpt_opt = checkpoint['opt']
            ckpt_state_dict = {}
            if isinstance(optim, Optimizer):  # Backward compatibility.
                ckpt_state_dict['training_step'] = optim._step + 1
                ckpt_state_dict['decay_step'] = optim._step + 1
                ckpt_state_dict['optimizer'] = optim.optimizer.state_dict()
            else:
                ckpt_state_dict = optim

            if config["reset_optim"] == 'none':
                # Load everything from the checkpoint.
                optim_opt = ckpt_opt
                optim_state_dict = ckpt_state_dict
            elif config["reset_optim"] == 'all':
                # Build everything from scratch.
                pass
            elif config["reset_optim"] == 'states':
                # Reset optimizer, keep options.
                optim_opt = ckpt_opt
                optim_state_dict = ckpt_state_dict
                del optim_state_dict['optimizer']
            elif config["reset_optim"] == 'keep_states':
                # Reset options, keep optimizer.
                optim_state_dict = ckpt_state_dict

        learning_rates = [
            optim_opt["learning_rate_bert"], 
            optim_opt["learning_rate_task"]
        ]
        decay_fn = [
            make_learning_rate_decay_fn(optim_opt['decay_method_bert'], 
                                        optim_opt['train_steps'],
                                        warmup_steps=optim_opt['warmup_steps'],
                                        decay_exp=optim_opt['decay_exp']), 
            make_learning_rate_decay_fn(optim_opt['decay_method_task'], 
                                        optim_opt['train_steps'],
                                        warmup_steps=optim_opt['warmup_steps'],
                                        decay_exp=optim_opt['decay_exp']), 
        ]
        optimizer = cls(
            build_torch_optimizer(model, optim_opt),
            learning_rates,
            learning_rate_decay_fn=decay_fn,
            max_grad_norm=optim_opt["max_grad_norm"])
        if optim_state_dict:
            optimizer.load_state_dict(optim_state_dict)
        return optimizer

    @property
    def training_step(self):
        """The current training step."""
        return self._training_step

    def learning_rate(self):
        """Returns the current learning rate."""
        if self._learning_rate_decay_fn is None:
            return self._learning_rate
        return [decay_fn(self._decay_step) * learning_rate \
            for decay_fn, learning_rate in \
            zip(self._learning_rate_decay_fn, self._learning_rate)]

    def state_dict(self):
        return {
            'training_step': self._training_step,
            'decay_step': self._decay_step,
            'optimizer': self._optimizer.state_dict()
        }

    def load_state_dict(self, state_dict):
        self._training_step = state_dict['training_step']
        # State can be partially restored.
        if 'decay_step' in state_dict:
            self._decay_step = state_dict['decay_step']
        if 'optimizer' in state_dict:
            self._optimizer.load_state_dict(state_dict['optimizer'])

    def zero_grad(self):
        """Zero the gradients of optimized parameters."""
        self._optimizer.zero_grad()

    def backward(self, loss):
        """Wrapper for backward pass. Some optimizer requires ownership of the
        backward pass."""
        loss.backward()

    def step(self):
        """Update the model parameters based on current gradients.

        Optionally, will employ gradient modification or update learning
        rate.
        """
        learning_rate = self.learning_rate()

        if isinstance(self._optimizer, MultipleOptimizer):
            optimizers = self._optimizer.optimizers
        else:
            optimizers = [self._optimizer]
        for lr, op in zip(learning_rate, optimizers):
            for group in op.param_groups:
                group['lr'] = lr
                if self._max_grad_norm > 0:
                    clip_grad_norm_(group['params'], self._max_grad_norm)
        self._optimizer.step()
        self._decay_step += 1
        self._training_step += 1


class AdamWeightDecay(Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamWeightDecay, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamWeightDecay, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                update = (exp_avg / bias_correction1) / denom
                # from tensorflow BERT:
                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want ot decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay'] != 0:
                    update.add_(p, alpha=group['weight_decay'])

                p.add_(update, alpha=-group['lr'])

        return loss        
