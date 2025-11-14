"""
Optimization utilities including custom optimizers and learning rate schedulers.
"""

import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    Cosine annealing with warmup and restarts.
    
    Similar to cosine annealing but with:
    - Linear warmup phase
    - Optional restarts (like SGDR)
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1
    ):
        """
        Initialize the scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            first_cycle_steps: Number of steps in first cycle
            cycle_mult: Cycle length multiplier
            max_lr: Maximum learning rate
            min_lr: Minimum learning rate
            warmup_steps: Number of warmup steps
            gamma: Learning rate decay factor
            last_epoch: Last epoch index
        """
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super().__init__(optimizer, last_epoch)
        
        # Initialize learning rate
        self.init_lr()
        
    def init_lr(self):
        """Initialize learning rates."""
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
            
    def get_lr(self):
        """Calculate learning rate."""
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                    for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) * 
                   (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) / 
                                (self.cur_cycle_steps - self.warmup_steps))) / 2
                   for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        """Step the scheduler."""
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult
                ) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * 
                                    (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * 
                                                     (self.cycle_mult ** n - 1) / 
                                                     (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** n
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    Linear warmup followed by cosine annealing.
    
    This is a common schedule used in transformer training.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        """
        Initialize the scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr: Minimum learning rate
            last_epoch: Last epoch index
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        """Calculate learning rate."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps 
                   for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return [self.min_lr + (base_lr - self.min_lr) * 0.5 * 
                   (1 + math.cos(math.pi * progress))
                   for base_lr in self.base_lrs]


class AdamWScale(Optimizer):
    """
    AdamW optimizer with learning rate scaling.
    
    Implements AdamW with optional learning rate scaling based on parameter dimensions,
    as used in some modern language models.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        scale_parameter: bool = True,
        relative_step: bool = False
    ):
        """
        Initialize AdamWScale optimizer.
        
        Args:
            params: Model parameters
            lr: Learning rate
            betas: Adam beta coefficients
            eps: Adam epsilon
            weight_decay: Weight decay coefficient
            scale_parameter: Whether to scale learning rate by parameter dimension
            relative_step: Whether to use relative step sizes
        """
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step
        )
        super().__init__(params, defaults)
        
    def _get_lr(self, param_group, param_scale):
        """Calculate learning rate for parameter group."""
        if param_group['relative_step']:
            min_step = 1e-6 * param_group['step']
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_group['step']))
            param_group['lr'] = rel_step_sz * param_scale
        return param_group['lr']
        
    def _get_options(self, param_group, param_shape):
        """Get optimizer options for parameter."""
        factored = len(param_shape) >= 2
        use_first_moment = param_group['betas'][0] > 0
        return factored, use_first_moment
        
    def _rms(self, tensor):
        """Calculate RMS of tensor."""
        return tensor.norm(2) / (tensor.numel() ** 0.5)
        
    def step(self, closure=None):
        """Perform optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamWScale does not support sparse gradients')
                    
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Calculate step size with parameter scaling
                param_scale = 1
                if group['scale_parameter']:
                    param_scale = 1 / math.sqrt(p.data.numel())
                    
                step_size = self._get_lr(group, param_scale)
                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                
                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
                    
                # Update parameters
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
        return loss


def get_optimizer(
    model,
    optimizer_type: str = 'adamw',
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    **kwargs
) -> Optimizer:
    """
    Get optimizer for model.
    
    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer ('adamw', 'adam', 'sgd', 'adamw_scale')
        learning_rate: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments
        
    Returns:
        Configured optimizer
    """
    # Separate parameters with and without weight decay
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0
        }
    ]
    
    if optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            **kwargs
        )
    elif optimizer_type.lower() == 'adam':
        return torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=learning_rate,
            **kwargs
        )
    elif optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(
            optimizer_grouped_parameters,
            lr=learning_rate,
            momentum=kwargs.get('momentum', 0.9),
            **kwargs
        )
    elif optimizer_type.lower() == 'adamw_scale':
        return AdamWScale(
            optimizer_grouped_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def get_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = 'cosine',
    num_training_steps: int = 1000,
    num_warmup_steps: int = 100,
    **kwargs
) -> _LRScheduler:
    """
    Get learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler
        num_training_steps: Total training steps
        num_warmup_steps: Warmup steps
        **kwargs: Additional scheduler arguments
        
    Returns:
        Configured scheduler
    """
    if scheduler_type.lower() == 'cosine':
        return LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_steps=num_warmup_steps,
            total_steps=num_training_steps,
            **kwargs
        )
    elif scheduler_type.lower() == 'cosine_restarts':
        return CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=num_training_steps,
            warmup_steps=num_warmup_steps,
            **kwargs
        )
    elif scheduler_type.lower() == 'linear':
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=kwargs.get('min_lr', 0.0),
            total_iters=num_training_steps - num_warmup_steps
        )
    elif scheduler_type.lower() == 'constant':
        return torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=num_training_steps
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
